"""Carry this repo's enrichment onto a differently-sampled frame manifest.

The team server samples 3 keyframes per shot; this repo follows the
organiser's `map-keyframes`, roughly one per second. The two therefore
disagree about what `keyframe_n` means, and joining on it — the obvious
thing — silently gives every frame another frame's text. Nothing raises;
the index just answers wrong.

WHAT IS JOINED ON INSTEAD

`(video_id, shot_id)` plus time. For each target frame, the source keyframe
taken is the one nearest in `pts_sec` **that falls inside the target frame's
own shot**. A shot is the unit where borrowing is defensible: within one the
camera has not cut, so the ticker, the caption and the speech all still
describe what is on screen. Across a boundary none of that holds, and the
nearest keyframe in raw time is very often across a boundary — p90 of the
unconstrained time gap on this corpus is 1.72s, and half the shots are
shorter than 2.7s.

MEASURED ON THE REAL MANIFESTS

    target frames                       289,881
    source keyframe in the same shot    259,131   89.4%
    -> ocr_text or ocr_text_vlm         237,311   81.9% of all frames
    -> caption_vi                       251,004   86.6%
    -> asr_text                         247,669   85.4%

The 10.6% with no match are not random: they are short shots, median 2.0s
and none longer than 6.9s, that the source sampling never landed a keyframe
in. They are left empty here and are the work list for a recogniser run,
which is why `--report-missing` writes them out.

    python scripts/join_server_frames.py \\
        --source data/frames-enriched.parquet \\
        --target ../aic2026-be/data/manifests-v2/frames.parquet \\
        --out data/frames-joined.parquet \\
        --report-missing data/frames-missing-text.parquet
"""

import argparse
import bisect
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

# Columns carried across. `objects` and the video-level fields are left
# behind: they are per-video or per-shot already and belong to a different
# join, not this one.
CARRIED = (
    "ocr_text",
    "ocr_regions",
    "ocr_text_vlm",
    "caption_vi",
    "asr_text",
    "asr_text_corrected",
)

TARGET_KEYS = ("video_id", "shot_id", "pts_sec", "shot_start_sec", "shot_end_sec")


class JoinError(RuntimeError):
    pass


def source_index(table: pa.Table) -> dict[str, list[tuple[float, int]]]:
    """Per video, this repo's keyframes sorted by timestamp."""
    videos = table.column("video_id").to_pylist()
    stamps = table.column("pts_sec").to_pylist()

    index: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for row, (video_id, pts) in enumerate(zip(videos, stamps)):
        index[video_id].append((float(pts), row))
    for rows in index.values():
        rows.sort()
    return index


def match_row(
    rows: list[tuple[float, int]],
    pts: float,
    shot_start: float,
    shot_end: float,
) -> int | None:
    """The source row nearest `pts` within [shot_start, shot_end], or None.

    Bisecting the shot's own window first is what enforces the constraint:
    anything outside it is never a candidate, however close in time it is.
    """
    stamps = [stamp for stamp, _ in rows]
    lo = bisect_left(stamps, shot_start)
    hi = bisect_right(stamps, shot_end)
    if lo >= hi:
        return None
    best_row, best_gap = None, None
    for stamp, row in rows[lo:hi]:
        gap = abs(stamp - pts)
        if best_gap is None or gap < best_gap:
            best_row, best_gap = row, gap
    return best_row


def join(args: argparse.Namespace) -> int:
    source = pq.read_table(args.source)
    target = pq.read_table(args.target)

    for name in CARRIED:
        if name not in source.column_names:
            raise JoinError(f"source has no column '{name}'")
    for name in TARGET_KEYS:
        if name not in target.column_names:
            raise JoinError(f"target has no column '{name}'")

    index = source_index(source)
    carried = {name: source.column(name).to_pylist() for name in CARRIED}

    videos = target.column("video_id").to_pylist()
    stamps = target.column("pts_sec").to_pylist()
    starts = target.column("shot_start_sec").to_pylist()
    ends = target.column("shot_end_sec").to_pylist()

    matches: list[int | None] = []
    gaps: list[float] = []
    for video_id, pts, start, end in zip(videos, stamps, starts, ends):
        rows = index.get(video_id)
        row = None if not rows else match_row(rows, float(pts), float(start), float(end))
        matches.append(row)
        if row is not None:
            gaps.append(abs(source.column("pts_sec")[row].as_py() - float(pts)))

    out = target
    for name in CARRIED:
        values = carried[name]
        column = [
            (values[row] if row is not None else ("" if isinstance(values[0], str) else None))
            for row in matches
        ]
        if name in out.column_names:
            out = out.drop_columns([name])
        out = out.append_column(name, pa.array(column))

    pq.write_table(out, args.out)

    matched = sum(1 for row in matches if row is not None)
    total = len(matches)
    print(f"target frames    : {total:,}")
    print(f"matched in shot  : {matched:,} ({matched / total:.1%})")
    gaps.sort()
    if gaps:
        print(
            f"time gap         : p50 {gaps[len(gaps) // 2]:.2f}s  "
            f"p90 {gaps[int(len(gaps) * 0.9)]:.2f}s  max {gaps[-1]:.2f}s"
        )
    for name in CARRIED:
        filled = sum(1 for value in out.column(name).to_pylist() if value)
        print(f"  {name:<20}: {filled:,} ({filled / total:.1%})")
    print(f"-> {args.out}")

    if args.report_missing:
        keep = [row is None for row in matches]
        missing = target.filter(pa.array(keep))
        pq.write_table(missing, args.report_missing)
        print(f"-> {args.report_missing} ({missing.num_rows:,} frames need a recogniser)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="this repo's enriched frames")
    parser.add_argument("--target", required=True, help="the manifest to carry it onto")
    parser.add_argument("--out", required=True)
    parser.add_argument("--report-missing", default=None)
    args = parser.parse_args(argv)
    try:
        return join(args)
    except JoinError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
