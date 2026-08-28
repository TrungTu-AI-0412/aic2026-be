"""Re-express an eval set's answers in another manifest's shot numbering.

`eval_set.jsonl` records each answer as `answer_shot_ids`, and those ids come
from this repo's shot detection — 97,811 shots. The team server ran its own
and got 96,627. The two disagree about where cuts fall and, more importantly,
about what any given integer means, so scoring a run against the server's
collection with these ids compares numbers that were never the same quantity.
Nothing errors. Recall just comes out wrong, and plausibly so.

WHAT SURVIVES THE TRANSLATION

Time does. `answer_frame_ids` are indexes into the source video, which both
manifests describe, so each answer frame can be placed on the timeline and
the target shot that contains that instant looked up. A frame is not
ambiguous the way a shot id is: it is one moment in one video.

One source shot can map to several target shots and vice versa, because the
two detectors disagree about cuts. That is expected and is why the output is
a set. An answer that lands in no target shot at all is dropped and counted,
not silently discarded — if that number is large the two manifests are not
describing the same videos and nothing downstream should be believed.

    python scripts/remap_eval_shots.py \\
        --eval-set ../data/eval_set.jsonl \\
        --source-frames data/frames-enriched.parquet \\
        --target-clips ../aic2026-be/data/manifests-v2/clips.parquet \\
        --out data/eval_set_server.jsonl
"""

import argparse
import bisect
import json
import sys
from collections import defaultdict

import pyarrow.parquet as pq


class RemapError(RuntimeError):
    pass


def frame_times(path: str) -> dict[tuple[str, int], float]:
    """`(video_id, original_frame_id)` -> the instant it was sampled at.

    Taken from the source manifest rather than computed from fps, because
    that is where the eval set's frame ids came from; recomputing would
    reintroduce the rounding that makes two keyframes share a frame index in
    192 of the 873 videos here.
    """
    table = pq.read_table(
        path, columns=["video_id", "original_frame_id", "pts_sec"]
    )
    return {
        (video, int(frame)): float(pts)
        for video, frame, pts in zip(
            table.column("video_id").to_pylist(),
            table.column("original_frame_id").to_pylist(),
            table.column("pts_sec").to_pylist(),
        )
    }


def shot_spans(path: str) -> dict[str, tuple[list[float], list[float], list[int]]]:
    """Per video, the target shots sorted by start time."""
    table = pq.read_table(
        path, columns=["video_id", "shot_id", "start_sec", "end_sec"]
    )
    grouped = defaultdict(list)
    for video, shot, start, end in zip(
        table.column("video_id").to_pylist(),
        table.column("shot_id").to_pylist(),
        table.column("start_sec").to_pylist(),
        table.column("end_sec").to_pylist(),
    ):
        grouped[video].append((float(start), float(end), int(shot)))

    spans = {}
    for video, entries in grouped.items():
        entries.sort()
        spans[video] = (
            [entry[0] for entry in entries],
            [entry[1] for entry in entries],
            [entry[2] for entry in entries],
        )
    return spans


def shot_at(spans, video: str, when: float) -> int | None:
    """The target shot containing `when`, or the nearest if it falls in a gap.

    Shot boundaries from two detectors never line up to the millisecond, and
    an answer landing a few frames inside the neighbouring shot is still the
    same moment of the same video. Falling back to the nearest start is what
    keeps a boundary disagreement from being scored as a miss.
    """
    entry = spans.get(video)
    if entry is None:
        return None
    starts, ends, ids = entry

    index = bisect.bisect_right(starts, when) - 1
    if 0 <= index < len(ids) and when <= ends[index]:
        return ids[index]

    best, best_gap = None, None
    for candidate in (index, index + 1):
        if 0 <= candidate < len(ids):
            gap = min(abs(when - starts[candidate]), abs(when - ends[candidate]))
            if best_gap is None or gap < best_gap:
                best, best_gap = ids[candidate], gap
    return best


def remap(args: argparse.Namespace) -> int:
    times = frame_times(args.source_frames)
    spans = shot_spans(args.target_clips)

    with open(args.eval_set, encoding="utf-8") as handle:
        queries = [json.loads(line) for line in handle if line.strip()]
    if not queries:
        raise RemapError(f"{args.eval_set} is empty")

    written = dropped = no_time = 0
    sizes = []
    with open(args.out, "w", encoding="utf-8") as out:
        for query in queries:
            video = query["video_id"]
            frames = query.get("answer_frame_ids") or []
            if isinstance(frames, str):
                frames = json.loads(frames)

            shots = set()
            for frame in frames:
                when = times.get((video, int(frame)))
                if when is None:
                    no_time += 1
                    continue
                shot = shot_at(spans, video, when)
                if shot is not None:
                    shots.add(shot)

            if not shots:
                dropped += 1
                continue

            remapped = dict(query)
            remapped["answer_shot_ids"] = sorted(shots)
            remapped["source_shot_ids"] = query.get("answer_shot_ids")
            out.write(json.dumps(remapped, ensure_ascii=False) + "\n")
            written += 1
            sizes.append(len(shots))

    print(f"queries in      : {len(queries):,}")
    print(f"remapped        : {written:,}")
    print(f"dropped         : {dropped:,}")
    print(f"answer frames with no timestamp: {no_time:,}")
    if sizes:
        sizes.sort()
        print(
            f"target shots per query: median {sizes[len(sizes) // 2]}, "
            f"max {sizes[-1]}"
        )
    print(f"-> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--source-frames", required=True)
    parser.add_argument("--target-clips", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        return remap(args)
    except RemapError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
