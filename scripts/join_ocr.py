"""Fold OCR output into the frame and clip manifests.

Reads the per-lot JSONL the Kaggle OCR job writes, keyed on
`(video_id, keyframe_n)` — the keyframe identity, never the frame index — and
adds `ocr_text` / `ocr_regions` to both manifests.

⚠ DO NOT FILTER BY CONFIDENCE

The obvious cleanup is to drop regions the recogniser was unsure about. On this
corpus that deletes the single most valuable thing OCR found. Real regions,
verbatim, with their reported scores:

    'Tam DUnG LuU Thong'                 0.15
    'doi Voi Xe 3 BaNH TRO LeN'          0.11
    'NGuoi Dan Di Lai CHu Y Quan Sat'    0.20

Those are news tickers: "Tạm dừng lưu thông", "Đối với xe 3 bánh trở lên",
"Người dân đi lại chú ý quan sát". EasyOCR scores them low because they are set
in all-caps without diacritics, not because it read them wrong. A 0.3 cutoff
would throw away 29.2% of all recognised characters, weighted towards exactly
the on-screen headlines the job was run to capture.

They survive retrieval because `app.features.sparse` folds diacritics: the
stored "Tam DUnG LuU Thong" and a typed "tạm dừng lưu thông" both reduce to
"tam dung luu thong". Verified — 4 of 4 tokens overlap.

Length is the honest signal instead. Regions of 1–2 characters are 16.3% of
regions but 3.3% of characters, and are noise like 'IA', '1n', 'IH' that can
collide with real short query words. Regions of 3–4 characters are mixed
('giây' at 0.99 next to 'Hhd' at 0.28), so those alone are held to a score.
Anything 5 characters or longer is kept whatever its score.

SHOT-LEVEL, NOT FRAME-LEVEL

The job ran on the first and last keyframe of each shot, because the lower
third scrolls and a shot's opening and closing frames carry different halves of
one headline. Their text is unioned onto every keyframe of the shot, matching
how `asr_text` already works: `dedupe.dedupe_by_shot` collapses a shot to one
hit before results leave the engine, so shot granularity is what the ranking
actually sees.

    python scripts/join_ocr.py --ocr data/ocr_raw/ocr \\
        --frames data/frames.parquet --clips data/clips.parquet
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Below this a region is noise, whatever the recogniser claims.
MIN_REGION_CHARS = 3
# 3-4 character regions are the mixed band: real words like 'giây' sit beside
# junk like 'Hhd'. Only here does the score get a vote.
SHORT_REGION_CHARS = 4
SHORT_REGION_MIN_SCORE = 0.5


class JoinError(RuntimeError):
    pass


def keep_region(text: str, score: float) -> bool:
    """See the module docstring: length decides, score only breaks the tie."""
    stripped = text.strip()
    if len(stripped) < MIN_REGION_CHARS:
        return False
    if len(stripped) <= SHORT_REGION_CHARS:
        return score >= SHORT_REGION_MIN_SCORE
    return True


def load_ocr(directory: Path) -> dict[tuple[str, int], list[str]]:
    """Kept regions per OCR'd keyframe, in the order the recogniser found them."""
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        raise JoinError(f"no .jsonl under {directory}")

    per_frame: dict[tuple[str, int], list[str]] = {}
    for path in files:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                kept = [
                    text.strip()
                    for text, score in zip(row["texts"], row["scores"])
                    if keep_region(text, score)
                ]
                per_frame[(row["video_id"], int(row["keyframe_n"]))] = kept
    return per_frame


def shot_text(
    frames: pa.Table, per_frame: dict[tuple[str, int], list[str]]
) -> tuple[dict[tuple[str, int], str], dict[tuple[str, int], int]]:
    """Union each shot's OCR'd keyframes into one deduplicated string."""
    seen: dict[tuple[str, int], list[str]] = defaultdict(list)
    for video_id, shot_id, keyframe_n in zip(
        frames.column("video_id").to_pylist(),
        frames.column("shot_id").to_pylist(),
        frames.column("keyframe_n").to_pylist(),
    ):
        regions = per_frame.get((video_id, int(keyframe_n)))
        if not regions:
            continue
        bucket = seen[(video_id, int(shot_id))]
        for region in regions:
            # The scroll repeats words between the opening and closing frame;
            # storing them twice only inflates the lexical term counts.
            if region not in bucket:
                bucket.append(region)

    texts = {key: " ".join(regions) for key, regions in seen.items()}
    counts = {key: len(regions) for key, regions in seen.items()}
    return texts, counts


def with_ocr(
    table: pa.Table,
    texts: dict[tuple[str, int], str],
    counts: dict[tuple[str, int], int],
    with_counts: bool,
) -> pa.Table:
    keys = list(
        zip(
            table.column("video_id").to_pylist(),
            [int(s) for s in table.column("shot_id").to_pylist()],
        )
    )
    columns = {"ocr_text": pa.array([texts.get(k, "") for k in keys], pa.string())}
    if with_counts:
        columns["ocr_regions"] = pa.array(
            [counts.get(k, 0) for k in keys], pa.int32()
        )

    for name, column in columns.items():
        if name in table.column_names:
            table = table.drop_columns([name])
        table = table.append_column(name, column)
    return table


def join(args: argparse.Namespace) -> int:
    per_frame = load_ocr(args.ocr)
    ocr_frames = sum(1 for regions in per_frame.values() if regions)

    frames = pq.read_table(args.frames)
    texts, counts = shot_text(frames, per_frame)

    frames = with_ocr(frames, texts, counts, with_counts=True)
    pq.write_table(frames, args.frames)

    clips = pq.read_table(args.clips)
    clips = with_ocr(clips, texts, counts, with_counts=False)
    pq.write_table(clips, args.clips)

    total_frames = frames.num_rows
    total_clips = clips.num_rows
    frames_with = sum(1 for t in frames.column("ocr_text").to_pylist() if t)
    clips_with = sum(1 for t in clips.column("ocr_text").to_pylist() if t)
    kept_regions = sum(len(row) for row in per_frame.values())

    print(f"OCR'd keyframes    : {len(per_frame):,}")
    print(f"  with kept text   : {ocr_frames:,} ({ocr_frames / len(per_frame):.1%})")
    print(f"  regions kept     : {kept_regions:,}")
    print(f"shots with OCR     : {len(texts):,} / {total_clips:,}")
    print(f"frames with OCR    : {frames_with:,} / {total_frames:,} "
          f"({frames_with / total_frames:.1%})  -> {args.frames}")
    print(f"clips with OCR     : {clips_with:,} / {total_clips:,}  -> {args.clips}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr", type=Path, default=Path("data/ocr_raw/ocr"))
    parser.add_argument("--frames", type=Path, default=Path("data/frames.parquet"))
    parser.add_argument("--clips", type=Path, default=Path("data/clips.parquet"))
    args = parser.parse_args(argv)
    try:
        return join(args)
    except JoinError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
