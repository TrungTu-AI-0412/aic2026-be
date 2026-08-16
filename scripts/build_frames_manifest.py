"""Join the organiser's artefacts into `frames.parquet` and `clips.parquet`.

`app.ingestion.batch_builder` cannot be used for this dataset. It reads
`original_frame_id` out of the keyframe *filename*, which is correct for
keyframes this repo sampled itself but wrong for the organiser's: their files
are named after column `n` of `map-keyframes` — the keyframe's ordinal — while
`original_frame_id` is column `frame_idx`. Ingesting by filename would put `n`
on every submission.

So `map-keyframes/<video_id>.csv` is the authority here, and the two columns
stay strictly separated:

    n         -> keyframe_n, the identity, and the keyframe/objects filename
    frame_idx -> original_frame_id, the value a submission reports

Shot boundaries come from a shot-detection pass verified by
`scripts/verify_shots.py`; nothing here re-derives them.

Outputs satisfy `app.ingestion.manifest`'s required columns and carry extra
enrichment columns alongside. Pydantic ignores unknown fields, so the existing
ingestion path reads these files unchanged while the extra columns stay
available for payload work.
"""

import argparse
import ast
import bisect
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

FRAME_SCHEMA = pa.schema(
    [
        # Required by app.ingestion.manifest.KEYFRAME_ARROW_SCHEMA.
        ("video_id", pa.string()),
        ("shot_id", pa.int32()),
        ("keyframe_n", pa.int32()),
        ("original_frame_id", pa.int64()),
        ("pts_sec", pa.float64()),
        ("path", pa.string()),
        # Enrichment.
        ("objects", pa.list_(pa.string())),
        ("object_counts", pa.map_(pa.string(), pa.int32())),
        ("asr_text", pa.string()),
        ("asr_text_corrected", pa.string()),
        ("asr_entities", pa.list_(pa.string())),
        ("title", pa.string()),
        ("author", pa.string()),
        ("channel_id", pa.string()),
        ("publish_date", pa.string()),
        ("keywords", pa.list_(pa.string())),
        ("watch_url", pa.string()),
    ]
)

CLIP_SCHEMA = pa.schema(
    [
        ("video_id", pa.string()),
        ("shot_id", pa.int32()),
        ("start_frame", pa.int64()),
        ("end_frame", pa.int64()),
        ("start_sec", pa.float64()),
        ("end_sec", pa.float64()),
        ("path", pa.string()),
        ("keyframe_count", pa.int32()),
        ("asr_text", pa.string()),
        ("asr_text_corrected", pa.string()),
        ("asr_entities", pa.list_(pa.string())),
    ]
)


# Read by app.submissions.service to reject a submission row whose frame index
# could never exist. Deliberately not named videos.parquet: that name belongs
# to the probe manifest, whose schema this does not share.
VIDEO_SCHEMA = pa.schema(
    [
        ("video_id", pa.string()),
        ("fps", pa.float64()),
        ("length_sec", pa.int64()),
        ("frame_upper_bound", pa.int64()),
    ]
)


class BuildError(RuntimeError):
    pass


def frame_upper_bound(length_sec: int, fps: float, observed_max: int) -> int:
    """One past the last frame index a submission may name for this video.

    `length_sec` comes from media-info, where it is whole seconds, so
    `length_sec * fps` lands slightly short of the real end. In 43 of the 873
    videos here the last keyframe already sits past that product, by up to 10
    frames — bounding on the product alone would reject those keyframes even
    though the organiser sampled them.

    So the bound is the larger of the two pieces of evidence. Erring long
    admits a handful of frames that do not exist; erring short throws away a
    correct answer, and only one of those is recoverable.
    """
    return max(int(length_sec * fps), observed_max + 1)


def load_keyframes(path: Path) -> tuple[list[tuple[int, int, float]], float]:
    """Return [(n, frame_idx, pts_time)] plus the video's frame rate."""
    rows: list[tuple[int, int, float]] = []
    rates: list[float] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for record in csv.DictReader(handle):
            rows.append(
                (
                    int(float(record["n"])),
                    int(float(record["frame_idx"])),
                    float(record["pts_time"]),
                )
            )
            rates.append(float(record["fps"]))
    if not rows:
        raise BuildError("empty map-keyframes CSV")
    rows.sort()
    return rows, statistics.median(rates)


def assign_shot(starts: list[int], shots: list[tuple[int, int]], frame: int) -> int:
    """Index of the shot covering `frame`, snapping to the nearest on a miss.

    Shot detectors leave the transition frame itself unassigned, so a keyframe
    landing exactly on a cut falls in a one- or two-frame hole between shots;
    a few videos also start their first shot at frame 1, leaving frame 0
    outside. Both are boundary artefacts, not evidence of misalignment
    (`verify_shots.py` measures how many there are), and dropping the keyframe
    would lose a real image. Snapping to the adjacent shot is off by at most a
    couple of frames.
    """
    index = bisect.bisect_right(starts, frame) - 1
    if index < 0:
        return 0
    if frame <= shots[index][1]:
        return index
    return min(index + 1, len(shots) - 1)


def load_objects(path: Path, score_min: float) -> tuple[list[str], dict[str, int]]:
    """Distinct detected entities above `score_min`, plus their counts.

    Detections come sorted by descending score, so the scan stops at the first
    one below the threshold.
    """
    record = json.loads(path.read_text(encoding="utf-8"))
    entities = record.get("detection_class_entities") or []
    scores = record.get("detection_scores") or []

    counts: Counter[str] = Counter()
    for entity, score in zip(entities, scores):
        if float(score) < score_min:
            break
        counts[entity] += 1
    return sorted(counts), dict(counts)


def load_transcript_spans(path: Path) -> list[tuple[float, float, str]]:
    """Segments as (start, end, text) with *non-overlapping* spans.

    YouTube caption timings are rolling display windows: measured on this
    dataset, 98% of adjacent segments overlap in time even though their text
    does not repeat. Using `start + duration` therefore makes one moment match
    two or three segments and duplicates the words, which skews term
    frequencies in a sparse index. The next segment's start is the real end.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload.get("segments") or []
    spans: list[tuple[float, float, str]] = []
    for index, segment in enumerate(segments):
        start = float(segment["start"])
        if index + 1 < len(segments):
            end = float(segments[index + 1]["start"])
        else:
            end = start + float(segment.get("duration", 0.0))
        spans.append((start, max(end, start), segment["text"]))
    return spans


def load_asr_csv(path: Path) -> list[tuple[float, float, str, str, list[str]]]:
    """Read one `<video_id>_segments_enriched.csv` from the zzzlazy/aic-asr set.

    Returns (start, end, text, text_corrected, entities). Unlike YouTube caption
    windows these spans do not overlap, so `end` is trustworthy and no trimming
    is needed.

    Both text columns are kept. `text_corrected` is an LLM cleanup that adds
    punctuation and capitalisation but does *not* fix misheard content words —
    it rewrites "sục lúng" into fluent prose that is still wrong. Fluent-but-
    wrong text is more misleading than obviously-garbled text, so the raw
    column stays available for anything that needs to judge reliability.

    Rows are selected on having text, not on the `has_speech` flag: 162
    segments in this corpus carry a transcript while the flag reads False, and
    filtering on the flag threw all of them away. A segment with words in it
    has words in it whatever the voice-activity detector concluded.
    """
    out: list[tuple[float, float, str, str, list[str]]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                start, end = float(row["start"]), float(row["end"])
            except (KeyError, TypeError, ValueError):
                continue

            text = (row.get("text") or "").strip()
            corrected = (row.get("text_corrected") or "").strip()
            if corrected == "nan":
                corrected = ""

            entities: list[str] = []
            raw = (row.get("entities") or "").strip()
            if raw.startswith("{"):
                try:
                    parsed = ast.literal_eval(raw)
                    entities = [
                        str(v).strip()
                        for values in parsed.values()
                        for v in values
                        if str(v).strip()
                    ]
                except (ValueError, SyntaxError):
                    entities = []

            if text or corrected:
                out.append((start, max(end, start), text, corrected, entities))
    out.sort()
    return out


def entities_between(
    asr: list[tuple[float, float, str, str, list[str]]], lo: float, hi: float
) -> list[str]:
    """Distinct entities from every segment overlapping [lo, hi)."""
    found: dict[str, None] = {}
    for start, end, _, _, entities in asr:
        if start >= hi:
            break
        if end > lo:
            for entity in entities:
                found.setdefault(entity, None)
    return list(found)


def text_between(spans: list[tuple[float, float, str]], lo: float, hi: float) -> str:
    if not spans:
        return ""
    starts = [s for s, _, _ in spans]
    index = max(0, bisect.bisect_right(starts, lo) - 1)
    parts = []
    for start, end, text in spans[index:]:
        if start >= hi:
            break
        if end > lo:
            parts.append(text)
    return " ".join(parts)


def build(args: argparse.Namespace) -> int:
    keyframe_files = {p.stem: p for p in args.map_keyframes.glob("*.csv")}
    if not keyframe_files:
        raise BuildError(f"no map-keyframes CSVs under {args.map_keyframes}")

    shot_files = {
        f"{p.parent.name}_{p.stem}": p
        for p in args.shots.rglob("*.json")
        if p.parent.name[:1] in {"L", "K"}
    }
    media_files = {p.stem: p for p in args.media_info.glob("*.json")}

    frame_rows: list[dict] = []
    clip_rows: list[dict] = []
    video_rows: list[dict] = []
    skipped_no_shots: list[str] = []
    missing_objects = 0
    videos_with_asr = 0

    for video_id in sorted(keyframe_files):
        shot_path = shot_files.get(video_id)
        if shot_path is None:
            skipped_no_shots.append(video_id)
            continue

        keyframes, fps = load_keyframes(keyframe_files[video_id])
        shots = [(int(a), int(b)) for a, b in json.loads(shot_path.read_text())]
        shots.sort()
        starts = [s for s, _ in shots]

        media: dict = {}
        if video_id in media_files:
            try:
                media = json.loads(media_files[video_id].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                media = {}

        # Two ASR sources with the same shape. The enriched CSV set covers all
        # 873 videos and carries corrected text plus entities, so it wins when
        # both are present; the scraped YouTube captions stay as a fallback for
        # anything it misses.
        spans: list[tuple[float, float, str]] = []
        spans_corrected: list[tuple[float, float, str]] = []
        asr_rows: list[tuple[float, float, str, str, list[str]]] = []

        asr_path = (
            args.asr_csv / f"{video_id}_segments_enriched.csv"
            if args.asr_csv
            else None
        )
        if asr_path is not None and asr_path.is_file():
            asr_rows = load_asr_csv(asr_path)
            spans = [(s, e, t) for s, e, t, _, _ in asr_rows]
            spans_corrected = [(s, e, c) for s, e, _, c, _ in asr_rows]
        elif args.transcripts:
            transcript_path = args.transcripts / f"{video_id}.json"
            if transcript_path.is_file():
                spans = load_transcript_spans(transcript_path)
        if spans:
            videos_with_asr += 1

        shot_text = [
            text_between(spans, start / fps, (end + 1) / fps) for start, end in shots
        ]
        shot_text_corrected = [
            text_between(spans_corrected, start / fps, (end + 1) / fps)
            for start, end in shots
        ]
        shot_entities = [
            entities_between(asr_rows, start / fps, (end + 1) / fps)
            for start, end in shots
        ]
        shot_keyframes = Counter()

        for n, frame_idx, pts in keyframes:
            shot_id = assign_shot(starts, shots, frame_idx)
            shot_keyframes[shot_id] += 1

            objects: list[str] = []
            counts: dict[str, int] = {}
            if args.objects:
                object_path = args.objects / video_id / f"{n:03d}.json"
                if object_path.is_file():
                    objects, counts = load_objects(object_path, args.object_score_min)
                else:
                    missing_objects += 1

            frame_rows.append(
                {
                    "video_id": video_id,
                    "shot_id": shot_id,
                    "keyframe_n": n,
                    "original_frame_id": frame_idx,
                    "pts_sec": pts,
                    "path": f"{args.keyframes_root}/{video_id}/{n:03d}.jpg",
                    "objects": objects,
                    "object_counts": counts,
                    "asr_text": shot_text[shot_id],
                    "asr_text_corrected": shot_text_corrected[shot_id],
                    "asr_entities": shot_entities[shot_id],
                    "title": media.get("title") or "",
                    "author": media.get("author") or "",
                    "channel_id": media.get("channel_id") or "",
                    "publish_date": media.get("publish_date") or "",
                    "keywords": [str(k) for k in (media.get("keywords") or [])],
                    "watch_url": media.get("watch_url") or "",
                }
            )

        length_sec = int(media.get("length") or 0)
        observed_max = max(
            max((frame_idx for _, frame_idx, _ in keyframes), default=0),
            max((end for _, end in shots), default=0),
        )
        video_rows.append(
            {
                "video_id": video_id,
                "fps": fps,
                "length_sec": length_sec,
                "frame_upper_bound": frame_upper_bound(length_sec, fps, observed_max),
            }
        )

        for shot_id, (start, end) in enumerate(shots):
            clip_rows.append(
                {
                    "video_id": video_id,
                    "shot_id": shot_id,
                    "start_frame": start,
                    "end_frame": end,
                    "start_sec": start / fps,
                    "end_sec": (end + 1) / fps,
                    "path": f"{args.videos_root}/{video_id}.mp4",
                    "keyframe_count": shot_keyframes.get(shot_id, 0),
                    "asr_text": shot_text[shot_id],
                    "asr_text_corrected": shot_text_corrected[shot_id],
                    "asr_entities": shot_entities[shot_id],
                }
            )

    if not frame_rows:
        raise BuildError("no rows produced")

    args.out_frames.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(frame_rows, schema=FRAME_SCHEMA), args.out_frames
    )
    pq.write_table(pa.Table.from_pylist(clip_rows, schema=CLIP_SCHEMA), args.out_clips)
    pq.write_table(
        pa.Table.from_pylist(video_rows, schema=VIDEO_SCHEMA), args.out_videos
    )

    videos = len({r["video_id"] for r in frame_rows})
    empty_shots = sum(1 for r in clip_rows if r["keyframe_count"] == 0)
    with_objects = sum(1 for r in frame_rows if r["objects"])
    with_asr = sum(1 for r in frame_rows if r["asr_text"])

    print(f"videos            : {videos}")
    print(f"frames            : {len(frame_rows):,}  -> {args.out_frames}")
    print(f"shots             : {len(clip_rows):,}  -> {args.out_clips}")
    print(f"video bounds      : {len(video_rows):,}  -> {args.out_videos}")
    print(f"  shots with no keyframe: {empty_shots:,}")
    print(f"frames with objects: {with_objects:,} ({with_objects / len(frame_rows):.1%})")
    print(f"frames with ASR    : {with_asr:,} ({with_asr / len(frame_rows):.1%})")
    print(f"videos with ASR    : {videos_with_asr}")
    if missing_objects:
        print(f"missing object files: {missing_objects:,}")
    if skipped_no_shots:
        print(f"skipped, no shot list: {len(skipped_no_shots)} {skipped_no_shots[:5]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-keyframes", type=Path, required=True)
    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument("--media-info", type=Path, required=True)
    parser.add_argument("--objects", type=Path)
    parser.add_argument(
        "--asr-csv",
        type=Path,
        help=(
            "directory of <video_id>_segments_enriched.csv "
            "(zzzlazy/aic-asr, Apache-2.0); preferred over --transcripts"
        ),
    )
    parser.add_argument(
        "--transcripts",
        type=Path,
        help="fallback: scraped YouTube captions from scrape_transcripts.py",
    )
    parser.add_argument(
        "--keyframes-root",
        default="keyframes",
        help="prefix for the keyframe `path` column",
    )
    parser.add_argument(
        "--videos-root", default="videos", help="prefix for the clip `path` column"
    )
    parser.add_argument("--object-score-min", type=float, default=0.3)
    parser.add_argument("--out-frames", type=Path, default=Path("frames.parquet"))
    parser.add_argument("--out-clips", type=Path, default=Path("clips.parquet"))
    parser.add_argument(
        "--out-videos", type=Path, default=Path("video_bounds.parquet")
    )
    args = parser.parse_args(argv)

    try:
        return build(args)
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
