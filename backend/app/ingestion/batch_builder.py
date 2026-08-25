"""Build manifests from artifacts produced outside this pipeline.

Two entry points exist because shot detection may run elsewhere:

- `keyframes`: scan a directory of already-extracted keyframes, e.g. the ones
  shipped by the organisers.
- `shots`: import shot boundaries from a CSV, e.g. a TransNetV2 run on Colab.

Both need a frame rate to turn frame indexes into timestamps, so either a
probe manifest or an explicit `--fps` is required. The frame rate is never
guessed.
"""

import argparse
import bisect
import csv
import re
from collections import defaultdict
from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path

from app.ingestion.manifest import (
    CLIP_ARROW_SCHEMA,
    KEYFRAME_ARROW_SCHEMA,
    ClipManifestRow,
    KeyframeManifestRow,
    VideoManifestRow,
    iter_rows,
    iter_video_rows,
    write_rows,
)
from app.schemas.ingestions import IngestionEntity

FRAME_NAME_PATTERN = re.compile(
    r"^(?P<video_id>L\d{2}_V\d{3})_(?P<frame_id>\d+)\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)

SHOT_CSV_COLUMNS = {"video_id", "start_frame", "end_frame"}


class ShotIndex:
    """Maps a source frame index back to the shot that contains it."""

    def __init__(self, shots: Iterable[ClipManifestRow]) -> None:
        self._by_video: dict[str, list[ClipManifestRow]] = defaultdict(list)
        for shot in shots:
            self._by_video[shot.video_id].append(shot)

        self._starts: dict[str, list[int]] = {}
        for video_id, video_shots in self._by_video.items():
            video_shots.sort(key=lambda shot: shot.start_frame)
            self._starts[video_id] = [shot.start_frame for shot in video_shots]

    def find(self, video_id: str, frame_id: int) -> int | None:
        video_shots = self._by_video.get(video_id)
        if not video_shots:
            return None

        position = bisect.bisect_right(self._starts[video_id], frame_id) - 1
        if position < 0:
            return None

        shot = video_shots[position]
        if shot.start_frame <= frame_id <= shot.end_frame:
            return shot.shot_id
        return None


def parse_fps(value: str) -> Fraction:
    """Parse a frame rate as an exact fraction, e.g. '25' or '30000/1001'."""
    try:
        fps = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid fps '{value}'") from exc

    if fps <= 0:
        raise ValueError(f"fps must be positive, got '{value}'")
    return fps


def load_videos(
    videos_manifest: str | None, fps: str | None
) -> tuple[dict[str, VideoManifestRow], Fraction | None]:
    """Load the probe manifest once, plus an optional explicit frame rate."""
    if videos_manifest is None and fps is None:
        raise ValueError("either --videos-manifest or --fps is required")

    videos: dict[str, VideoManifestRow] = {}
    if videos_manifest is not None:
        videos = {row.video_id: row for row in iter_video_rows(videos_manifest)}

    return videos, parse_fps(fps) if fps is not None else None


def resolve_fps(
    video_id: str, videos: dict[str, VideoManifestRow], fallback: Fraction | None
) -> Fraction:
    video = videos.get(video_id)
    if video is not None:
        return video.require_constant_frame_rate()
    if fallback is not None:
        return fallback
    raise ValueError(
        f"no frame rate for '{video_id}': add it to the probe manifest or pass --fps"
    )


def scan_frames(source_dir: Path) -> list[dict]:
    rows = []
    for file_path in sorted(source_dir.rglob("*")):
        if not file_path.is_file():
            continue

        match = FRAME_NAME_PATTERN.match(file_path.name)
        if not match:
            continue

        rows.append(
            {
                "video_id": match["video_id"],
                "original_frame_id": int(match["frame_id"]),
                "path": str(file_path),
            }
        )

    return rows


def read_shot_csv(csv_path: str) -> list[ClipManifestRow]:
    """Read externally detected shot boundaries.

    Expects `video_id,start_frame,end_frame` with an inclusive end frame. An
    optional `shot_id` column is honoured; otherwise ids are assigned per
    video in start-frame order. Timestamps are filled in later, once a frame
    rate is known.
    """
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = SHOT_CSV_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"shot csv is missing columns: {sorted(missing)}")
        records = list(reader)

    by_video: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_video[record["video_id"]].append(record)

    shots: list[dict] = []
    for video_id, video_records in by_video.items():
        video_records.sort(key=lambda record: int(record["start_frame"]))
        for position, record in enumerate(video_records):
            declared = record.get("shot_id")
            shots.append(
                {
                    "video_id": video_id,
                    "shot_id": int(declared) if declared else position,
                    "start_frame": int(record["start_frame"]),
                    "end_frame": int(record["end_frame"]),
                }
            )

    return shots


def build_shot_manifest(
    csv_path: str,
    out_path: str,
    videos_manifest: str | None = None,
    fps: str | None = None,
) -> int:
    videos, fallback = load_videos(videos_manifest, fps)
    records = read_shot_csv(csv_path)
    if not records:
        raise ValueError(f"no shots found in '{csv_path}'")

    rows = []
    for record in records:
        video_id = record["video_id"]
        rate = resolve_fps(video_id, videos, fallback)
        video = videos.get(video_id)
        rows.append(
            ClipManifestRow(
                video_id=video_id,
                shot_id=record["shot_id"],
                start_frame=record["start_frame"],
                end_frame=record["end_frame"],
                start_sec=record["start_frame"] / rate,
                end_sec=record["end_frame"] / rate,
                # Without a probe manifest the source video location is
                # unknown; the id stands in until clip embedding needs it.
                path=video.path if video is not None else video_id,
            )
        )

    return write_rows(rows, out_path, CLIP_ARROW_SCHEMA)


def build_keyframe_manifest(
    source_dir: str,
    out_path: str,
    videos_manifest: str | None = None,
    fps: str | None = None,
    shots_manifest: str | None = None,
) -> int:
    source = Path(source_dir)
    if not source.is_dir():
        raise ValueError(f"source directory not found: {source_dir}")

    videos, fallback = load_videos(videos_manifest, fps)

    records = scan_frames(source)
    if not records:
        raise ValueError(f"no keyframe files found under '{source_dir}'")

    shot_index = _load_shot_index(shots_manifest)
    unmatched: list[str] = []
    fallback_shot_ids: dict[str, int] = defaultdict(int)
    keyframe_counts: dict[str, int] = defaultdict(int)

    rows = []
    for record in sorted(
        records, key=lambda item: (item["video_id"], item["original_frame_id"])
    ):
        video_id = record["video_id"]
        frame_id = record["original_frame_id"]
        keyframe_counts[video_id] += 1

        if shot_index is None:
            # No shot information available: give every keyframe its own shot
            # so that shot-level dedupe degrades to a no-op instead of
            # silently collapsing unrelated frames into one group.
            shot_id = fallback_shot_ids[video_id]
            fallback_shot_ids[video_id] += 1
        else:
            found = shot_index.find(video_id, frame_id)
            if found is None:
                unmatched.append(f"{video_id}:{frame_id}")
                continue
            shot_id = found

        rate = resolve_fps(video_id, videos, fallback)
        rows.append(
            KeyframeManifestRow(
                video_id=video_id,
                shot_id=shot_id,
                keyframe_n=keyframe_counts[video_id],
                original_frame_id=frame_id,
                pts_sec=frame_id / rate,
                path=record["path"],
            )
        )

    if unmatched:
        raise ValueError(
            f"{len(unmatched)} keyframes fall outside every shot "
            f"(first: {', '.join(unmatched[:5])})"
        )

    return write_rows(rows, out_path, KEYFRAME_ARROW_SCHEMA)


def _load_shot_index(shots_manifest: str | None) -> ShotIndex | None:
    if shots_manifest is None:
        return None

    return ShotIndex(iter_rows(shots_manifest, IngestionEntity.CLIPS))


def _add_frame_rate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--videos-manifest",
        help="videos.parquet from the probe step, used for per-video frame rates",
    )
    parser.add_argument(
        "--fps",
        help="explicit frame rate for videos absent from the probe manifest, "
        "e.g. '25' or '30000/1001'",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    keyframes = subparsers.add_parser(
        "keyframes", help="scan a directory of extracted keyframes"
    )
    keyframes.add_argument("--source", required=True, help="keyframe directory")
    keyframes.add_argument("--out", required=True, help="output keyframes .parquet")
    keyframes.add_argument(
        "--shots-manifest",
        help="shots.parquet used to attach shot_id; without it every keyframe "
        "becomes its own shot",
    )
    _add_frame_rate_args(keyframes)

    shots = subparsers.add_parser(
        "shots", help="import externally detected shot boundaries from CSV"
    )
    shots.add_argument(
        "--csv", required=True, help="CSV with video_id,start_frame,end_frame"
    )
    shots.add_argument("--out", required=True, help="output shots .parquet")
    _add_frame_rate_args(shots)

    args = parser.parse_args()

    if args.command == "keyframes":
        count = build_keyframe_manifest(
            args.source,
            args.out,
            videos_manifest=args.videos_manifest,
            fps=args.fps,
            shots_manifest=args.shots_manifest,
        )
    else:
        count = build_shot_manifest(
            args.csv,
            args.out,
            videos_manifest=args.videos_manifest,
            fps=args.fps,
        )

    print(f"wrote {count} rows to {args.out}")


if __name__ == "__main__":
    main()
