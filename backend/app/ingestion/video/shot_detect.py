"""Shot boundary detection over source videos.

Detection is content-based: consecutive frames are downscaled and compared,
and a cut is declared where the mean absolute difference crosses a threshold.
That catches hard cuts reliably and gradual transitions less so, which is the
trade-off for needing no model weights and no GPU.

`detect_shots` is written against a detector-shaped signature so a learned
detector (TransNetV2) can replace the scoring pass later without changing the
manifest contract or the CLI.

Frames are decoded at native rate and counted in presentation order, so the
frame index used here *is* `original_frame_id`. Nothing resamples, which is
what keeps that id trustworthy.
"""

import argparse
from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np

from app.ingestion.manifest import (
    CLIP_ARROW_SCHEMA,
    ClipManifestRow,
    VideoManifestRow,
    iter_video_rows,
    write_rows,
)

# Mean absolute RGB difference, on a 0-255 scale, that separates a cut from
# ordinary motion. Tuned in the same range as PySceneDetect's content detector.
DEFAULT_THRESHOLD = 27.0

# Shots shorter than this are treated as flicker rather than content. Roughly
# half a second at 25-30fps.
DEFAULT_MIN_SHOT_FRAMES = 15

# Frames are compared at a fixed small size. Aspect ratio is deliberately not
# preserved: the distortion is identical on every frame, so it cancels out of
# a frame-to-frame difference, and a fixed size keeps the cost predictable.
ANALYSIS_SIZE = 64


class ShotDetectionError(Exception):
    pass


def iter_difference_scores(
    video_path: str, analysis_size: int = ANALYSIS_SIZE
) -> Iterator[tuple[int, float]]:
    """Yield `(frame_index, difference_from_previous_frame)` pairs.

    The first frame has no predecessor and so yields nothing; indexes start
    at 1 and are source-video frame indexes.
    """
    previous: np.ndarray | None = None

    try:
        with av.open(video_path) as container:
            if not container.streams.video:
                raise ShotDetectionError(f"no video stream in {video_path}")

            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            for index, frame in enumerate(container.decode(stream)):
                current = frame.reformat(
                    width=analysis_size, height=analysis_size, format="rgb24"
                ).to_ndarray().astype(np.int16)

                if previous is not None:
                    yield index, float(np.abs(current - previous).mean())

                previous = current
    except av.FFmpegError as exc:
        raise ShotDetectionError(f"cannot decode {video_path}: {exc}") from exc


def find_cuts(
    scores: Iterator[tuple[int, float]],
    threshold: float = DEFAULT_THRESHOLD,
    min_shot_frames: int = DEFAULT_MIN_SHOT_FRAMES,
) -> tuple[list[int], int]:
    """Reduce difference scores to cut positions and a total frame count.

    Returns the indexes of frames that *start* a new shot. A cut is ignored
    when it lands too soon after the previous one, which suppresses the burst
    of scores a dissolve produces.
    """
    cuts: list[int] = []
    last_cut = 0
    last_index = 0

    for index, score in scores:
        last_index = index
        if score >= threshold and index - last_cut >= min_shot_frames:
            cuts.append(index)
            last_cut = index

    return cuts, last_index + 1


def detect_shots(
    video: VideoManifestRow,
    threshold: float = DEFAULT_THRESHOLD,
    min_shot_frames: int = DEFAULT_MIN_SHOT_FRAMES,
) -> list[ClipManifestRow]:
    rate = video.require_constant_frame_rate()

    cuts, total_frames = find_cuts(
        iter_difference_scores(video.path), threshold, min_shot_frames
    )
    if total_frames == 0:
        raise ShotDetectionError(f"no decodable frames in {video.path}")

    starts = [0, *cuts]
    ends = [*[cut - 1 for cut in cuts], total_frames - 1]

    return [
        ClipManifestRow(
            video_id=video.video_id,
            shot_id=shot_id,
            start_frame=start,
            end_frame=end,
            start_sec=start / rate,
            end_sec=end / rate,
            path=video.path,
        )
        for shot_id, (start, end) in enumerate(zip(starts, ends, strict=True))
    ]


def build_shot_manifest(
    videos_manifest: str,
    out_path: str,
    threshold: float = DEFAULT_THRESHOLD,
    min_shot_frames: int = DEFAULT_MIN_SHOT_FRAMES,
    on_progress=None,
) -> int:
    rows: list[ClipManifestRow] = []

    for video in iter_video_rows(videos_manifest):
        shots = detect_shots(video, threshold, min_shot_frames)
        rows.extend(shots)
        if on_progress is not None:
            on_progress(video.video_id, len(shots))

    if not rows:
        raise ShotDetectionError(f"no videos in '{videos_manifest}'")

    return write_rows(rows, out_path, CLIP_ARROW_SCHEMA)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect shot boundaries and write a shots.parquet manifest."
    )
    parser.add_argument(
        "--videos-manifest", required=True, help="videos.parquet from the probe step"
    )
    parser.add_argument("--out", required=True, help="output shots .parquet path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"cut sensitivity, lower detects more (default {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--min-shot-frames",
        type=int,
        default=DEFAULT_MIN_SHOT_FRAMES,
        help=f"shortest allowed shot (default {DEFAULT_MIN_SHOT_FRAMES})",
    )
    args = parser.parse_args()

    if not Path(args.videos_manifest).is_file():
        raise SystemExit(f"probe manifest not found: {args.videos_manifest}")

    def report(video_id: str, shot_count: int) -> None:
        print(f"{video_id}: {shot_count} shots")

    count = build_shot_manifest(
        args.videos_manifest,
        args.out,
        threshold=args.threshold,
        min_shot_frames=args.min_shot_frames,
        on_progress=report,
    )
    print(f"wrote {count} shots to {args.out}")


if __name__ == "__main__":
    main()
