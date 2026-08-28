"""Shot boundary detection over source videos.

Two detectors are available:

- `transnetv2` (default): a small CNN trained for the task. It recognises
  gradual transitions - dissolves, fades, wipes - that frame differencing
  cannot, and is the accurate choice when a GPU is available.
- `content`: consecutive frames are downscaled and compared, and a cut is
  declared where the mean absolute difference crosses a threshold. No model,
  no GPU, useful as a fallback and for fast tests.

Both feed the same manifest contract, so the detector can be swapped without
anything downstream noticing.

Frames are decoded at native rate and counted in presentation order, so the
frame index used here *is* `original_frame_id`. Nothing resamples, which is
what keeps that id trustworthy.
"""

import argparse
from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np
from tqdm import tqdm

from functools import partial

from app.ingestion.video import parallel
from app.ingestion.manifest import (
    CLIP_ARROW_SCHEMA,
    ClipManifestRow,
    VideoManifestRow,
    count_rows,
    existing_video_ids,
    iter_video_rows,
    write_rows,
)
from app.ingestion.video import transnet

DETECTOR_TRANSNETV2 = "transnetv2"
DETECTOR_CONTENT = "content"
DETECTORS = (DETECTOR_TRANSNETV2, DETECTOR_CONTENT)
DEFAULT_DETECTOR = DETECTOR_TRANSNETV2

# Mean absolute RGB difference, on a 0-255 scale, that separates a cut from
# ordinary motion. Tuned in the same range as PySceneDetect's content detector.
DEFAULT_THRESHOLD = 27.0

# Each detector scores on its own scale: a probability for TransNetV2, a pixel
# difference for the content detector.
DEFAULT_THRESHOLDS = {
    DETECTOR_TRANSNETV2: transnet.DEFAULT_THRESHOLD,
    DETECTOR_CONTENT: DEFAULT_THRESHOLD,
}

# Shots shorter than this are treated as flicker rather than content. Roughly
# half a second at 25-30fps.
DEFAULT_MIN_SHOT_FRAMES = 15

# Rows buffered before the manifest is rewritten. Parquet cannot be extended in
# place, so each flush rewrites the file; at ~100 shots per video this trades a
# rewrite every few videos for never losing more than that to a crash.
DEFAULT_FLUSH_ROWS = 2000

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
            # See app/features/media.py: threaded decode deadlocks against
            # PyAV's GIL-taking log callback on container teardown.
            stream.thread_type = "NONE"

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


def detect_ranges(
    video_path: str,
    detector: str = DEFAULT_DETECTOR,
    threshold: float | None = None,
    min_shot_frames: int = DEFAULT_MIN_SHOT_FRAMES,
) -> list[tuple[int, int]]:
    """Run the chosen detector and return inclusive shot ranges."""
    if detector not in DEFAULT_THRESHOLDS:
        raise ShotDetectionError(
            f"unknown detector '{detector}'; supported: {', '.join(DETECTORS)}"
        )

    if threshold is None:
        threshold = DEFAULT_THRESHOLDS[detector]

    if detector == DETECTOR_TRANSNETV2:
        return transnet.detect_ranges(video_path, threshold, min_shot_frames)

    cuts, total_frames = find_cuts(
        iter_difference_scores(video_path), threshold, min_shot_frames
    )
    starts = [0, *cuts]
    ends = [*[cut - 1 for cut in cuts], total_frames - 1]
    return list(zip(starts, ends, strict=True))


def detect_shots(
    video: VideoManifestRow,
    detector: str = DEFAULT_DETECTOR,
    threshold: float | None = None,
    min_shot_frames: int = DEFAULT_MIN_SHOT_FRAMES,
) -> list[ClipManifestRow]:
    rate = video.require_constant_frame_rate()

    ranges = detect_ranges(video.path, detector, threshold, min_shot_frames)
    if not ranges:
        raise ShotDetectionError(f"no decodable frames in {video.path}")

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
        for shot_id, (start, end) in enumerate(ranges)
    ]


def build_shot_manifest(
    videos_manifest: str,
    out_path: str,
    detector: str = DEFAULT_DETECTOR,
    threshold: float | None = None,
    min_shot_frames: int = DEFAULT_MIN_SHOT_FRAMES,
    on_progress=None,
    resume: bool = False,
    limit: int | None = None,
    workers: int | None = None,
    flush_every: int = DEFAULT_FLUSH_ROWS,
) -> int:
    """Detect shots for every video in the probe manifest.

    With `resume`, videos that already have shots in `out_path` are skipped and
    the new ones appended - the expensive detector never re-runs over a video a
    previous slice covered. `limit` caps how many *new* videos this run takes.
    """
    # Materialised so the progress bar knows how many videos are coming; the
    # rows are metadata only, a few hundred bytes each.
    videos = list(iter_video_rows(videos_manifest))
    if not videos:
        raise ShotDetectionError(f"no videos in '{videos_manifest}'")

    done = existing_video_ids(out_path) if resume else set()
    pending = [video for video in videos if video.video_id not in done][:limit]

    work = partial(
        _detect_for_video,
        detector=detector,
        threshold=threshold,
        min_shot_frames=min_shot_frames,
    )
    results = parallel.map_videos(
        work, pending, parallel.resolve_workers(workers), desc="shot detection"
    )

    total = count_rows(out_path) if resume and Path(out_path).is_file() else 0
    pending_rows: list[ClipManifestRow] = []
    appending = resume

    for video, shots in results:
        pending_rows.extend(shots)
        if on_progress is not None:
            on_progress(video.video_id, len(shots))

        # Flush periodically rather than once at the end. This pass decodes
        # every frame of 873 videos and takes hours; keeping all of it in
        # memory meant a crash threw away the whole run, and `--resume` could
        # only restart from nothing.
        if len(pending_rows) >= flush_every:
            total = write_rows(
                _by_video(pending_rows), out_path, CLIP_ARROW_SCHEMA, append=appending
            )
            pending_rows = []
            appending = True

    if pending_rows or not appending:
        total = write_rows(
            _by_video(pending_rows), out_path, CLIP_ARROW_SCHEMA, append=appending
        )
    return total


def _detect_for_video(
    video: VideoManifestRow,
    detector: str,
    threshold: float | None,
    min_shot_frames: int,
) -> list[ClipManifestRow]:
    """Module-level so the spawn context can pickle it into a worker."""
    return detect_shots(video, detector, threshold, min_shot_frames)


def _by_video(rows: list[ClipManifestRow]) -> list[ClipManifestRow]:
    """Stable order within a flush, since workers finish out of order."""
    return sorted(rows, key=lambda row: (row.video_id, row.shot_id))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect shot boundaries and write a shots.parquet manifest."
    )
    parser.add_argument(
        "--videos-manifest", required=True, help="videos.parquet from the probe step"
    )
    parser.add_argument("--out", required=True, help="output shots .parquet path")
    parser.add_argument(
        "--detector",
        choices=DETECTORS,
        default=DEFAULT_DETECTOR,
        help=f"detection model (default {DEFAULT_DETECTOR})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="cut sensitivity, lower detects more; defaults to "
        f"{DEFAULT_THRESHOLDS[DETECTOR_TRANSNETV2]} for {DETECTOR_TRANSNETV2} and "
        f"{DEFAULT_THRESHOLDS[DETECTOR_CONTENT]} for {DETECTOR_CONTENT}",
    )
    parser.add_argument(
        "--min-shot-frames",
        type=int,
        default=DEFAULT_MIN_SHOT_FRAMES,
        help=f"shortest allowed shot (default {DEFAULT_MIN_SHOT_FRAMES})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip videos already in --out and append the rest",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="detect at most this many new videos, for a trial slice",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="worker processes; defaults to one fewer than the core count",
    )
    args = parser.parse_args()

    if not Path(args.videos_manifest).is_file():
        raise SystemExit(f"probe manifest not found: {args.videos_manifest}")

    def report(video_id: str, shot_count: int) -> None:
        tqdm.write(f"{video_id}: {shot_count} shots")

    count = build_shot_manifest(
        args.videos_manifest,
        args.out,
        detector=args.detector,
        threshold=args.threshold,
        min_shot_frames=args.min_shot_frames,
        on_progress=report,
        resume=args.resume,
        limit=args.limit,
        workers=args.workers,
    )
    print(f"clips manifest now holds {count} shots: {args.out}")


if __name__ == "__main__":
    main()
