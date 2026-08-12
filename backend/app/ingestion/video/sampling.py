"""Sample keyframes from detected shots and extract them as JPEG files.

Sampling is uniform in time - roughly one keyframe per second of shot, with
at least one per shot however short it is. Uniform beats content-adaptive
here because the keyframe count stays predictable, which is what makes recall
and embedding cost possible to reason about before a run.

Two adjustments earn their keep:

- Frames are taken away from the shot edges, where transition artefacts and
  motion blur cluster.
- Within a small window around each target the sharpest frame wins. The pass
  already holds every decoded frame, so measuring a handful of neighbours
  costs almost nothing and keeps blurred frames out of the index.

The whole video is decoded sequentially rather than seeking to each target:
seeking rewinds to the preceding keyframe and decodes forward anyway, so it
is slower once targets are dense.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import av
import numpy as np

from app.ingestion.manifest import (
    KEYFRAME_ARROW_SCHEMA,
    ClipManifestRow,
    KeyframeManifestRow,
    VideoManifestRow,
    iter_rows,
    iter_video_rows,
    write_rows,
)
from app.schemas.ingestions import IngestionEntity

DEFAULT_FRAMES_PER_SECOND = 1.0

# 720p keeps result grids fast to load and still readable enough to verify an
# answer by eye. The embedding model resizes to a far smaller square anyway.
DEFAULT_MAX_HEIGHT = 720

# MJPEG quantiser scale, 2 (best) to 31 (worst).
DEFAULT_JPEG_QSCALE = 4

# Frames either side of a target considered when picking the sharpest one.
DEFAULT_SHARPNESS_WINDOW = 2

# Fraction of a shot trimmed from each end before placing targets.
DEFAULT_BOUNDARY_INSET = 0.1

SHARPNESS_ANALYSIS_SIZE = 128


class SamplingError(Exception):
    pass


class _Window:
    """A target frame plus the neighbouring frames allowed to replace it."""

    __slots__ = ("shot_id", "target", "low", "high")

    def __init__(self, shot_id: int, target: int, low: int, high: int) -> None:
        self.shot_id = shot_id
        self.target = target
        self.low = low
        self.high = high


def plan_targets(
    shot: ClipManifestRow,
    fps: float,
    frames_per_second: float = DEFAULT_FRAMES_PER_SECOND,
    boundary_inset: float = DEFAULT_BOUNDARY_INSET,
) -> list[int]:
    """Choose evenly spaced frame indexes inside one shot."""
    length = shot.end_frame - shot.start_frame + 1
    inset = int(length * boundary_inset)
    low = shot.start_frame + inset
    high = shot.end_frame - inset
    if high < low:
        low = high = (shot.start_frame + shot.end_frame) // 2

    duration = length / fps
    count = max(1, round(duration * frames_per_second))

    span = high - low + 1
    count = min(count, span)

    # Targets sit at the midpoint of equal sub-intervals, so none lands on the
    # very first or last frame of the usable range.
    targets = [low + int((position + 0.5) * span / count) for position in range(count)]
    return sorted(set(targets))


def plan_windows(
    shot: ClipManifestRow,
    targets: list[int],
    sharpness_window: int = DEFAULT_SHARPNESS_WINDOW,
) -> list[_Window]:
    """Expand targets into non-overlapping candidate windows."""
    windows: list[_Window] = []
    previous_high = shot.start_frame - 1

    for target in targets:
        low = max(shot.start_frame, target - sharpness_window, previous_high + 1)
        high = min(shot.end_frame, target + sharpness_window)
        if high < low:
            continue
        windows.append(_Window(shot.shot_id, target, low, high))
        previous_high = high

    return windows


def sharpness(frame) -> float:
    """Variance of a discrete Laplacian: higher means more in focus."""
    gray = (
        frame.reformat(
            width=SHARPNESS_ANALYSIS_SIZE,
            height=SHARPNESS_ANALYSIS_SIZE,
            format="gray",
        )
        .to_ndarray()
        .astype(np.float32)
    )
    laplacian = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(laplacian.var())


def keyframe_path(output_dir: Path, video_id: str, frame_id: int) -> Path:
    # Padded so the name still matches the manifest scanner's frame pattern.
    return output_dir / video_id / f"{video_id}_{frame_id:06d}.jpg"


def write_jpeg(
    frame,
    destination: Path,
    rotation: int = 0,
    max_height: int = DEFAULT_MAX_HEIGHT,
    qscale: int = DEFAULT_JPEG_QSCALE,
) -> None:
    """Scale, un-rotate and encode one frame as a JPEG.

    Rotation is baked in here rather than left to the viewer, because the same
    files feed both the console and the embedding model, and a sideways frame
    embeds as nonsense.
    """
    displayed_height = frame.width if rotation in (90, 270) else frame.height
    scale = min(1.0, max_height / displayed_height) if displayed_height else 1.0
    width = max(2, round(frame.width * scale / 2) * 2)
    height = max(2, round(frame.height * scale / 2) * 2)

    image = frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
    if rotation:
        image = np.ascontiguousarray(np.rot90(image, k=-(rotation // 90)))

    output = av.VideoFrame.from_ndarray(image, format="rgb24")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with av.open(str(destination), "w", format="image2") as container:
        stream = container.add_stream("mjpeg", rate=1)
        stream.width = output.width
        stream.height = output.height
        stream.pix_fmt = "yuvj420p"
        stream.codec_context.qmin = qscale
        stream.codec_context.qmax = qscale

        for packet in stream.encode(output):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def sample_video(
    video: VideoManifestRow,
    shots: list[ClipManifestRow],
    output_dir: str | Path,
    frames_per_second: float = DEFAULT_FRAMES_PER_SECOND,
    boundary_inset: float = DEFAULT_BOUNDARY_INSET,
    sharpness_window: int = DEFAULT_SHARPNESS_WINDOW,
    max_height: int = DEFAULT_MAX_HEIGHT,
    qscale: int = DEFAULT_JPEG_QSCALE,
) -> list[KeyframeManifestRow]:
    rate = video.require_constant_frame_rate()
    root = Path(output_dir)

    windows: list[_Window] = []
    for shot in sorted(shots, key=lambda item: item.start_frame):
        targets = plan_targets(shot, float(rate), frames_per_second, boundary_inset)
        windows.extend(plan_windows(shot, targets, sharpness_window))

    if not windows:
        raise SamplingError(f"no keyframes planned for {video.video_id}")

    rows: list[KeyframeManifestRow] = []
    pending = iter(windows)
    current = next(pending, None)
    best_score = float("-inf")
    best_frame = None
    best_index = 0

    try:
        with av.open(video.path) as container:
            if not container.streams.video:
                raise SamplingError(f"no video stream in {video.path}")

            stream = container.streams.video[0]
            # See app/features/media.py: threaded decode deadlocks against
            # PyAV's GIL-taking log callback on container teardown.
            stream.thread_type = "NONE"

            for index, frame in enumerate(container.decode(stream)):
                if current is None:
                    break
                if index < current.low:
                    continue

                # A single-frame window needs no comparison.
                score = (
                    0.0
                    if current.low == current.high
                    else sharpness(frame)
                )
                if best_frame is None or score > best_score:
                    best_score, best_frame, best_index = score, frame, index

                if index >= current.high:
                    destination = keyframe_path(root, video.video_id, best_index)
                    write_jpeg(
                        best_frame, destination, video.rotation, max_height, qscale
                    )
                    rows.append(
                        KeyframeManifestRow(
                            video_id=video.video_id,
                            shot_id=current.shot_id,
                            original_frame_id=best_index,
                            pts_sec=best_index / rate,
                            path=str(destination),
                        )
                    )
                    best_score, best_frame = float("-inf"), None
                    current = next(pending, None)
    except av.FFmpegError as exc:
        raise SamplingError(f"cannot decode {video.path}: {exc}") from exc

    if not rows:
        raise SamplingError(f"no decodable frames in {video.path}")

    return rows


def group_shots_by_video(shots_manifest: str) -> dict[str, list[ClipManifestRow]]:
    grouped: dict[str, list[ClipManifestRow]] = defaultdict(list)
    for shot in iter_rows(shots_manifest, IngestionEntity.CLIPS):
        grouped[shot.video_id].append(shot)
    return grouped


def build_keyframe_manifest(
    videos_manifest: str,
    shots_manifest: str,
    output_dir: str,
    out_path: str,
    frames_per_second: float = DEFAULT_FRAMES_PER_SECOND,
    boundary_inset: float = DEFAULT_BOUNDARY_INSET,
    sharpness_window: int = DEFAULT_SHARPNESS_WINDOW,
    max_height: int = DEFAULT_MAX_HEIGHT,
    qscale: int = DEFAULT_JPEG_QSCALE,
    on_progress=None,
) -> int:
    shots_by_video = group_shots_by_video(shots_manifest)
    rows: list[KeyframeManifestRow] = []

    for video in iter_video_rows(videos_manifest):
        shots = shots_by_video.get(video.video_id)
        if not shots:
            raise SamplingError(
                f"'{video.video_id}' has no shots in '{shots_manifest}'; "
                "run shot detection over the same probe manifest"
            )

        sampled = sample_video(
            video,
            shots,
            output_dir,
            frames_per_second=frames_per_second,
            boundary_inset=boundary_inset,
            sharpness_window=sharpness_window,
            max_height=max_height,
            qscale=qscale,
        )
        rows.extend(sampled)
        if on_progress is not None:
            on_progress(video.video_id, len(sampled))

    if not rows:
        raise SamplingError(f"no videos in '{videos_manifest}'")

    return write_rows(rows, out_path, KEYFRAME_ARROW_SCHEMA)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample keyframes from shots, extract JPEGs and write a manifest."
    )
    parser.add_argument("--videos-manifest", required=True, help="videos.parquet")
    parser.add_argument("--shots-manifest", required=True, help="shots.parquet")
    parser.add_argument("--output-dir", required=True, help="directory for JPEGs")
    parser.add_argument("--out", required=True, help="output keyframes .parquet")
    parser.add_argument(
        "--frames-per-second",
        type=float,
        default=DEFAULT_FRAMES_PER_SECOND,
        help=f"keyframes per second of shot (default {DEFAULT_FRAMES_PER_SECOND})",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=DEFAULT_MAX_HEIGHT,
        help=f"cap on displayed height (default {DEFAULT_MAX_HEIGHT})",
    )
    parser.add_argument(
        "--qscale",
        type=int,
        default=DEFAULT_JPEG_QSCALE,
        help=f"JPEG quantiser, 2 best to 31 worst (default {DEFAULT_JPEG_QSCALE})",
    )
    parser.add_argument(
        "--sharpness-window",
        type=int,
        default=DEFAULT_SHARPNESS_WINDOW,
        help="frames either side of a target to consider; 0 disables the search",
    )
    parser.add_argument(
        "--boundary-inset",
        type=float,
        default=DEFAULT_BOUNDARY_INSET,
        help=f"fraction trimmed from each shot end (default {DEFAULT_BOUNDARY_INSET})",
    )
    args = parser.parse_args()

    def report(video_id: str, count: int) -> None:
        print(f"{video_id}: {count} keyframes")

    count = build_keyframe_manifest(
        args.videos_manifest,
        args.shots_manifest,
        args.output_dir,
        args.out,
        frames_per_second=args.frames_per_second,
        boundary_inset=args.boundary_inset,
        sharpness_window=args.sharpness_window,
        max_height=args.max_height,
        qscale=args.qscale,
        on_progress=report,
    )
    print(f"wrote {count} keyframes to {args.out}")


if __name__ == "__main__":
    main()
