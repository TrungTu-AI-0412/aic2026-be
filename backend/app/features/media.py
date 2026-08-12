from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from app.features.errors import FeatureExtractionError


@dataclass(frozen=True)
class ClipSegment:
    """Media coordinates needed to sample one inclusive video segment."""

    path: str
    video_id: str
    shot_id: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float


def read_image(path: str) -> np.ndarray:
    """Decode one image file as an RGB array."""
    source = Path(path)
    if not source.is_file():
        raise FeatureExtractionError(f"keyframe not found: {path}")

    try:
        with av.open(str(source)) as container:
            if not container.streams.video:
                raise FeatureExtractionError(f"no image stream in {path}")
            frame = next(container.decode(video=0), None)
            if frame is None:
                raise FeatureExtractionError(f"cannot decode keyframe: {path}")
            return frame.to_ndarray(format="rgb24")
    except av.FFmpegError as exc:
        raise FeatureExtractionError(f"cannot decode keyframe '{path}': {exc}") from exc


def sample_clip_frames(segment: ClipSegment, frame_count: int) -> list[np.ndarray]:
    """Decode representative RGB frames from one inclusive shot range."""
    source = Path(segment.path)
    if not source.is_file():
        raise FeatureExtractionError(f"clip source video not found: {segment.path}")

    available = segment.end_frame - segment.start_frame + 1
    count = min(frame_count, available)
    target_times = np.linspace(segment.start_sec, segment.end_sec, count).tolist()
    images: list[np.ndarray] = []
    last_image: np.ndarray | None = None

    if segment.end_frame > segment.start_frame:
        half_frame = (segment.end_sec - segment.start_sec) / (
            2 * (segment.end_frame - segment.start_frame)
        )
    else:
        half_frame = 0.0

    try:
        with av.open(str(source)) as container:
            if not container.streams.video:
                raise FeatureExtractionError(f"no video stream in {segment.path}")

            stream = container.streams.video[0]
            # Single-threaded on purpose. PyAV forwards FFmpeg logs into Python
            # logging from whatever thread emits them, taking the GIL to do it,
            # and `transformers` reinstalls that callback when it is imported.
            # With frame threading, a decoder worker logging (these h264
            # sources spam "mmco: unref short failure") blocks on the GIL while
            # the main thread already holds it inside avcodec_free_context()
            # waiting for that same worker to exit: both hang forever, which is
            # how a clips ingestion stalled in `upserting`. Threading buys
            # roughly 1.5x on decode and is not worth a deadlock.
            stream.thread_type = "NONE"
            if segment.start_sec > 0 and stream.time_base is not None:
                offset = max(0, int(segment.start_sec / float(stream.time_base)))
                container.seek(offset, stream=stream, backward=True)

            target_index = 0
            for frame in container.decode(stream):
                if frame.time is None:
                    continue

                timestamp = float(frame.time)
                if timestamp + half_frame < segment.start_sec:
                    continue
                if (
                    timestamp - half_frame > segment.end_sec
                    and target_index < count
                ):
                    break

                current = frame.to_ndarray(format="rgb24")
                last_image = current
                while (
                    target_index < count
                    and timestamp + half_frame >= target_times[target_index]
                ):
                    images.append(current)
                    target_index += 1

                if target_index == count:
                    break
    except av.FFmpegError as exc:
        raise FeatureExtractionError(
            f"cannot decode clip '{segment.video_id}:{segment.shot_id}': {exc}"
        ) from exc

    # Timestamp rounding near the inclusive end frame can leave the last
    # target unmatched. Reuse is bounded to at most half a frame.
    if last_image is not None:
        images.extend([last_image] * (count - len(images)))
    if not images:
        raise FeatureExtractionError(
            "no decodable frames for clip "
            f"'{segment.video_id}:{segment.shot_id}'"
        )
    return images
