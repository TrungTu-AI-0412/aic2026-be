"""Read source-video metadata with PyAV.

Probing runs before shot detection because every later stage depends on it:
shot boundaries and sampled keyframes are recorded as source-video frame
indexes, and turning those into timestamps needs the exact frame rate. The
rate is kept as a fraction all the way through - rounding 30000/1001 to
29.97 drifts by whole frames over a long video, which would silently
corrupt `original_frame_id`.

Probing also decodes the first frame, so an unreadable video fails here
rather than part-way through a GPU shot-detection run.
"""

import argparse
import math
import struct
from pathlib import Path

import av
from av.sidedata.sidedata import Type as SideDataType

from app.ingestion.manifest import (
    VIDEO_ARROW_SCHEMA,
    VideoManifestRow,
    write_rows,
)

VIDEO_SUFFIXES = frozenset(
    {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".ts"}
)

_DISPLAY_MATRIX_FORMAT = "<9i"
_SUPPORTED_ROTATIONS = (0, 90, 180, 270)


class VideoProbeError(Exception):
    pass


def probe_video(path: str | Path, video_id: str | None = None) -> VideoManifestRow:
    source = Path(path)
    if not source.is_file():
        raise VideoProbeError(f"video not found: {source}")

    try:
        with av.open(str(source)) as container:
            if not container.streams.video:
                raise VideoProbeError(f"no video stream in {source}")

            stream = container.streams.video[0]
            rate = stream.average_rate or stream.guessed_rate
            if rate is None or rate <= 0:
                raise VideoProbeError(f"cannot determine frame rate for {source}")

            rotation = _probe_rotation(container)

            return VideoManifestRow(
                video_id=video_id or source.stem,
                path=str(source),
                fps_num=rate.numerator,
                fps_den=rate.denominator,
                nb_frames=stream.frames or None,
                duration_sec=_duration_sec(container, stream),
                width=stream.width,
                height=stream.height,
                rotation=rotation,
                is_vfr=_is_vfr(stream),
                codec=stream.codec_context.name,
            )
    except av.FFmpegError as exc:
        raise VideoProbeError(f"cannot probe {source}: {exc}") from exc


def probe_directory(source_dir: str | Path, out_path: str) -> int:
    source = Path(source_dir)
    if not source.is_dir():
        raise VideoProbeError(f"source directory not found: {source}")

    videos = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise VideoProbeError(f"no video files found under '{source}'")

    rows = [probe_video(path) for path in videos]
    return write_rows(rows, out_path, VIDEO_ARROW_SCHEMA)


def _is_vfr(stream) -> bool:
    """Detect variable frame rate by comparing the average and base rates.

    For a VFR source a single frame rate cannot map frame index to time, so
    downstream stages must fall back to presentation timestamps.
    """
    if stream.average_rate is None or stream.base_rate is None:
        return False
    return stream.average_rate != stream.base_rate


def _duration_sec(container, stream) -> float:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / av.time_base
    return 0.0


def _probe_rotation(container) -> int:
    """Return the display rotation in degrees.

    PyAV exposes the display matrix only on decoded frames, so this decodes
    the first frame. That doubles as a decodability check.
    """
    try:
        frame = next(container.decode(video=0))
    except StopIteration:
        raise VideoProbeError("video has no decodable frames") from None

    if frame.side_data is None:
        return 0

    for side_data in frame.side_data.keys():
        if side_data.type is not SideDataType.DISPLAYMATRIX:
            continue

        raw = bytes(side_data)
        if len(raw) != struct.calcsize(_DISPLAY_MATRIX_FORMAT):
            return 0
        return _rotation_from_display_matrix(struct.unpack(_DISPLAY_MATRIX_FORMAT, raw))

    return 0


def _rotation_from_display_matrix(matrix: tuple[int, ...]) -> int:
    """Decode a 3x3 display matrix into a clockwise rotation in degrees.

    Mirrors ffmpeg's `av_display_rotation_get`, which reports the angle the
    decoder must undo, hence the negation.
    """
    scale_x = math.hypot(matrix[0], matrix[3])
    scale_y = math.hypot(matrix[1], matrix[4])
    if scale_x == 0 or scale_y == 0:
        return 0

    angle = math.degrees(math.atan2(matrix[1] / scale_y, matrix[0] / scale_x))
    rotation = round(-angle) % 360
    if rotation not in _SUPPORTED_ROTATIONS:
        raise VideoProbeError(f"unsupported display rotation: {rotation} degrees")
    return rotation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe source videos and write a videos.parquet manifest."
    )
    parser.add_argument("--source", required=True, help="directory of source videos")
    parser.add_argument("--out", required=True, help="output videos .parquet path")
    args = parser.parse_args()

    count = probe_directory(args.source, args.out)
    print(f"probed {count} videos into {args.out}")


if __name__ == "__main__":
    main()
