from asyncio import to_thread
from bisect import bisect_left
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Protocol

import av
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from app.core.config import settings
from app.features.errors import FeatureExtractionError
from app.features.media import ClipSegment, sample_clip_frames
from app.ingestion.manifest import VideoManifestRow, iter_video_rows
from app.ingestion.video.sampling import keyframe_path
from app.schemas.media import FrameContext, NeighbourFrame

KEYFRAME_COLUMNS = (
    "video_id",
    "original_frame_id",
    "keyframe_n",
    "pts_sec",
    "shot_id",
    "shot_start_sec",
    "shot_end_sec",
)


@dataclass(frozen=True)
class FrameImage:
    content: bytes
    media_type: str


@dataclass(frozen=True)
class ClipVideo:
    content: bytes
    media_type: str


class VideoNotFoundError(Exception):
    pass


class FrameNotFoundError(Exception):
    pass


class MediaService(Protocol):
    async def get_frame(self, video_id: str, frame_id: int) -> FrameImage:
        ...

    async def get_frame_context(
        self, video_id: str, frame_id: int, radius: int
    ) -> FrameContext:
        ...

    async def get_video_path(self, video_id: str) -> Path:
        ...

    async def get_clip(self, video_id: str, start_frame: int, end_frame: int) -> ClipVideo:
        ...


@lru_cache(maxsize=1)
def _load_videos(manifest_path: str) -> dict[str, VideoManifestRow]:
    # ponytail: loaded once per process; restart the API after re-probing
    # videos. Add an mtime check if manifests change while serving.
    if not Path(manifest_path).is_file():
        return {}
    return {row.video_id: row for row in iter_video_rows(manifest_path)}


@lru_cache(maxsize=1)
def _load_keyframe_table(manifest_path: str):
    # Columns only: the manifest is ~290k rows and the arrow table stays in
    # RAM, so leaving `path` out keeps it small. Same caching caveat as
    # _load_videos: re-ingesting needs an API restart.
    if not Path(manifest_path).is_file():
        return None
    return pq.read_table(manifest_path, columns=list(KEYFRAME_COLUMNS))


@lru_cache(maxsize=64)
def _video_keyframes(manifest_path: str, video_id: str) -> list[tuple]:
    """Every keyframe of one video, ascending by frame index.

    Cached per video because a verify panel walks the neighbours of the same
    video repeatedly; without it every click rescans all 290k rows.
    """
    table = _load_keyframe_table(manifest_path)
    if table is None:
        return []
    rows = table.filter(pc.equal(table["video_id"], video_id))
    columns = [rows[name].to_pylist() for name in KEYFRAME_COLUMNS[1:]]
    return sorted(zip(*columns))


def _encode_mp4(images: list[np.ndarray], fps: Fraction) -> bytes:
    buffer = BytesIO()
    height, width = images[0].shape[:2]
    # No faststart: it needs a real file to rewrite. Clips are capped at
    # MAX_CLIP_FRAMES, so the whole body arrives before playback anyway.
    with av.open(buffer, "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        # yuv420p needs even dimensions; the encoder rescales frames to fit.
        stream.width = width - width % 2
        stream.height = height - height % 2
        stream.pix_fmt = "yuv420p"
        for image in images:
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return buffer.getvalue()


class LocalMediaService:
    """Serve keyframes off disk and cut clips out of the source videos.

    Keyframes are the JPEGs sampling already wrote, so a frame request is a
    file read. Clips are decoded from the source video on demand: the frame
    range is what identifies them, no clip files are ever extracted.
    """

    def __init__(
        self,
        data_root: str = settings.INGESTION_DATA_ROOT,
        videos_manifest: str | None = None,
        keyframes_dir: str | None = None,
        frames_manifest: str | None = None,
    ) -> None:
        self._root = Path(data_root).resolve()
        self._keyframes = Path(keyframes_dir).resolve() if keyframes_dir else self._root / "keyframes"
        self._videos_manifest = str(
            videos_manifest or self._root / "manifests" / "videos.parquet"
        )
        self._frames_manifest = str(
            frames_manifest or self._root / "manifests" / "frames.parquet"
        )

    async def get_frame(self, video_id: str, frame_id: int) -> FrameImage:
        path = keyframe_path(self._keyframes, video_id, frame_id).resolve()
        if not path.is_relative_to(self._keyframes):
            raise VideoNotFoundError(f"video '{video_id}' not found")
        if not path.parent.is_dir():
            raise VideoNotFoundError(f"video '{video_id}' has no extracted keyframes")
        if not path.is_file():
            raise FrameNotFoundError(
                f"frame {frame_id} of '{video_id}' was not sampled as a keyframe"
            )
        return FrameImage(await to_thread(path.read_bytes), "image/jpeg")

    async def _video(self, video_id: str) -> VideoManifestRow:
        videos = await to_thread(_load_videos, self._videos_manifest)
        video = videos.get(video_id)
        if video is None:
            raise VideoNotFoundError(
                f"video '{video_id}' is not in {self._videos_manifest}"
            )
        return video

    async def get_video_path(self, video_id: str) -> Path:
        """Locate the source file so it can be served by byte range.

        Playing from a timestamp needs no work here: the client seeks with a
        media fragment and the browser range-requests only the bytes it wants.
        """
        video = await self._video(video_id)
        path = Path(video.path).resolve()
        if not path.is_relative_to(self._root) or not path.is_file():
            raise VideoNotFoundError(f"video file for '{video_id}' is not readable")
        return path

    async def get_frame_context(
        self, video_id: str, frame_id: int, radius: int
    ) -> FrameContext:
        """Metadata for one keyframe plus the keyframes either side of it.

        Neighbours are counted in *keyframes*, not source frames: only sampled
        frames exist as JPEGs, and at 3 frames per shot a ±30 frame window is
        often empty.
        """
        video = await self._video(video_id)
        keyframes = await to_thread(
            _video_keyframes, self._frames_manifest, video_id
        )
        if not keyframes:
            raise FrameNotFoundError(
                f"'{video_id}' has no keyframes in {self._frames_manifest}"
            )

        frame_ids = [row[0] for row in keyframes]
        position = bisect_left(frame_ids, frame_id)
        if position == len(frame_ids) or frame_ids[position] != frame_id:
            raise FrameNotFoundError(
                f"frame {frame_id} of '{video_id}' is not a keyframe"
            )

        _, keyframe_n, pts_sec, shot_id, shot_start, shot_end = keyframes[position]
        window = keyframes[max(position - radius, 0) : position + radius + 1]
        return FrameContext(
            video_id=video_id,
            frame_id=frame_id,
            keyframe_n=keyframe_n,
            pts_sec=pts_sec,
            shot_id=shot_id,
            shot_start_sec=shot_start,
            shot_end_sec=shot_end,
            shot_start_frame=video.sec_to_frame(shot_start),
            shot_end_frame=video.sec_to_frame(shot_end),
            fps=float(video.fps),
            duration_sec=video.duration_sec,
            width=video.width,
            height=video.height,
            neighbours=[
                NeighbourFrame(
                    frame_id=row[0],
                    keyframe_n=row[1],
                    pts_sec=row[2],
                    shot_id=row[3],
                    is_same_shot=row[3] == shot_id,
                )
                for row in window
            ],
        )

    async def get_clip(
        self, video_id: str, start_frame: int, end_frame: int
    ) -> ClipVideo:
        video = await self._video(video_id)
        segment = ClipSegment(
            path=video.path,
            video_id=video_id,
            shot_id=0,
            start_frame=start_frame,
            end_frame=end_frame,
            start_sec=video.frame_to_sec(start_frame),
            end_sec=video.frame_to_sec(end_frame),
        )
        try:
            images = await to_thread(
                sample_clip_frames, segment, end_frame - start_frame + 1
            )
        except FeatureExtractionError as exc:
            raise FrameNotFoundError(str(exc)) from exc

        content = await to_thread(_encode_mp4, images, video.require_constant_frame_rate())
        return ClipVideo(content, "video/mp4")
