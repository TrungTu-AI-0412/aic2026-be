from asyncio import to_thread
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Protocol

import av
import numpy as np

from app.core.config import settings
from app.features.errors import FeatureExtractionError
from app.features.media import ClipSegment, sample_clip_frames
from app.ingestion.manifest import VideoManifestRow, iter_video_rows
from app.ingestion.video.sampling import keyframe_path


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

    async def get_clip(self, video_id: str, start_frame: int, end_frame: int) -> ClipVideo:
        ...


@lru_cache(maxsize=1)
def _load_videos(manifest_path: str) -> dict[str, VideoManifestRow]:
    # ponytail: loaded once per process; restart the API after re-probing
    # videos. Add an mtime check if manifests change while serving.
    if not Path(manifest_path).is_file():
        return {}
    return {row.video_id: row for row in iter_video_rows(manifest_path)}


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
    ) -> None:
        self._root = Path(data_root).resolve()
        self._keyframes = self._root / "keyframes"
        self._videos_manifest = str(
            videos_manifest or self._root / "manifests" / "videos.parquet"
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

    async def get_clip(
        self, video_id: str, start_frame: int, end_frame: int
    ) -> ClipVideo:
        videos = await to_thread(_load_videos, self._videos_manifest)
        video = videos.get(video_id)
        if video is None:
            raise VideoNotFoundError(
                f"video '{video_id}' is not in {self._videos_manifest}"
            )

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
