from dataclasses import dataclass
from typing import Protocol


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
