from dataclasses import dataclass
from typing import Protocol

from app.schemas.submissions import ExportRequest


@dataclass(frozen=True)
class ExportFile:
    content: bytes
    media_type: str
    filename: str


class VideoNotFoundError(Exception):
    pass


class FrameOutOfBoundsError(Exception):
    pass


class SubmissionService(Protocol):
    async def export(self, request: ExportRequest) -> ExportFile:
        ...
