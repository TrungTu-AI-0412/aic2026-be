from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_CANDIDATES = 100

VIDEO_ID_PATTERN = r"^L\d{2}_V\d{3}$"


class ExportFormat(str, Enum):
    csv = "csv"
    json = "json"


class KisCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Literal["kis"]
    video_id: str = Field(pattern=VIDEO_ID_PATTERN)
    frame_id: int = Field(ge=0)


class QaCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Literal["qa"]
    video_id: str = Field(pattern=VIDEO_ID_PATTERN)
    frame_id: int = Field(ge=0)
    answer: str = Field(min_length=1)


class TrakeCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Literal["trake"]
    video_id: str = Field(pattern=VIDEO_ID_PATTERN)
    event_frame_ids: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_event_frame_ids(self) -> "TrakeCandidate":
        if any(frame_id < 0 for frame_id in self.event_frame_ids):
            raise ValueError("event_frame_ids must be non-negative")
        return self


Candidate = Annotated[
    KisCandidate | QaCandidate | TrakeCandidate,
    Field(discriminator="task"),
]


class ExportRequest(BaseModel):
    task: Literal["kis", "qa", "trake"]
    format: ExportFormat = ExportFormat.csv
    event_slot_count: int | None = Field(default=None, ge=1)
    candidates: list[Candidate] = Field(min_length=1, max_length=MAX_CANDIDATES)

    @model_validator(mode="after")
    def validate_candidates(self) -> "ExportRequest":
        for candidate in self.candidates:
            if candidate.task != self.task:
                raise ValueError(
                    f"candidate task '{candidate.task}' does not match "
                    f"request task '{self.task}'"
                )

            if isinstance(candidate, TrakeCandidate):
                if self.event_slot_count is None:
                    raise ValueError(
                        "event_slot_count is required for trake submissions"
                    )
                if len(candidate.event_frame_ids) != self.event_slot_count:
                    raise ValueError(
                        f"expected {self.event_slot_count} event frames, "
                        f"got {len(candidate.event_frame_ids)}"
                    )

        return self
