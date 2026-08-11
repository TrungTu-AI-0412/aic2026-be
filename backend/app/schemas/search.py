from typing import Annotated, Literal

from pydantic import BaseModel, Field


class KisSearchRequest(BaseModel):
    task: Literal["kis"]
    description: str = Field(min_length=1)
    top_k: int = Field(default=100, ge=1, le=100)


class QaSearchRequest(BaseModel):
    task: Literal["qa"]
    description: str = Field(min_length=1)
    question: str = Field(min_length=1)
    top_k: int = Field(default=100, ge=1, le=100)


class TrakeSearchRequest(BaseModel):
    task: Literal["trake"]
    overview: str = Field(min_length=1)
    events: list[str] = Field(min_length=1)
    top_k: int = Field(default=100, ge=1, le=100)


SearchRequest = Annotated[
    KisSearchRequest | QaSearchRequest | TrakeSearchRequest,
    Field(discriminator="task"),
]


class SearchResult(BaseModel):
    rank: int
    video_id: str
    frame_ids: list[int]
    answer: str | None = None
    score: float


class SearchVersions(BaseModel):
    frames_collection: str | None = None
    clips_collection: str | None = None
    model_config: str


class SearchResponse(BaseModel):
    request_id: str
    task: Literal["kis", "qa", "trake"]
    results: list[SearchResult]
    versions: SearchVersions
    latency_ms: dict[str, float]