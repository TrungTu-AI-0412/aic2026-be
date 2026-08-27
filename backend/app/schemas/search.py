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


class OcrSearchRequest(BaseModel):
    """Find a shot by the text printed on it, not by what it looks like.

    Separate from KIS because the two are answered differently: `text` is
    matched literally against what the recognisers read off the frame, no
    image embedding is computed, and no reranker runs. Use it when the query
    is a name, a date, a channel bug or a headline - things the image model
    has no representation of.

    `video_ids` narrows the search to videos an earlier visual search already
    surfaced, which is the usual way the two get used together.
    """

    task: Literal["ocr"]
    text: str = Field(min_length=1)
    top_k: int = Field(default=100, ge=1, le=100)
    video_ids: list[str] | None = None


SearchRequest = Annotated[
    KisSearchRequest | QaSearchRequest | TrakeSearchRequest | OcrSearchRequest,
    Field(discriminator="task"),
]


class SearchResult(BaseModel):
    rank: int
    video_id: str
    frame_ids: list[int]
    answer: str | None = None
    score: float
    # What the recognisers read off this frame, when they read anything. On a
    # lexical hit it is the evidence for the hit; on a visual one it is
    # context. Present on every track so a client renders results one way.
    ocr_text: str | None = None


class SearchVersions(BaseModel):
    frames_collection: str | None = None
    clips_collection: str | None = None
    model_config_name: str


class SearchResponse(BaseModel):
    request_id: str
    task: Literal["kis", "qa", "trake", "ocr"]
    results: list[SearchResult]
    versions: SearchVersions
    latency_ms: dict[str, float]