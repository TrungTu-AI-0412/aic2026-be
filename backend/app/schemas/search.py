from typing import Annotated, Literal

from pydantic import BaseModel, Field


class AsrOverrides(BaseModel):
    """Per-request overrides for the speech-overlap bonus.

    Every field defaults to None, meaning "use the configured value". The stage
    is tuned during a run rather than between deployments: an operator who can
    see that speech is helping or hurting a particular query needs to move the
    weight without a restart.
    """

    asr_enabled: bool | None = Field(
        default=None, description="Toggle the ASR overlap bonus for this query."
    )
    asr_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Share of a frame's score the best overlapping segment adds.",
    )
    asr_dense_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Weight of the dense (semantic) half of the speech search.",
    )
    asr_sparse_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Weight of the lexical (BM25) half of the speech search.",
    )


class KisSearchRequest(AsrOverrides):
    task: Literal["kis"]
    description: str = Field(min_length=1)
    top_k: int = Field(default=100, ge=1, le=100)


class QaSearchRequest(AsrOverrides):
    task: Literal["qa"]
    description: str = Field(min_length=1)
    question: str = Field(min_length=1)
    top_k: int = Field(default=100, ge=1, le=100)


class TrakeSearchRequest(AsrOverrides):
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
    model_config_name: str


class SearchResponse(BaseModel):
    request_id: str
    task: Literal["kis", "qa", "trake"]
    results: list[SearchResult]
    versions: SearchVersions
    latency_ms: dict[str, float]