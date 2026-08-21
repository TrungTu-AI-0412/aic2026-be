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
    """Same retrieval as KIS: find the moment, the operator reads it.

    There is no `question` field. It was never encoded — SigLIP2 scores a
    caption against an image and "what colour is the bike?" describes no
    image — and no VQA model is wired in, so nothing downstream could use it.
    The operator answers off the frame and types it into the submission.
    """

    task: Literal["qa"]
    description: str = Field(min_length=1)
    top_k: int = Field(default=100, ge=1, le=100)


class TrakeSearchRequest(AsrOverrides):
    task: Literal["trake"]
    overview: str = Field(min_length=1)
    events: list[str] = Field(min_length=1)
    top_k: int = Field(
        default=100,
        ge=1,
        le=100,
        description=(
            "Sequences to return. Effectively capped by the number of candidate"
            " videos the overview stage keeps."
        ),
    )
    video_ids: list[str] | None = Field(
        default=None,
        description=(
            "Search only these videos, skipping video selection entirely. For an"
            " operator who already found the video with a KIS query."
        ),
    )
    max_gap_sec: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Widest gap in seconds between consecutive events. 0 disables the"
            " check; None uses the configured default."
        ),
    )


SearchRequest = Annotated[
    KisSearchRequest | QaSearchRequest | TrakeSearchRequest,
    Field(discriminator="task"),
]


class EventCandidate(BaseModel):
    """One frame proposed for one TRAKE event."""

    frame_id: int
    shot_id: int
    pts_sec: float | None = None
    score: float


class EventHit(EventCandidate):
    """The frame chosen for an event, plus the ones it beat.

    `alternates` are bounded by the neighbouring events' chosen frames, so an
    operator swapping one in cannot produce an out-of-order submission.
    """

    event_index: int
    alternates: list[EventCandidate] = Field(default_factory=list)


class SearchResult(BaseModel):
    rank: int
    video_id: str
    frame_ids: list[int]
    score: float
    # TRAKE only, and additive: `frame_ids` keeps its meaning, so a client that
    # ignores this field is unaffected. Four bare integers cannot tell an
    # operator which event landed where in time, which is what this carries.
    events: list[EventHit] | None = None


class SearchVersions(BaseModel):
    frames_collection: str | None = None
    clips_collection: str | None = None
    model_config_name: str


class SearchResponse(BaseModel):
    request_id: str
    task: Literal["kis", "qa", "trake"]
    # The English form the rewriting step produced, which is what the image
    # space and the reranker were given. `[description]` for KIS and QA;
    # `[overview, *events]` in request order for TRAKE. The step also returns a
    # cleaned form of the original that the speech stage searches with; it is
    # not reported, being close enough to the request to read off it. None when
    # the step is off or fell back, so an operator can tell a query the model
    # left alone from one it never saw.
    rewritten_queries: list[str] | None = Field(
        default=None,
        description=(
            "The English forms the rewriting step encoded against the image"
            " space: [description] for KIS/QA, [overview, *events] for TRAKE."
            " The speech stage searched a cleaned form of the original, which"
            " is not reported. Null when rewriting was off or failed, in which"
            " case the query was searched exactly as typed."
        ),
    )
    results: list[SearchResult]
    versions: SearchVersions
    latency_ms: dict[str, float]