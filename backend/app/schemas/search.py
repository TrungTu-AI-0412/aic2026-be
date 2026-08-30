from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RetrievalMode = Literal["visual", "visual_asr", "asr_only"]

# Events a prose query may be decomposed into. A cap because stage B is
# `videos x events` serial round trips, so an over-eager decomposition is paid
# for on every search that follows it. Explicit `E1:`-style queries are not
# capped: the competition decides how many moments its own task has.
MAX_DECOMPOSED_EVENTS = 6


class QueryForms(BaseModel):
    """One part of a decomposed query, in the forms retrieval needs.

    Produced by `POST /search/decompose` and accepted back by the search
    endpoints, so an operator edits what the model wrote and the search uses it
    verbatim - a search given these does no LLM work at all.

    `original` is the span of the operator's query this was cut from, and is
    null for an overview the model synthesised (the query never stated one).
    `vision` is null when the caption call failed: retrieval then puts `speech`
    through the image tower, which ranks worse than a caption but is not
    nothing.
    """

    original: str | None = None
    vision: str | None = None
    speech: str = Field(min_length=1)


class DecomposeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    max_events: int = Field(
        default=MAX_DECOMPOSED_EVENTS,
        ge=1,
        le=12,
        description=(
            "Cap on events for a query that does not enumerate its own. Ignored"
            " when the query carries explicit `E1:` markers - those are split"
            " deterministically and the competition's own count is kept."
        ),
    )


class DecomposeResponse(BaseModel):
    """What the operator reviews before searching.

    `source` says how the split happened: `markers` when the query enumerated
    its events (a regex, no model, so it cannot renumber or merge them) and
    `llm` when they were inferred from prose.
    """

    source: Literal["markers", "llm"]
    overview: QueryForms
    events: list[QueryForms]
    latency_ms: dict[str, float] = Field(default_factory=dict)


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


class FrameSearchOverrides(AsrOverrides):
    """Source selection for the one-frame KIS and Q&A tracks."""

    retrieval_mode: RetrievalMode | None = Field(
        default=None,
        description=(
            "Explicit retrieval source. Omit to keep the configured default and"
            " legacy asr_enabled behaviour. ASR-only is supported by KIS/Q&A only."
        ),
    )

    @model_validator(mode="after")
    def _validate_retrieval_mode(self) -> "FrameSearchOverrides":
        if self.retrieval_mode is not None and self.asr_enabled is not None:
            raise ValueError("retrieval_mode and asr_enabled cannot be combined")
        if self.retrieval_mode == "asr_only" and self.asr_weight is not None:
            raise ValueError("asr_weight does not apply to ASR-only retrieval")
        if self.retrieval_mode == "visual" and any(
            value is not None
            for value in (
                self.asr_weight,
                self.asr_dense_weight,
                self.asr_sparse_weight,
            )
        ):
            raise ValueError("ASR weights do not apply to visual retrieval")
        if (
            self.retrieval_mode == "asr_only"
            and self.asr_dense_weight == 0
            and self.asr_sparse_weight == 0
        ):
            raise ValueError("ASR-only retrieval needs a dense or lexical branch")
        return self


class TemporalOverrides(BaseModel):
    """Shape of the sequence a temporal search will accept.

    Shared by TRAKE and by a KIS query the operator chose to search
    temporally: both run the same two-stage pipeline, so both want the same
    dials on the chain it builds.
    """

    max_gap_sec: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Widest gap in seconds between consecutive events. 0 disables the"
            " check; None uses the configured default."
        ),
    )
    gap_weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "What the time gap between consecutive events costs, as a share of"
            " the sequence score - the events of one query describe a single"
            " continuous action, so a tighter chain is the better answer. 0"
            " disables the penalty; None uses the configured default."
        ),
    )


class KisSearchRequest(FrameSearchOverrides, TemporalOverrides):
    """One frame per row, found either directly or through a sequence.

    `overview` and `events` turn this into a *temporal* KIS: a description that
    walks through phases ("phân cảnh tiếp theo...") is decomposed by
    `/search/decompose`, each moment votes on the video independently, and the
    row still reports one frame - the highest-scoring event of the aligned
    chain. Averaging those phases into a single 40-word caption is what this
    replaces. The operator opts in; nothing is auto-detected.
    """

    task: Literal["kis"]
    description: str = Field(
        min_length=1,
        description=(
            "The query as typed. Ignored for retrieval when `events` is given,"
            " where the decomposed forms are searched instead."
        ),
    )
    top_k: int = Field(default=100, ge=1, le=100)
    overview: str | QueryForms | None = Field(
        default=None,
        description="Temporal KIS only. Must be given with `events`.",
    )
    events: list[str | QueryForms] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Temporal KIS only: the moments the description walks through, in"
            " order. Omit for an ordinary single-frame KIS search."
        ),
    )

    @model_validator(mode="after")
    def _validate_temporal(self) -> "KisSearchRequest":
        if (self.overview is None) != (self.events is None):
            raise ValueError("overview and events must be given together")
        if self.events is not None and self.retrieval_mode == "asr_only":
            raise ValueError("temporal KIS has no ASR-only form")
        return self


class QaSearchRequest(FrameSearchOverrides):
    """Same retrieval as KIS: find the moment, the operator reads it.

    There is no `question` field. It was never encoded — SigLIP2 scores a
    caption against an image and "what colour is the bike?" describes no
    image — and no VQA model is wired in, so nothing downstream could use it.
    The operator answers off the frame and types it into the submission.
    """

    task: Literal["qa"]
    description: str = Field(min_length=1)
    top_k: int = Field(default=100, ge=1, le=100)


class TrakeSearchRequest(AsrOverrides, TemporalOverrides):
    model_config = ConfigDict(extra="forbid")

    task: Literal["trake"]
    overview: str | QueryForms
    events: list[str | QueryForms] = Field(min_length=1)
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


class AsrEvidence(BaseModel):
    """The speech segment that produced an ASR-only frame result."""

    segment: int
    text: str
    start_sec: float
    end_sec: float
    score: float


class StageAScore(BaseModel):
    """Why the video-selection stage kept this video.

    The composite alone cannot answer the question an operator actually asks -
    did this video win on the overview and lose every event, or the reverse -
    so the parts it is made of are reported next to it. `event_scores` is the
    best global score each event reached in this video, 0.0 where the event
    reached it nowhere, in request order.
    """

    rank: int
    score: float
    overview_score: float
    event_scores: list[float]


class SearchResult(BaseModel):
    rank: int
    video_id: str
    frame_ids: list[int]
    score: float
    # Temporal tracks only, and null when `video_ids` was supplied: stage A
    # never ran, and reporting 0.0 there would read as "matched nothing".
    stage_a: StageAScore | None = None
    # TRAKE only, and additive: `frame_ids` keeps its meaning, so a client that
    # ignores this field is unaffected. Four bare integers cannot tell an
    # operator which event landed where in time, which is what this carries.
    events: list[EventHit] | None = None
    asr_evidence: AsrEvidence | None = None


class SearchVersions(BaseModel):
    frames_collection: str | None = None
    clips_collection: str | None = None
    model_config_name: str
    asr_collection: str | None = None
    asr_model_config_name: str | None = None


class SearchResponse(BaseModel):
    request_id: str
    task: Literal["kis", "qa", "trake"]
    effective_retrieval_mode: RetrievalMode = "visual"
    # The English form the rewriting step produced, which is what the image
    # space and the reranker were given. `[description]` for KIS and QA;
    # `[overview, *events]` in request order for TRAKE. `cleaned_queries`
    # reports the parallel original-language forms used by speech retrieval.
    # Both are None when the step is off or fell back.
    rewritten_queries: list[str] | None = Field(
        default=None,
        description=(
            "The English forms the rewriting step encoded against the image"
            " space: [description] for KIS/QA, [overview, *events] for TRAKE."
            " See cleaned_queries for the speech form. Null when rewriting was"
            " off or failed, in which"
            " case the query was searched exactly as typed."
        ),
    )
    cleaned_queries: list[str] | None = Field(
        default=None,
        description=(
            "The original-language forms with retrieval instructions removed,"
            " searched against speech transcripts. Null when rewriting was off"
            " or failed."
        ),
    )
    results: list[SearchResult]
    versions: SearchVersions
    latency_ms: dict[str, float]
