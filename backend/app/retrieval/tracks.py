"""Per-track orchestration on top of the shared retrieval engine.

Each track decides what text to encode and how to turn hits into results.
The search, deduplication and scoring underneath are shared, so a change to
ranking applies to every track at once.
"""

from dataclasses import replace
from uuid import uuid4

from app.retrieval.engine import (
    RetrievalConfig,
    Timings,
    retrieve,
    retrieve_per_video,
)
from app.schemas.search import (
    EventCandidate,
    EventHit,
    KisSearchRequest,
    QaSearchRequest,
    SearchResponse,
    SearchResult,
    SearchVersions,
    TrakeSearchRequest,
)
from app.vector_store.search import ScoredFrame

# Videos that survive the global stage and earn per-event searches of their own.
# This is the cost dial: the second stage runs one Qdrant query per
# (video, event) pair.
TRAKE_VIDEO_CANDIDATES = 30
# Hits per query in the video-selection stage.
TRAKE_STAGE_A_TOP_K = 100
# Candidates per event *within one video*, not globally as this once meant.
TRAKE_CANDIDATES_PER_EVENT = 20
# Runners-up offered per event so an operator can override a chosen frame.
TRAKE_ALTERNATES_PER_EVENT = 5
# Widest gap between consecutive events. The events of one TRAKE query describe
# a single continuous action, so a chain spread across half an hour is noise,
# not a hit; without this a sequence spanning a 40-minute video scores exactly
# as well as one spanning three seconds.
TRAKE_MAX_GAP_SEC = 300.0


def search_kis(request: KisSearchRequest, config: RetrievalConfig) -> SearchResponse:
    return _frame_search("kis", request.description, request.top_k, config)


def search_qa(request: QaSearchRequest, config: RetrievalConfig) -> SearchResponse:
    """Locate the moment the question is about.

    Byte-for-byte the same retrieval as KIS, because the question was never
    part of it: the description locates the scene, and the operator reads the
    answer off the frame and types it into the submission. The two tracks stay
    separate endpoints only because they export different row shapes.
    """
    return _frame_search("qa", request.description, request.top_k, config)


def _frame_search(
    task: str, text: str, top_k: int, config: RetrievalConfig
) -> SearchResponse:
    """One hit per shot, one frame per hit."""
    timings = Timings()
    hits = retrieve(text, top_k, config, timings)

    results = [
        SearchResult(
            rank=rank,
            video_id=hit.video_id,
            frame_ids=[hit.representative_frame],
            score=hit.score,
        )
        for rank, hit in enumerate(hits, start=1)
    ]
    return _response(task, results, config, timings)


def search_trake(request: TrakeSearchRequest, config: RetrievalConfig) -> SearchResponse:
    """Find one frame per event, in order, inside a single video.

    Two stages, because the two questions are different. Which video is a
    whole-video judgement the overview answers best; where each event sits is a
    within-video judgement that only makes sense once the video is fixed.
    Searching every event globally and intersecting afterwards - the previous
    shape - loses the correct video whenever any single event fails to reach a
    global top-N, which a fine-grained event routinely does.
    """
    timings = Timings()
    events = request.events

    if request.video_ids:
        videos = list(dict.fromkeys(request.video_ids))
    else:
        videos = _candidate_videos(request, config, timings)
    if not videos:
        return _response("trake", [], config, timings)

    # Reranking is off for the second stage. The cross-encoder head covers the
    # top `rerank_top_n` of one global list, so against per-video candidate
    # sets it would rescore an arbitrary slice of them; making it meaningful
    # costs one forward pass per (video, event, candidate). The video is
    # already chosen by here, and within one video the dense space is what
    # separates its own frames.
    localise = replace(config, rerank_enabled=False)
    per_event = [
        retrieve_per_video(
            event, videos, TRAKE_CANDIDATES_PER_EVENT, localise, timings
        )
        for event in events
    ]

    max_gap = (
        TRAKE_MAX_GAP_SEC if request.max_gap_sec is None else request.max_gap_sec
    )

    sequences: list[tuple[float, str, list[ScoredFrame], list[list[ScoredFrame]]]] = []
    for video_id in videos:
        slots = [hits.get(video_id) or [] for hits in per_event]
        # A partial sequence is not a submission, so a video that cannot fill
        # every event is dropped. Unlike before, every event was searched
        # *inside* this video, so an empty slot means the video really has no
        # candidate rather than that the event lost a global ranking.
        if any(not candidates for candidates in slots):
            continue
        best = _best_increasing_sequence(slots, max_gap)
        if best is not None:
            score, chosen = best
            sequences.append((score, video_id, chosen, slots))

    sequences.sort(key=lambda item: item[0], reverse=True)

    results = [
        SearchResult(
            rank=rank,
            video_id=video_id,
            frame_ids=[hit.representative_frame for hit in chosen],
            score=score,
            events=_event_hits(chosen, slots, max_gap),
        )
        for rank, (score, video_id, chosen, slots) in enumerate(
            sequences[: request.top_k], start=1
        )
    ]
    return _response("trake", results, config, timings)


def _candidate_videos(
    request: TrakeSearchRequest, config: RetrievalConfig, timings: Timings
) -> list[str]:
    """Videos worth a per-event search, best first.

    The overview describes the whole video and every event describes a moment
    in it, so both are evidence about the video and both are searched globally
    here. Coverage is deliberately *not* required at this stage: demanding a
    hit for every event is what dropped correct videos, and the events get
    another chance inside each candidate. A video the overview alone found
    still earns a full alignment pass.
    """
    overview = _best_per_video(
        retrieve(request.overview, TRAKE_STAGE_A_TOP_K, config, timings)
    )
    per_event = [
        _best_per_video(retrieve(event, TRAKE_STAGE_A_TOP_K, config, timings))
        for event in request.events
    ]

    scored: dict[str, float] = {}
    for video_id in set(overview) | {v for best in per_event for v in best}:
        events = sum(best.get(video_id, 0.0) for best in per_event) / len(per_event)
        scored[video_id] = overview.get(video_id, 0.0) + events

    ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
    return [video_id for video_id, _ in ranked[:TRAKE_VIDEO_CANDIDATES]]


def _best_per_video(hits: list[ScoredFrame]) -> dict[str, float]:
    """Each video's best score in one result list."""
    best: dict[str, float] = {}
    for hit in hits:
        if hit.score > best.get(hit.video_id, float("-inf")):
            best[hit.video_id] = hit.score
    return best


def _within_gap(previous: ScoredFrame, following: ScoredFrame, max_gap: float) -> bool:
    """Whether two consecutive events are close enough to be one action.

    Skipped when either hit has no timestamp: a missing time is not evidence
    that the events are far apart, and only clip-only points lack one.
    """
    if max_gap <= 0:
        return True
    if previous.pts_sec is None or following.pts_sec is None:
        return True
    return following.pts_sec - previous.pts_sec <= max_gap


def _best_increasing_sequence(
    slots: list[list[ScoredFrame]], max_gap: float = 0.0
) -> tuple[float, list[ScoredFrame]] | None:
    """Pick one candidate per event so that frame ids strictly increase.

    Events happen in the order given, so a valid sequence must move forward in
    time. This maximises the summed similarity over all valid orderings.

    Ordering is on the frame id rather than on `pts_sec`: it is what a
    submission reports and it rises with time within a video anyway. Seconds
    are used only for the gap, which is a duration and so cannot be expressed
    in frames when videos differ in frame rate.

    Still an exact search, not a greedy walk: whether a candidate may follow
    another depends only on that other candidate, so the best total ending at
    each candidate is a sufficient state.
    """
    sorted_slots = [
        sorted(candidates, key=lambda hit: hit.representative_frame)
        for candidates in slots
    ]

    # best[j] = (total score, hits) for a sequence ending at candidate j of
    # the event processed so far.
    best: list[tuple[float, list[ScoredFrame]]] = [
        (hit.score, [hit]) for hit in sorted_slots[0]
    ]

    for candidates in sorted_slots[1:]:
        updated: list[tuple[float, list[ScoredFrame]]] = []
        for hit in candidates:
            feasible = [
                (score, chain)
                for score, chain in best
                if chain[-1].representative_frame < hit.representative_frame
                and _within_gap(chain[-1], hit, max_gap)
            ]
            if not feasible:
                continue
            score, chain = max(feasible, key=lambda item: item[0])
            updated.append((score + hit.score, [*chain, hit]))

        if not updated:
            return None
        best = updated

    total, chain = max(best, key=lambda item: item[0])
    return total / len(slots), chain


def _event_hits(
    chosen: list[ScoredFrame], slots: list[list[ScoredFrame]], max_gap: float
) -> list[EventHit]:
    """Report where each event landed, with the runners-up it beat."""
    return [
        EventHit(
            event_index=index,
            frame_id=hit.representative_frame,
            shot_id=hit.shot_id,
            pts_sec=hit.pts_sec,
            score=hit.score,
            alternates=[
                _candidate(alternate)
                for alternate in _alternates(
                    slots[index],
                    hit,
                    chosen[index - 1] if index else None,
                    chosen[index + 1] if index + 1 < len(chosen) else None,
                    max_gap,
                )
            ],
        )
        for index, hit in enumerate(chosen)
    ]


def _alternates(
    candidates: list[ScoredFrame],
    chosen: ScoredFrame,
    previous: ScoredFrame | None,
    following: ScoredFrame | None,
    max_gap: float,
) -> list[ScoredFrame]:
    """Other candidates for one event that keep the sequence valid.

    Held to the same two rules the sequence itself had to satisfy against its
    neighbouring picks: after the previous event, before the next, and within
    the gap. Offering a candidate the ranker would have rejected would let an
    operator assemble a sequence the system does not consider a sequence.
    Candidates arrive score-ordered, so this is a filter and a slice.
    """
    return [
        hit
        for hit in candidates
        if hit.representative_frame != chosen.representative_frame
        and _follows(previous, hit, max_gap)
        and _follows(hit, following, max_gap)
    ][:TRAKE_ALTERNATES_PER_EVENT]


def _follows(
    previous: ScoredFrame | None, following: ScoredFrame | None, max_gap: float
) -> bool:
    """Whether two hits could be consecutive events, ignoring absent ends."""
    if previous is None or following is None:
        return True
    return (
        previous.representative_frame < following.representative_frame
        and _within_gap(previous, following, max_gap)
    )


def _candidate(hit: ScoredFrame) -> EventCandidate:
    return EventCandidate(
        frame_id=hit.representative_frame,
        shot_id=hit.shot_id,
        pts_sec=hit.pts_sec,
        score=hit.score,
    )


def _response(
    task: str,
    results: list[SearchResult],
    config: RetrievalConfig,
    timings: Timings,
) -> SearchResponse:
    return SearchResponse(
        request_id=str(uuid4()),
        task=task,
        results=results,
        versions=SearchVersions(
            frames_collection=config.frames_collection,
            clips_collection=config.clips_collection,
            model_config_name=config.feature_profile,
        ),
        latency_ms=timings.as_dict(),
    )
