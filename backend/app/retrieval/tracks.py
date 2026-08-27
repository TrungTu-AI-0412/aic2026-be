"""Per-track orchestration on top of the shared retrieval engine.

Each track decides what text to encode and how to turn hits into results.
The search, deduplication and scoring underneath are shared, so a change to
ranking applies to every track at once.
"""

import math
from dataclasses import dataclass, replace
from uuid import uuid4

from app.retrieval.engine import (
    RetrievalConfig,
    Timings,
    retrieve,
    retrieve_per_video,
    retrieve_video_scores,
    search_asr_only,
)
from app.retrieval.rewrite import Rewrite, rewrite_queries
from app.schemas.search import (
    EventCandidate,
    EventHit,
    AsrEvidence,
    KisSearchRequest,
    QaSearchRequest,
    QueryForms,
    SearchResponse,
    SearchResult,
    SearchVersions,
    RetrievalMode,
    StageAScore,
    TrakeSearchRequest,
)
from app.vector_store.search import ScoredFrame

# Videos that survive the global stage and earn per-event searches of their own.
# This is the cost dial: the second stage runs one Qdrant query per
# (video, event) pair, serially, so 100 videos x 5 events is 500 round trips.
# 100 because a submission ranks 100 lines and the track can return at most one
# result per candidate video.
TRAKE_VIDEO_CANDIDATES = 100
# Hits per query in the video-selection stage, over-fetched into the Qdrant
# limit like anywhere else - so this is really the width of the pool, since
# `retrieve_video_scores` neither collapses nor truncates what comes back. Far
# above TRAKE_VIDEO_CANDIDATES on purpose: dense hits cluster inside a few long
# videos, so the top 5000 frames of a query name a few hundred videos, not 5000.
TRAKE_STAGE_A_TOP_K = 1000
# Candidates per event *within one video*, not globally as this once meant.
TRAKE_CANDIDATES_PER_EVENT = 20
# Runners-up offered per event so an operator can override a chosen frame.
TRAKE_ALTERNATES_PER_EVENT = 5
# Widest gap between consecutive events. The events of one TRAKE query describe
# a single continuous action, so a chain spread across half an hour is noise,
# not a hit; without this a sequence spanning a 40-minute video scores exactly
# as well as one spanning three seconds.
TRAKE_MAX_GAP_SEC = 300.0
# What separation costs, as a share of the sequence score, the same units
# `asr_weight` is in. The hard cutoff above only kills the absurd; inside it a
# one-second gap and a 299-second gap used to score identically, which is wrong
# for a query describing one continuous action. Small on purpose: it reorders
# near-ties and cannot outvote a clearly better frame.
TRAKE_GAP_WEIGHT = 0.05


def search_kis(request: KisSearchRequest, config: RetrievalConfig) -> SearchResponse:
    """One frame, found directly or through the sequence it sits in.

    A description that walks through phases is averaged into one 40-word
    caption by the direct path, which is what temporal KIS exists to avoid: the
    operator decomposes it, every moment votes on the video on its own, and the
    row still reports a single frame.
    """
    if request.events is not None:
        return _temporal_search(
            "kis", request.overview, request.events, request, config, single_frame=True
        )
    return _frame_search(
        "kis", request.description, request.top_k, config, request.retrieval_mode
    )


def search_qa(request: QaSearchRequest, config: RetrievalConfig) -> SearchResponse:
    """Locate the moment the question is about.

    Byte-for-byte the same retrieval as KIS, because the question was never
    part of it: the description locates the scene, and the operator reads the
    answer off the frame and types it into the submission. The two tracks stay
    separate endpoints only because they export different row shapes.
    """
    return _frame_search(
        "qa", request.description, request.top_k, config, request.retrieval_mode
    )


def _frame_search(
    task: str,
    text: str,
    top_k: int,
    config: RetrievalConfig,
    requested_mode: RetrievalMode | None = None,
) -> SearchResponse:
    """One hit per shot, one frame per hit."""
    timings = Timings()
    rewritten = rewrite_queries([text], config, timings)
    query = rewritten[0] if rewritten else Rewrite(text, text)
    effective_mode: RetrievalMode = requested_mode or (
        "visual_asr"
        if config.asr_enabled and config.asr_collection and config.asr_weight > 0
        else "visual"
    )

    if effective_mode == "asr_only":
        hits = search_asr_only(query.speech, top_k, config, timings)
        results = [
            SearchResult(
                rank=rank,
                video_id=hit.frame.video_id,
                frame_ids=[hit.frame.representative_frame],
                score=hit.segment.score,
                asr_evidence=AsrEvidence(
                    segment=hit.segment.segment,
                    text=hit.segment.text,
                    start_sec=hit.segment.start_sec,
                    end_sec=hit.segment.end_sec,
                    score=hit.segment.score,
                ),
            )
            for rank, hit in enumerate(hits, start=1)
        ]
        return _response(
            task, results, config, timings, rewritten, effective_mode
        )

    hits = retrieve(
        query.vision, top_k, config, timings, speech_text=query.speech
    )

    results = [
        SearchResult(
            rank=rank,
            video_id=hit.video_id,
            frame_ids=[hit.representative_frame],
            score=hit.score,
        )
        for rank, hit in enumerate(hits, start=1)
    ]
    return _response(task, results, config, timings, rewritten, effective_mode)


def search_trake(request: TrakeSearchRequest, config: RetrievalConfig) -> SearchResponse:
    return _temporal_search(
        "trake", request.overview, request.events, request, config
    )


def _query_forms(
    parts: list[str | QueryForms], config: RetrievalConfig, timings: Timings
) -> tuple[list[Rewrite], list[Rewrite] | None]:
    """One `Rewrite` per part, plus what to report back as the rewritten forms.

    Parts that arrived already rewritten are used **verbatim** and cost no LLM
    call: that is the whole point of the review step, and re-captioning a
    caption would throw away the operator's edit. All or nothing, because a mix
    means the payload is not what the decompose screen produced, and the safe
    reading of that is to prepare the whole batch the ordinary way.
    """
    if all(isinstance(part, QueryForms) for part in parts):
        # A part whose caption never arrived searches the image tower with its
        # original-language text - worse than a caption, better than nothing.
        forms = [
            Rewrite(vision=part.vision or part.speech, speech=part.speech)
            for part in parts
        ]
        return forms, forms

    texts = [part if isinstance(part, str) else part.speech for part in parts]
    rewritten = rewrite_queries(texts, config, timings)
    return rewritten or [Rewrite(text, text) for text in texts], rewritten


def _temporal_search(
    task: str,
    overview: str | QueryForms,
    events: list[str | QueryForms],
    request: TrakeSearchRequest | KisSearchRequest,
    config: RetrievalConfig,
    single_frame: bool = False,
) -> SearchResponse:
    """Find one frame per event, in order, inside a single video.

    Two stages, because the two questions are different. Which video is a
    whole-video judgement the overview answers best; where each event sits is a
    within-video judgement that only makes sense once the video is fixed.
    Searching every event globally and intersecting afterwards - the previous
    shape - loses the correct video whenever any single event fails to reach a
    global top-N, which a fine-grained event routinely does.

    `single_frame` is temporal KIS: the same search, reported as the one frame
    that track submits. The sequence is still what found it, and still rides
    along in `events[]` for the operator to override from.
    """
    timings = Timings()
    # Overview and every event in one rewrite call. Each is searched twice - once
    # globally to choose videos, once inside each candidate - and a request-wide
    # batch keeps that at a single round trip either way.
    queries, rewritten = _query_forms([overview, *events], config, timings)

    video_ids = getattr(request, "video_ids", None)
    stage_a: dict[str, _StageAVideo] = {}
    if video_ids:
        videos = list(dict.fromkeys(video_ids))
    else:
        # A result needs a candidate video to live in, so aligning more videos
        # than the operator asked for rows is work that could never produce one
        # - and stage B pays `videos x events` serial round trips for it.
        ranked = _candidate_videos(
            queries, min(TRAKE_VIDEO_CANDIDATES, request.top_k), config, timings
        )
        stage_a = {entry.video_id: entry for entry in ranked}
        videos = [entry.video_id for entry in ranked]
    if not videos:
        return _response(task, [], config, timings, rewritten)

    # Reranking is off for the second stage. The cross-encoder head covers the
    # top `rerank_top_n` of one global list, so against per-video candidate
    # sets it would rescore an arbitrary slice of them; making it meaningful
    # costs one forward pass per (video, event, candidate). The video is
    # already chosen by here, and within one video the dense space is what
    # separates its own frames.
    localise = replace(config, rerank_enabled=False)
    per_event = [
        retrieve_per_video(
            event.vision,
            videos,
            TRAKE_CANDIDATES_PER_EVENT,
            localise,
            timings,
            speech_text=event.speech,
        )
        for event in queries[1:]
    ]

    max_gap = (
        TRAKE_MAX_GAP_SEC if request.max_gap_sec is None else request.max_gap_sec
    )
    gap_weight = (
        TRAKE_GAP_WEIGHT if request.gap_weight is None else request.gap_weight
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
        best = _best_increasing_sequence(slots, max_gap, gap_weight)
        if best is not None:
            score, chosen = best
            sequences.append((score, video_id, chosen, slots))

    sequences.sort(key=lambda item: item[0], reverse=True)

    results = [
        SearchResult(
            rank=rank,
            video_id=video_id,
            frame_ids=_frame_ids(chosen, single_frame),
            score=score,
            events=_event_hits(chosen, slots, max_gap, gap_weight),
            stage_a=_stage_a_score(stage_a.get(video_id)),
        )
        for rank, (score, video_id, chosen, slots) in enumerate(
            sequences[: request.top_k], start=1
        )
    ]
    return _response(task, results, config, timings, rewritten)


def _frame_ids(chosen: list[ScoredFrame], single_frame: bool) -> list[int]:
    """What the track submits: the whole chain, or the one frame KIS reports.

    Temporal KIS submits the highest-scoring event of the chain, not the first
    or the overview's: the alignment already put every event in one compact,
    in-order run inside the chosen video, and KIS ground truth is a *set* of
    acceptable frames covering the described action - so all of them are in the
    window, and the best-scoring one is the moment the model is surest it saw.
    """
    if not single_frame:
        return [hit.representative_frame for hit in chosen]
    return [max(chosen, key=lambda hit: hit.score).representative_frame]


def _stage_a_score(entry: "_StageAVideo | None") -> StageAScore | None:
    if entry is None:
        return None
    return StageAScore(
        rank=entry.rank,
        score=entry.score,
        overview_score=entry.overview_score,
        event_scores=list(entry.event_scores),
    )


@dataclass(frozen=True)
class _StageAVideo:
    """One video's case for being searched event by event."""

    video_id: str
    rank: int
    score: float
    overview_score: float
    event_scores: tuple[float, ...]


def _candidate_videos(
    queries: list[Rewrite], limit: int, config: RetrievalConfig, timings: Timings
) -> list[_StageAVideo]:
    """Videos worth a per-event search, best first.

    The overview describes the whole video and every event describes a moment
    in it, so both are evidence about the video and both are searched globally
    here. Coverage is deliberately *not* required at this stage: demanding a
    hit for every event is what dropped correct videos, and the events get
    another chance inside each candidate. A video the overview alone found
    still earns a full alignment pass.

    Each query is reduced to one score per video by `retrieve_video_scores`
    rather than to a page of hits: this stage is judged on how many *videos* it
    names, and a shot-collapsed top-N names hardly any, because dense hits pile
    up inside a few long videos.

    `queries` is the overview followed by the events, each carrying the form the
    image space searches with and the form the speech stage searches with.
    """
    overview = retrieve_video_scores(
        queries[0].vision,
        TRAKE_STAGE_A_TOP_K,
        config,
        timings,
        speech_text=queries[0].speech,
    )
    per_event = [
        retrieve_video_scores(
            event.vision,
            TRAKE_STAGE_A_TOP_K,
            config,
            timings,
            speech_text=event.speech,
        )
        for event in queries[1:]
    ]

    # The parts are kept, not just the sum: "won on the overview, lost every
    # event" and its reverse score the same here and mean opposite things to an
    # operator deciding whether to trust the video.
    scored: list[tuple[float, str, float, tuple[float, ...]]] = []
    for video_id in set(overview) | {v for best in per_event for v in best}:
        event_scores = tuple(best.get(video_id, 0.0) for best in per_event)
        overview_score = overview.get(video_id, 0.0)
        composite = overview_score + sum(event_scores) / len(event_scores)
        scored.append((composite, video_id, overview_score, event_scores))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        _StageAVideo(
            video_id=video_id,
            rank=rank,
            score=composite,
            overview_score=overview_score,
            event_scores=event_scores,
        )
        for rank, (composite, video_id, overview_score, event_scores) in enumerate(
            scored[:limit], start=1
        )
    ]


def _gap_sec(previous: ScoredFrame, following: ScoredFrame) -> float | None:
    """Seconds between two consecutive picks, or None when that is unknown.

    Only clip-only points lack a timestamp. Both the hard cutoff and the soft
    penalty read None as "no information", never as "far apart" or "adjacent" -
    so they ask the question here rather than each answering it their own way.
    """
    if previous.pts_sec is None or following.pts_sec is None:
        return None
    return following.pts_sec - previous.pts_sec


def _within_gap(previous: ScoredFrame, following: ScoredFrame, max_gap: float) -> bool:
    """Whether two consecutive events are close enough to be one action."""
    if max_gap <= 0:
        return True
    gap = _gap_sec(previous, following)
    return gap is None or gap <= max_gap


def _gap_penalty(previous: ScoredFrame, following: ScoredFrame) -> float:
    """What the separation between two consecutive events costs, 0 at no gap.

    Normalised so that a gap of `TRAKE_MAX_GAP_SEC` costs exactly 1.0 and the
    weight applied to this reads as a share of the score, like `asr_weight`.
    Deliberately normalised by the module constant rather than the request's
    `max_gap_sec`: an operator who sets that to 0 has disabled the cutoff, not
    asked for spread events to rank as well as tight ones - the penalty is the
    better version of the rule they just turned off. It would also divide by
    zero. Gaps beyond the constant simply score above 1.0; there is nothing to
    clamp, since a bigger gap really is worse.

    `log1p` is steepest near zero, so this separates small gaps sharply and
    large ones barely. With the cutoff above that is the right division of
    labour: the cutoff rejects the absurd, this ranks the plausible.
    """
    gap = _gap_sec(previous, following)
    if gap is None or gap <= 0:
        return 0.0
    return math.log1p(gap) / math.log1p(TRAKE_MAX_GAP_SEC)


def _best_increasing_sequence(
    slots: list[list[ScoredFrame]],
    max_gap: float = 0.0,
    gap_weight: float = 0.0,
) -> tuple[float, list[ScoredFrame]] | None:
    """Pick one candidate per event so that frame ids strictly increase.

    Events happen in the order given, so a valid sequence must move forward in
    time. This maximises `mean(event scores) - gap_weight * mean(gap penalty)`
    over all valid orderings.

    Ordering is on the frame id rather than on `pts_sec`: it is what a
    submission reports and it rises with time within a video anyway. Seconds
    are used only for the gap, which is a duration and so cannot be expressed
    in frames when videos differ in frame rate.

    Still an exact search, not a greedy walk. The gap penalty puts a cost on
    the *edge* between two picks, so it has to be charged before a predecessor
    is chosen - taking the best-scoring chain and then paying the gap is what
    would make this greedy, and it returns the wrong answer whenever a slightly
    worse predecessor sits closer in time. Folding the edge cost into the value
    being maximised keeps it exact, because feasibility and edge cost both look
    only at the previous candidate, which leaves "best total ending at each
    candidate" a sufficient state.
    """
    sorted_slots = [
        sorted(candidates, key=lambda hit: hit.representative_frame)
        for candidates in slots
    ]

    # An n-event sequence has n-1 gaps but is divided by n below, so a weight
    # tuned at two events would land 1.6x harder at five. Scaling here makes
    # the final division come out as mean(score) - gap_weight * mean(penalty),
    # so the knob means the same thing whatever the event count.
    edges = len(slots) - 1
    edge_weight = gap_weight * len(slots) / edges if edges else 0.0

    # best[j] = (total score, hits) for a sequence ending at candidate j of
    # the event processed so far.
    best: list[tuple[float, list[ScoredFrame]]] = [
        (hit.score, [hit]) for hit in sorted_slots[0]
    ]

    for candidates in sorted_slots[1:]:
        updated: list[tuple[float, list[ScoredFrame]]] = []
        for hit in candidates:
            feasible = [
                (score - edge_weight * _gap_penalty(chain[-1], hit), chain)
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
    chosen: list[ScoredFrame],
    slots: list[list[ScoredFrame]],
    max_gap: float,
    gap_weight: float = 0.0,
) -> list[EventHit]:
    """Report where each event landed, with the runners-up it beat.

    `score` is the event's own similarity, deliberately not net of the gap
    penalty: the penalty is a property of the sequence, not of one pick, and
    charging a share of it here would report a frame as worse than it scored.
    """
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
                    gap_weight,
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
    gap_weight: float = 0.0,
) -> list[ScoredFrame]:
    """Other candidates for one event that keep the sequence valid, best first.

    Held to the same two rules the sequence itself had to satisfy against its
    neighbouring picks: after the previous event, before the next, and within
    the gap. Offering a candidate the ranker would have rejected would let an
    operator assemble a sequence the system does not consider a sequence.

    Ranked by what the swap would actually cost, which is no longer the same as
    score order once a gap carries a penalty: a slightly worse frame seconds
    from its neighbours beats a better one minutes away, and offering them in
    score order would recommend the swap the sequence search itself ranks
    worst. Absent neighbours - the first and last events - contribute nothing.
    """
    offered = [
        hit
        for hit in candidates
        if hit.representative_frame != chosen.representative_frame
        and _follows(previous, hit, max_gap)
        and _follows(hit, following, max_gap)
    ]
    if gap_weight <= 0:
        # Candidates arrive score-ordered, so this stays a filter and a slice.
        return offered[:TRAKE_ALTERNATES_PER_EVENT]

    def swap_score(hit: ScoredFrame) -> float:
        penalty = 0.0
        if previous is not None:
            penalty += _gap_penalty(previous, hit)
        if following is not None:
            penalty += _gap_penalty(hit, following)
        return hit.score - gap_weight * penalty

    offered.sort(key=swap_score, reverse=True)
    return offered[:TRAKE_ALTERNATES_PER_EVENT]


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
    rewritten: list[Rewrite] | None = None,
    effective_mode: RetrievalMode | None = None,
) -> SearchResponse:
    resolved_mode: RetrievalMode = effective_mode or (
        "visual_asr"
        if config.asr_enabled and config.asr_collection and config.asr_weight > 0
        else "visual"
    )
    return SearchResponse(
        request_id=str(uuid4()),
        task=task,
        effective_retrieval_mode=resolved_mode,
        results=results,
        # Parallel lists preserve request order for both retrieval spaces.
        rewritten_queries=(
            [query.vision for query in rewritten] if rewritten else None
        ),
        cleaned_queries=(
            [query.speech for query in rewritten] if rewritten else None
        ),
        versions=SearchVersions(
            frames_collection=config.frames_collection,
            clips_collection=config.clips_collection,
            model_config_name=config.feature_profile,
            asr_collection=config.asr_collection,
            asr_model_config_name=config.asr_profile,
        ),
        latency_ms=timings.as_dict(),
    )
