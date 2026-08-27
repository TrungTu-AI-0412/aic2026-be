"""Per-track orchestration on top of the shared retrieval engine.

Each track decides what text to encode and how to turn hits into results.
The search, deduplication and scoring underneath are shared, so a change to
ranking applies to every track at once.
"""

from uuid import uuid4

from app.retrieval.engine import (
    RetrievalConfig,
    Timings,
    retrieve,
    retrieve_by_ocr,
)
from app.schemas.search import (
    KisSearchRequest,
    OcrSearchRequest,
    QaSearchRequest,
    SearchResponse,
    SearchResult,
    SearchVersions,
    TrakeSearchRequest,
)
from app.vector_store.search import ScoredFrame

# How many shots per video are considered when assembling an event sequence.
TRAKE_CANDIDATES_PER_EVENT = 20


def search_kis(request: KisSearchRequest, config: RetrievalConfig) -> SearchResponse:
    timings = Timings()
    hits = retrieve(request.description, request.top_k, config, timings)

    return _response("kis", _one_frame_each(hits), config, timings)


def search_ocr(request: OcrSearchRequest, config: RetrievalConfig) -> SearchResponse:
    """Rank shots by the text printed on them, ignoring the image entirely.

    This is the companion to the boost the other tracks get: there, on-screen
    text nudges a visual ranking; here it *is* the ranking. A query whose
    whole content is a name or a headline has nothing for the image model to
    work with, and fusing a dense branch in would spend most of the result
    budget on frames that only look plausible.
    """
    timings = Timings()
    hits = retrieve_by_ocr(
        request.text, request.top_k, config, timings, request.video_ids
    )
    return _response("ocr", _one_frame_each(hits), config, timings)


def search_qa(request: QaSearchRequest, config: RetrievalConfig) -> SearchResponse:
    """Locate the moment the question is about.

    Retrieval uses the description, which describes the scene; the question
    describes what to read off it. `answer` stays None because no visual
    question-answering model is wired in - reporting a guess would be worse
    than reporting nothing.
    """
    timings = Timings()
    hits = retrieve(request.description, request.top_k, config, timings)

    return _response("qa", _one_frame_each(hits), config, timings)


def _one_frame_each(hits: list[ScoredFrame]) -> list[SearchResult]:
    """One result per hit, ranked. `answer` stays unset for every track."""
    return [
        SearchResult(
            rank=rank,
            video_id=hit.video_id,
            frame_ids=[hit.representative_frame],
            score=hit.score,
            ocr_text=hit.ocr_text,
        )
        for rank, hit in enumerate(hits, start=1)
    ]


def search_trake(request: TrakeSearchRequest, config: RetrievalConfig) -> SearchResponse:
    """Find one frame per event, in order, inside a single video.

    Each event is searched independently, then per video the best temporally
    increasing selection is chosen. A video must supply a candidate for every
    event to be a result; a partial sequence is not a valid submission.
    """
    timings = Timings()

    per_event: list[list[ScoredFrame]] = [
        retrieve(event, TRAKE_CANDIDATES_PER_EVENT, config, timings)
        for event in request.events
    ]

    by_video: dict[str, list[list[ScoredFrame]]] = {}
    for event_index, hits in enumerate(per_event):
        for hit in hits:
            slots = by_video.setdefault(
                hit.video_id, [[] for _ in range(len(request.events))]
            )
            slots[event_index].append(hit)

    sequences: list[tuple[float, str, list[int]]] = []
    for video_id, slots in by_video.items():
        if any(not candidates for candidates in slots):
            continue
        best = _best_increasing_sequence(slots)
        if best is not None:
            score, frame_ids = best
            sequences.append((score, video_id, frame_ids))

    sequences.sort(key=lambda item: item[0], reverse=True)

    results = [
        SearchResult(rank=rank, video_id=video_id, frame_ids=frame_ids, score=score)
        for rank, (score, video_id, frame_ids) in enumerate(
            sequences[: request.top_k], start=1
        )
    ]
    return _response("trake", results, config, timings)


def _best_increasing_sequence(
    slots: list[list[ScoredFrame]],
) -> tuple[float, list[int]] | None:
    """Pick one candidate per event so that frame ids strictly increase.

    Events happen in the order given, so a valid sequence must move forward in
    time. This maximises the summed similarity over all valid orderings.
    """
    sorted_slots = [
        sorted(candidates, key=lambda hit: hit.representative_frame)
        for candidates in slots
    ]

    # best[j] = (total score, frames) for a sequence ending at candidate j of
    # the event processed so far.
    best: list[tuple[float, list[int]]] = [
        (hit.score, [hit.representative_frame]) for hit in sorted_slots[0]
    ]

    for candidates in sorted_slots[1:]:
        updated: list[tuple[float, list[int]]] = []
        for hit in candidates:
            frame = hit.representative_frame
            feasible = [
                (score, frames)
                for score, frames in best
                if frames[-1] < frame
            ]
            if not feasible:
                continue
            score, frames = max(feasible, key=lambda item: item[0])
            updated.append((score + hit.score, [*frames, frame]))

        if not updated:
            return None
        best = updated

    total, frames = max(best, key=lambda item: item[0])
    return total / len(slots), frames


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
