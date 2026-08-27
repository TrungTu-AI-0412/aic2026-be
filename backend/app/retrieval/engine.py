"""Shared retrieval path: encode a query, search, collapse, rank.

Every track goes through here so that scoring, deduplication and timing stay
identical between them. Track modules decide *what* to encode and how to
assemble the final answer; they never re-implement the search itself.
"""

import time
from dataclasses import dataclass, field

from app.features import sparse
from app.features.multimodal import embed_text
from app.features.sparse import SparseVector
from app.ranking import boost, dedupe, fusion, rerank
from app.vector_store import collections
from app.vector_store.client import get_qdrant_client
from app.vector_store.search import (
    ScoredFrame,
    build_filter,
    search,
    search_sparse,
)


@dataclass
class Timings:
    """Per-stage latency in milliseconds, reported back to the operator."""

    values: dict[str, float] = field(default_factory=dict)

    def record(self, stage: str, started: float) -> None:
        elapsed = (time.perf_counter() - started) * 1000.0
        self.values[stage] = self.values.get(stage, 0.0) + elapsed

    def as_dict(self) -> dict[str, float]:
        return {stage: round(value, 3) for stage, value in self.values.items()}


@dataclass(frozen=True)
class RetrievalConfig:
    frames_collection: str
    feature_profile: str
    clips_collection: str | None = None
    clip_weight: float = fusion.DEFAULT_CLIP_WEIGHT
    rerank_enabled: bool = True
    rerank_top_n: int = rerank.DEFAULT_TOP_N
    rerank_model: str = rerank.DEFAULT_MODEL
    # Off for collections ingested before the lexical vectors existed: they
    # have no sparse slots, and a prefetch against a vector the collection
    # does not declare fails the whole query rather than degrading.
    hybrid_enabled: bool = True
    # Run on-screen text as its own weighted channel, fused onto the visual
    # ranking by rank, instead of leaving it as one more equal branch inside
    # Qdrant's RRF. Costs one extra sparse query per retrieve() call, which is
    # cheap next to the dense branch but is per-event on TRAKE.
    ocr_boost_enabled: bool = True
    ocr_boost_weight: float = boost.DEFAULT_OCR_WEIGHT


def encode_query(text: str, config: RetrievalConfig, timings: Timings) -> list[float]:
    started = time.perf_counter()
    try:
        return embed_text(config.feature_profile, text)
    finally:
        timings.record("encode", started)


def encode_query_sparse(
    text: str, config: RetrievalConfig
) -> SparseVector | None:
    """Lexical form of the query, or None when hybrid search is off.

    Not timed as its own stage: tokenising a query string is microseconds
    against a transformer forward pass, and a timing entry that always reads
    0.0 is noise in every response.
    """
    if not config.hybrid_enabled:
        return None
    encoded = sparse.encode(text)
    return encoded or None


def fused_sparse_names(config: RetrievalConfig) -> tuple[str, ...]:
    """Which lexical slots the dense query fuses in server-side.

    When the OCR boost is on, `ocr` comes out: it is queried separately and
    fused back with a weight the operator controls. Leaving it in as well
    would count on-screen text twice, once at RRF's fixed weight and once
    again at the configured one.
    """
    if config.ocr_boost_enabled:
        return (collections.SPARSE_SPEECH, collections.SPARSE_CAPTION)
    return collections.SPARSE_VECTOR_NAMES


def search_ocr(
    sparse_query: SparseVector,
    top_k: int,
    config: RetrievalConfig,
    timings: Timings,
    video_ids: list[str] | None = None,
) -> list[ScoredFrame]:
    """Search the frame index on on-screen text alone.

    Frames only, never clips. A clip point knows a shot's frame range but not
    which frame inside it carries the text, and the whole reason to search
    on-screen text is that the operator wants the frame where it is legible.
    """
    started = time.perf_counter()
    try:
        return search_sparse(
            get_qdrant_client(),
            config.frames_collection,
            sparse_query,
            using=collections.SPARSE_OCR,
            limit=dedupe.overfetch_limit(top_k),
            query_filter=build_filter(video_ids=video_ids),
        )
    finally:
        timings.record("ocr", started)


def search_vector(
    vector: list[float],
    top_k: int,
    config: RetrievalConfig,
    timings: Timings,
    video_ids: list[str] | None = None,
    sparse_query: SparseVector | None = None,
) -> list[ScoredFrame]:
    """Search the frame index, fused with the clip index and the OCR channel."""
    started = time.perf_counter()
    try:
        client = get_qdrant_client()
        limit = dedupe.overfetch_limit(top_k)
        query_filter = build_filter(video_ids=video_ids)
        frames = search(
            client,
            config.frames_collection,
            vector,
            limit=limit,
            query_filter=query_filter,
            sparse_query=sparse_query,
            sparse_names=fused_sparse_names(config),
        )
        clips = (
            search(
                client,
                config.clips_collection,
                vector,
                limit=limit,
                query_filter=query_filter,
                sparse_query=sparse_query,
                sparse_names=fused_sparse_names(config),
            )
            if config.clips_collection
            else []
        )
    finally:
        timings.record("qdrant", started)

    on_screen: list[ScoredFrame] = []
    if sparse_query and config.ocr_boost_enabled:
        on_screen = search_ocr(sparse_query, top_k, config, timings, video_ids)

    started = time.perf_counter()
    try:
        fused = fusion.fuse_frames_and_clips(frames, clips, config.clip_weight)
        if not on_screen:
            return fused
        return boost.reciprocal_rank_fuse(
            fused, on_screen, config.ocr_boost_weight
        )
    finally:
        timings.record("fuse", started)


def rank(
    frames: list[ScoredFrame],
    text: str,
    top_k: int,
    config: RetrievalConfig,
    timings: Timings,
) -> list[ScoredFrame]:
    started = time.perf_counter()
    try:
        hits = dedupe.dedupe_by_shot(frames, top_k)
    finally:
        timings.record("dedupe", started)

    if not config.rerank_enabled or not hits:
        return hits

    started = time.perf_counter()
    try:
        return rerank.rerank(text, hits, config.rerank_top_n, config.rerank_model)
    finally:
        timings.record("rerank", started)


def retrieve(
    text: str,
    top_k: int,
    config: RetrievalConfig,
    timings: Timings,
    video_ids: list[str] | None = None,
) -> list[ScoredFrame]:
    """Encode one text query and return deduplicated, reranked hits."""
    vector = encode_query(text, config, timings)
    sparse_query = encode_query_sparse(text, config)
    hits = search_vector(vector, top_k, config, timings, video_ids, sparse_query)
    return rank(hits, text, top_k, config, timings)


def retrieve_by_ocr(
    text: str,
    top_k: int,
    config: RetrievalConfig,
    timings: Timings,
    video_ids: list[str] | None = None,
) -> list[ScoredFrame]:
    """Return shots whose on-screen text matches, with no visual signal at all.

    Neither the image encoder nor the reranker runs. Skipping the encoder is
    the point - this path answers in milliseconds because it never touches a
    transformer. Skipping the reranker is a correctness matter: BLIP ITM
    scores how well an image *depicts* a caption, and it cannot read a ticker,
    so letting it reorder these hits would demote exactly the frames the query
    asked for in favour of ones that merely look like the words.

    `fold_diacritics` is left on. The ingest folded too, and this corpus's OCR
    fails almost exclusively on diacritics, so a query typed with correct
    Vietnamese has to reach the damaged spelling to find anything.
    """
    sparse_query = sparse.encode(text)
    if not sparse_query:
        return []
    hits = search_ocr(sparse_query, top_k, config, timings, video_ids)

    started = time.perf_counter()
    try:
        return dedupe.dedupe_by_shot(hits, top_k)
    finally:
        timings.record("dedupe", started)
