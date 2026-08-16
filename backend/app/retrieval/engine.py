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
from app.ranking import dedupe, fusion, rerank
from app.vector_store.client import get_qdrant_client
from app.vector_store.search import ScoredFrame, build_filter, search


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


def search_vector(
    vector: list[float],
    top_k: int,
    config: RetrievalConfig,
    timings: Timings,
    video_ids: list[str] | None = None,
    sparse_query: SparseVector | None = None,
) -> list[ScoredFrame]:
    """Search the frame index, fused with the clip index when one is set."""
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
        )
        if not config.clips_collection:
            return frames
        clips = search(
            client,
            config.clips_collection,
            vector,
            limit=limit,
            query_filter=query_filter,
            sparse_query=sparse_query,
        )
    finally:
        timings.record("qdrant", started)

    started = time.perf_counter()
    try:
        return fusion.fuse_frames_and_clips(frames, clips, config.clip_weight)
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
