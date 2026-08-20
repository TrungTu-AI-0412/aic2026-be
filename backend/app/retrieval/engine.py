"""Shared retrieval path: encode a query, search, collapse, rank.

Every track goes through here so that scoring, deduplication and timing stay
identical between them. Track modules decide *what* to encode and how to
assemble the final answer; they never re-implement the search itself.
"""

import time
from dataclasses import dataclass, field

from app.features import sparse, text as text_features
from app.features.multimodal import embed_text
from app.features.sparse import SparseVector
from app.ranking import asr, dedupe, fusion, rerank
from app.vector_store.client import get_qdrant_client
from app.vector_store.search import (
    AsrSegment,
    ScoredFrame,
    build_filter,
    search,
    search_asr,
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
    sparse_method: str = "bm25"
    splade_model: str | None = None
    # Lexical slots the frame collection actually populates. Empty by default
    # because querying a declared-but-unfilled slot raises rather than degrading;
    # becomes ("ocr",) once on-screen text is upserted.
    frame_sparse_names: tuple[str, ...] = ()

    # Speech overlap. A query also searches the segment collection, and each
    # frame gains a share of the best-scoring segment that covers it in time.
    # Unset collection or zero weight disables the stage outright.
    asr_collection: str | None = None
    asr_enabled: bool = True
    asr_profile: str = "qwen3-embed-0.6b-v1"
    asr_weight: float = asr.DEFAULT_WEIGHT
    asr_dense_weight: float = asr.DEFAULT_DENSE_WEIGHT
    asr_sparse_weight: float = asr.DEFAULT_SPARSE_WEIGHT
    asr_pad_sec: float = asr.DEFAULT_PAD_SEC


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

    Not timed as its own stage for BM25: tokenising a query string is microseconds
    against a transformer forward pass.
    """
    if not config.hybrid_enabled:
        return None
    kwargs = {}
    if config.splade_model:
        kwargs["model_id"] = config.splade_model
    encoded = sparse.encode(text, method=config.sparse_method, **kwargs)
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
            sparse_names=config.frame_sparse_names,
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
            sparse_names=config.frame_sparse_names,
        )
    finally:
        timings.record("qdrant", started)

    started = time.perf_counter()
    try:
        return fusion.fuse_frames_and_clips(frames, clips, config.clip_weight)
    finally:
        timings.record("fuse", started)


def search_speech(
    text: str,
    top_k: int,
    config: RetrievalConfig,
    timings: Timings,
    video_ids: list[str] | None = None,
) -> list[AsrSegment]:
    """Retrieve speech segments matching the query, dense and lexical fused.

    Returns an empty list when the stage is off or unconfigured, so the caller
    can treat "no speech collection" and "no matching speech" identically.
    """
    if not config.asr_enabled or not config.asr_collection or config.asr_weight <= 0:
        return []

    started = time.perf_counter()
    try:
        dense = (
            text_features.embed_query(config.asr_profile, text)
            if config.asr_dense_weight > 0
            else None
        )
        lexical = (
            encode_query_sparse(text, config)
            if config.asr_sparse_weight > 0
            else None
        )
        if dense is None and lexical is None:
            return []

        dense_hits, sparse_hits = search_asr(
            get_qdrant_client(),
            config.asr_collection,
            dense,
            lexical,
            limit=dedupe.overfetch_limit(top_k),
            query_filter=build_filter(video_ids=video_ids),
        )
        return asr.fuse_asr(
            dense_hits,
            sparse_hits,
            config.asr_dense_weight,
            config.asr_sparse_weight,
        )
    finally:
        timings.record("asr", started)


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


def retrieve_per_video(
    text: str,
    video_ids: list[str],
    limit: int,
    config: RetrievalConfig,
    timings: Timings,
) -> dict[str, list[ScoredFrame]]:
    """Top `limit` hits for `text` inside each of `video_ids`, keyed by video.

    One encode, then one filtered query per video. A single query filtered to
    all the videos at once would return the global top-N *across* them, which
    starves a correct video that ranks low overall - the same recall failure
    the caller's two-stage split exists to fix, one level down.

    Shots are deliberately not collapsed. `dedupe_by_shot` keeps one frame per
    shot, and two events of a TRAKE query can happen inside one two-second
    shot, so collapsing would make that sequence unrepresentable rather than
    merely rank it worse.
    """
    if not video_ids:
        return {}

    vector = encode_query(text, config, timings)
    sparse_query = encode_query_sparse(text, config)

    per_video = {
        video_id: search_vector(
            vector, limit, config, timings, [video_id], sparse_query
        )
        for video_id in video_ids
    }

    # One speech query for the whole candidate set, and one bonus pass over the
    # flattened hits. `apply_asr_bonus` min-max normalises the segments it is
    # given, so boosting each video from its own segment list would make the
    # bonuses incomparable between videos - exactly what ranking them needs.
    segments = search_speech(text, limit, config, timings, video_ids)
    if segments:
        started = time.perf_counter()
        try:
            boosted = asr.apply_asr_bonus(
                [hit for hits in per_video.values() for hit in hits],
                segments,
                config.asr_weight,
                config.asr_pad_sec,
            )
        finally:
            timings.record("asr_bonus", started)
        per_video = {video_id: [] for video_id in video_ids}
        for hit in boosted:
            if hit.video_id in per_video:
                per_video[hit.video_id].append(hit)

    return {
        video_id: sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]
        for video_id, hits in per_video.items()
    }


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

    # Before dedupe on purpose. The bonus is applied per frame, and dedupe keeps
    # the best frame per shot, so boosting first lets speech decide *which*
    # frame represents a shot as well as where that shot ranks.
    segments = search_speech(text, top_k, config, timings, video_ids)
    if segments:
        started = time.perf_counter()
        try:
            hits = asr.apply_asr_bonus(
                hits, segments, config.asr_weight, config.asr_pad_sec
            )
        finally:
            timings.record("asr_bonus", started)

    return rank(hits, text, top_k, config, timings)
