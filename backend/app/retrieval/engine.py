"""Shared retrieval path: encode a query, search, collapse, rank.

Every track goes through here so that scoring, deduplication and timing stay
identical between them. Track modules decide *what* to encode and how to
assemble the final answer; they never re-implement the search itself.
"""

import time
from dataclasses import dataclass, field

from app.features.multimodal import embed_text
from app.ranking import dedupe
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


def encode_query(text: str, config: RetrievalConfig, timings: Timings) -> list[float]:
    started = time.perf_counter()
    try:
        return embed_text(config.feature_profile, text)
    finally:
        timings.record("encode", started)


def search_vector(
    vector: list[float],
    top_k: int,
    config: RetrievalConfig,
    timings: Timings,
    video_ids: list[str] | None = None,
) -> list[ScoredFrame]:
    started = time.perf_counter()
    try:
        return search(
            get_qdrant_client(),
            config.frames_collection,
            vector,
            limit=dedupe.overfetch_limit(top_k),
            query_filter=build_filter(video_ids=video_ids),
        )
    finally:
        timings.record("qdrant", started)


def rank(frames: list[ScoredFrame], top_k: int, timings: Timings) -> list[ScoredFrame]:
    started = time.perf_counter()
    try:
        return dedupe.dedupe_by_shot(frames, top_k)
    finally:
        timings.record("rerank", started)


def retrieve(
    text: str,
    top_k: int,
    config: RetrievalConfig,
    timings: Timings,
    video_ids: list[str] | None = None,
) -> list[ScoredFrame]:
    """Encode one text query and return deduplicated, ranked hits."""
    vector = encode_query(text, config, timings)
    hits = search_vector(vector, top_k, config, timings, video_ids)
    return rank(hits, top_k, timings)
