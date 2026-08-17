"""Vector similarity search against an ingested collection.

Everything Qdrant-specific for querying lives here: filter construction and
the query call itself. Callers receive plain dataclasses so ranking and the
API layer never import Qdrant types.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.features.sparse import SparseVector
from app.vector_store import collections

DEFAULT_LIMIT = 100


@dataclass(frozen=True)
class ScoredFrame:
    """One retrieved point, flattened to the fields ranking needs."""

    score: float
    video_id: str
    shot_id: int
    original_frame_id: int | None
    start_frame: int | None
    end_frame: int | None
    path: str | None
    # Clip points only: reranking has to decode the shot back out of the
    # source video, and that needs timestamps, not frame indexes.
    start_sec: float | None = None
    end_sec: float | None = None

    @property
    def representative_frame(self) -> int:
        """The frame id to report for this hit.

        Keyframe points carry an exact id; clip points only know their range,
        so the first frame of the shot stands in.
        """
        if self.original_frame_id is not None:
            return self.original_frame_id
        return self.start_frame or 0


def build_filter(
    video_ids: list[str] | None = None,
    shot_ids: list[int] | None = None,
) -> qmodels.Filter | None:
    """Build a payload filter, or None when nothing is constrained."""
    conditions: list[qmodels.FieldCondition] = []

    if video_ids:
        conditions.append(
            qmodels.FieldCondition(
                key="video_id", match=qmodels.MatchAny(any=list(video_ids))
            )
        )
    if shot_ids:
        conditions.append(
            qmodels.FieldCondition(
                key="shot_id", match=qmodels.MatchAny(any=list(shot_ids))
            )
        )

    return qmodels.Filter(must=conditions) if conditions else None


def search(
    client: QdrantClient,
    collection_name: str,
    vector: list[float],
    limit: int = DEFAULT_LIMIT,
    query_filter: qmodels.Filter | None = None,
    sparse_query: SparseVector | None = None,
    sparse_names: Sequence[str] = (
        collections.SPARSE_SPEECH,
        collections.SPARSE_OCR,
    ),
) -> list[ScoredFrame]:
    """Dense search, fused with lexical search when the query has terms.

    With no sparse query this is a plain dense lookup, which keeps the path
    unchanged for collections that predate the lexical vectors.

    With one, each branch runs as a prefetch and Qdrant fuses them with
    Reciprocal Rank Fusion. RRF combines *ranks*, not scores, which matters
    here because a cosine similarity and an IDF-weighted lexical score have no
    common scale — a weighted sum of the two would be dominated by whichever
    happens to have the wider range on a given query.

    `sparse_names` covers the slots that hold vectors, not every declared one.
    `caption` stays out: the collection declares it so it can be filled by a
    later re-upsert, but querying a slot no point carries is wasted work
    against a server and raises outright against the in-memory client, which
    only registers a slot's IDF statistics once a point uses it. Narrow this
    argument when ingesting a manifest that predates OCR.
    """
    if sparse_query:
        prefetch = [
            qmodels.Prefetch(
                query=vector,
                using=collections.DENSE_VECTOR_NAME,
                limit=limit,
                filter=query_filter,
            )
        ]
        prefetch += [
            qmodels.Prefetch(
                query=qmodels.SparseVector(
                    indices=sparse_query.indices, values=sparse_query.values
                ),
                using=name,
                limit=limit,
                filter=query_filter,
            )
            for name in sparse_names
        ]
        response = client.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
    else:
        response = client.query_points(
            collection_name=collection_name,
            query=vector,
            using=collections.DENSE_VECTOR_NAME,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
    return [_to_scored_frame(point) for point in response.points]


def _to_scored_frame(point) -> ScoredFrame:
    payload = point.payload or {}
    return ScoredFrame(
        score=float(point.score),
        video_id=str(payload.get("video_id", "")),
        shot_id=int(payload.get("shot_id", 0)),
        original_frame_id=_optional_int(payload.get("original_frame_id")),
        start_frame=_optional_int(payload.get("start_frame")),
        end_frame=_optional_int(payload.get("end_frame")),
        path=payload.get("path"),
        start_sec=_optional_float(payload.get("start_sec")),
        end_sec=_optional_float(payload.get("end_sec")),
    )


def _optional_int(value) -> int | None:
    return None if value is None else int(value)


def _optional_float(value) -> float | None:
    return None if value is None else float(value)
