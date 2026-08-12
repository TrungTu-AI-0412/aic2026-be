"""Vector similarity search against an ingested collection.

Everything Qdrant-specific for querying lives here: filter construction and
the query call itself. Callers receive plain dataclasses so ranking and the
API layer never import Qdrant types.
"""

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

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
) -> list[ScoredFrame]:
    response = client.query_points(
        collection_name=collection_name,
        query=vector,
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
