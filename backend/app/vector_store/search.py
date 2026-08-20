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
    # Keyframe points: when in the video this frame is, and the span of the shot
    # it came from. Both are read by the ASR overlap bonus, which has to ask
    # whether a stretch of speech covers this frame. They were already in the
    # payload and simply discarded before.
    pts_sec: float | None = None
    shot_start_sec: float | None = None
    shot_end_sec: float | None = None

    def time_window(self, pad_sec: float = 0.0) -> tuple[float, float] | None:
        """The span of video time this hit covers, or None if unknown.

        Prefers the shot range, falling back to the frame's own instant. A
        keyframe is a single moment but the speech describing it runs either
        side, so an instant alone would match almost no segment.
        """
        if self.shot_start_sec is not None and self.shot_end_sec is not None:
            if self.shot_end_sec > self.shot_start_sec:
                return (self.shot_start_sec - pad_sec, self.shot_end_sec + pad_sec)
        if self.start_sec is not None and self.end_sec is not None:
            return (self.start_sec - pad_sec, self.end_sec + pad_sec)
        if self.pts_sec is not None:
            return (self.pts_sec - pad_sec, self.pts_sec + pad_sec)
        return None

    @property
    def representative_frame(self) -> int:
        """The frame id to report for this hit.

        Keyframe points carry an exact id; clip points only know their range,
        so the first frame of the shot stands in.
        """
        if self.original_frame_id is not None:
            return self.original_frame_id
        return self.start_frame or 0


@dataclass(frozen=True)
class AsrSegment:
    """One speech segment retrieved from the ASR collection.

    Lives here beside `ScoredFrame` for the same reason: search returns plain
    dataclasses so ranking and the API never import Qdrant types. Ranking
    depends on this module, never the other way round.
    """

    score: float
    video_id: str
    start_sec: float
    end_sec: float
    segment: int = 0
    text: str = ""


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
    sparse_names: Sequence[str] = (),
    dense_name: str = collections.DENSE_VECTOR_NAME,
) -> list[ScoredFrame]:
    """Dense search, fused with lexical search when the query has terms.

    With no sparse query this is a plain dense lookup, which keeps the path
    unchanged for collections that predate the lexical vectors.

    With one, each branch runs as a prefetch and Qdrant fuses them with
    Reciprocal Rank Fusion. RRF combines *ranks*, not scores, which matters
    here because a cosine similarity and an IDF-weighted lexical score have no
    common scale — a weighted sum of the two would be dominated by whichever
    happens to have the wider range on a given query.

    `sparse_names` covers the slots that actually hold vectors, not every slot
    the collection declares. It defaults to empty because querying a slot no
    point carries is wasted work against a server and raises outright against
    the in-memory client, which only registers a slot's IDF statistics once a
    point uses it. Frames declare `ocr` but do not populate it yet, so callers
    pass the names in explicitly once they are filled.

    `dense_name` selects the dense space to search: frames hold image vectors in
    `dense_video`, ASR segments hold text vectors in `dense_text`, and the two
    are different dimensions.
    """
    # Both are needed for the hybrid path: a lexical query with nowhere to match
    # would fuse a single ranked list with itself, paying for RRF to change
    # nothing.
    if sparse_query and sparse_names:
        prefetch = [
            qmodels.Prefetch(
                query=vector,
                using=dense_name,
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
            using=dense_name,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
    return [_to_scored_frame(point) for point in response.points]


def search_asr(
    client: QdrantClient,
    collection_name: str,
    vector: list[float] | None,
    sparse_query: SparseVector | None,
    limit: int = DEFAULT_LIMIT,
    query_filter: qmodels.Filter | None = None,
) -> tuple[list[AsrSegment], list[AsrSegment]]:
    """Search the speech collection, returning the two branches separately.

    Deliberately *not* fused server-side. The dense and lexical halves have to
    be combined with an explicit weight, and Qdrant's RRF fuses ranks with no
    weight to give, so `ranking.asr.fuse_asr` does it once both lists are back.

    Either branch may be skipped by passing None, which is how the toggles turn
    a hybrid speech query into a purely dense or purely lexical one.
    """
    dense_hits: list[AsrSegment] = []
    sparse_hits: list[AsrSegment] = []

    if vector is not None:
        response = client.query_points(
            collection_name=collection_name,
            query=vector,
            using=collections.DENSE_TEXT_NAME,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        dense_hits = [_to_asr_segment(point) for point in response.points]

    if sparse_query:
        response = client.query_points(
            collection_name=collection_name,
            query=qmodels.SparseVector(
                indices=sparse_query.indices, values=sparse_query.values
            ),
            using=collections.SPARSE_SPEECH,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        sparse_hits = [_to_asr_segment(point) for point in response.points]

    return dense_hits, sparse_hits


def _to_asr_segment(point) -> AsrSegment:
    payload = point.payload or {}
    return AsrSegment(
        score=float(point.score),
        video_id=str(payload.get("video_id", "")),
        start_sec=float(payload.get("start_sec") or 0.0),
        end_sec=float(payload.get("end_sec") or 0.0),
        segment=int(payload.get("segment") or 0),
        text=str(payload.get("text_corrected") or ""),
    )


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
        pts_sec=_optional_float(payload.get("pts_sec")),
        shot_start_sec=_optional_float(payload.get("shot_start_sec")),
        shot_end_sec=_optional_float(payload.get("shot_end_sec")),
    )


def _optional_int(value) -> int | None:
    return None if value is None else int(value)


def _optional_float(value) -> float | None:
    return None if value is None else float(value)
