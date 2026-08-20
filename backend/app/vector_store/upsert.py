import hashlib
from collections.abc import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.features.sparse import SparseVector
from app.vector_store import collections

DEFAULT_BATCH_SIZE = 256


def deterministic_point_id(*parts: str) -> int:
    digest = hashlib.sha1(":".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def make_point(
    point_id: int,
    vector: list[float],
    payload: dict,
    sparse_vectors: dict[str, SparseVector] | None = None,
    dense_name: str = collections.DENSE_VECTOR_NAME,
) -> qmodels.PointStruct:
    """Build a point carrying the dense embedding plus any lexical vectors.

    `dense_name` selects which declared dense slot the vector lands in: frames
    embed images into `dense_video`, ASR segments embed text into `dense_text`,
    and the two are different spaces at different dimensions.

    Empty sparse vectors are omitted rather than written as zero-length: a
    point without a given sparse vector is simply not a candidate for that
    part of a hybrid query, which is the correct behaviour for a frame with no
    speech or no on-screen text.
    """
    vectors: dict[str, object] = {dense_name: vector}

    for name, sparse_vector in (sparse_vectors or {}).items():
        if not sparse_vector:
            continue
        vectors[name] = qmodels.SparseVector(
            indices=sparse_vector.indices, values=sparse_vector.values
        )

    return qmodels.PointStruct(id=point_id, vector=vectors, payload=payload)


def upsert_points(
    client: QdrantClient,
    collection_name: str,
    points: Iterable[qmodels.PointStruct],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    batch: list[qmodels.PointStruct] = []
    total = 0

    # `wait=False` returns once the write is journalled rather than once it is
    # applied, so the next batch can be embedded on the GPU while Qdrant is
    # still ingesting this one. Safe for a bulk load: the run ends with
    # `optimize_collection`, which blocks until the collection reports green,
    # and nothing reads the collection before then.
    for point in points:
        batch.append(point)
        if len(batch) >= batch_size:
            client.upsert(collection_name=collection_name, points=batch, wait=False)
            total += len(batch)
            batch = []

    if batch:
        client.upsert(collection_name=collection_name, points=batch, wait=False)
        total += len(batch)

    return total
