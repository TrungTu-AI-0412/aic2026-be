import hashlib
from collections.abc import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

DEFAULT_BATCH_SIZE = 256


def deterministic_point_id(*parts: str) -> int:
    digest = hashlib.sha1(":".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def make_point(
    point_id: int, vector: list[float], payload: dict
) -> qmodels.PointStruct:
    return qmodels.PointStruct(id=point_id, vector=vector, payload=payload)


def upsert_points(
    client: QdrantClient,
    collection_name: str,
    points: Iterable[qmodels.PointStruct],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    batch: list[qmodels.PointStruct] = []
    total = 0

    for point in points:
        batch.append(point)
        if len(batch) >= batch_size:
            client.upsert(collection_name=collection_name, points=batch)
            total += len(batch)
            batch = []

    if batch:
        client.upsert(collection_name=collection_name, points=batch)
        total += len(batch)

    return total
