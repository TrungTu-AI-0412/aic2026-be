from dataclasses import dataclass

from qdrant_client import QdrantClient


@dataclass(frozen=True)
class CollectionStatus:
    name: str
    status: str
    points_count: int
    vectors_count: int | None
    indexed_vectors_count: int | None


def get_collection_status(client: QdrantClient, collection_name: str) -> CollectionStatus:
    info = client.get_collection(collection_name)
    return CollectionStatus(
        name=collection_name,
        status=str(info.status),
        points_count=info.points_count or 0,
        vectors_count=info.vectors_count,
        indexed_vectors_count=info.indexed_vectors_count,
    )
