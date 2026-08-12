from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

DEFAULT_DISTANCE = qmodels.Distance.COSINE


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    return client.collection_exists(collection_name)


def create_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: qmodels.Distance = DEFAULT_DISTANCE,
) -> None:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=distance),
    )


def delete_collection(client: QdrantClient, collection_name: str) -> None:
    client.delete_collection(collection_name=collection_name)
