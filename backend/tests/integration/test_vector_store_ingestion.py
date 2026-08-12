import uuid

import pytest
from qdrant_client import QdrantClient

from app.vector_store import collections, payload_indexes, upsert
from app.vector_store.client import build_client
from app.vector_store.collection_status import get_collection_status

VECTOR_SIZE = 8
POINT_COUNT = 100


@pytest.fixture(scope="module")
def client():
    # Fall back to in-memory mode so this test runs without a local Qdrant deployment.
    qdrant_client = build_client()
    try:
        qdrant_client.get_collections()
        yield qdrant_client
        return
    except Exception:
        pass

    yield QdrantClient(location=":memory:")


@pytest.fixture
def collection_name(client):
    name = f"test_ingestion_{uuid.uuid4().hex[:8]}"
    yield name
    if collections.collection_exists(client, name):
        collections.delete_collection(client, name)


def _build_points(count: int) -> list:
    return [
        upsert.make_point(
            point_id=upsert.deterministic_point_id("L01_V001", str(frame_id)),
            vector=[float((frame_id + offset) % 7) for offset in range(VECTOR_SIZE)],
            payload={"video_id": "L01_V001", "frame_id": frame_id},
        )
        for frame_id in range(count)
    ]


def test_create_collection_upsert_and_read_status(client, collection_name):
    assert not collections.collection_exists(client, collection_name)

    collections.create_collection(client, collection_name, VECTOR_SIZE)
    payload_indexes.create_payload_indexes(client, collection_name)

    points = _build_points(POINT_COUNT)
    upserted = upsert.upsert_points(client, collection_name, points, batch_size=32)
    assert upserted == POINT_COUNT

    status = get_collection_status(client, collection_name)
    assert status.name == collection_name
    assert status.points_count == POINT_COUNT


def test_deterministic_point_id_is_stable_across_calls():
    first = upsert.deterministic_point_id("L01_V001", "42")
    second = upsert.deterministic_point_id("L01_V001", "42")
    different = upsert.deterministic_point_id("L01_V001", "43")

    assert first == second
    assert first != different
