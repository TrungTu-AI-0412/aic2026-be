import uuid

import pytest
from qdrant_client import QdrantClient

from app.schemas.ingestions import IngestionEntity
from app.vector_store import collections, payload_indexes, search, upsert
from app.vector_store.client import build_client

VECTOR_SIZE = 8


@pytest.fixture(scope="module")
def client():
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
    name = f"test_search_{uuid.uuid4().hex[:8]}"
    yield name
    if collections.collection_exists(client, name):
        collections.delete_collection(client, name)


def _vector(index: int) -> list[float]:
    """Vectors that get progressively less similar to `_vector(0)`."""
    vector = [0.0] * VECTOR_SIZE
    vector[0] = 1.0
    vector[1] = index / 10.0
    return vector


@pytest.fixture
def populated(client, collection_name):
    collections.create_collection(client, collection_name, VECTOR_SIZE)
    payload_indexes.create_payload_indexes(
        client, collection_name, IngestionEntity.FRAMES
    )

    points = []
    for index in range(9):
        video_id = "L01_V001" if index < 6 else "L01_V002"
        points.append(
            upsert.make_point(
                point_id=index,
                vector=_vector(index),
                payload={
                    "video_id": video_id,
                    "shot_id": index // 3,
                    "original_frame_id": index * 25,
                    "path": f"/kf/{video_id}_{index:06d}.jpg",
                },
            )
        )
    upsert.upsert_points(client, collection_name, points)
    return collection_name


class TestSearch:
    def test_returns_hits_ordered_by_similarity(self, client, populated):
        hits = search.search(client, populated, _vector(0), limit=3)

        assert len(hits) == 3
        assert hits[0].original_frame_id == 0
        assert hits[0].score >= hits[1].score >= hits[2].score

    def test_payload_is_mapped_onto_the_result(self, client, populated):
        hits = search.search(client, populated, _vector(0), limit=1)

        assert hits[0].video_id == "L01_V001"
        assert hits[0].shot_id == 0
        assert hits[0].path == "/kf/L01_V001_000000.jpg"

    def test_representative_frame_uses_the_exact_frame_id(self, client, populated):
        hits = search.search(client, populated, _vector(0), limit=1)

        assert hits[0].representative_frame == hits[0].original_frame_id

    def test_limit_is_respected(self, client, populated):
        assert len(search.search(client, populated, _vector(0), limit=2)) == 2

    def test_video_filter_restricts_results(self, client, populated):
        hits = search.search(
            client,
            populated,
            _vector(0),
            limit=10,
            query_filter=search.build_filter(video_ids=["L01_V002"]),
        )

        assert hits
        assert {hit.video_id for hit in hits} == {"L01_V002"}

    def test_shot_filter_restricts_results(self, client, populated):
        hits = search.search(
            client,
            populated,
            _vector(0),
            limit=10,
            query_filter=search.build_filter(shot_ids=[2]),
        )

        assert hits
        assert {hit.shot_id for hit in hits} == {2}

    def test_no_constraints_means_no_filter(self):
        assert search.build_filter() is None


class TestOptimizeCollection:
    def test_collection_reaches_green(self, client, collection_name):
        collections.create_collection(client, collection_name, VECTOR_SIZE)
        upsert.upsert_points(
            client,
            collection_name,
            [upsert.make_point(0, _vector(0), {"video_id": "L01_V001", "shot_id": 0})],
        )

        collections.optimize_collection(client, collection_name, timeout_sec=60)

        assert str(client.get_collection(collection_name).status).endswith("green")

    def test_searchable_after_optimizing(self, client, populated):
        collections.optimize_collection(client, populated, timeout_sec=60)

        assert search.search(client, populated, _vector(0), limit=1)
