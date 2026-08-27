"""The HTTP surface of /search/ocr.

Mounts the router alone rather than the whole app: `create_app` builds the
retrieval container, which drags in torch, and none of that is under test
here. What is under test is the request contract the frontend codegens from.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import get_search_service
from app.api.endpoints import search as endpoint
from app.schemas.search import (
    OcrSearchRequest,
    SearchResponse,
    SearchResult,
    SearchVersions,
)


class StubSearchService:
    def __init__(self) -> None:
        self.received: OcrSearchRequest | None = None

    async def search_ocr(self, request: OcrSearchRequest) -> SearchResponse:
        self.received = request
        return SearchResponse(
            request_id="req-1",
            task="ocr",
            results=[
                SearchResult(
                    rank=1,
                    video_id="L21_V001",
                    frame_ids=[540],
                    score=9.1,
                    ocr_text="TẠM DỪNG LƯU THÔNG",
                )
            ],
            versions=SearchVersions(
                frames_collection="frames-v1", model_config_name="clip-b32-v1"
            ),
            latency_ms={"ocr": 4.2},
        )


@pytest.fixture
def stub():
    return StubSearchService()


@pytest.fixture
def client(stub):
    app = FastAPI()
    app.include_router(endpoint.router)
    app.dependency_overrides[get_search_service] = lambda: stub
    return TestClient(app)


def test_a_lexical_hit_carries_the_text_that_matched(client):
    response = client.post(
        "/search/ocr", json={"task": "ocr", "text": "tạm dừng lưu thông"}
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["ocr_text"] == "TẠM DỪNG LƯU THÔNG"


def test_video_ids_reach_the_service(client, stub):
    client.post(
        "/search/ocr",
        json={"task": "ocr", "text": "sạt lở", "video_ids": ["L21_V001"]},
    )

    assert stub.received.video_ids == ["L21_V001"]


def test_video_ids_are_optional(client, stub):
    client.post("/search/ocr", json={"task": "ocr", "text": "sạt lở"})

    assert stub.received.video_ids is None


def test_an_empty_query_is_rejected(client):
    """There is nothing to match on, and an empty sparse vector would return
    an arbitrary page of the collection rather than no results."""
    response = client.post("/search/ocr", json={"task": "ocr", "text": ""})

    assert response.status_code == 422


def test_the_wrong_task_discriminator_is_rejected(client):
    response = client.post("/search/ocr", json={"task": "kis", "text": "x"})

    assert response.status_code == 422
