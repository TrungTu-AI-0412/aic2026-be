import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import get_search_service
from app.api.endpoints import search as endpoint
from app.retrieval.engine import AsrOnlyRequestError, AsrOnlyUnavailableError


class FakeSearchService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def search_kis(self, request):
        return self._result("kis")

    async def search_qa(self, request):
        return self._result("qa")

    def _result(self, task: str):
        if self.error is not None:
            raise self.error
        return {
            "request_id": "request-123",
            "task": task,
            "effective_retrieval_mode": "asr_only",
            "rewritten_queries": ["an English report"],
            "cleaned_queries": ["bản tin tiếng Việt"],
            "results": [],
            "versions": {
                "frames_collection": "frames-v1",
                "clips_collection": None,
                "model_config_name": "vision-v1",
                "asr_collection": "asr-v1",
                "asr_model_config_name": "qwen-v1",
            },
            "latency_ms": {"rewrite": 1.0, "asr": 2.0, "asr_frame_map": 0.0},
        }


def make_client(error: Exception | None = None) -> TestClient:
    app = FastAPI()
    app.include_router(endpoint.router, prefix="/api/v1")
    app.dependency_overrides[get_search_service] = lambda: FakeSearchService(error)
    return TestClient(app)


@pytest.mark.parametrize(
    ("path", "body"),
    [
        (
            "/api/v1/search/kis",
            {"task": "kis", "description": "lời thoại", "top_k": 5},
        ),
        (
            "/api/v1/search/qa",
            {"task": "qa", "description": "lời thoại", "top_k": 5},
        ),
    ],
)
def test_asr_unavailable_is_a_503(path, body) -> None:
    response = make_client(AsrOnlyUnavailableError("ASR index unavailable")).post(
        path,
        json={**body, "retrieval_mode": "asr_only"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "ASR index unavailable"


def test_runtime_invalid_asr_request_is_a_422() -> None:
    response = make_client(AsrOnlyRequestError("weights cannot both be zero")).post(
        "/api/v1/search/kis",
        json={
            "task": "kis",
            "description": "lời thoại",
            "top_k": 5,
            "retrieval_mode": "asr_only",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "weights cannot both be zero"


@pytest.mark.parametrize(
    "overrides",
    [
        {"retrieval_mode": "asr_only", "asr_enabled": True},
        {
            "retrieval_mode": "asr_only",
            "asr_dense_weight": 0,
            "asr_sparse_weight": 0,
        },
    ],
)
def test_incompatible_asr_only_fields_are_a_422(overrides) -> None:
    response = make_client().post(
        "/api/v1/search/kis",
        json={
            "task": "kis",
            "description": "lời thoại",
            "top_k": 5,
            **overrides,
        },
    )

    assert response.status_code == 422


def test_trake_rejects_asr_only_mode() -> None:
    response = make_client().post(
        "/api/v1/search/trake",
        json={
            "task": "trake",
            "overview": "một chuỗi sự kiện",
            "events": ["sự kiện một"],
            "top_k": 5,
            "retrieval_mode": "asr_only",
        },
    )

    assert response.status_code == 422


def test_valid_query_with_no_asr_hits_is_an_empty_200() -> None:
    response = make_client().post(
        "/api/v1/search/kis",
        json={
            "task": "kis",
            "description": "không có lời thoại phù hợp",
            "top_k": 5,
            "retrieval_mode": "asr_only",
        },
    )

    assert response.status_code == 200
    assert response.json()["results"] == []
    assert response.json()["effective_retrieval_mode"] == "asr_only"
