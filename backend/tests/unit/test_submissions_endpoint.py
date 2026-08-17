"""The HTTP surface of /submissions/export.

Mounts the router alone rather than the whole app: `create_app` builds the
retrieval container, which drags in torch, and none of that is under test
here. What is under test is the wiring the operator actually touches — the
download headers, and which failure maps to which status code.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import get_submission_service
from app.api.endpoints import submissions as endpoint
from app.submissions.service import LocalSubmissionService, load_bounds

KIS_BODY = {
    "task": "kis",
    "candidates": [{"task": "kis", "video_id": "L22_V001", "frame_id": 1200}],
}


@pytest.fixture
def client(tmp_path):
    load_bounds.cache_clear()
    path = tmp_path / "video_bounds.parquet"
    pq.write_table(
        pa.table(
            {
                "video_id": ["L22_V001"],
                "fps": [30.0],
                "length_sec": [1163],
                "frame_upper_bound": [34890],
            }
        ),
        path,
    )

    app = FastAPI()
    app.include_router(endpoint.router, prefix="/submissions")
    app.dependency_overrides[get_submission_service] = lambda: (
        LocalSubmissionService(bounds_manifest=str(path))
    )
    yield TestClient(app)
    load_bounds.cache_clear()


def test_returns_the_file_as_a_download(client) -> None:
    response = client.post("/submissions/export", json=KIS_BODY)

    assert response.status_code == 200
    assert response.text == "L22_V001,1200\n"
    assert response.headers["content-type"].startswith("text/csv")
    assert (
        response.headers["content-disposition"]
        == 'attachment; filename="submission-kis.csv"'
    )


def test_unknown_video_is_a_404(client) -> None:
    response = client.post(
        "/submissions/export",
        json={
            "task": "kis",
            "candidates": [{"task": "kis", "video_id": "L99_V999", "frame_id": 1}],
        },
    )

    assert response.status_code == 404
    assert "L99_V999" in response.json()["detail"]


def test_frame_past_the_end_is_a_422(client) -> None:
    response = client.post(
        "/submissions/export",
        json={
            "task": "kis",
            "candidates": [
                {"task": "kis", "video_id": "L22_V001", "frame_id": 999_999}
            ],
        },
    )

    assert response.status_code == 422


def test_task_and_candidate_must_agree(client) -> None:
    """Guards against a UI sending QA rows under a KIS submission."""
    response = client.post(
        "/submissions/export",
        json={
            "task": "kis",
            "candidates": [
                {
                    "task": "qa",
                    "video_id": "L22_V001",
                    "frame_id": 1,
                    "answer": "x",
                }
            ],
        },
    )

    assert response.status_code == 422
