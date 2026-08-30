"""The OCR channel: the endpoint, the track, and the boost over visual search.

Kept apart from `test_search_endpoint.py` so the fake service there stays the
shape main gave it. What these hold onto is the reason the channel exists as
its own path rather than another sparse slot in the hybrid prefetch.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import get_search_service
from app.api.endpoints import search as endpoint
from app.ranking import boost
from app.retrieval import engine, tracks
from app.retrieval.engine import RetrievalConfig, Timings
from app.schemas.search import OcrSearchRequest
from app.vector_store.search import ScoredFrame


def frame(video_id: str, shot_id: int, score: float, ocr: str | None = None):
    return ScoredFrame(
        score=score,
        video_id=video_id,
        shot_id=shot_id,
        original_frame_id=shot_id * 10,
        start_frame=None,
        end_frame=None,
        path=None,
        ocr_text=ocr,
    )


class FakeOcrService:
    def __init__(self) -> None:
        self.seen: OcrSearchRequest | None = None

    async def search_ocr(self, request):
        self.seen = request
        return {
            "request_id": "request-ocr",
            "task": "ocr",
            "effective_retrieval_mode": "visual",
            "results": [
                {
                    "rank": 1,
                    "video_id": "L21_V001",
                    "frame_ids": [4102],
                    "score": 0.91,
                    "ocr_text": "TẠM DỪNG LƯU THÔNG · TAM DUNG LUU THONG",
                }
            ],
            "versions": {
                "frames_collection": "frames-v1",
                "clips_collection": None,
                "model_config_name": "vision-v1",
            },
            "latency_ms": {"ocr": 4.0},
        }


def make_client(service: FakeOcrService) -> TestClient:
    app = FastAPI()
    app.include_router(endpoint.router, prefix="/api/v1")
    app.dependency_overrides[get_search_service] = lambda: service
    return TestClient(app)


def test_endpoint_returns_the_matched_text_as_evidence():
    """A rank alone cannot tell an operator what the query actually matched."""
    service = FakeOcrService()

    response = make_client(service).post(
        "/api/v1/search/ocr",
        json={"task": "ocr", "text": "tạm dừng lưu thông", "top_k": 10},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "ocr"
    assert body["results"][0]["ocr_text"].startswith("TẠM DỪNG")


def test_endpoint_rejects_an_empty_query():
    response = make_client(FakeOcrService()).post(
        "/api/v1/search/ocr", json={"task": "ocr", "text": ""}
    )
    assert response.status_code == 422


def test_endpoint_rejects_asr_dials_it_cannot_honour():
    """`extra="forbid"`: this path never touches the speech collection, and
    silently ignoring a weight the caller set would be worse than refusing."""
    response = make_client(FakeOcrService()).post(
        "/api/v1/search/ocr",
        json={"task": "ocr", "text": "bản tin", "asr_weight": 0.4},
    )
    assert response.status_code == 422


def test_track_carries_ocr_text_onto_every_row(monkeypatch):
    hits = [frame("L21_V001", 3, 0.9, "HTV9 TIN CHÍNH"), frame("L21_V002", 1, 0.4)]
    monkeypatch.setattr(tracks, "retrieve_by_ocr", lambda *a, **k: hits)

    response = tracks.search_ocr(
        OcrSearchRequest(task="ocr", text="tin chính", top_k=10),
        RetrievalConfig(frames_collection="frames-v1", feature_profile="clip-b32-v1"),
    )

    assert [row.ocr_text for row in response.results] == ["HTV9 TIN CHÍNH", None]
    assert [row.rank for row in response.results] == [1, 2]


def test_track_does_not_rewrite_the_query(monkeypatch):
    """The operator typed what they read off the screen. A rewriter turns prose
    into a caption, which would destroy the only signal this path searches."""
    monkeypatch.setattr(tracks, "retrieve_by_ocr", lambda *a, **k: [])

    response = tracks.search_ocr(
        OcrSearchRequest(task="ocr", text="SỐ 27/2025/NĐ-CP"),
        RetrievalConfig(frames_collection="frames-v1", feature_profile="clip-b32-v1"),
    )

    assert response.rewritten_queries is None
    assert response.cleaned_queries is None


def test_ocr_search_runs_no_reranker(monkeypatch):
    """BLIP ITM scores how well a caption describes a picture. A chyron is not
    what the picture is *of*, so reranking demotes the very frames that hit."""
    monkeypatch.setattr(
        engine, "search_ocr_channel", lambda *a, **k: [frame("L21_V001", 1, 0.8)]
    )

    def explode(*args, **kwargs):
        raise AssertionError("the OCR channel must not call the reranker")

    monkeypatch.setattr(engine.rerank, "rerank", explode)

    hits = engine.retrieve_by_ocr(
        "tin chính",
        5,
        RetrievalConfig(frames_collection="frames-v1", feature_profile="clip-b32-v1"),
        Timings(),
    )

    assert [hit.shot_id for hit in hits] == [1]


def test_ocr_search_survives_a_query_with_no_indexable_terms(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("an empty sparse query must not reach Qdrant")

    monkeypatch.setattr(engine, "search_ocr_channel", explode)

    assert (
        engine.retrieve_by_ocr(
            "   ",
            5,
            RetrievalConfig(
                frames_collection="frames-v1", feature_profile="clip-b32-v1"
            ),
            Timings(),
        )
        == []
    )


def test_boost_is_off_unless_asked_for(monkeypatch):
    """It costs one extra sparse query per `retrieve()`, which on TRAKE is per
    event. A default that quietly multiplies round trips is not a default."""
    monkeypatch.setattr(engine, "encode_query", lambda *a, **k: [0.1, 0.2])
    monkeypatch.setattr(
        engine, "search_vector", lambda *a, **k: [frame("L21_V001", 1, 0.8)]
    )

    def explode(*args, **kwargs):
        raise AssertionError("the boost queried OCR without being enabled")

    monkeypatch.setattr(engine, "search_ocr_channel", explode)

    engine._search_and_boost(
        "bản tin",
        5,
        RetrievalConfig(
            frames_collection="frames-v1",
            feature_profile="clip-b32-v1",
            asr_weight=0.0,
        ),
        Timings(),
    )


def test_boost_lets_lexical_evidence_ride_along(monkeypatch):
    """A shot the visual search found and OCR also found keeps the visual hit
    as its carrier, but gains the text that explains why it rose."""
    visual = [frame("L21_V001", 1, 0.8), frame("L21_V001", 2, 0.7)]
    lexical = [frame("L21_V001", 2, 0.95, "SẠT LỞ BỜ SÔNG")]
    monkeypatch.setattr(engine, "encode_query", lambda *a, **k: [0.1, 0.2])
    monkeypatch.setattr(engine, "search_vector", lambda *a, **k: visual)
    monkeypatch.setattr(engine, "search_ocr_channel", lambda *a, **k: lexical)

    hits = engine._search_and_boost(
        "sạt lở bờ sông",
        5,
        RetrievalConfig(
            frames_collection="frames-v1",
            feature_profile="clip-b32-v1",
            asr_weight=0.0,
            ocr_boost_enabled=True,
        ),
        Timings(),
    )

    by_shot = {hit.shot_id: hit for hit in hits}
    assert by_shot[2].ocr_text == "SẠT LỞ BỜ SÔNG"
    assert by_shot[1].ocr_text is None


def test_the_shipped_weight_is_the_measured_one():
    """0.05, not the 0.5 this shipped as. At 0.5 the channel scored worse than
    being switched off entirely (recall@1 0.140 vs 0.230) — the table is in
    `ranking/boost.py`. A regression here silently undoes that measurement."""
    assert boost.DEFAULT_OCR_WEIGHT == pytest.approx(0.05)
    assert RetrievalConfig(
        frames_collection="c", feature_profile="clip-b32-v1"
    ).ocr_boost_weight == pytest.approx(0.05)
