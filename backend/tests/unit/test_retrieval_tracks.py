import time

import pytest

from app.retrieval import tracks
from app.retrieval.engine import RetrievalConfig
from app.schemas.search import (
    KisSearchRequest,
    QaSearchRequest,
    TrakeSearchRequest,
)
from app.vector_store.search import ScoredFrame

CONFIG = RetrievalConfig(
    frames_collection="frames-v1",
    clips_collection="clips-v1",
    feature_profile="clip-b32-v1",
)


def frame(
    score: float, video_id: str = "L01_V001", shot_id: int = 0, frame_id: int = 0
) -> ScoredFrame:
    return ScoredFrame(
        score=score,
        video_id=video_id,
        shot_id=shot_id,
        original_frame_id=frame_id,
        start_frame=None,
        end_frame=None,
        path=None,
    )


@pytest.fixture
def fake_retrieve(monkeypatch):
    """Replace the shared engine so tracks are tested without a model."""
    calls: list[str] = []
    responses: dict[str, list[ScoredFrame]] = {}

    def _retrieve(text, top_k, config, timings, video_ids=None):
        calls.append(text)
        timings.record("encode", time.perf_counter())
        return responses.get(text, [])[:top_k]

    monkeypatch.setattr(tracks, "retrieve", _retrieve)
    return calls, responses


class TestKis:
    def test_hits_become_ranked_results(self, fake_retrieve):
        calls, responses = fake_retrieve
        responses["a red car"] = [
            frame(0.9, shot_id=1, frame_id=100),
            frame(0.8, shot_id=2, frame_id=200),
        ]

        response = tracks.search_kis(
            KisSearchRequest(task="kis", description="a red car", top_k=10), CONFIG
        )

        assert calls == ["a red car"]
        assert [(r.rank, r.frame_ids, r.score) for r in response.results] == [
            (1, [100], 0.9),
            (2, [200], 0.8),
        ]

    def test_response_reports_the_active_collection_and_profile(self, fake_retrieve):
        _, responses = fake_retrieve
        responses["x"] = [frame(0.5)]

        response = tracks.search_kis(
            KisSearchRequest(task="kis", description="x", top_k=5), CONFIG
        )

        assert response.versions.frames_collection == "frames-v1"
        assert response.versions.clips_collection == "clips-v1"
        assert response.versions.model_config_name == "clip-b32-v1"
        assert response.task == "kis"

    def test_no_hits_is_an_empty_result_list(self, fake_retrieve):
        response = tracks.search_kis(
            KisSearchRequest(task="kis", description="nothing", top_k=5), CONFIG
        )

        assert response.results == []

    def test_latency_is_reported(self, fake_retrieve):
        _, responses = fake_retrieve
        responses["x"] = [frame(0.5)]

        response = tracks.search_kis(
            KisSearchRequest(task="kis", description="x", top_k=5), CONFIG
        )

        assert "encode" in response.latency_ms


class TestQa:
    def test_retrieval_uses_the_description_not_the_question(self, fake_retrieve):
        calls, responses = fake_retrieve
        responses["a man on a bike"] = [frame(0.9, frame_id=42)]

        tracks.search_qa(
            QaSearchRequest(
                task="qa",
                description="a man on a bike",
                question="what colour is the bike?",
                top_k=5,
            ),
            CONFIG,
        )

        assert calls == ["a man on a bike"]

    def test_answer_is_left_unset(self, fake_retrieve):
        _, responses = fake_retrieve
        responses["scene"] = [frame(0.9, frame_id=42)]

        response = tracks.search_qa(
            QaSearchRequest(
                task="qa", description="scene", question="how many?", top_k=5
            ),
            CONFIG,
        )

        # No VQA model is wired in; a fabricated answer would be worse than none.
        assert response.results[0].answer is None


class TestTrake:
    def _request(self, events: list[str], top_k: int = 10) -> TrakeSearchRequest:
        return TrakeSearchRequest(
            task="trake", overview="a jump", events=events, top_k=top_k
        )

    def test_events_are_searched_separately(self, fake_retrieve):
        calls, responses = fake_retrieve
        responses["run"] = [frame(0.9, frame_id=10)]
        responses["jump"] = [frame(0.8, frame_id=50)]

        tracks.search_trake(self._request(["run", "jump"]), CONFIG)

        assert calls == ["run", "jump"]

    def test_sequence_frames_increase_in_time(self, fake_retrieve):
        _, responses = fake_retrieve
        responses["run"] = [frame(0.9, shot_id=1, frame_id=10)]
        responses["jump"] = [frame(0.8, shot_id=2, frame_id=50)]

        response = tracks.search_trake(self._request(["run", "jump"]), CONFIG)

        assert response.results[0].frame_ids == [10, 50]

    def test_a_later_event_cannot_use_an_earlier_frame(self, fake_retrieve):
        _, responses = fake_retrieve
        # The best "jump" hit is before the "run" hit, so it must be skipped in
        # favour of the weaker but temporally valid one.
        responses["run"] = [frame(0.9, shot_id=1, frame_id=100)]
        responses["jump"] = [
            frame(0.95, shot_id=0, frame_id=20),
            frame(0.60, shot_id=3, frame_id=300),
        ]

        response = tracks.search_trake(self._request(["run", "jump"]), CONFIG)

        assert response.results[0].frame_ids == [100, 300]

    def test_video_missing_an_event_is_dropped(self, fake_retrieve):
        _, responses = fake_retrieve
        responses["run"] = [
            frame(0.9, video_id="L01_V001", frame_id=10),
            frame(0.8, video_id="L01_V002", frame_id=10),
        ]
        responses["jump"] = [frame(0.7, video_id="L01_V001", frame_id=90)]

        response = tracks.search_trake(self._request(["run", "jump"]), CONFIG)

        assert [r.video_id for r in response.results] == ["L01_V001"]

    def test_video_with_no_valid_ordering_is_dropped(self, fake_retrieve):
        _, responses = fake_retrieve
        responses["run"] = [frame(0.9, frame_id=500)]
        responses["jump"] = [frame(0.9, frame_id=10)]

        response = tracks.search_trake(self._request(["run", "jump"]), CONFIG)

        assert response.results == []

    def test_score_is_the_mean_over_events(self, fake_retrieve):
        _, responses = fake_retrieve
        responses["run"] = [frame(0.8, frame_id=10)]
        responses["jump"] = [frame(0.6, frame_id=50)]

        response = tracks.search_trake(self._request(["run", "jump"]), CONFIG)

        assert response.results[0].score == pytest.approx(0.7)

    def test_videos_are_ranked_by_sequence_score(self, fake_retrieve):
        _, responses = fake_retrieve
        responses["run"] = [
            frame(0.5, video_id="L01_V001", frame_id=10),
            frame(0.9, video_id="L01_V002", frame_id=10),
        ]
        responses["jump"] = [
            frame(0.5, video_id="L01_V001", frame_id=90),
            frame(0.9, video_id="L01_V002", frame_id=90),
        ]

        response = tracks.search_trake(self._request(["run", "jump"]), CONFIG)

        assert [r.video_id for r in response.results] == ["L01_V002", "L01_V001"]
        assert [r.rank for r in response.results] == [1, 2]

    def test_three_events_are_supported(self, fake_retrieve):
        _, responses = fake_retrieve
        responses["a"] = [frame(0.9, frame_id=10)]
        responses["b"] = [frame(0.8, frame_id=50)]
        responses["c"] = [frame(0.7, frame_id=90)]

        response = tracks.search_trake(self._request(["a", "b", "c"]), CONFIG)

        assert response.results[0].frame_ids == [10, 50, 90]
