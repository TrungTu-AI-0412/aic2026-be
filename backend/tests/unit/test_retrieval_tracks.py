import time
from dataclasses import dataclass

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
    score: float,
    video_id: str = "L01_V001",
    shot_id: int = 0,
    frame_id: int = 0,
    pts_sec: float | None = None,
) -> ScoredFrame:
    return ScoredFrame(
        score=score,
        video_id=video_id,
        shot_id=shot_id,
        original_frame_id=frame_id,
        start_frame=None,
        end_frame=None,
        path=None,
        pts_sec=pts_sec,
    )


@dataclass
class FakeEngine:
    """What each stage was asked, and what each stage answers.

    TRAKE searches globally to choose videos and again inside each chosen
    video, and the point of the split is that the two can disagree - an event
    that loses a global ranking can still be found inside the right video. So
    the two stages get separate response tables; `per_video_responses` falls
    back to `responses` for the tests that do not care.
    """

    calls: list[str]
    responses: dict[str, list[ScoredFrame]]
    per_video_calls: list[tuple[str, list[str]]]
    per_video_responses: dict[str, list[ScoredFrame]]


@pytest.fixture
def fake_retrieve(monkeypatch):
    """Replace the shared engine so tracks are tested without a model."""
    engine = FakeEngine(
        calls=[], responses={}, per_video_calls=[], per_video_responses={}
    )

    def _retrieve(text, top_k, config, timings, video_ids=None):
        engine.calls.append(text)
        timings.record("encode", time.perf_counter())
        hits = engine.responses.get(text, [])
        if video_ids is not None:
            hits = [hit for hit in hits if hit.video_id in video_ids]
        return hits[:top_k]

    def _retrieve_per_video(text, video_ids, limit, config, timings):
        engine.per_video_calls.append((text, list(video_ids)))
        timings.record("encode", time.perf_counter())
        table = engine.per_video_responses or engine.responses
        grouped: dict[str, list[ScoredFrame]] = {
            video_id: [] for video_id in video_ids
        }
        for hit in table.get(text, []):
            if hit.video_id in grouped:
                grouped[hit.video_id].append(hit)
        return {
            video_id: sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]
            for video_id, hits in grouped.items()
        }

    monkeypatch.setattr(tracks, "retrieve", _retrieve)
    monkeypatch.setattr(tracks, "retrieve_per_video", _retrieve_per_video)
    return engine


class TestKis:
    def test_hits_become_ranked_results(self, fake_retrieve):
        calls, responses = fake_retrieve.calls, fake_retrieve.responses
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
        responses = fake_retrieve.responses
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
        responses = fake_retrieve.responses
        responses["x"] = [frame(0.5)]

        response = tracks.search_kis(
            KisSearchRequest(task="kis", description="x", top_k=5), CONFIG
        )

        assert "encode" in response.latency_ms


class TestQa:
    def test_only_the_description_is_encoded(self, fake_retrieve):
        calls, responses = fake_retrieve.calls, fake_retrieve.responses
        responses["a man on a bike"] = [frame(0.9, frame_id=42)]

        response = tracks.search_qa(
            QaSearchRequest(task="qa", description="a man on a bike", top_k=5),
            CONFIG,
        )

        assert calls == ["a man on a bike"]
        assert response.results[0].frame_ids == [42]

    def test_a_stale_question_field_is_ignored(self, fake_retrieve):
        """The field is gone, not forbidden: the backend ships before the UI."""
        responses = fake_retrieve.responses
        responses["scene"] = [frame(0.9, frame_id=42)]

        request = QaSearchRequest.model_validate(
            {"task": "qa", "description": "scene", "question": "how many?", "top_k": 5}
        )

        assert not hasattr(request, "question")
        assert tracks.search_qa(request, CONFIG).results[0].frame_ids == [42]


class TestTrake:
    def _request(self, events: list[str], top_k: int = 10) -> TrakeSearchRequest:
        return TrakeSearchRequest(
            task="trake", overview="a jump", events=events, top_k=top_k
        )

    def test_both_stages_run(self, fake_retrieve):
        fake_retrieve.responses["run"] = [frame(0.9, frame_id=10)]
        fake_retrieve.responses["jump"] = [frame(0.8, frame_id=50)]

        tracks.search_trake(self._request(["run", "jump"]), CONFIG)

        # The overview is evidence about the video, so it is encoded too - it
        # used to be required by the schema and then dropped.
        assert fake_retrieve.calls == ["a jump", "run", "jump"]
        # Then each event is searched again, inside the chosen videos only.
        assert fake_retrieve.per_video_calls == [
            ("run", ["L01_V001"]),
            ("jump", ["L01_V001"]),
        ]

    def test_sequence_frames_increase_in_time(self, fake_retrieve):
        responses = fake_retrieve.responses
        responses["run"] = [frame(0.9, shot_id=1, frame_id=10)]
        responses["jump"] = [frame(0.8, shot_id=2, frame_id=50)]

        response = tracks.search_trake(self._request(["run", "jump"]), CONFIG)

        assert response.results[0].frame_ids == [10, 50]

    def test_a_later_event_cannot_use_an_earlier_frame(self, fake_retrieve):
        responses = fake_retrieve.responses
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
        responses = fake_retrieve.responses
        responses["run"] = [
            frame(0.9, video_id="L01_V001", frame_id=10),
            frame(0.8, video_id="L01_V002", frame_id=10),
        ]
        responses["jump"] = [frame(0.7, video_id="L01_V001", frame_id=90)]

        response = tracks.search_trake(self._request(["run", "jump"]), CONFIG)

        assert [r.video_id for r in response.results] == ["L01_V001"]

    def test_video_with_no_valid_ordering_is_dropped(self, fake_retrieve):
        responses = fake_retrieve.responses
        responses["run"] = [frame(0.9, frame_id=500)]
        responses["jump"] = [frame(0.9, frame_id=10)]

        response = tracks.search_trake(self._request(["run", "jump"]), CONFIG)

        assert response.results == []

    def test_score_is_the_mean_over_events(self, fake_retrieve):
        responses = fake_retrieve.responses
        responses["run"] = [frame(0.8, frame_id=10)]
        responses["jump"] = [frame(0.6, frame_id=50)]

        response = tracks.search_trake(self._request(["run", "jump"]), CONFIG)

        assert response.results[0].score == pytest.approx(0.7)

    def test_videos_are_ranked_by_sequence_score(self, fake_retrieve):
        responses = fake_retrieve.responses
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
        responses = fake_retrieve.responses
        responses["a"] = [frame(0.9, frame_id=10)]
        responses["b"] = [frame(0.8, frame_id=50)]
        responses["c"] = [frame(0.7, frame_id=90)]

        response = tracks.search_trake(self._request(["a", "b", "c"]), CONFIG)

        assert response.results[0].frame_ids == [10, 50, 90]

    def test_a_video_only_the_overview_found_can_win(self, fake_retrieve):
        """The recall fix: one weak event no longer costs the whole video.

        Globally, only the overview finds L01_V009 - neither event surfaces it,
        which is what a fine-grained event ("the moment all four feet touch the
        ground") does against 290k frames. Searched *inside* that video both
        events are there, so it must still reach the results.
        """
        fake_retrieve.responses["a jump"] = [frame(0.9, video_id="L01_V009")]
        fake_retrieve.responses["run"] = [frame(0.4, video_id="L01_V001", frame_id=10)]
        fake_retrieve.responses["jump"] = [frame(0.4, video_id="L01_V001", frame_id=90)]
        fake_retrieve.per_video_responses.update(
            {
                "run": [
                    frame(0.7, video_id="L01_V009", frame_id=10),
                    frame(0.4, video_id="L01_V001", frame_id=10),
                ],
                "jump": [
                    frame(0.7, video_id="L01_V009", frame_id=90),
                    frame(0.4, video_id="L01_V001", frame_id=90),
                ],
            }
        )

        response = tracks.search_trake(self._request(["run", "jump"]), CONFIG)

        assert [r.video_id for r in response.results] == ["L01_V009", "L01_V001"]

    def test_request_video_ids_skip_video_selection(self, fake_retrieve):
        fake_retrieve.responses["run"] = [frame(0.9, video_id="L21_V042", frame_id=10)]
        fake_retrieve.responses["jump"] = [frame(0.8, video_id="L21_V042", frame_id=50)]
        request = TrakeSearchRequest(
            task="trake",
            overview="a jump",
            events=["run", "jump"],
            video_ids=["L21_V042"],
        )

        response = tracks.search_trake(request, CONFIG)

        # Nothing is searched globally: the operator already found the video.
        assert fake_retrieve.calls == []
        assert fake_retrieve.per_video_calls == [
            ("run", ["L21_V042"]),
            ("jump", ["L21_V042"]),
        ]
        assert response.results[0].frame_ids == [10, 50]

    def test_max_gap_rejects_a_spread_sequence(self, fake_retrieve):
        responses = fake_retrieve.responses
        responses["run"] = [frame(0.9, frame_id=10, pts_sec=10.0)]
        responses["jump"] = [frame(0.9, frame_id=50, pts_sec=400.0)]

        request = TrakeSearchRequest(
            task="trake", overview="a jump", events=["run", "jump"], max_gap_sec=60.0
        )
        assert tracks.search_trake(request, CONFIG).results == []

    def test_max_gap_zero_disables_the_check(self, fake_retrieve):
        responses = fake_retrieve.responses
        responses["run"] = [frame(0.9, frame_id=10, pts_sec=10.0)]
        responses["jump"] = [frame(0.9, frame_id=50, pts_sec=4000.0)]

        request = TrakeSearchRequest(
            task="trake", overview="a jump", events=["run", "jump"], max_gap_sec=0.0
        )
        assert tracks.search_trake(request, CONFIG).results[0].frame_ids == [10, 50]

    def test_missing_timestamps_do_not_trigger_the_gap_check(self, fake_retrieve):
        """Absence of a timestamp is not evidence the events are far apart."""
        responses = fake_retrieve.responses
        responses["run"] = [frame(0.9, frame_id=10)]
        responses["jump"] = [frame(0.9, frame_id=50, pts_sec=4000.0)]

        request = TrakeSearchRequest(
            task="trake", overview="a jump", events=["run", "jump"], max_gap_sec=1.0
        )
        assert tracks.search_trake(request, CONFIG).results[0].frame_ids == [10, 50]

    def test_two_events_in_one_shot_are_both_selectable(self, fake_retrieve):
        """Two moments a second apart share a shot, and must stay separable.

        Shots average under three seconds with three keyframes each, so "the
        lion starts to spin" and "all four feet land" fall inside one shot.
        The per-video stage does not collapse shots for exactly this reason.
        """
        candidates = [
            frame(0.9, shot_id=5, frame_id=100, pts_sec=4.0),
            frame(0.8, shot_id=5, frame_id=130, pts_sec=5.0),
        ]
        fake_retrieve.responses["run"] = candidates
        fake_retrieve.responses["jump"] = candidates

        response = tracks.search_trake(self._request(["run", "jump"]), CONFIG)

        assert response.results[0].frame_ids == [100, 130]
        assert [event.shot_id for event in response.results[0].events] == [5, 5]

    def test_event_hits_report_time_and_shot(self, fake_retrieve):
        responses = fake_retrieve.responses
        responses["run"] = [frame(0.8, shot_id=3, frame_id=10, pts_sec=1.5)]
        responses["jump"] = [frame(0.6, shot_id=7, frame_id=50, pts_sec=2.5)]

        events = tracks.search_trake(self._request(["run", "jump"]), CONFIG).results[
            0
        ].events

        assert [e.event_index for e in events] == [0, 1]
        assert [e.frame_id for e in events] == [10, 50]
        assert [e.shot_id for e in events] == [3, 7]
        assert [e.pts_sec for e in events] == [1.5, 2.5]
        assert events[0].score == pytest.approx(0.8)

    def test_alternates_stay_between_the_neighbouring_picks(self, fake_retrieve):
        """Swapping an alternate in must not break the ordering.

        Frame 300 scores well for the first event but sits after the second
        event's chosen frame, so offering it would let an operator build an
        out-of-order submission.
        """
        responses = fake_retrieve.responses
        responses["run"] = [
            frame(0.9, frame_id=100),
            frame(0.8, frame_id=50),
            frame(0.7, frame_id=300),
        ]
        responses["jump"] = [frame(0.6, frame_id=200)]

        result = tracks.search_trake(self._request(["run", "jump"]), CONFIG).results[0]

        assert result.frame_ids == [100, 200]
        assert [a.frame_id for a in result.events[0].alternates] == [50]
        assert result.events[1].alternates == []

    def test_alternates_respect_the_gap_too(self, fake_retrieve):
        """An alternate the ranker would have rejected must not be offered.

        Frame 900 sits after the second event's pick, so it passes the ordering
        test, but it is 300s past it - a swap would build a chain the sequence
        search itself refused.
        """
        responses = fake_retrieve.responses
        responses["run"] = [frame(0.9, frame_id=10, pts_sec=1.0)]
        responses["jump"] = [
            frame(0.8, frame_id=50, pts_sec=5.0),
            frame(0.7, frame_id=900, pts_sec=305.0),
        ]

        request = TrakeSearchRequest(
            task="trake", overview="a jump", events=["run", "jump"], max_gap_sec=60.0
        )
        result = tracks.search_trake(request, CONFIG).results[0]

        assert result.frame_ids == [10, 50]
        assert result.events[1].alternates == []

    def test_non_trake_results_carry_no_events(self, fake_retrieve):
        fake_retrieve.responses["a red car"] = [frame(0.9, frame_id=10)]

        response = tracks.search_kis(
            KisSearchRequest(task="kis", description="a red car", top_k=10), CONFIG
        )

        assert response.results[0].events is None
