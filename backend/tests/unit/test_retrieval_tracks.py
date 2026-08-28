import time
from dataclasses import dataclass

import pytest

from app.retrieval import tracks
from app.retrieval.engine import AsrOnlyHit, RetrievalConfig
from app.retrieval.rewrite import Rewrite
from app.schemas.search import (
    KisSearchRequest,
    QaSearchRequest,
    QueryForms,
    TrakeSearchRequest,
)
from app.vector_store.search import AsrSegment, ScoredFrame

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
    # What the speech stage was handed, which is the cleaned original rather
    # than the English form the image space was given.
    speech_calls: list[str]
    # Queries the video-selection stage asked about, in order.
    video_score_calls: list[str]


@pytest.fixture
def fake_retrieve(monkeypatch):
    """Replace the shared engine so tracks are tested without a model."""
    engine = FakeEngine(
        calls=[],
        responses={},
        per_video_calls=[],
        per_video_responses={},
        speech_calls=[],
        video_score_calls=[],
    )

    def _retrieve(text, top_k, config, timings, video_ids=None, speech_text=None):
        engine.calls.append(text)
        engine.speech_calls.append(speech_text or text)
        timings.record("encode", time.perf_counter())
        hits = engine.responses.get(text, [])
        if video_ids is not None:
            hits = [hit for hit in hits if hit.video_id in video_ids]
        return hits[:top_k]

    def _retrieve_per_video(
        text, video_ids, limit, config, timings, speech_text=None
    ):
        engine.per_video_calls.append((text, list(video_ids)))
        engine.speech_calls.append(speech_text or text)
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

    def _retrieve_video_scores(text, top_k, config, timings, speech_text=None):
        engine.video_score_calls.append(text)
        engine.calls.append(text)
        engine.speech_calls.append(speech_text or text)
        timings.record("encode", time.perf_counter())
        best: dict[str, float] = {}
        for hit in engine.responses.get(text, []):
            best[hit.video_id] = max(best.get(hit.video_id, 0.0), hit.score)
        return best

    monkeypatch.setattr(tracks, "retrieve", _retrieve)
    monkeypatch.setattr(tracks, "retrieve_per_video", _retrieve_per_video)
    monkeypatch.setattr(tracks, "retrieve_video_scores", _retrieve_video_scores)
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

    def test_gap_penalty_prefers_the_tighter_sequence(self, fake_retrieve):
        """Inside the hard window, closer events are the better answer.

        Both videos score identically on every event and both chains sit well
        inside `max_gap`, so before the penalty existed these tied and the order
        fell out of dict insertion. The events of one query describe a single
        continuous action, so 4 seconds apart beats 200.
        """
        responses = fake_retrieve.responses
        responses["run"] = [
            frame(0.6, video_id="L01_V001", frame_id=10, pts_sec=1.0),
            frame(0.6, video_id="L01_V002", frame_id=10, pts_sec=1.0),
        ]
        responses["jump"] = [
            frame(0.6, video_id="L01_V001", frame_id=90, pts_sec=201.0),
            frame(0.6, video_id="L01_V002", frame_id=90, pts_sec=5.0),
        ]

        response = tracks.search_trake(self._request(["run", "jump"]), CONFIG)

        assert [r.video_id for r in response.results] == ["L01_V002", "L01_V001"]

    def test_gap_penalty_cannot_outvote_a_much_better_frame(self, fake_retrieve):
        """The weight is small on purpose: it reorders near-ties, nothing more.

        The spread video scores 0.3 higher per event, which no gap inside the
        window can pay for at `TRAKE_GAP_WEIGHT`. A penalty that could flip this
        would be picking frames on their timestamps rather than their content.
        """
        responses = fake_retrieve.responses
        responses["run"] = [
            frame(0.9, video_id="L01_V001", frame_id=10, pts_sec=1.0),
            frame(0.6, video_id="L01_V002", frame_id=10, pts_sec=1.0),
        ]
        responses["jump"] = [
            frame(0.9, video_id="L01_V001", frame_id=90, pts_sec=280.0),
            frame(0.6, video_id="L01_V002", frame_id=90, pts_sec=2.0),
        ]

        response = tracks.search_trake(self._request(["run", "jump"]), CONFIG)

        assert [r.video_id for r in response.results] == ["L01_V001", "L01_V002"]

    def test_the_best_predecessor_accounts_for_the_gap(self, fake_retrieve):
        """The edge cost has to be paid *before* the predecessor is chosen.

        Candidate A scores higher than B for the first event, so picking the
        best chain by score alone takes A. But the second event's only candidate
        sits 5s after B and 105s after A, and that gap is charged to the chain:

            A: 0.50 - 0.10 x penalty(105s) = 0.50 - 0.0817 = 0.4183
            B: 0.48 - 0.10 x penalty(5s)   = 0.48 - 0.0314 = 0.4486

        So B wins on the lower raw score. Choosing the predecessor first and
        charging the gap afterwards turns the exact search into a greedy walk
        and returns A - this is the regression test for that.
        """
        responses = fake_retrieve.responses
        responses["run"] = [
            frame(0.50, frame_id=10, pts_sec=0.0),
            frame(0.48, frame_id=20, pts_sec=100.0),
        ]
        responses["jump"] = [frame(0.7, frame_id=30, pts_sec=105.0)]

        result = tracks.search_trake(self._request(["run", "jump"]), CONFIG).results[0]

        assert result.frame_ids == [20, 30]

    def test_gap_penalty_is_off_without_timestamps(self, fake_retrieve):
        """No timestamp is no information, so the score stays the plain mean.

        Only clip-only points lack one. Reading absence as "close together"
        would hand them the best possible penalty and let them outrank real
        keyframes on a gap nobody measured.
        """
        responses = fake_retrieve.responses
        responses["run"] = [frame(0.8, frame_id=10)]
        responses["jump"] = [frame(0.6, frame_id=50)]

        response = tracks.search_trake(self._request(["run", "jump"]), CONFIG)

        assert response.results[0].score == pytest.approx(0.7)

    def test_gap_weight_zero_restores_the_mean(self, fake_retrieve):
        responses = fake_retrieve.responses
        responses["run"] = [frame(0.8, frame_id=10, pts_sec=1.0)]
        responses["jump"] = [frame(0.6, frame_id=50, pts_sec=200.0)]

        request = TrakeSearchRequest(
            task="trake", overview="a jump", events=["run", "jump"], gap_weight=0.0
        )

        assert tracks.search_trake(request, CONFIG).results[0].score == pytest.approx(
            0.7
        )

    def test_gap_penalty_applies_when_the_hard_check_is_disabled(self, fake_retrieve):
        """`max_gap_sec=0` drops the cutoff, not the preference for tightness.

        An operator who disables the hard rule still wants closer events ranked
        first - the penalty is the better version of the rule they turned off.
        So it is normalised by the module constant, never by `max_gap_sec`,
        which would also divide by zero here.
        """
        responses = fake_retrieve.responses
        responses["run"] = [
            frame(0.6, video_id="L01_V001", frame_id=10, pts_sec=1.0),
            frame(0.6, video_id="L01_V002", frame_id=10, pts_sec=1.0),
        ]
        responses["jump"] = [
            frame(0.6, video_id="L01_V001", frame_id=90, pts_sec=4000.0),
            frame(0.6, video_id="L01_V002", frame_id=90, pts_sec=6.0),
        ]

        request = TrakeSearchRequest(
            task="trake", overview="a jump", events=["run", "jump"], max_gap_sec=0.0
        )
        response = tracks.search_trake(request, CONFIG)

        assert [r.video_id for r in response.results] == ["L01_V002", "L01_V001"]

    def test_alternates_are_ranked_by_swap_cost(self, fake_retrieve):
        """Alternates are offered in the order the ranker would pick them.

        Frame 60 scores lower than frame 800 but sits seconds from the chosen
        first event instead of four minutes away, so swapping it in costs less.
        Listing 800 first would recommend the swap the sequence search itself
        ranks worst.
        """
        responses = fake_retrieve.responses
        responses["run"] = [frame(0.9, frame_id=10, pts_sec=1.0)]
        responses["jump"] = [
            frame(0.90, frame_id=900, pts_sec=250.0),
            frame(0.62, frame_id=800, pts_sec=240.0),
            frame(0.60, frame_id=60, pts_sec=4.0),
        ]

        result = tracks.search_trake(self._request(["run", "jump"]), CONFIG).results[0]

        assert result.frame_ids == [10, 900]
        assert [a.frame_id for a in result.events[1].alternates] == [60, 800]

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


class TestRewriting:
    """What gets searched with the rewrite, and what keeps the original.

    The rewrite is English for the sake of the image space and the reranker; the
    speech collection holds Vietnamese transcripts, so it is searched with the
    query as typed. Every test above runs with rewriting a no-op, because
    `CONFIG` sets no endpoint - which is also what protects them from the
    network.
    """

    @pytest.fixture
    def rewriter(self, monkeypatch):
        """Stand in for the LLM: prefix each query, preserving order."""
        batches: list[list[str]] = []

        def _rewrite_queries(texts, config, timings):
            batches.append(list(texts))
            return [Rewrite(f"EN {text}", f"VI {text}") for text in texts]

        monkeypatch.setattr(tracks, "rewrite_queries", _rewrite_queries)
        return batches

    def test_kis_encodes_the_rewrite_and_hears_the_original(
        self, fake_retrieve, rewriter
    ):
        fake_retrieve.responses["EN xe hơi đỏ"] = [frame(0.9, frame_id=10)]

        response = tracks.search_kis(
            KisSearchRequest(task="kis", description="xe hơi đỏ", top_k=10), CONFIG
        )

        assert fake_retrieve.calls == ["EN xe hơi đỏ"]
        # Not the raw input: the speech stage gets the cleaned Vietnamese, so
        # "hãy tìm trong video" stops scoring as a BM25 term against transcripts.
        assert fake_retrieve.speech_calls == ["VI xe hơi đỏ"]
        assert response.rewritten_queries == ["EN xe hơi đỏ"]
        assert response.cleaned_queries == ["VI xe hơi đỏ"]
        assert response.results[0].frame_ids == [10]

    def test_asr_only_searches_the_cleaned_form_without_visual_retrieval(
        self, fake_retrieve, rewriter, monkeypatch
    ):
        spoken: list[str] = []

        def _search_asr_only(text, top_k, config, timings):
            spoken.append(text)
            return [
                AsrOnlyHit(
                    frame=frame(
                        0.0,
                        video_id="L01_V002",
                        shot_id=4,
                        frame_id=240,
                        pts_sec=8.0,
                    ),
                    segment=AsrSegment(
                        score=0.92,
                        video_id="L01_V002",
                        segment=3,
                        start_sec=7.5,
                        end_sec=9.0,
                        text="nội dung lời thoại",
                    ),
                )
            ]

        monkeypatch.setattr(tracks, "search_asr_only", _search_asr_only)

        response = tracks.search_qa(
            QaSearchRequest(
                task="qa",
                description="hãy tìm lời thoại",
                retrieval_mode="asr_only",
            ),
            CONFIG,
        )

        assert rewriter == [["hãy tìm lời thoại"]]
        assert spoken == ["VI hãy tìm lời thoại"]
        assert fake_retrieve.calls == []
        assert response.effective_retrieval_mode == "asr_only"
        assert response.rewritten_queries == ["EN hãy tìm lời thoại"]
        assert response.cleaned_queries == ["VI hãy tìm lời thoại"]
        assert response.results[0].frame_ids == [240]
        assert response.results[0].asr_evidence.text == "nội dung lời thoại"

    def test_trake_rewrites_the_whole_request_in_one_call(
        self, fake_retrieve, rewriter
    ):
        """One batch, and the events stay in the order they were given.

        Each event is searched twice - globally, then inside each candidate -
        and the second stage must reuse the first stage's rewrite rather than
        ask again.
        """
        fake_retrieve.responses["EN chạy"] = [frame(0.9, frame_id=10)]
        fake_retrieve.responses["EN nhảy"] = [frame(0.8, frame_id=50)]

        response = tracks.search_trake(
            TrakeSearchRequest(
                task="trake", overview="cú nhảy", events=["chạy", "nhảy"], top_k=10
            ),
            CONFIG,
        )

        assert rewriter == [["cú nhảy", "chạy", "nhảy"]]
        assert fake_retrieve.calls == ["EN cú nhảy", "EN chạy", "EN nhảy"]
        assert fake_retrieve.per_video_calls == [
            ("EN chạy", ["L01_V001"]),
            ("EN nhảy", ["L01_V001"]),
        ]
        # Both stages hear the cleaned form, event order preserved.
        assert fake_retrieve.speech_calls == [
            "VI cú nhảy",
            "VI chạy",
            "VI nhảy",
            "VI chạy",
            "VI nhảy",
        ]
        assert response.rewritten_queries == ["EN cú nhảy", "EN chạy", "EN nhảy"]
        assert response.results[0].frame_ids == [10, 50]

    def test_a_failed_rewrite_leaves_the_query_alone(self, fake_retrieve, monkeypatch):
        """None means "nothing was rewritten", and the query still runs."""
        monkeypatch.setattr(
            tracks, "rewrite_queries", lambda texts, config, timings: None
        )
        fake_retrieve.responses["xe hơi đỏ"] = [frame(0.9, frame_id=10)]

        response = tracks.search_kis(
            KisSearchRequest(task="kis", description="xe hơi đỏ", top_k=10), CONFIG
        )

        assert fake_retrieve.calls == ["xe hơi đỏ"]
        assert response.rewritten_queries is None
        assert response.cleaned_queries is None
        assert response.results[0].frame_ids == [10]


class TestPreRewrittenForms:
    """What `/search/decompose` produced is searched verbatim.

    The review screen only means something if the operator's edit is what runs,
    so a request carrying both forms must make no LLM call at all - and a
    caption the model never returned must not cost the query its speech form.
    """

    def _forms(self, vision: str | None, speech: str) -> QueryForms:
        return QueryForms(original=speech, vision=vision, speech=speech)

    def test_supplied_forms_skip_rewriting_entirely(self, fake_retrieve, monkeypatch):
        monkeypatch.setattr(
            tracks,
            "rewrite_queries",
            lambda *args, **kwargs: pytest.fail("rewriting ran on a decomposed query"),
        )
        fake_retrieve.responses["EN run"] = [frame(0.9, frame_id=10)]
        fake_retrieve.responses["EN jump"] = [frame(0.8, frame_id=50)]

        response = tracks.search_trake(
            TrakeSearchRequest(
                task="trake",
                overview=self._forms("EN overview", "VI overview"),
                events=[self._forms("EN run", "VI run"), self._forms("EN jump", "VI jump")],
                top_k=10,
            ),
            CONFIG,
        )

        assert fake_retrieve.calls == ["EN overview", "EN run", "EN jump"]
        assert fake_retrieve.speech_calls[:3] == ["VI overview", "VI run", "VI jump"]
        assert response.rewritten_queries == ["EN overview", "EN run", "EN jump"]
        assert response.cleaned_queries == ["VI overview", "VI run", "VI jump"]

    def test_a_missing_caption_searches_the_original_language(self, fake_retrieve):
        fake_retrieve.responses["VI run"] = [frame(0.9, frame_id=10)]
        fake_retrieve.responses["VI jump"] = [frame(0.8, frame_id=50)]

        tracks.search_trake(
            TrakeSearchRequest(
                task="trake",
                overview=self._forms(None, "VI overview"),
                events=[self._forms(None, "VI run"), self._forms(None, "VI jump")],
                top_k=10,
            ),
            CONFIG,
        )

        assert fake_retrieve.calls == ["VI overview", "VI run", "VI jump"]

    def test_a_mixed_payload_is_rewritten_as_a_whole(self, fake_retrieve, monkeypatch):
        """A payload the decompose screen did not produce gets the ordinary path."""
        seen: list[list[str]] = []

        def _rewrite(texts, config, timings):
            seen.append(list(texts))
            return None

        monkeypatch.setattr(tracks, "rewrite_queries", _rewrite)
        fake_retrieve.responses["VI run"] = [frame(0.9, frame_id=10)]

        tracks.search_trake(
            TrakeSearchRequest(
                task="trake",
                overview="VI overview",
                events=[self._forms("EN run", "VI run")],
                top_k=10,
            ),
            CONFIG,
        )

        assert seen == [["VI overview", "VI run"]]


class TestStageAScores:
    def _request(self, events: list[str], top_k: int = 10) -> TrakeSearchRequest:
        return TrakeSearchRequest(
            task="trake", overview="a jump", events=events, top_k=top_k
        )

    def test_the_parts_of_the_video_score_are_reported(self, fake_retrieve):
        responses = fake_retrieve.responses
        responses["a jump"] = [frame(0.5, frame_id=1)]
        responses["run"] = [frame(0.9, shot_id=1, frame_id=10)]
        responses["jump"] = [frame(0.7, shot_id=2, frame_id=50)]

        response = tracks.search_trake(self._request(["run", "jump"]), CONFIG)

        stage_a = response.results[0].stage_a
        assert stage_a.rank == 1
        assert stage_a.overview_score == 0.5
        assert stage_a.event_scores == [0.9, 0.7]
        # Composite is the overview plus the mean of the events, as stage A ranks.
        assert stage_a.score == pytest.approx(0.5 + 0.8)

    def test_an_event_the_video_never_matched_scores_zero(self, fake_retrieve):
        responses = fake_retrieve.responses
        responses["a jump"] = [frame(0.5, frame_id=1)]
        responses["run"] = [frame(0.9, shot_id=1, frame_id=10)]
        # "jump" is found nowhere globally, but is found inside the video.
        fake_retrieve.per_video_responses = {
            "run": responses["run"],
            "jump": [frame(0.4, shot_id=2, frame_id=50)],
        }

        response = tracks.search_trake(self._request(["run", "jump"]), CONFIG)

        assert response.results[0].stage_a.event_scores == [0.9, 0.0]

    def test_supplied_video_ids_report_no_stage_a(self, fake_retrieve):
        responses = fake_retrieve.responses
        responses["run"] = [frame(0.9, shot_id=1, frame_id=10)]
        responses["jump"] = [frame(0.8, shot_id=2, frame_id=50)]
        request = TrakeSearchRequest(
            task="trake",
            overview="a jump",
            events=["run", "jump"],
            top_k=10,
            video_ids=["L01_V001"],
        )

        response = tracks.search_trake(request, CONFIG)

        # Stage A never ran, and 0.0 would read as "matched nothing".
        assert response.results[0].stage_a is None

    def test_the_candidate_pool_never_exceeds_the_rows_asked_for(self, fake_retrieve):
        responses = fake_retrieve.responses
        responses["a jump"] = [
            frame(0.9 - index / 100, video_id=f"L01_V{index:03d}", frame_id=1)
            for index in range(10)
        ]
        responses["run"] = list(responses["a jump"])
        responses["jump"] = [
            frame(0.5, video_id=f"L01_V{index:03d}", shot_id=2, frame_id=50)
            for index in range(10)
        ]

        tracks.search_trake(self._request(["run", "jump"], top_k=3), CONFIG)

        # Stage B is videos x events serial round trips, so a pool wider than
        # the rows requested is work that could never produce a row.
        assert [len(videos) for _, videos in fake_retrieve.per_video_calls] == [3, 3]


class TestTemporalKis:
    def _request(self, top_k: int = 10) -> KisSearchRequest:
        return KisSearchRequest(
            task="kis",
            description="một người vệ sinh máy ảnh, tháo rời rồi lau ống kính",
            overview="cleaning a camera",
            events=["taking the camera apart", "wiping the lens"],
            top_k=top_k,
        )

    def test_one_frame_is_reported_and_it_is_the_best_scoring_event(
        self, fake_retrieve
    ):
        responses = fake_retrieve.responses
        responses["cleaning a camera"] = [frame(0.4, frame_id=1)]
        responses["taking the camera apart"] = [frame(0.6, shot_id=1, frame_id=10)]
        responses["wiping the lens"] = [frame(0.95, shot_id=2, frame_id=50)]

        response = tracks.search_kis(self._request(), CONFIG)

        assert response.task == "kis"
        assert response.results[0].frame_ids == [50]

    def test_the_whole_sequence_is_still_reported_for_review(self, fake_retrieve):
        responses = fake_retrieve.responses
        responses["cleaning a camera"] = [frame(0.4, frame_id=1)]
        responses["taking the camera apart"] = [frame(0.9, shot_id=1, frame_id=10)]
        responses["wiping the lens"] = [frame(0.7, shot_id=2, frame_id=50)]

        response = tracks.search_kis(self._request(), CONFIG)

        assert response.results[0].frame_ids == [10]
        assert [event.frame_id for event in response.results[0].events] == [10, 50]
        assert response.results[0].stage_a.event_scores == [0.9, 0.7]

    def test_every_event_votes_on_the_video(self, fake_retrieve):
        """The point of the track: a moment the overview cannot find still counts."""
        responses = fake_retrieve.responses
        responses["cleaning a camera"] = [frame(0.3, video_id="L01_V002", frame_id=1)]
        responses["taking the camera apart"] = [
            frame(0.9, video_id="L01_V001", shot_id=1, frame_id=10)
        ]
        responses["wiping the lens"] = [
            frame(0.9, video_id="L01_V001", shot_id=2, frame_id=50)
        ]

        response = tracks.search_kis(self._request(), CONFIG)

        assert response.results[0].video_id == "L01_V001"

    def test_a_plain_description_takes_the_direct_path(self, fake_retrieve):
        fake_retrieve.responses["xe hơi đỏ"] = [frame(0.9, frame_id=10)]

        response = tracks.search_kis(
            KisSearchRequest(task="kis", description="xe hơi đỏ", top_k=10), CONFIG
        )

        assert fake_retrieve.per_video_calls == []
        assert response.results[0].frame_ids == [10]
        assert response.results[0].events is None
