"""Decomposing one pasted query into an overview and its events.

Two paths that fail in different ways on purpose. The marker path is a regex,
so it cannot fail at all - it only loses its captions when the box is down. The
prose path *is* the model, so when the model is gone there is nothing to review
and the request is refused rather than answered with a decomposition nobody
performed.

The queries here are the real ones from `data/evaluation_set_p1.csv`, including
the task the organiser numbered `E1, E2, E2, E4`.
"""

import pytest

from app.retrieval import decompose, rewrite
from app.retrieval.decompose import (
    EVENT_CAPTION_PROMPT,
    DecompositionUnavailableError,
    split_markers,
)
from app.retrieval.engine import RetrievalConfig, Timings
from app.retrieval.rewrite import CLEAN_PROMPT

CONFIG = RetrievalConfig(
    frames_collection="frames-v2",
    feature_profile="siglip2-giant-opt-patch16-384-v1",
    rewrite_base_url="http://vlm.invalid/v1",
    rewrite_model="Qwen/Qwen3.6-27B",
)

# Numbered E1, E2, E2, E4 by the organiser. Four events all the same.
MISNUMBERED = (
    "Trong đoạn video nấu ăn một món ăn về nấm, gồm các khoảnh khắc sơ chế:\n"
    "E1: Khoảnh khắc đầu tiên thấy cắt nấm.\n"
    "E2: Khoảnh khắc đầu tiên cắt củ năng.\n"
    "E2: Khoảnh khắc đầu tiên cắt đậu hủ.\n"
    "E4: Khoảnh khắc chảo đặt lên bếp, đầu bếp mở lửa và thấy lửa bắt đầu xuất hiện"
)
NO_OVERVIEW = (
    "E1: Khoảnh khắc đầu tiên bột được bỏ vào tô măng tây.\n"
    "E2: Khoảnh khắc đầu tiên thấy miến măng tây tiếp xúc với dầu trong chảo.\n"
    "E3: Khoảnh khắc miếng măng tây đầu tiên rời khỏi chảo dầu."
)
PROSE = (
    "Đoạn video mô tả một người ngồi vệ sinh máy ảnh. Công đoạn này bắt đầu"
    " bằng việc tháo rời máy ảnh. Tiếp theo, chiếc ống kính đã được tháo rời và"
    " được đặt ngay ngắn trên một chiếc khăn màu tím hồng. Phân cảnh cuối cùng"
    " là vệ sinh ống kính bằng một chiếc tăm bông."
)
PROSE_LINES = [
    "một người ngồi vệ sinh máy ảnh",
    "tháo rời máy ảnh",
    "chiếc ống kính đã được tháo rời, đặt trên một chiếc khăn màu tím hồng",
    "vệ sinh ống kính bằng một chiếc tăm bông",
]


def numbered(lines: list[str]) -> str:
    return "\n".join(f"{index}. {line}" for index, line in enumerate(lines, start=1))


@pytest.fixture(autouse=True)
def clear_cache():
    rewrite._call.cache_clear()
    yield
    rewrite._call.cache_clear()


@pytest.fixture
def responder(monkeypatch):
    """Answer each prompt by canned content, recording the order they were sent.

    Keyed on the system prompt: the calls are told apart by nothing else, and
    which of them ran - and in what order - is most of what these tests check.
    """
    sent: list[dict] = []
    replies: dict[str, object] = {}

    def _post(base_url, payload, api_key, timeout):
        system = payload["messages"][0]["content"]
        sent.append(payload)
        reply = replies[system]
        if isinstance(reply, Exception):
            raise reply
        return reply, "stop"

    monkeypatch.setattr(rewrite, "_post", _post)
    return sent, replies


def systems(sent: list[dict]) -> list[str]:
    """The system prompt of every call made, in order."""
    return [payload["messages"][0]["content"] for payload in sent]


def sent_under(sent: list[dict], system: str) -> str:
    """What one prompt was actually handed."""
    return next(
        payload["messages"][1]["content"]
        for payload in sent
        if payload["messages"][0]["content"] == system
    )


class TestMarkerSplit:
    def test_events_are_counted_by_marker_not_by_number(self):
        overview, events = split_markers(MISNUMBERED)

        assert len(events) == 4
        assert events[1] == "Khoảnh khắc đầu tiên cắt củ năng."
        assert events[2] == "Khoảnh khắc đầu tiên cắt đậu hủ."
        assert overview.startswith("Trong đoạn video nấu ăn")

    def test_a_task_can_open_straight_into_its_first_event(self):
        overview, events = split_markers(NO_OVERVIEW)

        assert overview == ""
        assert len(events) == 3

    def test_prose_carries_no_markers(self):
        assert split_markers(PROSE) is None

    def test_one_marker_is_not_an_enumeration(self):
        assert split_markers("E1: chỉ một khoảnh khắc duy nhất") is None

    def test_markers_pasted_onto_one_line_still_split(self):
        overview, events = split_markers(
            "Cảnh múa lân, tìm các sự kiện sau: E1: lân xoay vòng. E2: lân tiếp đất."
        )

        assert overview == "Cảnh múa lân, tìm các sự kiện sau:"
        assert events == ["lân xoay vòng.", "lân tiếp đất."]


class TestMarkerPath:
    def test_the_decomposer_is_never_called(self, responder):
        sent, replies = responder
        replies[CLEAN_PROMPT] = numbered(["nấu ăn món nấm", "a", "b", "c", "d"])
        replies[EVENT_CAPTION_PROMPT] = numbered(["EN0", "EN1", "EN2", "EN3", "EN4"])

        result = decompose.decompose(MISNUMBERED, 6, CONFIG, Timings())

        assert sorted(systems(sent)) == sorted([CLEAN_PROMPT, EVENT_CAPTION_PROMPT])
        assert result.source == "markers"
        assert len(result.events) == 4
        assert result.overview.speech == "nấu ăn món nấm"
        assert [event.vision for event in result.events] == ["EN1", "EN2", "EN3", "EN4"]

    def test_every_event_reports_the_text_it_was_cut_from(self, responder):
        sent, replies = responder
        replies[CLEAN_PROMPT] = numbered(["overview", "a", "b", "c", "d"])
        replies[EVENT_CAPTION_PROMPT] = numbered(["EN0", "EN1", "EN2", "EN3", "EN4"])

        result = decompose.decompose(MISNUMBERED, 6, CONFIG, Timings())

        assert result.events[0].original == "Khoảnh khắc đầu tiên thấy cắt nấm."
        assert result.overview.original.startswith("Trong đoạn video nấu ăn")

    def test_a_missing_overview_is_the_events_own_words(self, responder):
        sent, replies = responder
        replies[CLEAN_PROMPT] = numbered(["joined", "a", "b", "c"])
        replies[EVENT_CAPTION_PROMPT] = numbered(["EN0", "EN1", "EN2", "EN3"])

        result = decompose.decompose(NO_OVERVIEW, 6, CONFIG, Timings())

        # Sent for cleaning as the events joined, and reported as nobody's text.
        assert result.overview.original is None
        assert "măng tây" in sent_under(sent, CLEAN_PROMPT).splitlines()[0]

    def test_a_failed_deletion_leaves_the_markers_as_typed(self, responder):
        sent, replies = responder
        replies[CLEAN_PROMPT] = ValueError("chatty")
        replies[EVENT_CAPTION_PROMPT] = numbered(["EN0", "EN1", "EN2", "EN3", "EN4"])

        result = decompose.decompose(MISNUMBERED, 6, CONFIG, Timings())

        assert result.events[0].speech == "Khoảnh khắc đầu tiên thấy cắt nấm."
        assert result.events[0].vision == "EN1"

    def test_a_failed_caption_costs_only_the_vision_form(self, responder):
        sent, replies = responder
        replies[CLEAN_PROMPT] = numbered(["overview", "a", "b", "c", "d"])
        replies[EVENT_CAPTION_PROMPT] = ValueError("box down")

        result = decompose.decompose(MISNUMBERED, 6, CONFIG, Timings())

        assert [event.vision for event in result.events] == [None] * 4
        assert result.events[1].speech == "b"

    def test_markers_still_split_with_no_endpoint_configured(self):
        offline = RetrievalConfig(
            frames_collection="frames-v2", feature_profile="siglip2-giant-v1"
        )

        result = decompose.decompose(MISNUMBERED, 6, offline, Timings())

        assert len(result.events) == 4
        assert result.events[0].vision is None
        assert result.events[0].speech == "Khoảnh khắc đầu tiên thấy cắt nấm."


class TestProsePath:
    def test_the_split_happens_before_the_captions(self, responder):
        sent, replies = responder
        replies[decompose._decompose_prompt(6)] = numbered(PROSE_LINES)
        replies[EVENT_CAPTION_PROMPT] = numbered(["EN0", "EN1", "EN2", "EN3"])

        result = decompose.decompose(PROSE, 6, CONFIG, Timings())

        assert systems(sent) == [decompose._decompose_prompt(6), EVENT_CAPTION_PROMPT]
        assert result.source == "llm"
        assert [event.speech for event in result.events] == PROSE_LINES[1:]
        assert result.overview.speech == PROSE_LINES[0]

    def test_the_deletion_prompt_does_not_run_a_second_time(self, responder):
        sent, replies = responder
        replies[decompose._decompose_prompt(6)] = numbered(PROSE_LINES)
        replies[EVENT_CAPTION_PROMPT] = numbered(["EN0", "EN1", "EN2", "EN3"])

        decompose.decompose(PROSE, 6, CONFIG, Timings())

        assert CLEAN_PROMPT not in systems(sent)

    def test_events_point_at_no_span_of_the_query(self, responder):
        sent, replies = responder
        replies[decompose._decompose_prompt(6)] = numbered(PROSE_LINES)
        replies[EVENT_CAPTION_PROMPT] = numbered(["EN0", "EN1", "EN2", "EN3"])

        result = decompose.decompose(PROSE, 6, CONFIG, Timings())

        assert [event.original for event in result.events] == [None] * 3

    def test_a_failed_decomposition_is_fatal(self, responder):
        sent, replies = responder
        replies[decompose._decompose_prompt(6)] = ValueError("box down")

        with pytest.raises(DecompositionUnavailableError):
            decompose.decompose(PROSE, 6, CONFIG, Timings())

    def test_an_unconfigured_endpoint_cannot_decompose_prose(self):
        offline = RetrievalConfig(
            frames_collection="frames-v2", feature_profile="siglip2-giant-v1"
        )

        with pytest.raises(DecompositionUnavailableError):
            decompose.decompose(PROSE, 6, offline, Timings())

    def test_more_events_than_the_cap_is_refused_not_truncated(self, responder):
        sent, replies = responder
        replies[decompose._decompose_prompt(2)] = numbered(PROSE_LINES)

        with pytest.raises(DecompositionUnavailableError):
            decompose.decompose(PROSE, 2, CONFIG, Timings())

    def test_an_overview_with_no_events_is_refused(self, responder):
        sent, replies = responder
        replies[decompose._decompose_prompt(6)] = numbered(["chỉ có tổng quan"])

        with pytest.raises(DecompositionUnavailableError):
            decompose.decompose(PROSE, 6, CONFIG, Timings())

    def test_a_gap_in_the_numbering_is_refused(self, responder):
        sent, replies = responder
        replies[decompose._decompose_prompt(6)] = "1. tổng quan\n3. sự kiện hai"

        with pytest.raises(DecompositionUnavailableError):
            decompose.decompose(PROSE, 6, CONFIG, Timings())

    def test_latency_reports_both_stages(self, responder):
        sent, replies = responder
        replies[decompose._decompose_prompt(6)] = numbered(PROSE_LINES)
        replies[EVENT_CAPTION_PROMPT] = numbered(["EN0", "EN1", "EN2", "EN3"])

        result = decompose.decompose(PROSE, 6, CONFIG, Timings())

        assert set(result.latency_ms) == {"decompose", "caption"}
