"""Query rewriting, without the endpoint.

The value of this step is entirely in what it does when the model misbehaves.
The query path is not allowed a hard network dependency, so every failure -
unreachable box, timeout, chatty, misnumbered or truncated output - has to come
back as "not rewritten" and cost the query nothing but the timeout. And because
the caption and the cleaning are separate calls, each has to fail on its own
without taking the other down.
"""

from dataclasses import replace

import httpx
import pytest

from app.retrieval import rewrite
from app.retrieval.engine import RetrievalConfig, Timings
from app.retrieval.rewrite import CAPTION_PROMPT, CLEAN_PROMPT, Rewrite

CONFIG = RetrievalConfig(
    frames_collection="frames-v2",
    feature_profile="siglip2-giant-opt-patch16-384-v1",
    rewrite_base_url="http://vlm.invalid/v1",
    rewrite_model="Qwen/Qwen3.6-27B",
)

# As typed, captioned for the image space, and stripped for the transcripts.
VI = [
    "Hãy tìm trong video một người đàn ông mặc áo đỏ đang chạy trên bãi biển",
    "Đoạn video về tường thuật một cuộc đua xe đạp",
]
CAPTIONS = [
    "a man in a red shirt running on the beach",
    "three cyclists riding in a line, white jerseys and yellow-green shorts",
]
CLEANED = [
    "một người đàn ông mặc áo đỏ đang chạy trên bãi biển",
    "tường thuật một cuộc đua xe đạp",
]
BOTH = [Rewrite(v, s) for v, s in zip(CAPTIONS, CLEANED)]


def numbered(lines: list[str]) -> str:
    return "\n".join(f"{index}. {line}" for index, line in enumerate(lines, start=1))


@pytest.fixture(autouse=True)
def clear_cache():
    """Successes are memoised for the session, so tests must not share them."""
    rewrite._call.cache_clear()
    yield
    rewrite._call.cache_clear()


@pytest.fixture
def responder(monkeypatch):
    """Answer each prompt with canned content, recording every payload sent.

    Keyed on the system prompt, because the two calls are told apart by nothing
    else - and a test that swaps one of them has to leave the other working.
    """
    sent: list[dict] = []
    replies: dict[str, object] = {
        CAPTION_PROMPT: numbered(CAPTIONS),
        CLEAN_PROMPT: numbered(CLEANED),
    }

    def _post(base_url, payload, api_key, timeout):
        sent.append(payload)
        reply = replies[payload["messages"][0]["content"]]
        if isinstance(reply, Exception):
            raise reply
        # A plain string is a complete answer; a tuple carries a finish reason.
        return reply if isinstance(reply, tuple) else (reply, "stop")

    monkeypatch.setattr(rewrite, "_post", _post)
    return sent, replies


def sent_for(sent: list[dict], prompt: str) -> dict:
    return next(p for p in sent if p["messages"][0]["content"] == prompt)


def test_both_forms_come_back_in_input_order(responder):
    sent, _ = responder

    assert rewrite.rewrite_queries(VI, CONFIG, Timings()) == BOTH
    # Two calls, one per job, each carrying the whole batch numbered the way the
    # reply is matched: the two orderings are the same fact.
    assert len(sent) == 2
    for payload in sent:
        assert payload["messages"][1]["content"] == numbered(VI)


def test_a_failed_caption_does_not_cost_the_cleaned_form(responder):
    """The reason the two jobs are separate calls."""
    _, replies = responder
    replies[CAPTION_PROMPT] = httpx.ConnectError("no route to host")

    result = rewrite.rewrite_queries(VI, CONFIG, Timings())

    assert [r.speech for r in result] == CLEANED
    # The caption falls back to the query as typed, which is then what really is
    # encoded - so reporting it as the rewritten query stays honest.
    assert [r.vision for r in result] == VI


def test_a_failed_cleaning_does_not_cost_the_caption(responder):
    _, replies = responder
    replies[CLEAN_PROMPT] = httpx.ReadTimeout("too slow")

    result = rewrite.rewrite_queries(VI, CONFIG, Timings())

    assert [r.vision for r in result] == CAPTIONS
    assert [r.speech for r in result] == VI


def test_both_failing_is_reported_as_not_rewritten(responder):
    _, replies = responder
    replies[CAPTION_PROMPT] = httpx.ConnectError("down")
    replies[CLEAN_PROMPT] = httpx.ConnectError("down")

    assert rewrite.rewrite_queries(VI, CONFIG, Timings()) is None


def test_a_reasoning_block_is_stripped(responder):
    """Thinking is disabled in the request, but not every server obeys."""
    _, replies = responder
    replies[CAPTION_PROMPT] = f"<think>Let me translate.</think>\n{numbered(CAPTIONS)}"

    assert rewrite.rewrite_queries(VI, CONFIG, Timings()) == BOTH


def test_surrounding_quotes_and_a_preamble_are_tolerated(responder):
    _, replies = responder
    replies[CAPTION_PROMPT] = f'Here you go:\n1. "{CAPTIONS[0]}"\n2. {CAPTIONS[1]}\n'

    assert rewrite.rewrite_queries(VI, CONFIG, Timings()) == BOTH


def test_a_missing_line_falls_back_rather_than_shifting_the_rest(responder):
    """A partial parse is worse than none: it moves a TRAKE event one query left."""
    _, replies = responder
    replies[CLEAN_PROMPT] = f"1. {CLEANED[0]}\n3. something else"

    result = rewrite.rewrite_queries(VI, CONFIG, Timings())

    assert [r.speech for r in result] == VI
    assert [r.vision for r in result] == CAPTIONS


def test_a_truncated_answer_falls_back(responder):
    """`finish_reason` is the only signal that the tail was cut.

    A truncated reply still parses - the early lines are intact - so the last
    query would come back as half a sentence and be searched that way.
    """
    _, replies = responder
    replies[CLEAN_PROMPT] = (numbered([CLEANED[0], CLEANED[1][:12]]), "length")

    assert [r.speech for r in rewrite.rewrite_queries(VI, CONFIG, Timings())] == VI


def test_the_cleaning_budget_is_sized_from_the_input(responder):
    """A 700-character description is echoed back nearly whole."""
    sent, replies = responder
    long_query = "Đoạn video mô tả cảnh trang trí bánh rán. " * 17  # ~700 chars
    replies[CAPTION_PROMPT] = "1. chef decorating donuts"
    replies[CLEAN_PROMPT] = "1. cảnh trang trí bánh rán"

    rewrite.rewrite_queries([long_query], CONFIG, Timings())

    assert sent_for(sent, CLEAN_PROMPT)["max_tokens"] > len(long_query) // 3
    # The caption is capped at 40 words however long the query is, so its budget
    # must not inflate with the input.
    assert sent_for(sent, CAPTION_PROMPT)["max_tokens"] < 200


def test_a_dead_endpoint_costs_the_query_nothing_and_is_not_cached(monkeypatch):
    attempts: list[int] = []

    def _post(base_url, payload, api_key, timeout):
        attempts.append(1)
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(rewrite, "_post", _post)
    timings = Timings()

    assert rewrite.rewrite_queries(VI, CONFIG, timings) is None
    assert rewrite.rewrite_queries(VI, CONFIG, timings) is None

    # `lru_cache` does not memoise exceptions, so the box coming back mid-session
    # is picked up on the next query rather than after a restart. Two calls per
    # attempt, one per job.
    assert len(attempts) == 4
    assert "rewrite" in timings.as_dict()


def test_a_success_is_reused_for_the_rest_of_the_session(responder):
    sent, _ = responder

    first = rewrite.rewrite_queries(VI, CONFIG, Timings())
    second = rewrite.rewrite_queries(VI, CONFIG, Timings())

    assert first == second == BOTH
    assert len(sent) == 2


def test_no_endpoint_means_no_call_at_all(monkeypatch):
    """The default config: rewriting is opt-in on the URL, not just the flag."""

    def _unreachable(*args, **kwargs):
        raise AssertionError("the endpoint must not be called")

    monkeypatch.setattr(rewrite, "_post", _unreachable)

    for config in (
        replace(CONFIG, rewrite_base_url=None),
        replace(CONFIG, rewrite_enabled=False),
    ):
        assert rewrite.rewrite_queries(VI, config, Timings()) is None
    assert rewrite.rewrite_queries([], CONFIG, Timings()) is None
