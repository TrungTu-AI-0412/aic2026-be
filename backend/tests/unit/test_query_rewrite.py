"""Query rewriting, without the endpoint.

The value of this step is entirely in what it does when the model misbehaves.
The query path is not allowed a hard network dependency, so every failure -
unreachable box, timeout, chatty or misnumbered output - has to come back as
"nothing was rewritten" and cost the query nothing but the timeout.
"""

from dataclasses import replace

import httpx
import pytest

from app.retrieval import rewrite
from app.retrieval.engine import RetrievalConfig, Timings

CONFIG = RetrievalConfig(
    frames_collection="frames-v2",
    feature_profile="siglip2-giant-opt-patch16-384-v1",
    rewrite_base_url="http://vlm.invalid/v1",
    rewrite_model="Qwen/Qwen3.6-27B",
)

VI = [
    "Hãy tìm trong video một người đàn ông mặc áo đỏ đang chạy trên bãi biển",
    "cảnh sát giao thông thổi còi ở ngã tư đường Nguyễn Huệ",
]
EN = [
    "a man in a red shirt running on the beach",
    "a traffic police officer blowing a whistle at the Nguyen Hue intersection",
]


@pytest.fixture(autouse=True)
def clear_cache():
    """Successes are memoised for the session, so tests must not share them."""
    rewrite._rewrite.cache_clear()
    yield
    rewrite._rewrite.cache_clear()


@pytest.fixture
def responder(monkeypatch):
    """Answer the chat completion with canned content, recording the payloads."""
    sent: list[dict] = []
    replies: list[str] = []

    def _post(base_url, payload, api_key, timeout):
        sent.append(payload)
        return replies.pop(0)

    monkeypatch.setattr(rewrite, "_post", _post)
    return sent, replies


def test_numbered_output_comes_back_in_input_order(responder):
    sent, replies = responder
    replies.append(f"1. {EN[0]}\n2. {EN[1]}")

    assert rewrite.rewrite_queries(VI, CONFIG, Timings()) == EN
    # One call for the batch, and the input is numbered the way the reply is
    # matched: the two orderings are the same fact.
    assert len(sent) == 1
    assert sent[0]["messages"][1]["content"] == f"1. {VI[0]}\n2. {VI[1]}"


def test_a_reasoning_block_is_stripped(responder):
    """Thinking is disabled in the request, but not every server obeys."""
    _, replies = responder
    replies.append(
        f"<think>The user wants Vietnamese translated.</think>\n1. {EN[0]}\n2. {EN[1]}"
    )

    assert rewrite.rewrite_queries(VI, CONFIG, Timings()) == EN


def test_surrounding_quotes_and_a_preamble_are_tolerated(responder):
    _, replies = responder
    replies.append(f'Here you go:\n1. "{EN[0]}"\n2. {EN[1]}\n')

    assert rewrite.rewrite_queries(VI, CONFIG, Timings()) == EN


def test_a_missing_line_falls_back_rather_than_shifting_the_rest(responder):
    """A partial parse is worse than none: it moves a TRAKE event one query left."""
    _, replies = responder
    replies.append(f"1. {EN[0]}\n3. something else")

    assert rewrite.rewrite_queries(VI, CONFIG, Timings()) is None


def test_unnumbered_output_falls_back(responder):
    _, replies = responder
    replies.append("\n".join(EN))

    assert rewrite.rewrite_queries(VI, CONFIG, Timings()) is None


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
    # is picked up on the next query rather than after a restart.
    assert len(attempts) == 2
    assert "rewrite" in timings.as_dict()


def test_a_success_is_reused_for_the_rest_of_the_session(responder):
    sent, replies = responder
    replies.append(f"1. {EN[0]}\n2. {EN[1]}")

    first = rewrite.rewrite_queries(VI, CONFIG, Timings())
    second = rewrite.rewrite_queries(VI, CONFIG, Timings())

    assert first == second == EN
    # A second reply was never queued: `replies.pop(0)` would have raised.
    assert len(sent) == 1


def test_no_endpoint_means_no_call_at_all(monkeypatch):
    """The default config: rewriting is opt-in on the URL, not just the flag."""

    def _unreachable(*args, **kwargs):
        raise AssertionError("the endpoint must not be called")

    monkeypatch.setattr(rewrite, "_post", _unreachable)

    assert rewrite.rewrite_queries(VI, replace(CONFIG, rewrite_base_url=None), Timings()) is None
    assert rewrite.rewrite_queries(VI, replace(CONFIG, rewrite_enabled=False), Timings()) is None
    assert rewrite.rewrite_queries([], CONFIG, Timings()) is None
