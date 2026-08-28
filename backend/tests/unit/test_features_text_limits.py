"""The text tower's context length, and the warning when a query overruns it.

`embed_text` truncates. That is the right behaviour — a query answered from
its first 64 tokens still ranks usefully, and refusing it outright during a
competition run would be worse. What is not acceptable is doing it silently,
because the symptom of a truncated query is a plausible ranking, not an error.
"""

import logging

import numpy as np
import pytest

from app.features import multimodal
from app.features.profiles import FEATURE_PROFILES, get_profile


class FakeTokenizer:
    """Counts one token per whitespace-separated word."""

    def __call__(self, text, truncation=False):
        return {"input_ids": text.split()}


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()


class FakeRuntime:
    def __init__(self):
        self.processor = FakeProcessor()


def test_siglip_profiles_declare_the_64_token_tower():
    """SigLIP2's text tower is the tightest budget any caller has to respect."""
    for name in (
        "siglip2-giant-opt-patch16-384-v1",
        "siglip2-so400m-patch14-384-v1",
    ):
        assert get_profile(name).max_text_tokens == 64


def test_jina_declares_room_a_rewriter_need_not_budget_for():
    assert get_profile("jina-clip-v2").max_text_tokens == 8192


def test_every_profile_declares_a_limit():
    """A missing limit would read as "unbounded" to a rewriter. None may be 0."""
    for name, profile in FEATURE_PROFILES.items():
        assert profile.max_text_tokens > 0, name


def test_warns_when_the_query_overruns_the_tower(caplog):
    profile = get_profile("siglip2-giant-opt-patch16-384-v1")
    text = " ".join(["từ"] * 70)

    with caplog.at_level(logging.WARNING, logger=multimodal.__name__):
        multimodal._warn_if_truncated(FakeRuntime(), profile, text)

    assert "70 tokens" in caplog.text
    # The operator needs to know how much was lost, not merely that something
    # was: six tokens off a rewrite is noise, forty is a different query.
    assert "last 6 tokens" in caplog.text


def test_says_nothing_when_the_query_fits(caplog):
    profile = get_profile("siglip2-giant-opt-patch16-384-v1")

    with caplog.at_level(logging.WARNING, logger=multimodal.__name__):
        multimodal._warn_if_truncated(FakeRuntime(), profile, " ".join(["từ"] * 64))

    assert caplog.text == ""


def test_a_query_at_the_limit_is_not_a_truncation(caplog):
    """Exactly `max_text_tokens` reaches the encoder whole; only past it is lost."""
    profile = get_profile("clip-b32-v1")

    with caplog.at_level(logging.WARNING, logger=multimodal.__name__):
        multimodal._warn_if_truncated(FakeRuntime(), profile, " ".join(["w"] * 77))

    assert caplog.text == ""


def test_a_runtime_without_a_tokenizer_is_not_an_error():
    """The Jina branch has no processor at all and never reaches this path."""
    profile = get_profile("jina-clip-v2")

    class NoTokenizer:
        processor = None

    multimodal._warn_if_truncated(NoTokenizer(), profile, "bất kỳ")


def test_pool_features_still_rejects_a_dimension_mismatch():
    """The guard that would catch a profile pointed at the wrong collection."""
    with pytest.raises(multimodal.FeatureExtractionError, match="dimension"):
        multimodal.pool_features(np.ones((1, 512), dtype=np.float32), 1024)
