"""The caption cap follows the profile's text tower, not a number in a prompt.

40 words is right for SigLIP2 and wrong for anything else. These hold the
derivation to the two things that actually matter: a SigLIP2 deployment must
keep the exact prompt it shipped with, and a wider tower must stop being
throttled to a window it does not have.
"""

import pytest

from app.retrieval import decompose
from app.retrieval.engine import RetrievalConfig
from app.retrieval.rewrite import (
    CAPTION_PROMPT,
    DEFAULT_CAPTION_WORDS,
    MAX_CAPTION_WORDS,
    MIN_CAPTION_WORDS,
    _caption_budget,
    caption_prompt,
    caption_word_cap,
)


def config(profile: str) -> RetrievalConfig:
    return RetrievalConfig(frames_collection="frames-v1", feature_profile=profile)


def test_siglip_still_gets_the_forty_words_it_was_tuned_for():
    """The number that shipped. A derivation that moved it would be a silent
    change to every caption the competition run produces."""
    assert caption_word_cap(config("siglip2-giant-opt-patch16-384-v1")) == 40
    assert caption_word_cap(config("siglip2-so400m-patch14-384-v1")) == 40


def test_the_default_prompt_is_byte_for_byte_what_it_was():
    """`CAPTION_PROMPT` is still importable and still the SigLIP2 prompt, so
    the existing rewrite and decompose tests are testing the shipped string."""
    assert caption_prompt(DEFAULT_CAPTION_WORDS) == CAPTION_PROMPT
    assert "AT MOST 40 words" in CAPTION_PROMPT
    assert decompose.event_caption_prompt(40) == decompose.EVENT_CAPTION_PROMPT


def test_jina_is_not_throttled_to_a_window_it_does_not_have():
    """8192 tokens. Capping this at 40 words throws away detail the encoder
    would have read — the whole reason the number left the prompt string."""
    cap = caption_word_cap(config("jina-clip-v2"))

    assert cap == MAX_CAPTION_WORDS
    assert cap > DEFAULT_CAPTION_WORDS
    assert f"AT MOST {cap} words" in caption_prompt(cap)


def test_clip_b32_gets_its_own_slightly_wider_window():
    """77 tokens, so 48 words. Not 40, and not the ceiling — the derivation has
    to actually track the profile rather than picking one of two answers."""
    assert caption_word_cap(config("clip-b32-v1")) == 48


def test_an_unknown_profile_does_not_take_the_query_path_down():
    """A rewrite is an improvement, never a dependency. Raising here would fail
    a live search over a profile name, which is not a trade worth making."""
    assert caption_word_cap(config("not-a-real-profile")) == DEFAULT_CAPTION_WORDS


def test_the_derived_cap_stays_inside_its_bounds():
    for name in ("clip-b32-v1", "jina-clip-v2", "siglip2-giant-opt-patch16-384-v1"):
        cap = caption_word_cap(config(name))
        assert MIN_CAPTION_WORDS <= cap <= MAX_CAPTION_WORDS, name


def test_the_output_budget_follows_the_cap_not_the_query():
    """Capping is what makes the budget independent of input length: a long
    query and a short one both produce a caption of at most `word_cap` words."""
    short = ("mèo",)
    long = ("x" * 700,)

    assert _caption_budget(short, 40) == _caption_budget(long, 40)
    assert _caption_budget(short, 120) > _caption_budget(short, 40)


def test_the_budget_scales_with_the_batch():
    """One caption per query, so a TRAKE overview plus five events needs six."""
    one = _caption_budget(("a",), 40)
    six = _caption_budget(tuple("abcdef"), 40)

    assert six > one
    assert six - 64 == 6 * (one - 64)


def test_the_shipped_budget_for_a_single_siglip_query_is_unchanged():
    """64 tokens of room per line plus 64, which is what this always sent."""
    assert _caption_budget(("một câu truy vấn",), 40) == 64 + 64
