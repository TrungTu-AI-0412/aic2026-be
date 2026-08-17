"""Tests for the evaluation-set builder.

Run from the repository root; `scripts` is not importable from `backend/`.
"""

from collections import Counter

from scripts.build_eval_set import (
    MIN_TOKEN_DF,
    best_window,
    distinctiveness,
    document_frequency,
)


def frequency_over(*texts: str) -> tuple[Counter, int]:
    return document_frequency(list(texts))


class TestDocumentFrequency:
    def test_counts_shots_not_occurrences(self) -> None:
        """A word said five times in one shot appears in one shot."""
        frequency, total = frequency_over("bão bão bão", "lụt")

        assert frequency["bão"] == 1
        assert total == 2


class TestDistinctiveness:
    def test_a_rare_word_beats_common_ones(self) -> None:
        # Above MIN_TOKEN_DF: rare enough to identify a shot, common enough
        # that the corpus agrees it is a real word.
        frequency = Counter({"chào": 900, "các": 900, "bạn": 900, "hezbollah": 6})
        common = distinctiveness("xin chào các bạn", [], frequency, 1000)
        rare = distinctiveness("hezbollah các bạn", [], frequency, 1000)

        assert rare > common

    def test_length_alone_does_not_win(self) -> None:
        """Summing every token would rank a long bland shot above a sharp one.

        A four-minute lecture accumulates hundreds of ordinary words. If volume
        counted, the set would fill with static-camera lectures whose moment is
        not findable in any interesting sense.
        """
        frequency = Counter({f"w{i}": 500 for i in range(60)})
        frequency["angelina"] = 4
        long_bland = " ".join(f"w{i}" for i in range(60))
        short_sharp = "angelina w1 w2"

        assert distinctiveness(short_sharp, [], frequency, 1000) > distinctiveness(
            long_bland, [], frequency, 1000
        )

    def test_entities_count_double(self) -> None:
        frequency = Counter({"movistar": 5, "khác": 5})
        plain = distinctiveness("movistar", [], frequency, 1000)
        tagged = distinctiveness("movistar", ["Movistar"], frequency, 1000)

        assert tagged > plain

    def test_a_token_below_the_df_floor_is_ignored(self) -> None:
        """ASR misrecognitions are unique by construction, so IDF loves them.

        `Bi tơ ri Cu nốp` for `Petr Vakoc` scores higher than any real name if
        rarity alone decides. Anything the corpus says fewer than MIN_TOKEN_DF
        times is treated as noise.
        """
        frequency = Counter({"bitoricunop": MIN_TOKEN_DF - 1, "chào": 900})

        assert distinctiveness("bitoricunop chào", [], frequency, 1000) == (
            distinctiveness("chào", [], frequency, 1000)
        )

    def test_text_with_nothing_above_the_floor_scores_zero(self) -> None:
        frequency = Counter({"lạ": 1})

        assert distinctiveness("lạ", [], frequency, 1000) == 0.0


class TestBestWindow:
    def test_short_text_is_returned_whole(self) -> None:
        frequency, total = frequency_over("một hai ba")

        assert best_window("một hai ba", [], frequency, total, {}, 32) == "một hai ba"

    def test_skips_the_greeting_for_the_informative_span(self) -> None:
        """Broadcast shots open with filler; the first N words are the worst N."""
        frequency = Counter(
            {"xin": 900, "chào": 900, "quý": 900, "vị": 900, "movistar": 5, "vuelta": 5}
        )
        text = "xin chào quý vị movistar vuelta"

        assert best_window(text, [], frequency, 1000, {}, 2) == "movistar vuelta"

    def test_window_length_is_respected(self) -> None:
        frequency, total = frequency_over("a b c d e f")

        assert len(best_window("a b c d e f", [], frequency, total, {}, 3).split()) == 3

    def test_memo_does_not_change_the_answer(self) -> None:
        """The word-score cache is an optimisation, not a behaviour change."""
        frequency = Counter({"xin": 900, "chào": 900, "movistar": 5})
        text = "xin chào movistar"
        shared: dict = {}

        first = best_window(text, [], frequency, 1000, {}, 1)
        second = best_window(text, [], frequency, 1000, shared, 1)
        third = best_window(text, [], frequency, 1000, shared, 1)

        assert first == second == third == "movistar"
