from app.eval.metrics import (
    OFFICIAL_K,
    QueryResult,
    mean_reciprocal_rank,
    mean_top_k_recall,
    rank_of,
    recall_at,
    summarise,
)

HITS = [("L21_V001", 0), ("L22_V005", 3), ("L21_V001", 7), ("L23_V002", 1)]


class TestRankOf:
    def test_finds_the_first_matching_shot(self) -> None:
        assert rank_of(HITS, {("L21_V001", 7)}) == 3

    def test_any_of_several_right_shots_counts(self) -> None:
        """An ASR span covers consecutive shots; each really does contain it."""
        assert rank_of(HITS, {("L21_V001", 7), ("L22_V005", 3)}) == 2

    def test_missing_shot_is_none(self) -> None:
        assert rank_of(HITS, {("L99_V999", 0)}) is None

    def test_video_must_match_too(self) -> None:
        """Shot ids are per-video, so an id alone identifies nothing."""
        assert rank_of(HITS, {("L23_V002", 0)}) is None

    def test_beyond_the_submission_limit_counts_as_missing(self) -> None:
        """The competition takes 100 rows; rank 101 scores like no answer."""
        hits = [("L00_V000", i) for i in range(120)]

        assert rank_of(hits, {("L00_V000", 99)}) == 100
        assert rank_of(hits, {("L00_V000", 100)}) is None

    def test_limit_is_adjustable(self) -> None:
        hits = [("L00_V000", i) for i in range(10)]

        assert rank_of(hits, {("L00_V000", 5)}, limit=5) is None


class TestQueryResult:
    def test_reciprocal_rank_of_a_miss_is_zero(self) -> None:
        assert QueryResult("q1", None).reciprocal_rank == 0.0

    def test_reciprocal_rank_of_first_place_is_one(self) -> None:
        assert QueryResult("q1", 1).reciprocal_rank == 1.0

    def test_hit_at_is_inclusive_of_k(self) -> None:
        result = QueryResult("q1", 5)

        assert result.hit_at(5)
        assert not result.hit_at(4)

    def test_a_miss_hits_at_no_k(self) -> None:
        assert not QueryResult("q1", None).hit_at(100)


class TestAggregates:
    RESULTS = [
        QueryResult("q1", 1),
        QueryResult("q2", 4),
        QueryResult("q3", None),
        QueryResult("q4", 30),
    ]

    def test_recall_counts_misses_in_the_denominator(self) -> None:
        assert recall_at(self.RESULTS, 5) == 0.5

    def test_recall_grows_with_k(self) -> None:
        assert recall_at(self.RESULTS, 1) == 0.25
        assert recall_at(self.RESULTS, 50) == 0.75

    def test_mrr_weights_the_top_of_the_list(self) -> None:
        assert mean_reciprocal_rank(self.RESULTS) == (1.0 + 0.25 + 0 + 1 / 30) / 4

    def test_mean_top_k_averages_over_the_official_ks(self) -> None:
        expected = sum(recall_at(self.RESULTS, k) for k in OFFICIAL_K) / len(OFFICIAL_K)

        assert mean_top_k_recall(self.RESULTS) == expected

    def test_empty_run_scores_zero_rather_than_dividing_by_zero(self) -> None:
        assert recall_at([], 5) == 0.0
        assert mean_reciprocal_rank([]) == 0.0
        assert mean_top_k_recall([]) == 0.0


class TestSummarise:
    def test_reports_every_official_k(self) -> None:
        summary = summarise([QueryResult("q1", 3)])

        for k in OFFICIAL_K:
            assert f"recall@{k}" in summary

    def test_carries_the_query_count_so_runs_are_comparable(self) -> None:
        """Two configurations scored on different set sizes are not comparable."""
        summary = summarise([QueryResult("q1", 1), QueryResult("q2", None)])

        assert summary["queries"] == 2.0
        assert summary["found"] == 0.5
