"""Scoring for a run of the evaluation set.

A hit is judged at **shot** granularity, not frame. `dedupe.dedupe_by_shot`
collapses each shot to one hit before results ever leave the engine, so asking
whether a specific frame came back would measure the sampler, not the search.

The organiser scores "Mean of Top-k R-Score" over k in {1, 5, 20, 50, 100},
with up to 100 ranked results per query — both AIC 2025 write-ups report the
same protocol. `OFFICIAL_K` mirrors that k set so local numbers move for the
same reasons the competition's do. The exact per-k R-Score formula is not
published, so `mean_top_k_recall` uses binary hit-at-k; it tracks the official
metric in shape but must not be quoted as if it were the official score.
"""

from dataclasses import dataclass

OFFICIAL_K: tuple[int, ...] = (1, 5, 20, 50, 100)
MAX_SUBMITTED = 100


@dataclass(frozen=True)
class QueryResult:
    """One evaluated query: where the correct shot landed, if at all."""

    query_id: str
    rank: int | None  # 1-based; None when the shot never came back

    @property
    def reciprocal_rank(self) -> float:
        return 0.0 if self.rank is None else 1.0 / self.rank

    def hit_at(self, k: int) -> bool:
        return self.rank is not None and self.rank <= k


def rank_of(
    hits: list[tuple[str, int]],
    answers: set[tuple[str, int]],
    limit: int = MAX_SUBMITTED,
) -> int | None:
    """1-based rank of the first hit in `answers`, or None past `limit`.

    `answers` is a set, not one shot, because a question can have several right
    answers: an ASR span runs across consecutive shots, so each of them really
    does contain the moment being described. Insisting on one would mark a
    correct retrieval wrong for picking the neighbouring shot.

    Truncating at `limit` is not a detail: the competition accepts 100 rows, so
    a correct answer at rank 140 scores exactly the same as one that never
    appeared. Counting it would flatter every configuration equally and hide
    which one actually fits inside the submission.
    """
    for position, hit in enumerate(hits[:limit], start=1):
        if hit in answers:
            return position
    return None


def recall_at(results: list[QueryResult], k: int) -> float:
    if not results:
        return 0.0
    return sum(r.hit_at(k) for r in results) / len(results)


def mean_reciprocal_rank(results: list[QueryResult]) -> float:
    if not results:
        return 0.0
    return sum(r.reciprocal_rank for r in results) / len(results)


def mean_top_k_recall(
    results: list[QueryResult], ks: tuple[int, ...] = OFFICIAL_K
) -> float:
    """Mean of recall@k across the organiser's k set. See module docstring."""
    if not ks:
        return 0.0
    return sum(recall_at(results, k) for k in ks) / len(ks)


def summarise(
    results: list[QueryResult], ks: tuple[int, ...] = OFFICIAL_K
) -> dict[str, float]:
    """Every headline number for one configuration, ready to diff against another."""
    summary = {f"recall@{k}": recall_at(results, k) for k in ks}
    summary["mrr"] = mean_reciprocal_rank(results)
    summary["mean_top_k"] = mean_top_k_recall(results, ks)
    summary["found"] = sum(r.rank is not None for r in results) / (
        len(results) or 1
    )
    summary["queries"] = float(len(results))
    return summary
