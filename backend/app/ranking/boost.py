"""Lift shots whose on-screen text matches, on top of the visual ranking.

`fusion.py` combines the frame and clip indexes arithmetically because both
sides are cosine similarities in one shared space. That argument does not
extend to on-screen text: a lexical hit is scored by summed IDF, which has no
upper bound and shifts with the query's rarest token, while a cosine
similarity on this corpus lives in a narrow band around 0.2-0.35. Adding them
would let one query be decided entirely by the dense side and the next
entirely by the lexical side, for no reason an operator could see.

So this fuses on *rank*. Reciprocal rank fusion ignores magnitude by
construction, which is the same choice `search.py` already makes when it hands
a hybrid query to Qdrant, and the same one `fusion.py` records as its fallback
if the frame and clip scales ever drift apart.

WHAT THE WEIGHT BUYS, AND WHAT IT COST TO FIND OUT

This module shipped at weight 0.5 on the reasoning that half strength would
keep on-screen text a tie-breaker. Measured against 300 queries on the real
index, that reasoning was wrong by an order of magnitude:

    weight   recall@1   recall@5     MRR
    1.0       0.1067     0.3200    0.2307
    0.5       0.1400     0.5867    0.3322   <- the shipped default
    0.25      0.2167     0.7300    0.4261
    off       0.2300     0.8000    0.4528   <- ocr left inside Qdrant's RRF
    0.1       0.2567     0.8100    0.4813
    0.05      0.2800     0.8067    0.5093   <- current default

Monotonic, with no inflection: every increase in the weight cost accuracy.
At 0.5 the channel was worse than not running it at all. At 0.05 it is worth
+22% relative recall@1 over leaving `ocr` as one equal branch of the
server-side fusion.

The reason the curve is that steep is the fusion itself. Folding the primary
list into reciprocal ranks discards how far ahead its leader was: a hit that
won by a mile becomes 1/61 against 1/62 for the runner-up. A single OCR match
at rank 1 then contributes weight/61, which at weight 0.5 is worth about
thirty places. The channel stops supplementing the visual ranking and starts
overwriting it.

CAVEAT ON THAT NUMBER

The query set those figures come from is derived from speech transcripts, so
on-screen text is a *secondary* signal throughout it and a low weight is what
that measures. A query set written from what is printed on the frame would
almost certainly prefer a higher one. 0.05 is the honest default for the only
evidence that exists; it is not a claim about queries nobody has tested yet.
`/search/ocr` is unaffected either way - it never consults this weight.
"""

from dataclasses import replace

from app.ranking.dedupe import best_per_shot
from app.vector_store.search import ScoredFrame

# The constant from the original RRF paper. Large relative to the ranks that
# matter, so the difference between rank 1 and rank 5 stays modest and no
# single channel can dominate on the strength of one placement.
RRF_K = 60

# Measured, not reasoned. See the table in the module docstring: this was 0.5
# on argument alone and every step down from there bought accuracy.
DEFAULT_OCR_WEIGHT = 0.05


def _ranked_shots(
    hits: list[ScoredFrame],
) -> dict[tuple[str, int], tuple[int, ScoredFrame]]:
    """Collapse to one hit per shot and number them from 1, best first."""
    ordered = sorted(
        best_per_shot(hits).items(), key=lambda item: item[1].score, reverse=True
    )
    return {key: (rank, hit) for rank, (key, hit) in enumerate(ordered, start=1)}


def reciprocal_rank_fuse(
    primary: list[ScoredFrame],
    secondary: list[ScoredFrame],
    weight: float = DEFAULT_OCR_WEIGHT,
    k: int = RRF_K,
) -> list[ScoredFrame]:
    """Fuse two ranked lists on `(video_id, shot_id)`, primary unweighted.

    Returns hits carrying RRF scores, not similarities. Nothing downstream
    reads the magnitude - `dedupe_by_shot` only orders by it and `rerank`
    replaces it outright - so the change of scale stays inside ranking.
    """
    if not secondary or weight <= 0:
        return list(primary)

    ranked_primary = _ranked_shots(primary)
    ranked_secondary = _ranked_shots(secondary)

    fused: list[ScoredFrame] = []
    for key in ranked_primary.keys() | ranked_secondary.keys():
        score = 0.0
        carrier = None

        if key in ranked_primary:
            rank, carrier = ranked_primary[key]
            score += 1.0 / (k + rank)

        if key in ranked_secondary:
            rank, lexical = ranked_secondary[key]
            score += weight / (k + rank)
            if carrier is None:
                carrier = lexical
            elif carrier.ocr_text is None:
                # The visual hit may be a clip point, which knows only the
                # shot's frame range. Carrying the text across is what lets a
                # UI show why the shot moved up.
                carrier = replace(carrier, ocr_text=lexical.ocr_text)

        fused.append(replace(carrier, score=score))

    # Ties are common: two shots each seen by one channel at the same rank
    # score identically. Sorting on the key as well keeps the output stable
    # between runs rather than following set iteration order.
    fused.sort(key=lambda hit: (-hit.score, hit.video_id, hit.shot_id))
    return fused
