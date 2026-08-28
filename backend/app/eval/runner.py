"""Run the evaluation set through retrieval and print the metrics.

⚠ NEVER EXECUTED. Every other piece in `app/eval/` and `scripts/build_eval_set.py`
has been run against the real manifests; this module has not, because it needs a
live Qdrant holding an ingested collection and this machine has neither. Treat
it as a draft until it has produced a number.

This is the ablation harness `task.md` section 3 asks for: each flag turns one
retrieval component off, and the summary lines are directly comparable because
the same query set feeds all of them.

⚠ Read `scripts/build_eval_set.py` before believing a number from this. The
queries in the shipped set are derived from ASR, so a run with the speech
vectors ON is scoring the lexical index against its own text. Pass
`--no-hybrid` for an honest read on the visual index, or compare two runs that
share the same channel setting.

    python -m app.eval.runner --eval-set ../data/eval_set.jsonl --no-hybrid
    python -m app.eval.runner --eval-set ../data/eval_set.jsonl --no-rerank
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from app.core.config import settings
from app.eval.metrics import MAX_SUBMITTED, QueryResult, rank_of, summarise
from app.retrieval.engine import RetrievalConfig, Timings, retrieve


def load_queries(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(queries: list[dict], config: RetrievalConfig) -> list[QueryResult]:
    results = []
    for query in queries:
        answers = {
            (query["video_id"], int(shot_id))
            for shot_id in query["answer_shot_ids"]
        }
        hits = retrieve(query["query_text"], MAX_SUBMITTED, config, Timings())
        results.append(
            QueryResult(
                query_id=query["query_id"],
                rank=rank_of([(h.video_id, h.shot_id) for h in hits], answers),
            )
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", default="../data/eval_set.jsonl")
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-clips", action="store_true")
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args(argv)

    if not Path(args.eval_set).is_file():
        print(f"error: no eval set at {args.eval_set}", file=sys.stderr)
        return 1

    config = RetrievalConfig(
        frames_collection=settings.QDRANT_FRAMES_COLLECTION,
        clips_collection=None if args.no_clips else settings.QDRANT_CLIPS_COLLECTION,
        feature_profile=settings.FEATURE_PROFILE,
        clip_weight=settings.CLIP_FUSION_WEIGHT,
        rerank_enabled=not args.no_rerank,
        rerank_top_n=settings.RERANK_TOP_N,
        rerank_model=settings.RERANK_MODEL,
    )
    config = replace(config, hybrid_enabled=not args.no_hybrid)

    queries = load_queries(args.eval_set)
    summary = summarise(evaluate(queries, config))

    print(f"label   : {args.label}")
    print(f"hybrid  : {config.hybrid_enabled}")
    print(f"rerank  : {config.rerank_enabled}")
    print(f"clips   : {bool(config.clips_collection)}")
    for name, value in summary.items():
        print(f"{name:<12}: {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
