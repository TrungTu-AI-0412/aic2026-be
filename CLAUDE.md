# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Product boundary

- This is a competition video retrieval engine, not a chatbot.
- Optimize retrieval accuracy, top-rank quality and query latency.
- Preserve `original_frame_id` throughout the pipeline.
- Qdrant is the vector database and metadata-filtering source.
- No cloud dependency is allowed in the competition query path (model weights
  must already be in the local Hugging Face cache).

## Commands

All Python commands run from `backend/` with the venv activated and `.env`
exported; `pytest.ini` sets `pythonpath = .`, so `app.*` imports resolve only
from that directory. `pytest` from the repo root works because `testpaths`/
`pythonpath` are relative to `backend/pytest.ini`.

```bash
# Run the API
docker compose up -d qdrant                  # from repo root
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1

# Tests
python -m pytest tests/unit -q               # no Qdrant needed
python -m pytest tests/integration -q        # falls back to :memory: if no live Qdrant
python -m pytest tests/unit/test_ranking_fusion.py -q
python -m pytest tests/unit/test_ranking_fusion.py::test_name -q

# Regenerate the OpenAPI contract after any schema/endpoint change
python -c "import json; from pathlib import Path; from app.main import create_app; Path('../docs/openapi.json').write_text(json.dumps(create_app().openapi(), indent=2) + '\n', encoding='utf-8')"
```

Preprocessing CLIs (each writes a Parquet manifest; `--resume` is supported by
probe and shot detection):

```bash
python -m app.ingestion.video.probe --source DIR --out videos.parquet --resume
python -m app.ingestion.video.shot_detect --videos-manifest videos.parquet --out clips.parquet --detector transnetv2 --resume
python -m app.ingestion.video.sampling --videos-manifest videos.parquet --shots-manifest clips.parquet --output-dir keyframes/ --out frames.parquet
python -m app.ingestion.batch_builder keyframes|shots ...   # import externally produced artifacts
python -m app.ingestion.runner --job-id ing-xxxx            # normally spawned by the API
python3 ../scripts/qdrant_snapshot.py create|restore ...    # collection hand-off between machines
```

Repo-root tools for the AIC 2025 batch-1 dataset (run from the repo root, not
`backend/`; their tests import `scripts.*` and only resolve from there):

```bash
python scripts/scrape_transcripts.py --media-info DIR --out DIR [--report]
python scripts/verify_shots.py --shots DIR --map-keyframes DIR --media-info DIR
python scripts/build_frames_manifest.py --map-keyframes DIR --shots DIR \
    --media-info DIR --objects DIR --transcripts DIR \
    --out-frames frames.parquet --out-clips clips.parquet \
    --out-videos video_bounds.parquet
python scripts/join_ocr.py --ocr data/ocr_raw/ocr   # fold OCR into both manifests
python scripts/join_captions.py                   # fold Vietnamese VLM captions in
python scripts/build_eval_set.py --limit 300      # candidate eval queries
```

Evaluation (`app/eval/`) scores a run of that set. `app.eval.runner` is the
ablation harness — each flag disables one retrieval component and the summaries
are comparable because the query set is shared. It needs a live Qdrant with an
ingested collection and **has never been executed**; the metrics and the set
builder have. The shipped set is ASR-derived, so a run with the speech vectors
on is scoring the lexical index against its own text — see `data/ARTIFACTS.md`.

```bash
python -m app.eval.runner --eval-set ../data/eval_set.jsonl --no-hybrid
```

`build_frames_manifest.py` exists because `batch_builder keyframes` cannot read
this dataset: it takes `original_frame_id` from the keyframe filename, but the
organiser names files after `map-keyframes` column `n` (the ordinal), not
`frame_idx`. Treat `map-keyframes` as the authority for both columns.

No linter/formatter is configured; there is no `pyproject.toml`.

## Architecture

FastAPI (`app/main.py`) → router (`app/api/router.py`) → endpoints that only
`Depends` on service protocols from `app/services/`. The lifespan builds a
`Container` (`app/runtime/container.py`) from `Settings` and stores it on
`app.state`; that container is the single place where protocols are bound to
implementations and where the active collection names are injected.

**Query path** (`app/retrieval/`):

`QdrantSearchService` wraps the whole synchronous path in `run_in_threadpool` —
a transformer forward pass plus blocking Qdrant IO would otherwise stall the
event loop. `engine.retrieve()` is the one shared path for every track:

1. `encode_query` → `features.multimodal.embed_text` with the configured profile
2. `search_vector` → frame collection, plus the clip collection when configured,
   over-fetched by `dedupe.DEFAULT_OVERFETCH`
3. `fusion.fuse_frames_and_clips` → combines both lists on `(video_id, shot_id)`,
   imputing each list's worst observed score for one-sided shots
4. `dedupe.dedupe_by_shot` → one hit per shot, since ~1 keyframe/sec means a
   single shot produces many near-identical vectors
5. `rerank.rerank` → BLIP ITM cross-encoder over the top `RERANK_TOP_N`; the
   reranked head stays a block because ITM probabilities are not comparable
   with the cosine scores below it

`tracks.py` decides only *what* to encode and how to shape results. KIS/QA return
one frame per hit; QA leaves `answer=None` (no VQA model is wired in); TRAKE
searches each event separately and picks, per video, the highest-scoring
strictly frame-increasing selection covering every event.

**Ingestion path** (`app/ingestion/`):

`POST /ingestions` writes a job row to SQLite and detaches a
`python -m app.ingestion.runner` subprocess, so ingestion survives independently
of the API process; progress is polled back out of SQLite. The runner walks
validate → create collection (indexing disabled) → payload indexes → streamed
batched upsert → `optimize_collection` (re-enables indexing and blocks until the
collection is green). `manifest.py` owns the row schema — subclasses of
`ManifestRow` decide their own point id and payload so the pipeline never
branches on entity type.

A keyframe's identity is `(video_id, keyframe_n)`, never
`(video_id, original_frame_id)`. Frame indexes are derived from rounded
presentation timestamps, so two consecutive keyframes can share one — it
happens in 192 of the 873 videos in this dataset. Point ids built from the
frame index made those 614 keyframes overwrite each other during upsert with
nothing raised. `original_frame_id` is what a submission reports; it is not a
key.

**Submission export** (`app/submissions/`): `formats.py` renders the graded
file (headerless UTF-8 CSV, LF, no BOM); `LocalSubmissionService` first rejects
rows that could never score, against the per-video frame bounds in
`video_bounds.parquet`. That bound is deliberately generous — see
`scripts/build_frames_manifest.frame_upper_bound`. A missing bounds file
disables the check rather than failing the export.

**Feature profiles** (`app/features/profiles.py`) are the contract that ties a
collection to a query. `FEATURE_PROFILE` in `.env` must match the profile the
active collection was ingested with — same model, same dimension, same space.
Model runtimes are `lru_cache`d and auto-select cuda/mps/cpu with fp16/fp32.
All vectors are L2-normalized (mean-pooled then re-normalized for clips), so
cosine similarity is a dot product.

**`app/vector_store/`** is the only place that imports `qdrant_client`; search
returns plain `ScoredFrame` dataclasses so ranking and the API never see Qdrant
types.

## Architecture rules

- API routes must delegate to application services.
- Qdrant-specific code belongs in `vector_store/`.
- Track modules orchestrate shared retrieval code; do not duplicate it.
- Ingestion must build versioned collections and never modify the active
  collection during competition mode.
- This is a single-team competition tool, not a multi-user production
  system: do not add validation/snapshot/activation gating to ingestion.
- Parquet manifests remain the rebuild/audit source of truth;
  `docs/data-pipeline.md` is the current state of that data — what is in
  each column, where it came from, and what is still missing.
- Manifest paths are constrained to `INGESTION_DATA_ROOT`; keep that check.

## Review rules

- Keep changes scoped to one implementation-plan item.
- Add tests for mapping, scoring, filtering or API-contract changes.
- Run relevant formatting, linting and tests before handoff.
- Explain new production dependencies before adding them.
- Do not change API contracts silently; regenerate `docs/openapi.json` when they
  change. The frontend (`../aic2026-fe`) codegens from it and is a separate
  repository — never mix backend and frontend changes in one commit.
- `AGENTS.md` is a copy of this file; keep the two in sync.

## Known gaps

- `batch_builder.scan_clips` raises `NotImplementedError` pending a clip file
  naming convention.
- `docs/architecture.md` is empty.
