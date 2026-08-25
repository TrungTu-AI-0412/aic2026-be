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

All three video stages take `--workers` (default 3) and `--resume`, and flush
their manifest periodically, so an interrupted run is re-entrant.

```bash
python -m app.ingestion.video.probe --source DIR --out videos.parquet --workers 3 --resume
python -m app.ingestion.video.shot_detect --videos-manifest videos.parquet --out clips.parquet --detector transnetv2 --workers 3 --resume
python -m app.ingestion.video.sampling --videos-manifest videos.parquet --shots-manifest clips.parquet --output-dir keyframes/ --frames-per-shot 3 --out frames.parquet --workers 3 --resume
python -m app.ingestion.batch_builder keyframes|shots ...   # import externally produced artifacts
python -m app.ingestion.runner --job-id ing-xxxx            # normally spawned by the API
python3 ../scripts/qdrant_snapshot.py create|restore ...    # collection hand-off between machines
```

Full rebuild from raw video, ~9 hours under tmux (see `docs/data-pipeline.md` §8):

```bash
./scripts/ingest_all.sh
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
python scripts/build_eval_set.py --limit 300      # candidate eval queries
python scripts/build_asr_manifest.py --transcripts data/transcripts --out asr_segments.parquet
```

`build_frames_manifest.py` needs `--map-keyframes` and the organiser's keyframe
images. **Neither is on this machine**, so it cannot be run here: every path in
the old `frames.parquet` points at a missing file. Rebuild from `data/videos`
instead, via the three video CLIs above.

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
event loop.

Ahead of the engine, `rewrite.rewrite_queries` prepares every query of the
request twice, because the two collections want opposite things from the same
string. Two **separate prompts on two concurrent calls** to `VLM_BASE_URL`, not
one call asking for both: merged, the caption rules bled into the deletion rules
and the model variously deleted the subject of the query (`lễ hội đèn lồng`,
`4 phi hành gia mặc áo đen`) or deleted nothing at all. Split, each call does
its own job, and the wall clock is the slower of the two rather than their sum.

- `Rewrite.vision` — an English **caption**, at most 40 words, for the SigLIP2
  text tower and the BLIP reranker. Not a translation: they would score "hãy tìm
  trong video" as part of the scene, and the text tower reads exactly **64
  tokens** (~45 English words), so a literal translation of a 700-character KIS
  description is silently cut in half — losing the tail, which is where the
  distinguishing detail usually sits. 40 words is the budget matched to that
  window; capping it lower threw away detail that discriminates, including the
  camera angle, which *is* visible. The prompt keeps a stated shot type
  (overhead, head-on, close-up) and is told never to invent one.
- `Rewrite.speech` — the query in its original language with the narration
  phrases **deleted**, for the transcripts. Not translated, because they are
  Vietnamese and are matched by term overlap as well as densely, so English
  would drop the lexical half to nothing; not left as typed either, because
  `đoạn video mô tả`, `phân cảnh bắt đầu là`, `hãy tìm` are live BM25 terms
  scoring against those transcripts. Deletion, never rewording: whatever
  survives is word-for-word what the operator typed.

`retrieve(text, ..., speech_text=)` is where the two part company. Both forms
come out of one call, so this costs one round trip, not two. It is the only
network hop the query path takes, so it is on a timeout that has to cover a
whole TRAKE batch (measured 1.8s for an overview plus five events) and **every**
failure — box down, timeout, a misnumbered line, a `finish_reason` other than
`stop` — falls back to the query as typed. The two halves fail **independently**:
a caption that does not arrive does not cost the cleaned form, which is the point
of separate calls. `None` means both failed. The `finish_reason` check is
load-bearing: a truncated reply still parses, because the early lines are
intact, so without it the last query is searched as half a sentence.
`SearchResponse.rewritten_queries` reports the English forms and
`cleaned_queries` reports the speech forms; both are `None` when the step did
not run. `docs/research/mervin.md` argues the whole step
away in favour of a Vietnamese-native embedding model — the argument is against
SigLIP2, not against translating for it.

`engine.retrieve()` is the one shared path for every track:

1. `encode_query` → `features.multimodal.embed_text` with the configured profile
2. `search_vector` → frame collection, plus the clip collection when configured,
   over-fetched by `dedupe.DEFAULT_OVERFETCH`
3. `fusion.fuse_frames_and_clips` → combines both lists on `(video_id, shot_id)`,
   imputing each list's worst observed score for one-sided shots
4. `ranking.asr.apply_asr_bonus` → adds `asr_weight ×` the best-scoring speech
   segment whose time range covers each frame. Before dedupe on purpose, so
   speech decides *which* frame represents a shot as well as where it ranks.
   Additive, never multiplicative: 22 of 873 videos have no transcript, so
   silence is not evidence against a frame
5. `dedupe.dedupe_by_shot` → one hit per shot, since several keyframes per shot
   produce many near-identical vectors
6. `rerank.rerank` → BLIP ITM cross-encoder over the top `RERANK_TOP_N`; the
   reranked head stays a block because ITM probabilities are not comparable
   with the cosine scores below it

`tracks.py` decides only *what* to encode and how to shape results. KIS/QA return
one frame per hit; QA leaves `answer=None` (no VQA model is wired in).

**TRAKE runs in two stages**, because "which video" and "where in it" are
different questions. `_candidate_videos` searches the overview *and* every event
globally and keeps the top `TRAKE_VIDEO_CANDIDATES` videos, scoring each as
`best overview hit + mean of best per-event hits`. Coverage is deliberately not
required here: demanding a global hit for every event is what dropped correct
videos, since a fine-grained event ("the moment all four feet touch the ground")
does not reach a global top-N against 290k frames. That stage reads
`engine.retrieve_video_scores`, not `retrieve`: collapsing per shot and cutting
to a page is what a result list wants, and it named 28 videos for a 100-video
pool, because dense hits pile up inside a few long videos (the top 5000 frames
of one query name 277 of them). It also skips reranking, for the reason stage B
does - ITM probabilities near 1.0 summed against cosine scores near 0.2 made the
candidate pool whatever BLIP happened to see, and cost seconds per request. Then
`engine.retrieve_per_video` searches each event again *inside* each candidate —
one filtered query per (video, event), because one query over all of them
returns the global top-N across them and starves a video that ranks low
overall. That stage does not collapse shots and turns reranking off: two events
can happen inside one two-second shot, and the cross-encoder head covers the
top-N of a single global list, so per-video it would rescore an arbitrary slice.
Finally `_best_increasing_sequence` picks, per video, the highest-scoring
strictly frame-increasing selection covering every event, subject to
`max_gap_sec` between consecutive events. Ordering is on `original_frame_id`
(what a submission reports, and monotone with time within a video); `pts_sec` is
used only for the gap, which is a duration and so cannot be expressed in frames
across videos with different frame rates. The result carries `events[]` —
per-event frame, shot, timestamp, score and the runners-up it beat, bounded by
the neighbouring picks so a swap cannot produce an out-of-order submission.

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

**Collections** are two, not one. A speech segment is a time *range* and a
keyframe is an *instant*, so they share no key: `frames` holds `dense_video`
(image) plus reserved `dense_text`/`ocr` slots, and `asr` holds `dense_text`
(Qwen3-Embedding-0.6B) plus a populated `speech` sparse vector. Frames declare
**no** `speech` slot — pooling ASR onto frames as well would score the same
speech twice, once through Qdrant RRF and once through the overlap bonus. Every
slot is declared at creation because Qdrant cannot add a vector to an existing
collection, so a slot declared now is a re-upsert later instead of re-embedding
293k points. While `ocr` is unpopulated, frame search is dense-only and
`sparse_names` must stay empty: querying an empty slot raises.

**Feature profiles** (`app/features/profiles.py`) are the contract that ties a
collection to a query. `FEATURE_PROFILE` in `.env` must match the profile the
active collection was ingested with — same model, same dimension, same space; a
mismatch returns plausible nonsense rather than an error. `kind` separates image
profiles from text ones, and the slot a job writes is sized from that job's own
profile, so the same manifest can be ingested under two models and compared.
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
- Video decode stays single-threaded (`thread_type = "NONE"`); parallelise with
  processes. PyAV's log callback takes the GIL from a decoder thread while the
  main thread holds it inside `avcodec_free_context()` — a real deadlock, see
  `app/features/media.py`. Processes are also faster here: 3 × 428 fps beats the
  705 fps threaded decode measured.
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

- Ingestion has no resume: a failed upsert re-embeds from row 0, and the
  collection is created unconditionally so a retry over an existing one raises.
  The three video stages *do* resume, per video.

- `batch_builder.scan_clips` raises `NotImplementedError` pending a clip file
  naming convention.
- TRAKE event localisation is only as fine as the keyframes: three per shot, so
  a sub-second moment is bracketed, not pinned. `events[].alternates` exists so
  an operator closes that last gap by hand. The VLM endpoint in `.env` is now
  read, but only to rewrite queries; asking it *where* an event is inside a
  bracketed shot is the obvious next lever.
- No TRAKE eval set exists (`data/eval_set.jsonl` is absent), so retrieval
  changes to it are checked by unit tests and by eye, not measured.
- `docs/architecture.md` is empty.
