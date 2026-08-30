# AGENTS.md

This file provides guidance to AI coding agents working with code in this repository.

## Product boundary

- This is a competition video retrieval engine, not a chatbot.
- Optimize retrieval accuracy, top-rank quality, temporal localization accuracy,
  and query latency.
- Preserve `original_frame_id` throughout the pipeline.
- The final retrieval unit is always an **original-video frame**, even when
  intermediate retrieval happens at shot, clip, ASR-segment, or video level.
- Qdrant is the vector database and metadata-filtering source.
- No cloud dependency is allowed in the competition query path. Model weights
  must already be available locally, e.g. in the Hugging Face cache.
- Do not treat all competition queries as the same retrieval problem. KIS,
  VQA/Q&A, and TRAKE have different input semantics, output contracts, and
  correctness conditions.

---

## Competition search task contracts

Before changing query parsing, retrieval, fusion, reranking, temporal
localization, or submission code, determine which task contract the change
affects.

The three competition tasks share retrieval infrastructure but **must not be
conceptually collapsed into one generic `text -> frame` search**.

### 1. KIS — Textual Known Item Search

KIS describes one target scene, event, or temporal region in natural language.

Example:

```text
Đây là phần giới thiệu việc phóng tàu vũ trụ tư nhân.
Đoạn clip bắt đầu với hình ảnh 4 phi hành gia mặc áo đen.
Một trong những nhiệm vụ dự kiến của tàu vũ trụ là nghiên cứu
ánh sáng cực quang ở vùng cực.
```

Logical input:

```text
query_text: str
```

Required competition output:

```text
video_id, frame_id
```

Conceptual schema:

```python
class KISResult:
    video_id: str
    frame_id: int
```

Correctness condition:

```text
video_id == ground_truth_video
AND
frame_id lies inside the accepted temporal interval
```

Important implications:

- KIS is fundamentally **single-target temporal retrieval**.
- The system does not need to reproduce the exact organizer-selected frame.
  Any original frame inside the accepted interval can score.
- Coarse frame/shot retrieval is therefore sufficient to identify the region,
  followed by optional local temporal refinement.
- Retrieval may use frame, clip, OCR, ASR, metadata, objects, or other evidence,
  but the returned entity is still `(video_id, original_frame_id)`.
- Ranking should preserve alternative videos and temporal regions because the
  competition evaluates ranked candidates up to top 100.
- Do not return `shot_id`, keyframe ordinal, Qdrant point id, PTS, or ASR segment
  id as the competition frame id.

**Temporal KIS.** A KIS description that walks through phases ("phân cảnh tiếp
theo...", "sau đó...") is one target reached through a sequence, and averaging
those phases into a single 40-word caption is what loses it. Given `overview` +
`events` (from `/search/decompose`, reviewed by the operator), `/search/kis`
runs the TRAKE two-stage path so each moment votes on the video independently,
and still reports **one** frame: the highest-scoring event of the aligned chain.
Not the overview's frame - "cảnh trang trí bánh rán" exists to select the video,
not the frame - and this is safe precisely because of the correctness condition
above: the chain is compact and in order inside one video, so every event frame
is inside the accepted interval. The whole chain rides along in `events[]` for a
one-click override. The operator opts in; nothing is auto-detected.

---

### 2. VQA / Q&A — Retrieval first, answer second

The competition specification calls this Q&A. Within this repository and team
discussion, `VQA` may also be used to refer to the same task family.

A VQA query asks for information that can only be answered after finding the
correct video moment.

Example:

```text
Trong đoạn video có 2 câu thơ của một nhà thơ ca ngợi anh hùng
Nguyễn Trung Trực trong đình thần Nguyễn Trung Trực tại Kiên Giang.
Hai câu thơ đó là gì?
```

Official logical output:

```text
video_id, frame_id, answer
```

Conceptual schema:

```python
class VQAResult:
    video_id: str
    frame_id: int
    answer: str | None
```

However, the current competition workflow for this project is:

1. the retrieval system finds the correct video/frame;
2. the contestant inspects the retrieved evidence;
3. the contestant manually enters the textual answer.

Therefore the current automated retrieval target is effectively:

```text
video_id, frame_id
```

while `answer` remains optional/unimplemented in the backend.

Correctness conceptually requires:

```text
correct video
AND
frame inside the accepted interval
AND
correct answer
```

but the current retrieval engine is responsible primarily for the first two.

Important implications:

- Do **not** optimize VQA as generic answer generation before retrieval.
- Retrieval quality is the first hard gate: an answer grounded in the wrong
  video or temporal region is useless.
- Search should retrieve frames that make the answer easy for the contestant
  to inspect manually.
- OCR and ASR can be substantially more important for VQA than for ordinary KIS.
  For example, questions asking for written text, names, spoken sentences,
  numbers, signs, or quotations may depend primarily on OCR/ASR evidence.
- Visual similarity should therefore not automatically dominate every VQA
  query.
- The query-rewriting/parser layer may separate:
  - retrieval description,
  - asked information,
  - likely evidence modality such as visual/OCR/ASR.
- `answer=None` is intentional until an automated VQA answering component is
  explicitly implemented.
- Do not silently introduce an LLM/VLM answer-generation dependency into the
  competition query path.

---

### 3. TRAKE — Temporal Retrieval and Alignment of Key Events

TRAKE is not ordinary frame search.

A TRAKE query specifies:

1. a coarse description identifying the target video; and
2. an ordered sequence of semantic events `E1 ... EN`.

Example:

```text
Đoạn video múa lân một con lân màu vàng đen trắng, tìm các sự kiện sau:

E1:
Lân quay vòng trên cột số 4 bằng 2 chân trước rồi tiếp đất.
Khoảnh khắc đầu tiên mà lân bắt đầu xoay vòng.

E2:
Khoảnh khắc 4 chân hoàn toàn chạm đất đầu tiên.

E3:
Khoảnh khắc đầu tiên 2 người biểu diễn lân cuối chào ban giám khảo.

E4:
Sau đó lân tiến lại chào một con rồng.
Khoảnh khắc đầu tiên con rồng cử động đầu.
```

Required competition output:

```text
video_id,
frame_id_1,
frame_id_2,
frame_id_3,
frame_id_4
```

General schema:

```python
class TRAKEResult:
    video_id: str
    frame_ids: list[int]
```

where:

```text
frame_ids[j] corresponds exactly to event Ej
```

For the example:

```text
frame_id_1 -> E1
frame_id_2 -> E2
frame_id_3 -> E3
frame_id_4 -> E4
```

Each submitted frame must fall inside the accepted interval for its own event:

```text
frame_id_1 in GT_interval_E1
frame_id_2 in GT_interval_E2
...
frame_id_N in GT_interval_EN
```

The output is therefore one **complete sequence hypothesis**:

```text
(video_id, frame_for_E1, ..., frame_for_EN)
```

and not N unrelated retrieval results.

#### TRAKE has two distinct retrieval problems

##### Stage A — video-level retrieval

Use evidence from the complete sequence to identify the most likely video.

The overview plus `E1 ... EN` should contribute to video confidence.

A video that strongly matches only one event is not necessarily the correct
TRAKE video.

Conceptually:

```text
query
  -> candidate videos
  -> video-level sequence confidence
```

Useful evidence may include:

- frame similarities,
- clip similarities,
- ASR,
- OCR,
- metadata,
- detected entities,
- event coverage,
- temporal ordering.

##### Stage B — event-level temporal alignment

Once candidate videos are identified, locate each event independently within
the same video:

```text
E1 -> temporal neighborhood -> exact frame
E2 -> temporal neighborhood -> exact frame
...
EN -> temporal neighborhood -> exact frame
```

Then combine them into one valid sequence.

TRAKE alignment frequently asks for semantic boundaries such as:

```text
first moment ...
first frame where ...
starts to ...
completely touches ...
completely leaves ...
highest point ...
after that ...
```

These are boundary-localization problems, not just semantic similarity
problems.

A coarse keyframe may correctly identify the event but still be several frames
outside the accepted interval.

#### Temporal ordering

When the query describes events chronologically, returned frames should satisfy:

```text
frame_id_1 < frame_id_2 < ... < frame_id_N
```

unless the competition query explicitly implies otherwise.

Temporal ordering is a structural constraint, not merely another similarity
score.

#### TRAKE scoring implications

A wrong video makes all event localization irrelevant.

Therefore retrieval should generally prioritize:

```text
correct video identification
        ↓
event localization
        ↓
exact boundary refinement
```

Within the correct video, every event should still be localized separately
because each event contributes independently to sequence quality.

Do not implement TRAKE as:

```python
for event in events:
    global_frame_search(event)
return top_1_of_each_event
```

because this can easily select frames from different videos or produce an
incoherent event sequence.

Instead, use a hierarchical approach conceptually similar to:

```text
all events
   ↓
video candidate generation / fusion
   ↓
top candidate videos
   ↓
per-event search constrained to each video
   ↓
temporal sequence selection
   ↓
local boundary refinement
```

---

## Shared output invariants

All competition tasks ultimately return original-video coordinates.

### Frame IDs

`frame_id` always means:

```text
original_frame_id in the source video
```

It does **not** mean:

- Qdrant point id,
- keyframe filename ordinal,
- keyframe sequence number,
- shot id,
- clip id,
- ASR segment id,
- PTS in seconds.

Intermediate entities must retain enough provenance to recover
`original_frame_id`.

### Video IDs

`video_id` must correspond to the competition video identifier, not an internal
database id.

### Ranked results

The system may generate multiple ranked candidate answers.

A candidate is atomic at task level:

```text
KIS:
(video_id, frame_id)

VQA:
(video_id, frame_id[, answer])

TRAKE:
(video_id, frame_E1, ..., frame_EN)
```

For TRAKE in particular, ranks represent **complete sequence hypotheses**.

Do not independently rank each event and then interpret their rank positions as
one TRAKE candidate.

---

## Retrieval evidence versus output entity

Search evidence and submission output are deliberately different concepts.

The retrieval engine may search over:

```text
frames
clips
shots
videos
ASR segments
OCR text
metadata
detected entities
```

but those are evidence sources.

The competition output remains frame-based:

```text
KIS  -> one original frame
VQA  -> one original frame
TRAKE -> one original frame per event
```

ASR/OCR/metadata should therefore be treated as:

```text
retrieval evidence
reranking evidence
video-level evidence
temporal localization evidence
```

rather than alternative output entities.

---

## Commands

All Python commands run from `backend/` with the venv activated and `.env`
exported; `pytest.ini` sets `pythonpath = .`, so `app.*` imports resolve only
from that directory. `pytest` from the repo root works because `testpaths`/
`pythonpath` are relative to `backend/pytest.ini`.

```bash
# Run the API
docker compose up -d qdrant
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1

# Tests
python -m pytest tests/unit -q
python -m pytest tests/integration -q
python -m pytest tests/unit/test_ranking_fusion.py -q
python -m pytest tests/unit/test_ranking_fusion.py::test_name -q

# Regenerate the OpenAPI contract after any schema/endpoint change
python -c "import json; from pathlib import Path; from app.main import create_app; Path('../docs/openapi.json').write_text(json.dumps(create_app().openapi(), indent=2) + '\n', encoding='utf-8')"
```

Preprocessing CLIs each write a Parquet manifest. `--resume` is supported by
probe and shot detection.

All three video stages take `--workers` (default 3) and `--resume`, and flush
their manifest periodically, so an interrupted run is re-entrant.

```bash
python -m app.ingestion.video.probe \
    --source DIR \
    --out videos.parquet \
    --workers 3 \
    --resume

python -m app.ingestion.video.shot_detect \
    --videos-manifest videos.parquet \
    --out clips.parquet \
    --detector transnetv2 \
    --workers 3 \
    --resume

python -m app.ingestion.video.sampling \
    --videos-manifest videos.parquet \
    --shots-manifest clips.parquet \
    --output-dir keyframes/ \
    --frames-per-shot 3 \
    --out frames.parquet \
    --workers 3 \
    --resume

python -m app.ingestion.batch_builder keyframes|shots ...
python -m app.ingestion.runner --job-id ing-xxxx
python3 ../scripts/qdrant_snapshot.py create|restore ...
```

Full rebuild from raw video, approximately 9 hours under `tmux`:

```bash
./scripts/ingest_all.sh
```

See `docs/data-pipeline.md` §8.

Repo-root tools for the AIC 2025 batch-1 dataset must be run from the repo root,
not `backend/`.

Their tests import `scripts.*` and only resolve correctly from there.

```bash
python scripts/scrape_transcripts.py \
    --media-info DIR \
    --out DIR \
    [--report]

python scripts/verify_shots.py \
    --shots DIR \
    --map-keyframes DIR \
    --media-info DIR

python scripts/build_frames_manifest.py \
    --map-keyframes DIR \
    --shots DIR \
    --media-info DIR \
    --objects DIR \
    --transcripts DIR \
    --out-frames frames.parquet \
    --out-clips clips.parquet \
    --out-videos video_bounds.parquet

python scripts/join_ocr.py \
    --ocr data/ocr_raw/ocr

python scripts/build_eval_set.py \
    --limit 300

python scripts/build_asr_manifest.py \
    --transcripts data/transcripts \
    --out asr_segments.parquet
```

`build_frames_manifest.py` needs `--map-keyframes` and the organiser's keyframe
images.

**Neither is on this machine**, so it cannot be run here. Every path in the old
`frames.parquet` points at a missing file.

Rebuild from `data/videos` instead, via the three video CLIs above.

Evaluation under `app/eval/` scores a run of the shared evaluation set.

`app.eval.runner` is the ablation harness. Each flag disables one retrieval
component and summaries remain comparable because the query set is shared.

It needs a live Qdrant containing an ingested collection and **has never been
executed**; the metrics and set builder have.

The shipped evaluation set is ASR-derived, so running it with speech retrieval
enabled partially evaluates the lexical index against text derived from the
same source. Interpret those scores carefully; see `data/ARTIFACTS.md`.

```bash
python -m app.eval.runner \
    --eval-set ../data/eval_set.jsonl \
    --no-hybrid
```

`build_frames_manifest.py` exists because `batch_builder keyframes` cannot
directly read this dataset.

`batch_builder keyframes` takes `original_frame_id` from the keyframe filename,
but the organiser names files using the `map-keyframes` column `n`, which is an
ordinal rather than `frame_idx`.

Treat `map-keyframes` as the authority for both values.

No linter or formatter is currently configured. There is no `pyproject.toml`.

---

## Architecture

FastAPI:

```text
app/main.py
    ↓
app/api/router.py
    ↓
endpoint
    ↓
service protocol from app/services/
```

The lifespan creates a `Container` in:

```text
app/runtime/container.py
```

from `Settings` and stores it on `app.state`.

The container is the single location where:

- protocols are bound to implementations;
- runtime dependencies are constructed;
- active collection names are injected.

---

## Query architecture

`QdrantSearchService` wraps the synchronous retrieval path in
`run_in_threadpool`.

A transformer forward pass plus blocking Qdrant IO must not block FastAPI's
event loop.

`engine.retrieve()` is the shared low-level retrieval path.

Conceptually:

```text
query
  ↓
encode
  ↓
candidate retrieval
  ↓
cross-modal / cross-source fusion
  ↓
supporting evidence
  ↓
dedupe
  ↓
reranking
  ↓
task-specific shaping
```

Current shared stages are:

### 0a. Query decomposition (optional, operator-driven)

```text
POST /search/decompose  (decompose.py)
  markers "E1:/E2:" -> split_markers (regex)   ->| CLEAN_PROMPT        |concurrent
                                                 | EVENT_CAPTION_PROMPT|
  prose             -> _decompose_prompt(N) -> EVENT_CAPTION_PROMPT   (serial)
  -> {overview, events[]} each as {original, vision, speech}
  -> operator reviews and edits, then posts them to /search/trake or /search/kis
```

Only runs when the operator asks for it, and **retrieval does not run here**. A
wrong decomposition costs the whole task, so it is reviewed before it is
searched; the search endpoints then accept those `{vision, speech}` objects and
make no LLM call at all, which is what keeps the operator's edit from being
re-captioned away.

Events are counted by **marker line**, never by the number inside the marker:
`data/evaluation_set_p1.csv` contains a TRAKE task the organiser numbered
`E1, E2, E2, E4`, which has four events. Prose is capped at
`MAX_DECOMPOSED_EVENTS` (6); over the cap raises rather than truncating, because
the tail of these descriptions carries the distinguishing detail. Both paths end
at `EVENT_CAPTION_PROMPT`, which sees the overview on line 1 and carries the
scene's lasting visual detail into every event caption - an event is searched
alone against single frames, and "khoảnh khắc 4 chân hoàn toàn chạm đất" alone
describes nothing.

Failure is deliberately not uniform: no decomposition is a **503** (it is the
entire product of the call), while a failed caption is a 200 with `vision: null`
and a failed `CLEAN_PROMPT` leaves the marker text as typed. Full rationale in
`docs/trake-retrieval.md` §11.

### 0. Query rewriting

```text
rewrite.rewrite_queries
  -> CAPTION_PROMPT ->| concurrent |-> VLM_BASE_URL /chat/completions
  -> CLEAN_PROMPT   ->|            |
  -> Rewrite(vision=<English caption>, speech=<original, narration deleted>)
```

Runs in `tracks.py`, ahead of the engine. Every query of the request goes in one
batch per call — a TRAKE overview plus N events is two round trips, not 2(N+1).

The two jobs get **separate prompts on separate concurrent calls**. This is not
incidental: asking one prompt for both forms on one line was measurably worse at
both, because the caption rules and the deletion rules contaminated each other.
Across three prompt versions the merged call variously deleted the subject of the
query (`lễ hội đèn lồng`, `4 phi hành gia mặc áo đen`, the cyclists and their
uniforms) or performed no deletion at all. Split, each call is reliable, and
because they run concurrently the wall clock is the slower of the two rather
than their sum — measured 2.8s for a 700-character query where the cleaning call
alone was 2.76s.

The two forms, because the two collections want opposite things:

- `vision` — an English **caption** of what is on screen, capped at 40 words. This is what `encode_query` and the BLIP reranker get. Deliberately
  not a translation, for two reasons: both models are English-centric and would
  otherwise score "hãy tìm trong video" as part of the scene, and the SigLIP2
  text tower reads exactly **64 tokens** (`padding="max_length"`,
  `truncation=True` in `features/multimodal.embed_text`). A literal translation
  of a 700-character KIS description runs well past that and is cut without a
  word of warning, losing the tail where the distinguishing detail sits. 40 words
  (~50 tokens) is the budget matched to that window. Capping it lower is not
  free: at 20 words the model dropped the clothing, the setting and the *camera
  angle*, which is visible and highly discriminative. The prompt keeps a stated
  shot type and is told never to invent one - instructed to keep the angle, it
  will otherwise hallucinate "close-up of" onto queries that said nothing about
  the camera.
- `speech` — the query in its **original language** with the narration phrases
  deleted. This is what `search_speech` gets, and in `asr_only` mode it is the
  only query there is. Deletion, not rewriting: no translation, paraphrase,
  reordering or spelling correction, so whatever survives is word-for-word what
  the operator typed. Translating for it would drop the lexical half of the
  speech search to nothing, since the transcripts are Vietnamese; leaving it as
  typed makes `đoạn video mô tả`, `phân cảnh bắt đầu là`, `hãy tìm` live BM25
  terms scoring against those transcripts.

`retrieve(text, ..., speech_text=)` and `retrieve_per_video` are where the two
part company. A caller that passes only `text` gets the old behaviour, one
string for both.

This is the only network hop on the query path, so:

- the timeout (`QUERY_REWRITE_TIMEOUT_SEC`, default 6s) has to cover a whole
  TRAKE batch, not one query — the step is output-token-bound, and an overview
  plus five events in two forms measured 1.8s;
- **every** failure — unreachable box, timeout, misnumbered output, a
  `finish_reason` other than `stop` — falls back to the query as typed, and the
  two halves fail independently: a caption that does not arrive must not cost
  the cleaned form. `None` is returned only when both calls failed;
- the `finish_reason` check is load-bearing rather than belt-and-braces: a
  truncated reply still parses, since the separator sits on the first line long
  before the cut, so without it the last query is silently searched as half a
  sentence. `max_tokens` is sized from the input length for the same reason —
  the cleaned form is nearly as long as the query itself;
- a partial parse is treated as a failure, since a shifted line would move a
  TRAKE event onto the wrong query, and a line with only one form gives no way
  to tell which form it is;
- successes are `lru_cache`d and failures are not, so the endpoint coming back
  mid-session is picked up on the next query;
- `SearchResponse.rewritten_queries` reports the **English** forms and
  `cleaned_queries` reports the speech forms. Both are `None` when the step did
  not run.

`docs/research/mervin.md` argues the step away in favour of a Vietnamese-native
embedding model. That is an argument against SigLIP2, not against translating
for the SigLIP2 that ships.

### 1. Query encoding

```text
encode_query
  -> features.multimodal.embed_text
  -> configured feature profile
```

### 2. Vector search

Search the frame collection and, when configured, the clip collection.

Candidate counts are over-fetched according to:

```text
dedupe.DEFAULT_OVERFETCH
```

because multiple retrieved points may collapse to the same shot.

### 3. Frame/clip fusion

```python
fusion.fuse_frames_and_clips
```

combines both lists on:

```text
(video_id, shot_id)
```

and imputes each list's worst observed score for one-sided shots.

### 4. ASR supporting evidence

```python
ranking.asr.apply_asr_bonus
```

adds:

```text
asr_weight × best overlapping ASR score
```

for the speech segment whose time range overlaps the candidate frame.

This happens before shot deduplication intentionally, allowing speech evidence
to influence both:

- which frame represents a shot;
- where the shot ranks.

The bonus is additive rather than multiplicative.

Some videos contain no transcript, so missing speech must not be interpreted as
negative visual evidence.

ASR is supporting evidence, not the competition output entity.

### 5. Shot deduplication

```python
dedupe.dedupe_by_shot
```

keeps one representative hit per shot.

Several keyframes from the same shot are usually near duplicates and otherwise
waste top-K ranking capacity.

### 6. Reranking

```python
rerank.rerank
```

applies the BLIP ITM cross-encoder over the top:

```text
RERANK_TOP_N
```

The reranked head remains a separate ranking block because ITM probabilities
are not numerically comparable with the cosine scores below it.

---

## Task orchestration in `tracks.py`

`tracks.py` decides:

- how the competition query is interpreted;
- what gets encoded/retrieved;
- what task-level constraints are applied;
- how low-level retrieval hits become competition candidates.

It should **not** duplicate common frame/clip/ASR retrieval logic.

### KIS

KIS consumes one event description.

It returns one frame per candidate:

```text
(video_id, original_frame_id)
```

### VQA / Q&A

VQA currently uses the same retrieval substrate as KIS but the query semantics
may include both:

```text
event context
+
question / requested information
```

It returns:

```text
(video_id, original_frame_id, answer=None)
```

`answer=None` is intentional because automatic answer generation is not
currently wired into the backend.

The contestant can inspect the retrieved frame/video and enter the final answer
manually.

Future automated answer generation should be implemented as a separate grounded
stage after retrieval, not by changing the fundamental retrieval output.

### TRAKE

TRAKE must be treated hierarchically.

At minimum:

```text
parse events
    ↓
search each event / overall query for evidence
    ↓
aggregate evidence by video
    ↓
rank candidate videos
    ↓
within each candidate video, align each event
    ↓
enforce event-slot identity
    ↓
enforce temporal order when applicable
    ↓
return complete sequence candidates
```

The implementation now follows that hierarchy. `tracks._candidate_videos`
searches the overview *and* every event globally and keeps the top
`min(TRAKE_VIDEO_CANDIDATES, top_k)` videos - a result needs a candidate video to
live in, and stage B pays `videos x events` serial round trips for every extra
one - scored as `best overview hit + mean of best per-event hits`. That score and
the parts it is made of come back on every result as `SearchResult.stage_a`
(null when `video_ids` was supplied and the stage never ran). Coverage is deliberately not required at that stage: demanding
a global hit for every event is what dropped correct videos, since a
fine-grained event does not reach a global top-N against 290k frames. That stage reads
`engine.retrieve_video_scores`, not `retrieve`: collapsing per shot and cutting
to a page is what a result list wants, and it named 28 videos for a 100-video
pool, because dense hits pile up inside a few long videos (the top 5000 frames
of one query name 277 of them). It also skips reranking, for the reason stage B
does - ITM probabilities near 1.0 summed against cosine scores near 0.2 made the
candidate pool whatever BLIP happened to see, and cost seconds per request. Then
`engine.retrieve_per_video` aligns each event *inside* each candidate, one
filtered query per (video, event) - one query over all of them would return the
global top-N across them and starve a video that ranks low overall. That stage
collapses no shots and reranks nothing: two events can fall inside one
two-second shot, and the cross-encoder head covers the top-N of a single global
list. `tracks._best_increasing_sequence` then returns complete sequence
hypotheses, strictly frame-increasing and subject to `max_gap_sec` between
consecutive events, each carrying per-event frame/shot/timestamp/score plus
runners-up bounded by the neighbouring picks.

Any modification to TRAKE must preserve the distinction between:

```text
video selection
```

and:

```text
event alignment inside that video
```

Future video-level or temporal-aware fusion belongs naturally between these two
stages.

---

## Ingestion architecture

`POST /ingestions` creates a job row in SQLite and launches:

```bash
python -m app.ingestion.runner
```

as a detached subprocess.

Ingestion therefore survives independently of the API process.

Progress is read from SQLite.

The runner performs:

```text
validate
  ↓
create collection with indexing disabled
  ↓
create payload indexes
  ↓
streamed batched upsert
  ↓
optimize_collection
  ↓
collection green
```

`manifest.py` owns row schemas.

Subclasses of `ManifestRow` determine:

- point id;
- payload.

The ingestion pipeline should not branch on entity type.

---

## Frame identity

A keyframe's internal identity is:

```text
(video_id, keyframe_n)
```

not:

```text
(video_id, original_frame_id)
```

Frame indices are derived from rounded presentation timestamps.

Two adjacent keyframes can therefore map to the same original frame index.

This occurs in the current dataset.

Using `original_frame_id` as the Qdrant point key caused hundreds of keyframes
to silently overwrite each other during upsert.

Therefore:

```text
keyframe identity != competition frame identity
```

`original_frame_id` is the coordinate reported to the evaluator.

It is not the database primary key.

---

## Submission export

Submission code lives under:

```text
app/submissions/
```

`formats.py` renders the graded output:

```text
headerless UTF-8 CSV
LF line endings
no BOM
```

`LocalSubmissionService` rejects rows that could never score using per-video
frame bounds from:

```text
video_bounds.parquet
```

The bound is deliberately generous. See:

```python
scripts/build_frames_manifest.frame_upper_bound
```

If the bounds file is missing, this validation is disabled rather than causing
submission export to fail.

Submission-layer code must enforce the task contract:

```text
KIS:
video_id, frame_id

VQA:
video_id, frame_id, answer
when automated answer submission is used

TRAKE:
video_id, frame_id_1, ..., frame_id_N
```

---

## Collections

The current architecture uses two logical collections rather than pooling all
entities into one.

### `frames`

Contains frame points with:

```text
dense_video
```

and reserved slots for:

```text
dense_text
ocr
```

### `asr`

Contains speech segments with:

```text
dense_text
speech sparse vector
```

A speech segment describes a time range.

A keyframe describes an instant.

They therefore do not share a natural point key.

Frames intentionally declare **no `speech` slot**.

Pooling ASR onto frame points as well would score the same speech evidence
twice:

1. once through Qdrant hybrid/RRF retrieval;
2. once through the temporal ASR-overlap bonus.

Every vector slot is declared at collection creation because Qdrant cannot add
a new vector configuration to an existing collection.

A reserved but currently empty slot therefore permits a later re-upsert without
forcing re-embedding of every point.

While `ocr` is unpopulated:

```text
frame search must remain dense-only
sparse_names must remain empty
```

Querying an empty vector slot raises an error.

---

## Feature profiles

Feature profiles live in:

```text
app/features/profiles.py
```

A feature profile is the contract tying a collection to a query encoder.

`FEATURE_PROFILE` must match the profile used to ingest the active collection:

```text
same model
same dimension
same similarity space
same normalization assumptions
```

A mismatch can return plausible-looking but meaningless rankings without
raising an obvious error.

`kind` distinguishes image profiles from text profiles.

The vector slot written by an ingestion job is sized according to that job's
own profile.

This allows the same manifest to be ingested under multiple embedding models
for controlled comparison.

Model runtimes are `lru_cache`d and automatically select:

```text
cuda
mps
cpu
```

with appropriate FP16/FP32 behavior.

All vectors are L2-normalized.

Clip vectors are mean-pooled and then normalized again.

With normalized vectors, cosine similarity is equivalent to a dot product.

---

## Vector-store boundary

`app/vector_store/` is the only package that should import:

```python
qdrant_client
```

Search functions return plain application dataclasses such as:

```python
ScoredFrame
```

Ranking, track orchestration, and API layers must not depend directly on Qdrant
types.

This separation is intentional.

---

## Architecture rules

- API routes must delegate to application services.
- Qdrant-specific code belongs in `vector_store/`.
- Track modules orchestrate shared retrieval code; do not duplicate it.
- Keep evidence-source logic separate from competition output formatting.
- Preserve `original_frame_id` across every transformation.
- Do not substitute keyframe ordinal for original frame id.
- KIS, VQA, and TRAKE may share retrieval primitives but not necessarily the
  same ranking strategy.
- TRAKE changes must reason explicitly about both video-level retrieval and
  event-level temporal alignment.
- ASR/OCR are supporting retrieval evidence unless a specific feature requires
  otherwise.
- Ingestion must build versioned collections and never modify the active
  collection during competition mode.
- This is a single-team competition tool, not a multi-user production system.
  Do not add production-style validation/snapshot/activation gating without a
  concrete competition need.
- Video decode stays single-threaded:

```text
thread_type = "NONE"
```

Parallelize across processes.

PyAV's log callback can take the GIL from a decoder thread while the main
thread holds it inside `avcodec_free_context()`, producing a real deadlock.

See:

```text
app/features/media.py
```

Processes are also faster in the measured workload:

```text
3 processes × 428 fps > 705 fps threaded decode
```

- Parquet manifests remain the rebuild and audit source of truth.
- `docs/data-pipeline.md` documents current data state, column provenance, and
  missing artifacts.
- Manifest paths must remain constrained to `INGESTION_DATA_ROOT`.

---

## Review rules

- Keep changes scoped to one implementation-plan item.
- Add tests for mapping, scoring, filtering, temporal constraints, or
  API-contract changes.
- For retrieval changes, identify which task or tasks are affected:
  `KIS`, `VQA`, `TRAKE`, or shared infrastructure.
- For TRAKE ranking changes, test complete sequence behavior rather than only
  individual event rankings.
- For any frame-localization change, test that exported IDs are
  `original_frame_id`.
- Run relevant tests before handoff.
- Explain new production dependencies before adding them.
- Do not change API contracts silently.
- Regenerate:

```text
docs/openapi.json
```

when API schemas or endpoints change.

The frontend repository:

```text
../aic2026-fe
```

generates types from this OpenAPI contract and is a separate repository.

Never mix backend and frontend changes in one commit.

`CLAUDE.md` and `AGENTS.md` describe the same repository behavior. Keep them in
sync when either file contains rules intended for all coding agents.

---

## Known gaps

- Ingestion has no resume at the Qdrant upsert level. A failed upsert currently
  restarts embedding from row 0.
- The collection is created unconditionally, so retrying against an existing
  collection raises.
- The three raw-video preprocessing stages do support resume per video.
- `batch_builder.scan_clips` raises `NotImplementedError` pending a clip file
  naming convention.
- TRAKE event localisation is only as fine as the keyframes: three per shot, so
  a sub-second moment is bracketed rather than pinned. `events[].alternates`
  exists so an operator closes that gap by hand. The VLM endpoint in `.env` is
  now read, but only to rewrite queries; asking it *where* an event is inside a
  bracketed shot is the obvious next lever.
- Nothing in TRAKE is fitted. `data/evaluation_set_p1.csv` (24 queries: 18 KIS,
  3 TRAKE with frame-level ground truth, 3 QA) has never been run against this
  path, so stage A's `overview + mean(events)`, `TRAKE_MAX_GAP_SEC` and
  `TRAKE_GAP_WEIGHT` are reasoned from the score scale rather than measured -
  and decomposition now feeds them inferred events and synthesised overviews,
  which is a different input distribution. `data/eval_set.jsonl` is a separate,
  ASR-derived set and is absent.
- `docs/architecture.md` is empty.
- Automated VQA answer generation is not implemented.
- Exact TRAKE semantic-boundary refinement beyond coarse retrieved frames
  remains an area for further development.
