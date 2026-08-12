# Ingestion

Để triển khai toàn bộ backend từ một Ubuntu server mới, xem
[Ubuntu deployment runbook](runbook-ubuntu.md).

Internal pipeline for building a new Qdrant collection from a dataset
version. This is **not** a public upload API — data is copied/mounted onto
local storage out-of-band, and the API only creates the job that turns a
manifest into a collection.

For deploying already-built collections without recomputing embeddings, see
[Qdrant deployment and snapshot hand-off](qdrant-operations.md).

Each time data or the embedding model changes, ingestion builds a **new**
collection rather than mutating an existing one. The best collection is
then picked for serving via config/alias once it has been evaluated
offline — that activation step is intentionally not part of this API.

## Pipeline

```
Copy/mount data into local storage
        ↓
python -m app.ingestion.batch_builder --source ... --entity frames --out ...
        ↓
Manifest written as Parquet (video_id, frame_id, path)
        ↓
POST /api/v1/ingestions with manifest_path + collection_name
        ↓
Job record created in SQLite, runner subprocess launched
        ↓
GET /api/v1/ingestions/{job_id} reads progress back from SQLite
```

`app/ingestion/batch_builder.py` is the CLI that turns a locally
copied/mounted directory into a manifest — it scans for frame files named
`{video_id}_{frame_id}.jpg|png` and writes one Parquet row per frame
(`video_id`, `frame_id`, `path`). Clip manifests aren't scanned yet
(`scan_clips` is a deliberate `NotImplementedError` until that naming
convention is decided).

`app/ingestion/manifest.py` defines the manifest's row schema
(`ManifestRow`) and the Parquet reader used by the runner: `count_rows`
(for `progress.total`), `validate_columns` (required columns present), and
`iter_rows` (streamed row-by-row for `upsert_points`). It depends on
`pyarrow`, added to `requirements.txt` for this.

## Embedding design choices

The frame, clip, and text-query vectors must come from the same feature
profile. A collection built with one profile must therefore be queried with
`app.features.multimodal.embed_text` using that exact profile; dimensions
alone are not enough to make vectors from different models comparable.

Embedding concerns are kept outside the ingestion orchestrator:

- `app/features/profiles.py` owns versioned model identifiers, dimensions,
  clip sampling limits, and inference batch sizes.
- `app/features/media.py` owns image decoding, video seeking, and clip-frame
  sampling.
- `app/features/multimodal.py` owns model loading, image/text inference,
  normalization, and pooling without depending on ingestion schemas.
- `app/ingestion/embedder.py` is the boundary adapter that maps frame/clip
  manifest rows to generic feature-layer inputs.
- `app/ingestion/pipeline.py` only coordinates manifest iteration, collection
  creation, point construction, batched upsert, and progress reporting.

The recommended accuracy-first profile is
`siglip2-giant-opt-patch16-384-v1`. It uses the multilingual SigLIP 2 Giant
image and text encoders, whose outputs share a 1,536-dimensional retrieval
space. `siglip2-so400m-patch14-384-v1` is the lower-memory alternative, while
`clip-b32-v1` remains for compatibility and cheap experiments. Changing a
profile requires a new versioned collection.

### Keyframe embedding

A keyframe manifest row points to one JPEG/PNG. Ingestion decodes that file to
RGB and passes the pixels to the checkpoint's `AutoProcessor`, which applies
the model-specific resize and pixel preprocessing. The image encoder then
produces one feature vector, which is L2-normalized before it is stored.

Pixel preprocessing and L2 feature normalization are separate operations:
the image is fed to the model first, and only the model output is
L2-normalized. For an embedding `v`, the stored vector is:

```text
v_unit = v / ||v||₂
```

Retrieval is based on vector direction rather than uncontrolled differences
in output magnitude. Because both stored image vectors and query-text vectors
have unit length, their dot product equals cosine similarity. Qdrant is also
configured with cosine distance, but normalizing before storage preserves the
same invariant for offline evaluation and detects zero or invalid vectors
before an upsert.

### Clip embedding

A clip manifest row represents one inclusive shot range in a source video; it
does not point to a separately encoded clip file. It supplies `path`,
`start_frame`, `end_frame`, `start_sec`, and `end_sec`. Ingestion performs the
following steps:

1. Choose up to eight timestamps uniformly across `[start_sec, end_sec]`. A
   shot containing fewer than eight source frames uses its actual frame count.
2. Seek backward to the nearest decodable video keyframe, then decode forward
   into the requested range. Video codecs generally cannot begin decoding at
   an arbitrary inter-frame without its preceding references.
3. Select RGB frames as decoding crosses the target timestamps, allowing a
   half-frame tolerance for container time-base and floating-point rounding.
4. Encode the sampled frames in small image batches with the same image encoder
   used for standalone keyframes.
5. L2-normalize each frame feature, mean-pool the frame features, and
   L2-normalize the pooled result:

```text
nᵢ = frame_embeddingᵢ / ||frame_embeddingᵢ||₂
clip_raw = mean(n₁, n₂, ..., nₖ)
clip_embedding = clip_raw / ||clip_raw||₂
```

Normalizing each frame before pooling gives every sampled moment equal weight;
otherwise a frame with a larger feature norm could dominate the clip for a
reason unrelated to semantic relevance. The final normalization makes the
pooled vector directly comparable with normalized text queries under cosine
similarity.

Uniform sampling plus mean pooling was chosen as a bounded-cost baseline for
shot-level semantic retrieval. It covers the beginning, middle, and end of a
shot, keeps each shot to one Qdrant point, and remains in SigLIP's shared
image-text space. Eight frames is a quality/cost default, not a claim of a
universally optimal sample count.

This representation is intentionally order-insensitive. It is suitable for
queries about objects, people, places, and scene content, but it does not model
motion direction or distinguish action order such as `A then B` from `B then
A`. Brief events can also be diluted by mean pooling. If temporal retrieval
becomes a benchmark bottleneck, evaluate per-frame multi-vectors, max-similarity
aggregation, or a dedicated video-text encoder rather than silently changing
this collection contract.

## Endpoints

Three endpoints, no more:

- `POST /api/v1/ingestions` — create a job.
- `GET /api/v1/ingestions` — list jobs.
- `GET /api/v1/ingestions/{job_id}` — read one job's status.

There is deliberately no separate validate/snapshot/activate/rollback/cancel
API at this stage — those are follow-ups once there is a real need for them.

### Create job — `POST /api/v1/ingestions`

```json
{
  "entity": "frames",
  "manifest_path": "/data/manifests/batch1-frames.parquet",
  "collection_name": "aic_frames_r001",
  "feature_profile": "siglip2-giant-opt-patch16-384-v1"
}
```

Validated before a job is accepted:

- `manifest_path` resolves inside the allowed data root
  (`settings.INGESTION_DATA_ROOT`) — no path escape.
- `collection_name` isn't already used by another non-failed job.

Response (`202 Accepted`):

```json
{
  "job_id": "ing-001",
  "status": "queued",
  "collection_name": "aic_frames_r001"
}
```

### List / status — `GET /api/v1/ingestions`, `GET /api/v1/ingestions/{job_id}`

```json
{
  "job_id": "ing-001",
  "status": "running",
  "stage": "upserting",
  "progress": { "completed": 320000, "total": 500000, "percent": 64.0 },
  "collection_name": "aic_frames_r001",
  "error": null
}
```

Status is one of `queued` / `running` / `succeeded` / `failed` — no richer
state machine. `stage` is informational only, one of: `validating`,
`creating_collection`, `creating_payload_indexes`, `upserting`,
`optimizing`, `completed`.

## Why a subprocess, not a FastAPI background task

Ingestion jobs run for minutes to hours, use GPU for feature extraction,
and must survive an API process restart. A `BackgroundTasks` callback dies
with the request's worker process, so it can't give any of that.

```
POST /api/v1/ingestions
       ↓
Create a row in SQLite (status=queued)
       ↓
Launch a local subprocess: python -m app.ingestion.runner --job-id ing-001
       ↓
Subprocess runs the pipeline and writes progress back into SQLite
       ↓
GET /api/v1/ingestions/{job_id} just reads that SQLite row
```

- The API (`app/ingestion/service.py`) only validates the request, writes
  the job row, and spawns the runner — it never runs the pipeline itself.
- The runner (`app/ingestion/runner.py`) is a plain CLI entry point invoked
  as its own OS process (`start_new_session=True`), so it keeps running
  even if the API process restarts.
- Job state lives in SQLite (`app/ingestion/store.py`), not in memory, so
  `GET` always reflects the runner's last write regardless of which
  process (or restart) is asking.

## What's implemented vs. stubbed

- API layer, SQLite job store, manifest building/reading, and the
  runner's stage-by-stage state transitions are real and wired
  end-to-end — `validate_manifest` and the row-iteration side of
  `upsert_points` genuinely read the Parquet manifest.
- Feature extraction and Qdrant upsert are implemented. The highest-capacity
  profile `siglip2-giant-opt-patch16-384-v1` uses the multilingual SigLIP 2
  Giant checkpoint. Keyframes are embedded directly; clips are represented by
  up to eight uniformly sampled frames, normalized and mean-pooled in the same
  vector space used by `app.features.multimodal.embed_text`. The So400m and
  CLIP B/32 profiles remain available when ingestion memory is constrained.
- Model weights must be downloaded into the local Hugging Face cache before
  an offline competition run. The SigLIP 2 Giant checkpoint is roughly 7.5 GB,
  and GPU inference is strongly recommended.
- Collection optimization is implemented: ingestion disables HNSW indexing
  during the bulk load, restores the indexing threshold afterward, and waits
  for Qdrant to report the collection as green before the job succeeds.
