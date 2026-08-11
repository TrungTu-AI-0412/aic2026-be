# Ingestion

Internal pipeline for building a new Qdrant collection from a dataset
version. This is **not** a public upload API — data is copied/mounted onto
local storage out-of-band, and the API only creates the job that turns a
manifest into a collection.

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
  "feature_profile": "clip-b32-v1"
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
- The actual feature-extraction and Qdrant-upsert work (the
  `_embed_and_upsert` call inside `app/ingestion/pipeline.py`, plus
  `create_collection`/`create_payload_indexes`/`optimize_collection`) is
  stubbed with `NotImplementedError` — that depends on the embedding
  model and the `vector_store/` client, which don't exist yet. Wiring
  those in is the next step; the runner already calls everything in the
  right order and persists whatever progress they report.
