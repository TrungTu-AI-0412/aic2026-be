#!/usr/bin/env bash
# Rebuild both collections from the raw videos and transcripts.
#
# Run under tmux: the whole thing takes roughly nine hours, dominated by two
# full decode passes over 78GB of video. Every stage supports --resume and
# flushes its manifest periodically, so an interrupted run is re-entrant --
# re-running this script picks up where it stopped rather than starting over.
#
#   tmux new -s ingest
#   ./scripts/ingest_all.sh 2>&1 | tee logs/ingest.log
#
# Stage-by-stage timings, measured on 1x L40S / 4 vCPU:
#   probe            ~1 min      container headers only, no decode
#   shot detection   ~4.6 h      12.26M frames through TransNetV2 at 737 fps
#   sampling         ~3.1 h      second decode pass, writes ~293k JPEGs (~26GB)
#   asr manifest     ~30 s       35k segments from 873 CSVs
#   frame embed      ~1-2 h      SigLIP2 giant; so400m measured 83 pts/s
#   asr embed        ~2 min      Qwen3-Embedding-0.6B
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA="${DATA:-$ROOT/data}"
# Versioned, not `data/manifests`: that directory holds manifests from an older
# schema, and resuming into one would be refused (correctly) part-way through a
# multi-hour run. A new pipeline gets new files.
MANIFESTS="${MANIFESTS:-$DATA/manifests-v2}"
KEYFRAMES="${KEYFRAMES:-$DATA/keyframes-v2}"
WORKERS="${WORKERS:-3}"
FRAMES_PER_SHOT="${FRAMES_PER_SHOT:-3}"
IMAGE_PROFILE="${IMAGE_PROFILE:-siglip2-giant-opt-patch16-384-v1}"
TEXT_PROFILE="${TEXT_PROFILE:-qwen3-embed-0.6b-v1}"
FRAMES_COLLECTION="${FRAMES_COLLECTION:-aic2026-frames-v2}"
ASR_COLLECTION="${ASR_COLLECTION:-aic2026-asr-v1}"
API="${API:-http://127.0.0.1:8000/api/v1}"

mkdir -p "$MANIFESTS" "$ROOT/logs"
# shellcheck disable=SC1091
source venv/bin/activate
set -a; [ -f .env ] && . ./.env; set +a

step() { printf '\n=== %s === %s\n' "$1" "$(date -Is)"; }

step "0/6 preflight"
# The API is checked here rather than at stage 5, where discovering it is down
# would mean eight hours of decoding followed by an exit. Stages 1-4 do resume,
# so nothing would be lost -- but nothing would be gained either.
if ! curl -sf "$API/health/ready" >/dev/null 2>&1; then
    cat <<EOF
The API is not answering on $API, and stage 5 needs it to queue the ingestion.
Start it in another window, then re-run this script:

  cd $ROOT/backend && python -m uvicorn app.main:app --port 8000 --workers 1

EOF
    exit 1
fi
echo "API ready on $API"

# SigLIP2 giant is not in the local cache by default and the competition query
# path is not allowed to reach the network, so fetch it now and prove it
# resolves offline afterwards. Failing here is far cheaper than failing after
# the two decode passes.
python - <<'PY'
from transformers import AutoImageProcessor, AutoModel
for model_id in ("google/siglip2-giant-opt-patch16-384", "Qwen/Qwen3-Embedding-0.6B"):
    AutoModel.from_pretrained(model_id)
    print(f"cached: {model_id}")
AutoImageProcessor.from_pretrained(
    "google/siglip2-giant-opt-patch16-384", use_fast=True
)
PY
HF_HUB_OFFLINE=1 python - <<'PY'
from transformers import AutoModel
for model_id in ("google/siglip2-giant-opt-patch16-384", "Qwen/Qwen3-Embedding-0.6B"):
    AutoModel.from_pretrained(model_id)
print("offline resolution OK")
PY

cd backend

step "1/6 probe"
python -m app.ingestion.video.probe \
    --source "$DATA/videos" \
    --out "$MANIFESTS/videos.parquet" \
    --workers "$WORKERS" --resume

step "2/6 shot detection (TransNetV2, ~4.6h)"
python -m app.ingestion.video.shot_detect \
    --videos-manifest "$MANIFESTS/videos.parquet" \
    --out "$MANIFESTS/clips.parquet" \
    --detector transnetv2 --workers "$WORKERS" --resume

step "3/6 sampling ($FRAMES_PER_SHOT keyframes/shot, ~3h)"
python -m app.ingestion.video.sampling \
    --videos-manifest "$MANIFESTS/videos.parquet" \
    --shots-manifest "$MANIFESTS/clips.parquet" \
    --output-dir "$KEYFRAMES" \
    --frames-per-shot "$FRAMES_PER_SHOT" \
    --out "$MANIFESTS/frames.parquet" \
    --workers "$WORKERS" --resume

cd "$ROOT"

step "4/6 asr segment manifest"
python scripts/build_asr_manifest.py \
    --transcripts "$DATA/transcripts" \
    --out "$MANIFESTS/asr_segments.parquet"

step "5/6 ingest"
# Manifest paths are resolved against INGESTION_DATA_ROOT, so they are passed
# relative to it rather than absolute.
#
# Re-checked because the run above takes hours and the API may have died in the
# meantime. Stages 1-4 resume, so re-running after restarting it is cheap.
if ! curl -sf "$API/health/ready" >/dev/null 2>&1; then
    echo "The API stopped answering on $API during the run."
    echo "Restart it and re-run -- stages 1-4 resume from their manifests:"
    echo "  cd $ROOT/backend && python -m uvicorn app.main:app --port 8000 --workers 1"
    exit 1
fi

queue() {
    # Body captured rather than piped straight into python: a 4xx from the API
    # otherwise surfaced as a JSONDecodeError on empty stdin, hiding the reason.
    local body
    body=$(curl -s -X POST "$API/ingestions" -H 'content-type: application/json' \
        -d "{\"entity\":\"$1\",\"manifest_path\":\"$2\",\
             \"collection_name\":\"$3\",\"feature_profile\":\"$4\"}")
    python -c 'import json,sys; print(json.load(sys.stdin)["job_id"])' <<<"$body" \
        || { echo "queueing $1 failed: $body" >&2; return 1; }
}

wait_for() {
    local job="$1"
    while true; do
        local state
        state=$(curl -sf "$API/ingestions/$job" \
            | python -c 'import json,sys; d=json.load(sys.stdin); p=d.get("progress") or {}; print(d["status"], d.get("stage"), str(p.get("completed", 0)) + "/" + str(p.get("total", 0)), d.get("error") or "")')
        echo "  $job: $state"
        case "$state" in
            succeeded*) return 0 ;;
            failed*)    echo "job $job failed"; return 1 ;;
        esac
        sleep 30
    done
}

# Derived from $MANIFESTS rather than hardcoded, so overriding the directory
# cannot leave these two pointing at the previous one.
REL_MANIFESTS="${MANIFESTS#"$DATA"/}"

frames_job=$(queue frames "$REL_MANIFESTS/frames.parquet" "$FRAMES_COLLECTION" "$IMAGE_PROFILE")
wait_for "$frames_job"

asr_job=$(queue asr_segments "$REL_MANIFESTS/asr_segments.parquet" "$ASR_COLLECTION" "$TEXT_PROFILE")
wait_for "$asr_job"

step "6/6 done"
cat <<EOF
Point .env at the new collections and restart the API. The profile and the
collection must change together -- a mismatch searches the wrong vector space
and returns plausible nonsense instead of an error.

  FEATURE_PROFILE=$IMAGE_PROFILE
  ASR_FEATURE_PROFILE=$TEXT_PROFILE
  QDRANT_FRAMES_COLLECTION=$FRAMES_COLLECTION
  QDRANT_ASR_COLLECTION=$ASR_COLLECTION
  QDRANT_CLIPS_COLLECTION=
EOF
