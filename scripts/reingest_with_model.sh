#!/usr/bin/env bash
# Ingest keyframes and/or ASR into a new Qdrant collection using a different
# feature profile (embedding model), reusing existing clips.parquet / videos.parquet / keyframes.
#
# Usage:
#   ./scripts/reingest_with_model.sh [options]
#
# Examples:
#   # Ingest frames using SigLIP2-so400m into a new collection:
#   ./scripts/reingest_with_model.sh \
#       --image-profile siglip2-so400m-patch14-384-v1 \
#       --frames-collection aic2026-frames-so400m
#
#   # Ingest frames using CLIP ViT-B/32:
#   ./scripts/reingest_with_model.sh \
#       --image-profile clip-b32-v1 \
#       --frames-collection aic2026-frames-clip-b32
#
#   # Re-sample keyframes (e.g. 5 frames per shot) and embed with a new model:
#   ./scripts/reingest_with_model.sh \
#       --image-profile siglip2-so400m-patch14-384-v1 \
#       --frames-collection aic2026-frames-so400m-5fps \
#       --frames-per-shot 5 \
#       --force-sampling
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Defaults
DATA="${DATA:-$ROOT/data}"
MANIFESTS="${MANIFESTS:-$DATA/manifests-v2}"
KEYFRAMES="${KEYFRAMES:-$DATA/keyframes-v2}"
WORKERS="${WORKERS:-3}"
FRAMES_PER_SHOT="${FRAMES_PER_SHOT:-3}"
API="${API:-http://127.0.0.1:8000/api/v1}"

IMAGE_PROFILE="${IMAGE_PROFILE:-siglip2-so400m-patch14-384-v1}"
FRAMES_COLLECTION="${FRAMES_COLLECTION:-aic2026-frames-so400m}"
TEXT_PROFILE="${TEXT_PROFILE:-qwen3-embed-0.6b-v1}"
ASR_COLLECTION="${ASR_COLLECTION:-}"  # If empty, skip ASR ingestion

FORCE_SAMPLING=0
SKIP_FRAMES=0
SKIP_ASR=0

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --image-profile <name>      Image feature profile (default: $IMAGE_PROFILE)
                              Supported: siglip2-giant-opt-patch16-384-v1,
                                         siglip2-so400m-patch14-384-v1,
                                         clip-b32-v1
  --frames-collection <name>  Target Qdrant collection for frames (default: $FRAMES_COLLECTION)
  --text-profile <name>       Text feature profile for ASR (default: $TEXT_PROFILE)
                              Supported: qwen3-embed-0.6b-v1
  --asr-collection <name>     Target Qdrant collection for ASR (optional, e.g. aic2026-asr-v2)
  --manifests-dir <dir>       Path to manifests dir (default: $MANIFESTS)
  --keyframes-dir <dir>       Path to keyframes dir (default: $KEYFRAMES)
  --frames-per-shot <num>     Number of keyframes per shot (default: $FRAMES_PER_SHOT)
  --workers <num>             Worker processes for sampling (default: $WORKERS)
  --force-sampling            Force re-sampling keyframes even if frames.parquet exists
  --skip-frames               Skip keyframes ingestion
  --skip-asr                  Skip ASR ingestion
  -h, --help                  Show this help message
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image-profile)      IMAGE_PROFILE="$2"; shift 2 ;;
        --frames-collection)  FRAMES_COLLECTION="$2"; shift 2 ;;
        --text-profile)       TEXT_PROFILE="$2"; shift 2 ;;
        --asr-collection)     ASR_COLLECTION="$2"; shift 2 ;;
        --manifests-dir)      MANIFESTS="$2"; shift 2 ;;
        --keyframes-dir)      KEYFRAMES="$2"; shift 2 ;;
        --frames-per-shot)    FRAMES_PER_SHOT="$2"; shift 2 ;;
        --workers)            WORKERS="$2"; shift 2 ;;
        --force-sampling)     FORCE_SAMPLING=1; shift ;;
        --skip-frames)        SKIP_FRAMES=1; shift ;;
        --skip-asr)           SKIP_ASR=1; shift ;;
        -h|--help)            usage ;;
        *)                    echo "Unknown option: $1"; usage ;;
    esac
done

mkdir -p "$MANIFESTS" "$ROOT/logs"

# shellcheck disable=SC1091
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi
set -a; [ -f .env ] && . ./.env; set +a

step() { printf '\n=== %s === %s\n' "$1" "$(date -Is)"; }

step "Preflight Checks"

# 1. Check required manifests
if [ ! -f "$MANIFESTS/videos.parquet" ]; then
    echo "Error: $MANIFESTS/videos.parquet not found."
    echo "Run probe first or check your --manifests-dir."
    exit 1
fi

if [ ! -f "$MANIFESTS/clips.parquet" ]; then
    echo "Error: $MANIFESTS/clips.parquet not found."
    echo "Run shot detection first or check your --manifests-dir."
    exit 1
fi
echo "Found videos.parquet and clips.parquet in $MANIFESTS"

# 2. Check API readiness
if ! curl -sf "$API/health/ready" >/dev/null 2>&1; then
    cat <<EOF
The API is not answering on $API.
Please start it in another window:

  cd $ROOT/backend && python -m uvicorn app.main:app --port 8000 --workers 1

EOF
    exit 1
fi
echo "API ready on $API"

# 3. Pre-cache model weights in HuggingFace cache
step "Pre-caching Model Weights ($IMAGE_PROFILE / $TEXT_PROFILE)"
# The text profile is only resolved when ASR will actually be ingested, so a
# frames-only run is not aborted by an unused profile name.
CACHE_TEXT_PROFILE=""
if [ "$SKIP_ASR" -eq 0 ] && [ -n "$ASR_COLLECTION" ]; then
    CACHE_TEXT_PROFILE="$TEXT_PROFILE"
fi
# From backend/: `pythonpath` for `app.*` resolves only there.
(cd "$ROOT/backend" && python - <<PY
import sys
from app.features.profiles import get_profile

for prof_name in ("$IMAGE_PROFILE", "$CACHE_TEXT_PROFILE"):
    if not prof_name:
        continue
    try:
        prof = get_profile(prof_name)
        model_id = prof.model_id
        print(f"Resolving model for profile '{prof_name}': {model_id}")
        if prof.kind == "image":
            from transformers import AutoImageProcessor, AutoModel
            AutoModel.from_pretrained(model_id)
            try:
                AutoImageProcessor.from_pretrained(model_id, use_fast=True)
            except Exception:
                pass
        else:
            from transformers import AutoModel, AutoTokenizer
            AutoModel.from_pretrained(model_id)
            AutoTokenizer.from_pretrained(model_id)
        print(f"Cached successfully: {model_id}")
    except Exception as e:
        print(f"Failed to cache model for profile '{prof_name}': {e}", file=sys.stderr)
        sys.exit(1)
PY
)

# Step: Keyframe Sampling (if needed)
#
# A re-sample gets its own manifest and keyframe directory, keyed by the rate.
# Writing back into frames.parquet would both destroy the manifest the existing
# collections were built from and, because --resume skips videos already in the
# output, silently keep the old rate's rows -- so a "5 frames/shot" collection
# would have been ingested from 3-frame rows with nothing raised.
FRAMES_MANIFEST="$MANIFESTS/frames.parquet"
if [ "$FORCE_SAMPLING" -eq 1 ]; then
    FRAMES_MANIFEST="$MANIFESTS/frames-${FRAMES_PER_SHOT}fps.parquet"
    KEYFRAMES="$KEYFRAMES-${FRAMES_PER_SHOT}fps"
fi

if [ "$SKIP_FRAMES" -eq 0 ]; then
    if [ ! -f "$FRAMES_MANIFEST" ] || [ "$FORCE_SAMPLING" -eq 1 ]; then
        step "Sampling Keyframes ($FRAMES_PER_SHOT frames/shot -> $FRAMES_MANIFEST)"
        # --resume kept: a fresh manifest resumes nothing on the first run, and
        # it makes a re-run of this multi-hour stage re-entrant.
        (cd "$ROOT/backend" && python -m app.ingestion.video.sampling \
            --videos-manifest "$MANIFESTS/videos.parquet" \
            --shots-manifest "$MANIFESTS/clips.parquet" \
            --output-dir "$KEYFRAMES" \
            --frames-per-shot "$FRAMES_PER_SHOT" \
            --out "$FRAMES_MANIFEST" \
            --workers "$WORKERS" --resume)
    else
        echo "Found existing $FRAMES_MANIFEST. Skipping sampling (use --force-sampling to re-sample)."
    fi
fi

# Step: ASR manifest (if needed)
if [ "$SKIP_ASR" -eq 0 ] && [ -n "$ASR_COLLECTION" ]; then
    if [ ! -f "$MANIFESTS/asr_segments.parquet" ]; then
        step "Building ASR Segments Manifest"
        python scripts/build_asr_manifest.py \
            --transcripts "$DATA/transcripts" \
            --out "$MANIFESTS/asr_segments.parquet"
    else
        echo "Found existing $MANIFESTS/asr_segments.parquet."
    fi
fi

# Queue ingestion helper functions
queue() {
    # Body captured rather than piped straight into python: a 4xx from the API
    # otherwise surfaced as a JSONDecodeError on empty stdin, hiding the reason
    # (a re-used collection name is the common one).
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
        sleep 15
    done
}

REL_MANIFESTS="${MANIFESTS#"$DATA"/}"

# Ingest Frames
if [ "$SKIP_FRAMES" -eq 0 ]; then
    step "Ingesting Frames into Collection: $FRAMES_COLLECTION (Profile: $IMAGE_PROFILE)"
    frames_job=$(queue frames "${FRAMES_MANIFEST#"$DATA"/}" "$FRAMES_COLLECTION" "$IMAGE_PROFILE")
    echo "Queued frames job: $frames_job"
    wait_for "$frames_job"
fi

# Ingest ASR
if [ "$SKIP_ASR" -eq 0 ] && [ -n "$ASR_COLLECTION" ]; then
    step "Ingesting ASR into Collection: $ASR_COLLECTION (Profile: $TEXT_PROFILE)"
    asr_job=$(queue asr_segments "$REL_MANIFESTS/asr_segments.parquet" "$ASR_COLLECTION" "$TEXT_PROFILE")
    echo "Queued ASR job: $asr_job"
    wait_for "$asr_job"
fi

step "Ingestion Completed Successfully!"
cat <<EOF
To use your new collection with the API, update your backend/.env (or root .env) with:

  FEATURE_PROFILE=$IMAGE_PROFILE
  QDRANT_FRAMES_COLLECTION=$FRAMES_COLLECTION
EOF

if [ -n "$ASR_COLLECTION" ]; then
cat <<EOF
  ASR_FEATURE_PROFILE=$TEXT_PROFILE
  QDRANT_ASR_COLLECTION=$ASR_COLLECTION
EOF
fi

cat <<EOF

Then restart the API server to apply the changes.
EOF
