from pathlib import Path
from typing import Dict, List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute so settings load the same whether the process starts in the repo
# root or in backend/ (the runbook's uvicorn working directory).
_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_ignore_empty=True,
        extra="ignore",
    )

    PROJECT_NAME: str = "AIC 2026 Retrieval System"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"

    INGESTION_DATA_ROOT: str = "./data"
    INGESTION_DB_PATH: str = "./data/ingestion.db"

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "aic2026"
    QDRANT_BATCH_SIZE: int = 256

    # Collections are versioned per ingestion run, so these name whichever
    # build the API should serve. Ingestion never writes to the active one.
    QDRANT_FRAMES_COLLECTION: str = "aic2026-frames-v1"
    QDRANT_CLIPS_COLLECTION: Optional[str] = None
    # Speech segments, one point per ASR segment rather than per frame. Unset
    # disables the ASR stage entirely, so an older deployment keeps working.
    QDRANT_ASR_COLLECTION: Optional[str] = None

    # Must match the profile the active collection was ingested with: the
    # query vector has to land in the same space, at the same dimension.
    FEATURE_PROFILE: str = "siglip2-giant-opt-patch16-384-v1"
    # Text profile for the ASR collection. Separate from FEATURE_PROFILE
    # because speech is matched text-to-text, not against an image space.
    ASR_FEATURE_PROFILE: str = "qwen3-embed-0.6b-v1"

    # Lexical retrieval. Sparse method is "bm25" or "splade"; keep bm25 unless
    # a Vietnamese SPLADE model is cached, since the default one is
    # English-only and its subword ids are incompatible with the CRC32 slots.
    HYBRID_ENABLED: bool = True
    # On-screen text folded over the visual ranking as a second, rank-fused
    # query. Off by default: it costs one extra sparse query per `retrieve()`,
    # which on TRAKE is per event.
    OCR_BOOST_ENABLED: bool = False
    # Measured, not chosen. This shipped at 0.5 and was worse than turning the
    # channel off; every step down bought accuracy. See `ranking/boost.py`.
    OCR_BOOST_WEIGHT: float = 0.05
    SPARSE_METHOD: str = "bm25"
    SPLADE_MODEL: str = "naver/splade-cocondenser-ensembledistil"

    # ASR overlap bonus. A query also searches the speech collection, and each
    # frame is boosted by the best-scoring segment whose time range covers it.
    ASR_ENABLED: bool = True
    ASR_WEIGHT: float = 0.3
    # Dense is weighted above sparse: the transcript is fluent Vietnamese, so
    # semantic similarity carries more of the signal than term overlap, which
    # is there to catch the names and numbers dense retrieval loses.
    ASR_DENSE_WEIGHT: float = 0.7
    ASR_SPARSE_WEIGHT: float = 0.3
    # Segment bounds in the source are rounded to whole seconds, and 4.5% of
    # video time has no segment at all, so overlap is tested with slack.
    ASR_PAD_SEC: float = 1.0

    # Weight of the clip index when fusing it with the frame index. 0 disables
    # fusion and searches frames only.
    CLIP_FUSION_WEIGHT: float = 0.5

    # Cross-encoder rerank of the head of each result list. It costs one
    # forward pass per candidate, so RERANK_TOP_N is the latency dial. The
    # weights must be in the local Hugging Face cache before the competition:
    # the query path is not allowed to reach the network.
    RERANK_ENABLED: bool = True
    RERANK_TOP_N: int = 30
    RERANK_MODEL: str = "Salesforce/blip-itm-large-coco"

    # Query rewriting. One LLM call returns two forms of every query: translated
    # to English for the image space and the reranker, which are both
    # English-centric, and merely stripped of the operator's phrasing ("hãy tìm
    # trong video...") for the speech stage, whose transcripts are Vietnamese.
    # This is the one network hop the query path takes, so any failure falls
    # back to the query as typed. The timeout has to cover a whole TRAKE batch:
    # the step is output-token-bound, and an overview plus five events in two
    # forms measured ~3.2s.
    QUERY_REWRITE_ENABLED: bool = True
    QUERY_REWRITE_TIMEOUT_SEC: float = 6.0
    # OpenAI-compatible chat completions endpoint, including the /v1. Unset
    # leaves rewriting a silent no-op even with QUERY_REWRITE_ENABLED=true.
    VLM_BASE_URL: Optional[str] = None
    VLM_MODEL: str = "Qwen/Qwen3.6-27B"
    # Plain str, not Optional: env_ignore_empty drops the blank value in .env, so
    # this default is what a keyless local server actually gets.
    VLM_API_KEY: str = ""

    # Per-video frame bounds used to reject a submission row that could never
    # score. Defaults to video_bounds.parquet under INGESTION_DATA_ROOT; when
    # the file is missing the export still works, just without the check.
    SUBMISSION_BOUNDS_PATH: Optional[str] = None

    # Keyframe JPEGs and the video probe manifest the media endpoints read.
    # Both default under INGESTION_DATA_ROOT, but a re-ingest writes to a new
    # versioned directory, so point these at whatever the active collection
    # was built from.
    MEDIA_KEYFRAMES_DIR: Optional[str] = None
    MEDIA_VIDEOS_MANIFEST: Optional[str] = None
    MEDIA_FRAMES_MANIFEST: Optional[str] = None


settings = Settings()