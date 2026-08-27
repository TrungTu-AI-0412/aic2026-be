from typing import Dict, List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
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

    # Must match the profile the active collection was ingested with: the
    # query vector has to land in the same space, at the same dimension.
    FEATURE_PROFILE: str = "siglip2-so400m-patch14-384-v1"

    # Weight of the clip index when fusing it with the frame index. 0 disables
    # fusion and searches frames only.
    CLIP_FUSION_WEIGHT: float = 0.5

    # Fuse the lexical vectors into every query. Must be off for a collection
    # ingested before they existed: prefetching a vector the collection does
    # not declare fails the query outright rather than degrading.
    HYBRID_ENABLED: bool = True

    # Search on-screen text as its own channel and fuse it onto the visual
    # ranking by rank. 0 weight, or the flag off, leaves `ocr` as one more
    # equal branch inside Qdrant's own fusion. /search/ocr is unaffected
    # either way - it never consults these.
    #
    # The weight is small because it was measured, not argued: on 300 queries
    # every step above 0.05 cost accuracy, and the 0.5 this shipped with was
    # worse than turning the channel off. `app/ranking/boost.py` carries the
    # table and the caveat that the query set is speech-derived.
    OCR_BOOST_ENABLED: bool = True
    OCR_BOOST_WEIGHT: float = 0.05

    # Cross-encoder rerank of the head of each result list. It costs one
    # forward pass per candidate, so RERANK_TOP_N is the latency dial. The
    # weights must be in the local Hugging Face cache before the competition:
    # the query path is not allowed to reach the network.
    RERANK_ENABLED: bool = True
    RERANK_TOP_N: int = 30
    RERANK_MODEL: str = "Salesforce/blip-itm-large-coco"

    # Per-video frame bounds used to reject a submission row that could never
    # score. Defaults to video_bounds.parquet under INGESTION_DATA_ROOT; when
    # the file is missing the export still works, just without the check.
    SUBMISSION_BOUNDS_PATH: Optional[str] = None


settings = Settings()