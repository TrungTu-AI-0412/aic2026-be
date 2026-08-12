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


settings = Settings()