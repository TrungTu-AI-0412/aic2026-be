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
    
    # QDRANT_VECTOR_SIZE: int = 1048
    

settings = Settings()