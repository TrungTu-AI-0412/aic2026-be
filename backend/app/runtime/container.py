from dataclasses import dataclass

from app.core.config import Settings
from app.ingestion.service import SqliteIngestionService
from app.retrieval.engine import RetrievalConfig
from app.retrieval.service import QdrantSearchService
from app.services.ingestions import IngestionService
from app.services.media import LocalMediaService, MediaService
from app.services.search import SearchService
from app.services.submissions import SubmissionService


@dataclass
class Container:
    search_service: SearchService
    ingestion_service: IngestionService
    media_service: MediaService
    submission_service: SubmissionService | None = None

    async def close(self) -> None:
        pass


async def build_container(settings: Settings) -> Container:
    return Container(
        search_service=QdrantSearchService(
            RetrievalConfig(
                frames_collection=settings.QDRANT_FRAMES_COLLECTION,
                clips_collection=settings.QDRANT_CLIPS_COLLECTION,
                feature_profile=settings.FEATURE_PROFILE,
            )
        ),
        ingestion_service=SqliteIngestionService(
            db_path=settings.INGESTION_DB_PATH,
            data_root=settings.INGESTION_DATA_ROOT,
        ),
        media_service=LocalMediaService(data_root=settings.INGESTION_DATA_ROOT),
    )
