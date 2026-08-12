from dataclasses import dataclass

from app.core.config import Settings
from app.ingestion.service import SqliteIngestionService
from app.services.ingestions import IngestionService
from app.services.media import MediaService
from app.services.search import SearchService
from app.services.submissions import SubmissionService
from app.stubs.search import StubSearchService


@dataclass
class Container:
    search_service: SearchService
    ingestion_service: IngestionService
    media_service: MediaService | None = None
    submission_service: SubmissionService | None = None

    async def close(self) -> None:
        pass


async def build_container(settings: Settings) -> Container:
    return Container(
        search_service=StubSearchService(),
        ingestion_service=SqliteIngestionService(
            db_path=settings.INGESTION_DB_PATH,
            data_root=settings.INGESTION_DATA_ROOT,
        ),
    )
