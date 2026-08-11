import subprocess
import sys
from asyncio import to_thread
from pathlib import Path
from uuid import uuid4

from app.core.config import settings
from app.ingestion import store
from app.schemas.ingestions import (
    CreateIngestionJobRequest,
    CreateIngestionJobResponse,
    IngestionJobStatusResponse,
    IngestionProgress,
)
from app.services.ingestions import (
    CollectionAlreadyExistsError,
    IngestionJobNotFoundError,
    ManifestPathNotAllowedError,
)


def _row_to_status(row) -> IngestionJobStatusResponse:
    progress = None
    if row["progress_total"] is not None:
        completed = row["progress_completed"] or 0
        total = row["progress_total"]
        percent = (completed / total * 100) if total else 0.0
        progress = IngestionProgress(completed=completed, total=total, percent=percent)

    return IngestionJobStatusResponse(
        job_id=row["job_id"],
        status=row["status"],
        stage=row["stage"],
        progress=progress,
        collection_name=row["collection_name"],
        error=row["error"],
    )


class SqliteIngestionService:
    def __init__(
        self,
        db_path: str = settings.INGESTION_DB_PATH,
        data_root: str = settings.INGESTION_DATA_ROOT,
    ) -> None:
        self._db_path = db_path
        self._data_root = Path(data_root).resolve()

    async def create_job(
        self, request: CreateIngestionJobRequest
    ) -> CreateIngestionJobResponse:
        manifest_path = Path(request.manifest_path).resolve()
        if not manifest_path.is_relative_to(self._data_root):
            raise ManifestPathNotAllowedError(
                f"manifest_path '{request.manifest_path}' is outside "
                f"the allowed data root '{self._data_root}'"
            )

        exists = await to_thread(
            store.collection_name_exists, self._db_path, request.collection_name
        )
        if exists:
            raise CollectionAlreadyExistsError(
                f"collection '{request.collection_name}' is already used by another job"
            )

        job_id = f"ing-{uuid4().hex[:10]}"

        await to_thread(
            store.create_job,
            self._db_path,
            job_id,
            request.entity.value,
            str(manifest_path),
            request.collection_name,
            request.feature_profile,
        )

        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "app.ingestion.runner",
                "--job-id",
                job_id,
                "--db-path",
                self._db_path,
            ],
            start_new_session=True,
        )

        return CreateIngestionJobResponse(
            job_id=job_id, collection_name=request.collection_name
        )

    async def list_jobs(self) -> list[IngestionJobStatusResponse]:
        rows = await to_thread(store.list_jobs, self._db_path)
        return [_row_to_status(row) for row in rows]

    async def get_job_status(self, job_id: str) -> IngestionJobStatusResponse:
        row = await to_thread(store.get_job, self._db_path, job_id)
        if row is None:
            raise IngestionJobNotFoundError(f"ingestion job '{job_id}' not found")
        return _row_to_status(row)
