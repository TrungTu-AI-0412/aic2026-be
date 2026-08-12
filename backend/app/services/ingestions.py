from typing import Protocol

from app.schemas.ingestions import (
    CreateIngestionJobRequest,
    CreateIngestionJobResponse,
    IngestionFeatureProfilesResponse,
    IngestionJobStatusResponse,
)


class ManifestPathNotAllowedError(Exception):
    pass


class CollectionAlreadyExistsError(Exception):
    pass


class IngestionJobNotFoundError(Exception):
    pass


class UnsupportedFeatureProfileError(Exception):
    pass


class IngestionService(Protocol):
    async def list_feature_profiles(self) -> IngestionFeatureProfilesResponse:
        ...

    async def create_job(
        self, request: CreateIngestionJobRequest
    ) -> CreateIngestionJobResponse:
        ...

    async def list_jobs(self) -> list[IngestionJobStatusResponse]:
        ...

    async def get_job_status(self, job_id: str) -> IngestionJobStatusResponse:
        ...
