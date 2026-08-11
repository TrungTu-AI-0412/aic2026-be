from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_ingestion_service
from app.schemas.ingestions import (
    CreateIngestionJobRequest,
    CreateIngestionJobResponse,
    IngestionJobListResponse,
    IngestionJobStatusResponse,
)
from app.services.ingestions import (
    CollectionAlreadyExistsError,
    IngestionJobNotFoundError,
    IngestionService,
    ManifestPathNotAllowedError,
)

router = APIRouter()


@router.post(
    "",
    response_model=CreateIngestionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_ingestion_job(
    request: CreateIngestionJobRequest,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> CreateIngestionJobResponse:
    try:
        return await ingestion_service.create_job(request)
    except ManifestPathNotAllowedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CollectionAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("", response_model=IngestionJobListResponse)
async def list_ingestion_jobs(
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> IngestionJobListResponse:
    jobs = await ingestion_service.list_jobs()
    return IngestionJobListResponse(jobs=jobs)


@router.get("/{job_id}", response_model=IngestionJobStatusResponse)
async def get_ingestion_job(
    job_id: str,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> IngestionJobStatusResponse:
    try:
        return await ingestion_service.get_job_status(job_id)
    except IngestionJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
