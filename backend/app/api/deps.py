from fastapi import Request

from app.services.ingestions import IngestionService
from app.services.media import MediaService
from app.services.search import SearchService
from app.services.submissions import SubmissionService


def get_search_service(request: Request) -> SearchService:
    return request.app.state.container.search_service


def get_media_service(request: Request) -> MediaService:
    return request.app.state.container.media_service


def get_submission_service(request: Request) -> SubmissionService:
    return request.app.state.container.submission_service


def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.container.ingestion_service
