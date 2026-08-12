from fastapi import APIRouter, Depends

from app.api.deps import get_search_service
from app.schemas.search import (
    KisSearchRequest,
    QaSearchRequest,
    SearchResponse,
    TrakeSearchRequest,
)
from app.services.search import SearchService

router = APIRouter()


@router.post("/search/kis", response_model=SearchResponse)
async def search_kis(
    request: KisSearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    return await search_service.search_kis(request)


@router.post("/search/qa", response_model=SearchResponse)
async def search_qa(
    request: QaSearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    return await search_service.search_qa(request)


@router.post("/search/trake", response_model=SearchResponse)
async def search_trake(
    request: TrakeSearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    return await search_service.search_trake(request)
