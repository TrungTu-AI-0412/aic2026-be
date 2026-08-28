from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_search_service
from app.retrieval.decompose import DecompositionUnavailableError
from app.retrieval.engine import AsrOnlyRequestError, AsrOnlyUnavailableError
from app.schemas.search import (
    DecomposeRequest,
    DecomposeResponse,
    KisSearchRequest,
    OcrSearchRequest,
    QaSearchRequest,
    SearchResponse,
    TrakeSearchRequest,
)
from app.services.search import SearchService

router = APIRouter()


@router.post(
    "/search/kis",
    response_model=SearchResponse,
    responses={503: {"description": "ASR-only retrieval is unavailable"}},
)
async def search_kis(
    request: KisSearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    try:
        return await search_service.search_kis(request)
    except AsrOnlyRequestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AsrOnlyUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/search/qa",
    response_model=SearchResponse,
    responses={503: {"description": "ASR-only retrieval is unavailable"}},
)
async def search_qa(
    request: QaSearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    try:
        return await search_service.search_qa(request)
    except AsrOnlyRequestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AsrOnlyUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/search/decompose",
    response_model=DecomposeResponse,
    responses={503: {"description": "The query could not be decomposed"}},
)
async def decompose(
    request: DecomposeRequest,
    search_service: SearchService = Depends(get_search_service),
) -> DecomposeResponse:
    """Split one pasted query into the overview and events, for review.

    Retrieval does not run here. The operator reads what came back, fixes what
    the model got wrong, and posts it to `/search/trake` or `/search/kis`,
    which then search those forms verbatim and make no LLM call of their own.
    """
    try:
        return await search_service.decompose(request)
    except DecompositionUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/search/trake", response_model=SearchResponse)
async def search_trake(
    request: TrakeSearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    return await search_service.search_trake(request)


@router.post("/search/ocr", response_model=SearchResponse)
async def search_ocr(
    request: OcrSearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    return await search_service.search_ocr(request)
