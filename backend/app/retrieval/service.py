"""SearchService backed by Qdrant."""

from starlette.concurrency import run_in_threadpool

from app.retrieval import tracks
from app.retrieval.engine import RetrievalConfig
from app.schemas.search import (
    KisSearchRequest,
    OcrSearchRequest,
    QaSearchRequest,
    SearchResponse,
    TrakeSearchRequest,
)


class QdrantSearchService:
    """Async facade over the synchronous retrieval path.

    Encoding a query runs a transformer forward pass and Qdrant calls block on
    IO, so each request is handed to a worker thread. Running them inline would
    stall the event loop and serialise every other request behind one search.
    """

    def __init__(self, config: RetrievalConfig) -> None:
        self._config = config

    async def search_kis(self, request: KisSearchRequest) -> SearchResponse:
        return await run_in_threadpool(tracks.search_kis, request, self._config)

    async def search_qa(self, request: QaSearchRequest) -> SearchResponse:
        return await run_in_threadpool(tracks.search_qa, request, self._config)

    async def search_trake(self, request: TrakeSearchRequest) -> SearchResponse:
        return await run_in_threadpool(tracks.search_trake, request, self._config)

    async def search_ocr(self, request: OcrSearchRequest) -> SearchResponse:
        # Off the event loop like the rest, even though this one runs no
        # model: the Qdrant call still blocks on IO.
        return await run_in_threadpool(tracks.search_ocr, request, self._config)
