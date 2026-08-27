"""SearchService backed by Qdrant."""

from dataclasses import replace

from starlette.concurrency import run_in_threadpool

from app.retrieval import decompose as decompose_query
from app.retrieval import tracks
from app.retrieval.engine import RetrievalConfig, Timings
from app.schemas.search import (
    AsrOverrides,
    DecomposeRequest,
    DecomposeResponse,
    KisSearchRequest,
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

    def _resolve(self, request: AsrOverrides) -> RetrievalConfig:
        """Apply this request's overrides on top of the configured defaults.

        Done here rather than in `engine.retrieve`, whose signature the track
        modules and their tests depend on. The config is frozen, so `replace`
        yields a per-request copy and one query can never leak its tuning into
        the next.
        """
        overrides = {
            field: value
            for field, value in request.model_dump(
                include=set(AsrOverrides.model_fields)
            ).items()
            if value is not None
        }
        mode = getattr(request, "retrieval_mode", None)
        if mode == "visual":
            overrides["asr_enabled"] = False
        elif mode in ("visual_asr", "asr_only"):
            overrides["asr_enabled"] = True
        return replace(self._config, **overrides) if overrides else self._config

    async def search_kis(self, request: KisSearchRequest) -> SearchResponse:
        return await run_in_threadpool(
            tracks.search_kis, request, self._resolve(request)
        )

    async def search_qa(self, request: QaSearchRequest) -> SearchResponse:
        return await run_in_threadpool(
            tracks.search_qa, request, self._resolve(request)
        )

    async def search_trake(self, request: TrakeSearchRequest) -> SearchResponse:
        return await run_in_threadpool(
            tracks.search_trake, request, self._resolve(request)
        )

    async def decompose(self, request: DecomposeRequest) -> DecomposeResponse:
        """Split a pasted query for review. No retrieval, only the LLM calls."""
        return await run_in_threadpool(
            decompose_query.decompose,
            request.query,
            request.max_events,
            self._config,
            Timings(),
        )
