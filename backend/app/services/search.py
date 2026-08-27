from typing import Protocol

from app.schemas.search import (
    DecomposeRequest,
    DecomposeResponse,
    KisSearchRequest,
    QaSearchRequest,
    SearchResponse,
    TrakeSearchRequest,
)

class SearchService(Protocol):
    async def search_kis(self, request: KisSearchRequest) -> SearchResponse:
        ...

    async def search_qa(self, request: QaSearchRequest) -> SearchResponse:
        ...

    async def search_trake(self, request: TrakeSearchRequest) -> SearchResponse:
        ...

    async def decompose(self, request: DecomposeRequest) -> DecomposeResponse:
        ...
