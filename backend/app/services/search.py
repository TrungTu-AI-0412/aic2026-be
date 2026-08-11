from typing import Protocol

from app.schemas.search import SearchRequest, SearchResponse


class SearchService(Protocol):
    async def search(self, request: SearchRequest) -> SearchResponse:                                                                                               
        ...