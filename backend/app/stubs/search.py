from uuid import uuid4

from app.schemas.search import (
    KisSearchRequest,
    QaSearchRequest,
    SearchResponse,
    SearchResult,
    SearchVersions,
    TrakeSearchRequest,
)


class StubSearchService:
    async def search_kis(self, request: KisSearchRequest) -> SearchResponse:
        return self._stub_response(task="kis")

    async def search_qa(self, request: QaSearchRequest) -> SearchResponse:
        return self._stub_response(task="qa", answer="5")

    async def search_trake(self, request: TrakeSearchRequest) -> SearchResponse:
        return self._stub_response(task="trake")

    def _stub_response(self, task: str, answer: str | None = None) -> SearchResponse:
        return SearchResponse(
            request_id=str(uuid4()),
            task=task,
            results=[
                SearchResult(
                    rank=1,
                    video_id="L01_V001",
                    frame_ids=[505],
                    answer=answer,
                    score=0.91,
                )
            ],
            versions=SearchVersions(
                frames_collection="stub-frames-v1",
                clips_collection="stub-clips-v1",
                model_config_name="stub-v1",
            ),
            latency_ms={
                "encode": 0,
                "qdrant": 0,
                "rerank": 0,
                "refine": 0,
            },
        )
