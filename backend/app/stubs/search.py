from uuid import uuid4

from app.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchVersions,
)


class StubSearchService:
    async def search(self, request: SearchRequest) -> SearchResponse:
        answer = "5" if request.task == "qa" else None

        return SearchResponse(
            request_id=str(uuid4()),
            task=request.task,
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
                model_config="stub-v1",
            ),
            latency_ms={
                "encode": 0,
                "qdrant": 0,
                "rerank": 0,
                "refine": 0,
            },
        )