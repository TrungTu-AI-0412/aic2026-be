from qdrant_client import QdrantClient

from app.core.config import settings

_client: QdrantClient | None = None


def build_client(url: str | None = None, api_key: str | None = None) -> QdrantClient:
    return QdrantClient(
        url=url or settings.QDRANT_URL,
        api_key=api_key or settings.QDRANT_API_KEY or None,
    )


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = build_client()
    return _client
