from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

PAYLOAD_INDEXES: dict[str, qmodels.PayloadSchemaType] = {
    "video_id": qmodels.PayloadSchemaType.KEYWORD,
    "frame_id": qmodels.PayloadSchemaType.INTEGER,
}


def create_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    for field_name, field_schema in PAYLOAD_INDEXES.items():
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )
