from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.schemas.ingestions import IngestionEntity

_COMMON_INDEXES: dict[str, qmodels.PayloadSchemaType] = {
    "video_id": qmodels.PayloadSchemaType.KEYWORD,
    "shot_id": qmodels.PayloadSchemaType.INTEGER,
}

PAYLOAD_INDEXES: dict[IngestionEntity, dict[str, qmodels.PayloadSchemaType]] = {
    IngestionEntity.FRAMES: {
        **_COMMON_INDEXES,
        "original_frame_id": qmodels.PayloadSchemaType.INTEGER,
    },
    IngestionEntity.CLIPS: {
        **_COMMON_INDEXES,
        "start_frame": qmodels.PayloadSchemaType.INTEGER,
        "end_frame": qmodels.PayloadSchemaType.INTEGER,
    },
}


def create_payload_indexes(
    client: QdrantClient, collection_name: str, entity: IngestionEntity
) -> None:
    for field_name, field_schema in PAYLOAD_INDEXES[entity].items():
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )
