from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.schemas.ingestions import IngestionEntity

_COMMON_INDEXES: dict[str, qmodels.PayloadSchemaType] = {
    "video_id": qmodels.PayloadSchemaType.KEYWORD,
    "shot_id": qmodels.PayloadSchemaType.INTEGER,
    # Narrowing filters. During a run the operator recognises the programme or
    # an object long before they find the exact frame, so these are the fields
    # that convert "somewhere in 177k frames" into "somewhere in a few
    # hundred". Without an index Qdrant falls back to a full scan of the
    # collection for every one of them.
    "objects": qmodels.PayloadSchemaType.KEYWORD,
    "asr_entities": qmodels.PayloadSchemaType.KEYWORD,
    "channel_id": qmodels.PayloadSchemaType.KEYWORD,
    "author": qmodels.PayloadSchemaType.KEYWORD,
    "publish_date": qmodels.PayloadSchemaType.KEYWORD,
}

PAYLOAD_INDEXES: dict[IngestionEntity, dict[str, qmodels.PayloadSchemaType]] = {
    IngestionEntity.FRAMES: {
        **_COMMON_INDEXES,
        "original_frame_id": qmodels.PayloadSchemaType.INTEGER,
        "keyframe_n": qmodels.PayloadSchemaType.INTEGER,
    },
    IngestionEntity.CLIPS: {
        **_COMMON_INDEXES,
        "start_frame": qmodels.PayloadSchemaType.INTEGER,
        "end_frame": qmodels.PayloadSchemaType.INTEGER,
    },
}

# Full-text payload indexes are declared separately: they need a tokenizer
# config rather than a bare schema type. These back substring/phrase filters
# on speech; ranking by relevance is the sparse vectors' job, not theirs.
_TEXT_FIELDS = ("asr_text", "asr_text_corrected", "ocr_text", "title")

_TEXT_INDEX = qmodels.TextIndexParams(
    type=qmodels.TextIndexType.TEXT,
    # Vietnamese words are written as space-separated syllables, so a word
    # tokenizer indexes syllables, not words. That is the correct unit here:
    # matching "sông Cửu Long" then relies on all three syllables being
    # present rather than on a segmenter this pipeline does not have.
    tokenizer=qmodels.TokenizerType.WORD,
    min_token_len=1,
    max_token_len=20,
    lowercase=True,
)


def create_payload_indexes(
    client: QdrantClient, collection_name: str, entity: IngestionEntity
) -> None:
    for field_name, field_schema in PAYLOAD_INDEXES[entity].items():
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )
    for field_name in _TEXT_FIELDS:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=_TEXT_INDEX,
        )
