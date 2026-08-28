from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.schemas.ingestions import IngestionEntity

_VIDEO_INDEXES: dict[str, qmodels.PayloadSchemaType] = {
    "video_id": qmodels.PayloadSchemaType.KEYWORD,
    "channel_id": qmodels.PayloadSchemaType.KEYWORD,
    "author": qmodels.PayloadSchemaType.KEYWORD,
}

# Narrowing filters. During a run the operator recognises the programme or an
# object long before they find the exact frame, so these are the fields that
# turn "somewhere in 293k frames" into "somewhere in a few hundred". Without an
# index Qdrant falls back to a full scan for every one of them.
_SHOT_INDEXES: dict[str, qmodels.PayloadSchemaType] = {
    **_VIDEO_INDEXES,
    "shot_id": qmodels.PayloadSchemaType.INTEGER,
    "objects": qmodels.PayloadSchemaType.KEYWORD,
}

PAYLOAD_INDEXES: dict[IngestionEntity, dict[str, qmodels.PayloadSchemaType]] = {
    IngestionEntity.FRAMES: {
        **_SHOT_INDEXES,
        "original_frame_id": qmodels.PayloadSchemaType.INTEGER,
        "keyframe_n": qmodels.PayloadSchemaType.INTEGER,
        # Float, not integer: the ASR overlap bonus and any time-window filter
        # work in seconds, and a shot boundary is not on a second boundary.
        "shot_start_sec": qmodels.PayloadSchemaType.FLOAT,
        "shot_end_sec": qmodels.PayloadSchemaType.FLOAT,
    },
    IngestionEntity.CLIPS: {
        **_SHOT_INDEXES,
        "start_frame": qmodels.PayloadSchemaType.INTEGER,
        "end_frame": qmodels.PayloadSchemaType.INTEGER,
    },
    IngestionEntity.ASR_SEGMENTS: {
        **_VIDEO_INDEXES,
        "segment": qmodels.PayloadSchemaType.INTEGER,
        "start_sec": qmodels.PayloadSchemaType.FLOAT,
        "end_sec": qmodels.PayloadSchemaType.FLOAT,
        # Entity types stay separate so a filter on a person cannot be
        # satisfied by a location that happens to share the name.
        "asr_persons": qmodels.PayloadSchemaType.KEYWORD,
        "asr_orgs": qmodels.PayloadSchemaType.KEYWORD,
        "asr_locations": qmodels.PayloadSchemaType.KEYWORD,
    },
}

# Full-text payload indexes are declared separately: they need a tokenizer
# config rather than a bare schema type. These back substring/phrase filters;
# ranking by relevance is the sparse vectors' job, not theirs.
#
# Per entity, because indexing a field the entity never writes creates an index
# over nothing — and the set genuinely differs now that speech is its own
# collection rather than a column on every frame.
_TEXT_FIELDS: dict[IngestionEntity, tuple[str, ...]] = {
    IngestionEntity.FRAMES: ("ocr_text", "title"),
    IngestionEntity.CLIPS: ("ocr_text", "title"),
    IngestionEntity.ASR_SEGMENTS: ("text_corrected",),
}

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
    for field_name in _TEXT_FIELDS[entity]:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=_TEXT_INDEX,
        )
