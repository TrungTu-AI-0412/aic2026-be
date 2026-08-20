"""What each collection declares at creation time.

Worth pinning because the cost of getting it wrong is asymmetric: Qdrant cannot
add a vector to a collection that already exists, so a slot missed here is a
full re-ingest of ~293k images, while a slot declared and never filled is free
until something queries it.
"""

import pytest

from app.features.profiles import (
    DEFAULT_IMAGE_PROFILE,
    DEFAULT_TEXT_PROFILE,
    embedding_dimension,
)
from app.ingestion import pipeline
from app.schemas.ingestions import IngestionEntity
from app.vector_store import collections


class _RecordingClient:
    """Captures the create_collection call instead of talking to a server."""

    def __init__(self):
        self.kwargs = None

    def create_collection(self, **kwargs):
        self.kwargs = kwargs


def declared(entity: IngestionEntity, profile: str) -> tuple[dict, tuple]:
    """The dense/sparse slots `create_collection` would declare for a job."""
    captured = {}

    class _Client(_RecordingClient):
        def create_collection(self, **kwargs):
            captured.update(kwargs)

    import app.ingestion.pipeline as module

    original = module.get_qdrant_client
    module.get_qdrant_client = lambda: _Client()
    try:
        module.create_collection("c", profile, entity)
    finally:
        module.get_qdrant_client = original

    return (
        {name: params.size for name, params in captured["vectors_config"].items()},
        tuple(captured["sparse_vectors_config"]),
    )


class TestDeclaredVectors:
    def test_frames_declare_an_image_space_and_a_reserved_text_space(self):
        dense, sparse = declared(IngestionEntity.FRAMES, DEFAULT_IMAGE_PROFILE)

        assert dense == {
            collections.DENSE_VECTOR_NAME: embedding_dimension(DEFAULT_IMAGE_PROFILE),
            collections.DENSE_TEXT_NAME: embedding_dimension(DEFAULT_TEXT_PROFILE),
        }
        assert sparse == (collections.SPARSE_OCR,)

    def test_asr_declares_a_text_space_and_a_speech_vector(self):
        dense, sparse = declared(IngestionEntity.ASR_SEGMENTS, DEFAULT_TEXT_PROFILE)

        assert dense == {
            collections.DENSE_TEXT_NAME: embedding_dimension(DEFAULT_TEXT_PROFILE)
        }
        assert sparse == (collections.SPARSE_SPEECH,)

    def test_the_populated_slot_is_sized_from_the_jobs_own_profile(self):
        """So the same manifest can be ingested under two models and compared.
        Pinning the size to one profile would make that impossible."""
        dense, _ = declared(IngestionEntity.FRAMES, "clip-b32-v1")

        assert dense[collections.DENSE_VECTOR_NAME] == embedding_dimension(
            "clip-b32-v1"
        )

    def test_frames_declare_no_speech_slot(self):
        """Speech lives in its own collection and reaches frames through the
        overlap bonus. A frame-level speech vector would let Qdrant's fusion and
        the bonus score the same transcript twice."""
        _, sparse = declared(IngestionEntity.FRAMES, DEFAULT_IMAGE_PROFILE)

        assert collections.SPARSE_SPEECH not in sparse

    def test_no_caption_slot_anywhere(self):
        """`dense_text` covers caption text, so a sparse caption slot would be a
        second, worse copy of the same signal."""
        assert not hasattr(collections, "SPARSE_CAPTION")

        for entity, profile in (
            (IngestionEntity.FRAMES, DEFAULT_IMAGE_PROFILE),
            (IngestionEntity.ASR_SEGMENTS, DEFAULT_TEXT_PROFILE),
        ):
            _, sparse = declared(entity, profile)
            assert "caption" not in sparse

    def test_the_two_dense_spaces_have_different_dimensions(self):
        """Which is exactly why they cannot share one slot: an image vector and a
        text vector are not interchangeable even before you ask what they mean."""
        assert embedding_dimension(DEFAULT_IMAGE_PROFILE) != embedding_dimension(
            DEFAULT_TEXT_PROFILE
        )


class TestCreateCollection:
    def test_every_declared_slot_reaches_qdrant(self):
        client = _RecordingClient()
        collections.create_collection(
            client,
            "c",
            dense_vectors={"dense_video": 1536, "dense_text": 1024},
            sparse_vectors=("ocr",),
        )

        assert set(client.kwargs["vectors_config"]) == {"dense_video", "dense_text"}
        assert client.kwargs["vectors_config"]["dense_video"].size == 1536
        assert client.kwargs["vectors_config"]["dense_text"].size == 1024
        assert set(client.kwargs["sparse_vectors_config"]) == {"ocr"}

    def test_sparse_slots_use_idf_so_scoring_is_bm25_equivalent(self):
        client = _RecordingClient()
        collections.create_collection(
            client, "c", dense_vectors={"d": 8}, sparse_vectors=("speech",)
        )

        assert client.kwargs["sparse_vectors_config"]["speech"].modifier == "idf"

    def test_a_collection_with_no_dense_vector_is_refused(self):
        with pytest.raises(ValueError, match="dense vector"):
            collections.create_collection(_RecordingClient(), "c", dense_vectors={})

    def test_indexing_is_disabled_during_bulk_load(self):
        client = _RecordingClient()
        collections.create_collection(client, "c", dense_vectors={"d": 8})

        assert client.kwargs["optimizers_config"].indexing_threshold == 0


class TestProfileGuard:
    def test_a_text_profile_cannot_ingest_frames(self, monkeypatch):
        """Caught at job start rather than as a shape error thousands of rows in.
        The dimensions alone would not catch it."""
        monkeypatch.setattr(pipeline, "get_qdrant_client", lambda: _RecordingClient())

        with pytest.raises(ValueError, match="needs an? 'image' profile"):
            pipeline.create_collection("c", DEFAULT_TEXT_PROFILE, IngestionEntity.FRAMES)

    def test_an_image_profile_cannot_ingest_speech(self, monkeypatch):
        monkeypatch.setattr(pipeline, "get_qdrant_client", lambda: _RecordingClient())

        with pytest.raises(ValueError, match="needs an? 'text' profile"):
            pipeline.create_collection(
                "c", DEFAULT_IMAGE_PROFILE, IngestionEntity.ASR_SEGMENTS
            )

    def test_any_image_profile_is_accepted_for_frames(self, monkeypatch):
        """Including a smaller one: comparing profiles is a supported workflow."""
        monkeypatch.setattr(pipeline, "get_qdrant_client", lambda: _RecordingClient())

        pipeline.create_collection("c", DEFAULT_IMAGE_PROFILE, IngestionEntity.FRAMES)
        pipeline.create_collection("c", "clip-b32-v1", IngestionEntity.FRAMES)


class TestDenseVectorName:
    def test_frames_write_the_image_slot(self):
        assert (
            pipeline.dense_vector_name(IngestionEntity.FRAMES)
            == collections.DENSE_VECTOR_NAME
        )

    def test_speech_writes_the_text_slot(self):
        assert (
            pipeline.dense_vector_name(IngestionEntity.ASR_SEGMENTS)
            == collections.DENSE_TEXT_NAME
        )
