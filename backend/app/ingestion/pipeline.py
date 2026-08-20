from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.features import sparse
from app.features.profiles import (
    DEFAULT_TEXT_PROFILE,
    embedding_dimension,
    get_profile,
)
from app.ingestion import embedder, manifest as manifest_module
from app.schemas.ingestions import IngestionEntity
from app.vector_store import collections, payload_indexes, upsert
from app.vector_store.client import get_qdrant_client


class ManifestNotFoundError(Exception):
    pass


def _sparse_vectors(
    row: manifest_module.ManifestRow, method: str = "bm25"
) -> dict[str, sparse.SparseVector]:
    """Lexical vectors for one row, keyed by the slot they belong in.

    Each text source gets its own vector rather than being pooled into one.
    Pooling would let a long passage of speech swamp a short OCR line, and the
    OCR line is usually the more precise signal — it is what the broadcast
    itself chose to write down. Separate vectors also mean each source
    contributes its own ranked list to the fusion, so one bad source degrades a
    query instead of poisoning it.

    Frames return nothing while OCR is unavailable. Qdrant registers a sparse
    vector's IDF statistics from the points that use it, so writing an empty
    vector would skew the term statistics without making anything findable.
    Speech is deliberately absent from frames entirely: it lives in its own
    collection and reaches frames through the overlap bonus.
    """
    if isinstance(row, manifest_module.AsrSegmentManifestRow):
        # Entity mentions ride along with the transcript they came from. They
        # are the tokens a competition query actually hangs on, and repeating
        # them lifts their term frequency in the segment that names them.
        return {
            collections.SPARSE_SPEECH: sparse.encode(
                row.text_corrected,
                " ".join(row.entity_terms()),
                method=method,
            )
        }

    if getattr(row, "ocr_text", ""):
        return {collections.SPARSE_OCR: sparse.encode(row.ocr_text, method=method)}
    return {}


def validate_manifest(manifest_path: str, entity: IngestionEntity) -> int:
    if not Path(manifest_path).is_file():
        raise ManifestNotFoundError(f"manifest not found: {manifest_path}")

    manifest_module.validate_columns(manifest_path, entity)
    return manifest_module.count_rows(manifest_path)


# What each entity's collection declares, and which slot its job writes into.
#
# `populated` is sized from the job's own profile, so the same manifest can be
# ingested twice under different models into different collections and compared.
# `reserved` slots hold no vectors yet and are sized from a default profile;
# they exist because Qdrant cannot add a vector to a collection that already
# exists, making a slot declared now a re-upsert later instead of re-embedding
# every point.
@dataclass(frozen=True)
class _Layout:
    populated: str
    kind: str
    reserved: tuple[tuple[str, str], ...] = ()
    sparse: tuple[str, ...] = ()


_FRAME_LAYOUT = _Layout(
    populated=collections.DENSE_VECTOR_NAME,
    kind="image",
    reserved=((collections.DENSE_TEXT_NAME, DEFAULT_TEXT_PROFILE),),
    sparse=collections.FRAME_SPARSE_VECTORS,
)

_LAYOUTS: dict[IngestionEntity, _Layout] = {
    IngestionEntity.FRAMES: _FRAME_LAYOUT,
    IngestionEntity.CLIPS: _FRAME_LAYOUT,
    IngestionEntity.ASR_SEGMENTS: _Layout(
        populated=collections.DENSE_TEXT_NAME,
        kind="text",
        sparse=collections.ASR_SPARSE_VECTORS,
    ),
}


def dense_vector_name(entity: IngestionEntity) -> str:
    """The slot this entity's points write their dense vector into."""
    return _LAYOUTS[entity].populated


def create_collection(
    collection_name: str, feature_profile: str, entity: IngestionEntity
) -> None:
    layout = _LAYOUTS[entity]
    profile = get_profile(feature_profile)

    # An image profile cannot embed a speech segment, or the reverse. The
    # dimensions alone would not catch it, and the failure downstream is an
    # obscure shape error thousands of rows into the run rather than a refusal
    # here.
    if profile.kind != layout.kind:
        raise ValueError(
            f"{entity.value} needs a '{layout.kind}' profile for "
            f"'{layout.populated}'; '{feature_profile}' is '{profile.kind}'"
        )

    dense_vectors = {layout.populated: profile.dimension}
    for name, reserved_profile in layout.reserved:
        dense_vectors.setdefault(name, embedding_dimension(reserved_profile))

    client = get_qdrant_client()
    collections.create_collection(
        client,
        collection_name,
        dense_vectors=dense_vectors,
        sparse_vectors=layout.sparse,
    )


def create_payload_indexes(collection_name: str, entity: IngestionEntity) -> None:
    client = get_qdrant_client()
    payload_indexes.create_payload_indexes(client, collection_name, entity)


def upsert_points(
    collection_name: str,
    manifest_path: str,
    entity: IngestionEntity,
    feature_profile: str,
    on_progress: Callable[[int], None],
) -> None:
    client = get_qdrant_client()
    dense_name = dense_vector_name(entity)
    batch_size = max(1, settings.QDRANT_BATCH_SIZE)
    completed = 0

    for rows in _chunked(manifest_module.iter_rows(manifest_path, entity), batch_size):
        # One forward pass for the whole chunk. Embedding row by row was the
        # single largest cost in this stage: it ran the image model at batch
        # size 1 and left the GPU idle between frames.
        vectors = embedder.embed_rows(feature_profile, rows)
        points = [
            upsert.make_point(
                point_id=upsert.deterministic_point_id(*row.point_parts()),
                vector=vector,
                payload=row.payload(),
                sparse_vectors=_sparse_vectors(row, settings.SPARSE_METHOD),
                dense_name=dense_name,
            )
            for row, vector in zip(rows, vectors, strict=True)
        ]
        completed += upsert.upsert_points(client, collection_name, points)
        on_progress(completed)


def _chunked(
    rows: Iterator[manifest_module.ManifestRow], size: int
) -> Iterator[list[manifest_module.ManifestRow]]:
    """Group a row stream into lists, without materialising the manifest."""
    batch: list[manifest_module.ManifestRow] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def optimize_collection(collection_name: str) -> None:
    client = get_qdrant_client()
    collections.optimize_collection(client, collection_name)
