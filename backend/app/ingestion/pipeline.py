from collections.abc import Callable
from pathlib import Path

from app.features import sparse
from app.features.profiles import embedding_dimension
from app.ingestion import embedder, manifest as manifest_module
from app.schemas.ingestions import IngestionEntity
from app.vector_store import collections, payload_indexes, upsert
from app.vector_store.client import get_qdrant_client


class ManifestNotFoundError(Exception):
    pass


def _sparse_vectors(row: manifest_module.ManifestRow) -> dict[str, sparse.SparseVector]:
    """Lexical vectors for one row, one per text source.

    Speech, on-screen text and captions stay in separate vectors rather than
    being pooled into one. Pooling would let a long ASR passage swamp a short
    OCR line, and the OCR line is usually the more precise signal — it is what
    the broadcast itself chose to write down. Separate vectors also mean each
    source contributes its own ranked list to the fusion, so one bad source
    degrades a query instead of poisoning it.

    `title` and `keywords` ride along with speech: they describe the video the
    speech belongs to, and they are the only lexical signal on the 25 videos
    that carry no speech at all.
    """
    vectors = {
        collections.SPARSE_SPEECH: sparse.encode(
            row.asr_text,
            row.asr_text_corrected,
            " ".join(row.asr_entities),
            row.title,
            " ".join(row.keywords),
        )
    }
    # Only when the shot actually carries on-screen text. Qdrant registers a
    # sparse vector's IDF statistics from the points that use it, so writing an
    # empty vector for the 20% of shots with no text would add nothing to find
    # and skew the term statistics of the ones that do.
    if row.ocr_text:
        vectors[collections.SPARSE_OCR] = sparse.encode(row.ocr_text)
    return vectors


def validate_manifest(manifest_path: str, entity: IngestionEntity) -> int:
    if not Path(manifest_path).is_file():
        raise ManifestNotFoundError(f"manifest not found: {manifest_path}")

    manifest_module.validate_columns(manifest_path, entity)
    return manifest_module.count_rows(manifest_path)


def create_collection(collection_name: str, feature_profile: str) -> None:
    client = get_qdrant_client()
    collections.create_collection(
        client,
        collection_name,
        embedding_dimension(feature_profile),
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
    completed = 0
    batch = []

    for row in manifest_module.iter_rows(manifest_path, entity):
        batch.append(
            upsert.make_point(
                point_id=upsert.deterministic_point_id(*row.point_parts()),
                vector=embedder.embed_row(feature_profile, row),
                payload=row.payload(),
                sparse_vectors=_sparse_vectors(row),
            )
        )

        if len(batch) >= upsert.DEFAULT_BATCH_SIZE:
            completed += upsert.upsert_points(client, collection_name, batch)
            on_progress(completed)
            batch = []

    if batch:
        completed += upsert.upsert_points(client, collection_name, batch)
        on_progress(completed)


def optimize_collection(collection_name: str) -> None:
    client = get_qdrant_client()
    collections.optimize_collection(client, collection_name)
