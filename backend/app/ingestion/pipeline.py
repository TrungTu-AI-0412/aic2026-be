from collections.abc import Callable
from pathlib import Path

from app.features.profiles import embedding_dimension
from app.ingestion import embedder, manifest as manifest_module
from app.schemas.ingestions import IngestionEntity
from app.vector_store import collections, payload_indexes, upsert
from app.vector_store.client import get_qdrant_client


class ManifestNotFoundError(Exception):
    pass


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
