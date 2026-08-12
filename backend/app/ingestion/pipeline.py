from collections.abc import Callable
from pathlib import Path

from app.ingestion import manifest as manifest_module
from app.vector_store import collections, payload_indexes, upsert
from app.vector_store.client import get_qdrant_client

# Placeholder registry until feature profiles are sourced from a real model
# registry. Extend this as new embedding models are wired in.
FEATURE_PROFILE_DIMENSIONS: dict[str, int] = {
    "clip-b32-v1": 512,
}


class ManifestNotFoundError(Exception):
    pass


class UnknownFeatureProfileError(Exception):
    pass


def validate_manifest(manifest_path: str) -> int:
    if not Path(manifest_path).is_file():
        raise ManifestNotFoundError(f"manifest not found: {manifest_path}")

    manifest_module.validate_columns(manifest_path)
    return manifest_module.count_rows(manifest_path)


def create_collection(collection_name: str, feature_profile: str) -> None:
    vector_size = _feature_profile_dimension(feature_profile)
    client = get_qdrant_client()
    collections.create_collection(client, collection_name, vector_size)


def create_payload_indexes(collection_name: str) -> None:
    client = get_qdrant_client()
    payload_indexes.create_payload_indexes(client, collection_name)


def upsert_points(
    collection_name: str,
    manifest_path: str,
    feature_profile: str,
    on_progress: Callable[[int], None],
) -> None:
    client = get_qdrant_client()
    completed = 0
    batch = []

    for row in manifest_module.iter_rows(manifest_path):
        vector = _embed(feature_profile, row)
        point_id = upsert.deterministic_point_id(row.video_id, str(row.frame_id))
        batch.append(
            upsert.make_point(
                point_id=point_id,
                vector=vector,
                payload={
                    "video_id": row.video_id,
                    "frame_id": row.frame_id,
                    "path": row.path,
                },
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
    raise NotImplementedError("collection optimization is not implemented yet")


def _feature_profile_dimension(feature_profile: str) -> int:
    try:
        return FEATURE_PROFILE_DIMENSIONS[feature_profile]
    except KeyError as exc:
        raise UnknownFeatureProfileError(
            f"unknown feature_profile '{feature_profile}'"
        ) from exc


def _embed(feature_profile: str, row: manifest_module.ManifestRow) -> list[float]:
    raise NotImplementedError(
        "feature extraction is not implemented yet - wire in the embedding "
        f"model for feature_profile '{feature_profile}'"
    )
