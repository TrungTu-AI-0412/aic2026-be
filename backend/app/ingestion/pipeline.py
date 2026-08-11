from collections.abc import Callable
from pathlib import Path

from app.ingestion import manifest as manifest_module


class ManifestNotFoundError(Exception):
    pass


def validate_manifest(manifest_path: str) -> int:
    if not Path(manifest_path).is_file():
        raise ManifestNotFoundError(f"manifest not found: {manifest_path}")

    manifest_module.validate_columns(manifest_path)
    return manifest_module.count_rows(manifest_path)


def create_collection(collection_name: str, feature_profile: str) -> None:
    raise NotImplementedError(
        "Qdrant collection creation belongs in vector_store/ - wire it in here"
    )


def create_payload_indexes(collection_name: str) -> None:
    raise NotImplementedError("payload index creation is not implemented yet")


def upsert_points(
    collection_name: str,
    manifest_path: str,
    feature_profile: str,
    on_progress: Callable[[int], None],
) -> None:
    completed = 0
    for row in manifest_module.iter_rows(manifest_path):
        _embed_and_upsert(collection_name, feature_profile, row)
        completed += 1
        on_progress(completed)


def _embed_and_upsert(
    collection_name: str, feature_profile: str, row: manifest_module.ManifestRow
) -> None:
    raise NotImplementedError(
        "feature extraction + Qdrant upsert is not implemented yet - wire "
        "in the embedding model and vector_store/ client here"
    )


def optimize_collection(collection_name: str) -> None:
    raise NotImplementedError("collection optimization is not implemented yet")
