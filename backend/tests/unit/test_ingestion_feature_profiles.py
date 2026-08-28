import pytest

from app.core.config import settings
from app.ingestion.service import SqliteIngestionService
from app.schemas.ingestions import CreateIngestionJobRequest, IngestionEntity
from app.services.ingestions import (
    ManifestPathNotAllowedError,
    UnsupportedFeatureProfileError,
)


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def service(tmp_path):
    return SqliteIngestionService(
        db_path=str(tmp_path / "ingestion.db"),
        data_root=str(tmp_path),
    )


async def test_lists_registered_feature_profiles(service):
    response = await service.list_feature_profiles()

    assert [profile.name for profile in response.profiles] == [
        "clip-b32-v1",
        "jina-clip-v2",
        "qwen3-embed-0.6b-v1",
        "siglip2-giant-opt-patch16-384-v1",
        "siglip2-so400m-patch14-384-v1",
    ]
    # Read from settings, not hardcoded: the default is whatever the deployment
    # configures, and a literal here just asserts the contents of `.env`.
    assert response.default_profile == settings.FEATURE_PROFILE
    assert response.profiles[0].model_id == "openai/clip-vit-base-patch32"
    assert response.profiles[0].dimension == 512


async def test_rejects_unregistered_feature_profile_before_queuing(service, tmp_path):
    manifest = tmp_path / "frames.parquet"
    manifest.touch()

    with pytest.raises(UnsupportedFeatureProfileError, match="supported"):
        await service.create_job(
            CreateIngestionJobRequest(
                entity=IngestionEntity.FRAMES,
                manifest_path=str(manifest),
                collection_name="aic2026-frames-invalid",
                feature_profile="not-a-real-profile",
            )
        )


async def test_relative_manifest_path_resolves_against_data_root(service, tmp_path):
    # `ingest_all.sh` passes paths relative to INGESTION_DATA_ROOT; resolving
    # them against the API process CWD instead rejected every one of them.
    (tmp_path / "manifests-v2").mkdir()
    (tmp_path / "manifests-v2/frames.parquet").touch()

    response = await service.create_job(
        CreateIngestionJobRequest(
            entity=IngestionEntity.FRAMES,
            manifest_path="manifests-v2/frames.parquet",
            collection_name="aic2026-frames-relative",
            feature_profile="clip-b32-v1",
        )
    )

    assert response.job_id.startswith("ing-")


async def test_rejects_manifest_path_escaping_data_root(service):
    with pytest.raises(ManifestPathNotAllowedError, match="outside"):
        await service.create_job(
            CreateIngestionJobRequest(
                entity=IngestionEntity.FRAMES,
                manifest_path="../etc/passwd",
                collection_name="aic2026-frames-escape",
                feature_profile="clip-b32-v1",
            )
        )
