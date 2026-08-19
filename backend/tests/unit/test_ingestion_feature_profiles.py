import pytest

from app.core.config import settings
from app.ingestion.service import SqliteIngestionService
from app.schemas.ingestions import CreateIngestionJobRequest, IngestionEntity
from app.services.ingestions import UnsupportedFeatureProfileError


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
