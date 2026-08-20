import pytest

from app.retrieval import tracks
from app.retrieval.engine import RetrievalConfig
from app.retrieval.service import QdrantSearchService
from app.schemas.search import KisSearchRequest

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def captured(monkeypatch):
    """Capture the config each track actually receives."""
    seen = {}

    def fake_kis(request, config):
        seen["config"] = config
        return "response"

    monkeypatch.setattr(tracks, "search_kis", fake_kis)
    return seen


def service() -> QdrantSearchService:
    return QdrantSearchService(
        RetrievalConfig(
            frames_collection="frames",
            feature_profile="clip-b32-v1",
            asr_collection="asr",
            asr_weight=0.3,
            asr_dense_weight=0.7,
            asr_sparse_weight=0.3,
        )
    )


async def test_request_without_overrides_uses_the_configured_values(captured):
    await service().search_kis(KisSearchRequest(task="kis", description="xin chào"))

    config = captured["config"]
    assert config.asr_enabled is True
    assert config.asr_weight == 0.3


async def test_weight_override_reaches_the_engine(captured):
    await service().search_kis(
        KisSearchRequest(task="kis", description="xin chào", asr_weight=0.9)
    )

    assert captured["config"].asr_weight == 0.9


async def test_the_bonus_can_be_switched_off_per_request(captured):
    await service().search_kis(
        KisSearchRequest(task="kis", description="xin chào", asr_enabled=False)
    )

    assert captured["config"].asr_enabled is False


async def test_an_override_does_not_leak_into_the_next_request(captured):
    """The config is frozen and replaced per request, so tuning one query must
    not silently retune every query after it."""
    svc = service()

    await svc.search_kis(
        KisSearchRequest(task="kis", description="a", asr_weight=0.9)
    )
    assert captured["config"].asr_weight == 0.9

    await svc.search_kis(KisSearchRequest(task="kis", description="b"))
    assert captured["config"].asr_weight == 0.3


async def test_unrelated_settings_are_preserved(captured):
    await service().search_kis(
        KisSearchRequest(task="kis", description="a", asr_weight=0.5)
    )

    config = captured["config"]
    assert config.frames_collection == "frames"
    assert config.asr_collection == "asr"
    assert config.asr_dense_weight == 0.7


async def test_weight_above_one_is_rejected_before_the_engine():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        KisSearchRequest(task="kis", description="a", asr_weight=1.5)
