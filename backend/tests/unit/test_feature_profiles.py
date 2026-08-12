import pytest

from app.features import profiles
from app.features.errors import UnknownFeatureProfileError


def test_siglip2_giant_dimension_matches_model_projection():
    assert profiles.embedding_dimension("siglip2-giant-opt-patch16-384-v1") == 1536


def test_siglip2_so400m_dimension_matches_model_projection():
    assert profiles.embedding_dimension("siglip2-so400m-patch14-384-v1") == 1152


def test_unknown_profile_lists_supported_profiles():
    with pytest.raises(UnknownFeatureProfileError, match="supported"):
        profiles.get_profile("does-not-exist")
