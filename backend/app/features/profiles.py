from dataclasses import dataclass

from app.features.errors import UnknownFeatureProfileError


@dataclass(frozen=True)
class FeatureProfile:
    """Versioned contract shared by ingestion and text-query encoding."""

    model_id: str
    dimension: int
    clip_frame_count: int = 8
    image_batch_size: int = 4


# SigLIP 2 Giant is the highest-capacity retrieval profile. So400m is kept as a
# lower-memory alternative, and CLIP B/32 remains available so existing jobs do
# not silently change vector size. Each profile's image and text encoders share
# one embedding space.
FEATURE_PROFILES: dict[str, FeatureProfile] = {
    "siglip2-giant-opt-patch16-384-v1": FeatureProfile(
        model_id="google/siglip2-giant-opt-patch16-384",
        dimension=1536,
        image_batch_size=2,
    ),
    "siglip2-so400m-patch14-384-v1": FeatureProfile(
        model_id="google/siglip2-so400m-patch14-384",
        dimension=1152,
    ),
    "clip-b32-v1": FeatureProfile(
        model_id="openai/clip-vit-base-patch32",
        dimension=512,
        image_batch_size=8,
    ),
}


def get_profile(name: str) -> FeatureProfile:
    try:
        return FEATURE_PROFILES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(FEATURE_PROFILES))
        raise UnknownFeatureProfileError(
            f"unknown feature_profile '{name}'; supported: {supported}"
        ) from exc


def embedding_dimension(name: str) -> int:
    return get_profile(name).dimension
