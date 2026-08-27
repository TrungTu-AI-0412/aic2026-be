from dataclasses import dataclass

from app.features.errors import UnknownFeatureProfileError


@dataclass(frozen=True)
class FeatureProfile:
    """Versioned contract shared by ingestion and text-query encoding."""

    model_id: str
    dimension: int
    clip_frame_count: int = 8
    image_batch_size: int = 4
    # Which encoder API the weights expose. "hf" is the CLIP-style pair of
    # `get_text_features` / `get_image_features` that every model here used
    # until Jina; "jina" is that model's own `encode_text` / `encode_image`,
    # which arrive through `trust_remote_code` and take raw text and PIL
    # images rather than a processor's tensors.
    api: str = "hf"
    trust_remote_code: bool = False


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
    # Multilingual, and Vietnamese is one of the 89 languages it was trained
    # on. That is the reason to try it here: SigLIP2's text tower handles
    # Vietnamese poorly enough that a query typed without diacritics — which
    # is how people actually type — returned a turtle for "sạt lở bờ sông".
    # Whether it beats SigLIP2 on this corpus is unmeasured; it is ingested
    # into its own collection so the two can be compared rather than swapped.
    "jina-clip-v2": FeatureProfile(
        model_id="jinaai/jina-clip-v2",
        dimension=1024,
        image_batch_size=8,
        api="jina",
        trust_remote_code=True,
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
