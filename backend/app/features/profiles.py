from dataclasses import dataclass

from app.features.errors import UnknownFeatureProfileError


@dataclass(frozen=True)
class FeatureProfile:
    """Versioned contract shared by ingestion and text-query encoding."""

    model_id: str
    dimension: int
    clip_frame_count: int = 8
    # Frames sampled *within one clip*, pooled into a single vector. Small
    # because a clip only has `clip_frame_count` of them.
    image_batch_size: int = 4
    # Independent images per forward pass when embedding keyframes, each keeping
    # its own vector. Far larger: this is the knob that decides whether the GPU
    # is busy, and at batch 4 the model spends most of its time idle between
    # passes. Bounded by GPU memory, so a bigger model gets a smaller value.
    embed_batch_size: int = 32
    # "image" profiles embed pixels into a shared image/text space; "text"
    # profiles embed text against text. They are not interchangeable, and the
    # dimension alone does not distinguish them, so entities that index frames
    # and entities that index speech each require their own kind.
    kind: str = "image"


# SigLIP 2 Giant is the highest-capacity retrieval profile. So400m is kept as a
# lower-memory alternative, and CLIP B/32 remains available so existing jobs do
# not silently change vector size. Each profile's image and text encoders share
# one embedding space.
FEATURE_PROFILES: dict[str, FeatureProfile] = {
    "siglip2-giant-opt-patch16-384-v1": FeatureProfile(
        model_id="google/siglip2-giant-opt-patch16-384",
        dimension=1536,
        image_batch_size=2,
        embed_batch_size=16,
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
    # Multilingual text retrieval for ASR segments. Qwen3-Embedding pools the
    # last token rather than the mean (`1_Pooling/config.json` sets
    # `pooling_mode_lasttoken`), which is why `features.text` left-pads.
    "qwen3-embed-0.6b-v1": FeatureProfile(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        dimension=1024,
        kind="text",
    ),
}

# Profiles used to size a *reserved* slot, one nothing populates yet. The slot a
# job actually writes is sized from that job's own profile, so two collections
# can be built from the same manifest with different models and compared.
DEFAULT_IMAGE_PROFILE = "siglip2-giant-opt-patch16-384-v1"
DEFAULT_TEXT_PROFILE = "qwen3-embed-0.6b-v1"


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
