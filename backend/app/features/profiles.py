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
    # Which encoder API the weights expose. "hf" is the CLIP-style pair of
    # `get_text_features` / `get_image_features` that every model here used
    # until Jina; "jina" is that model's own `encode_text` / `encode_image`,
    # which arrive through `trust_remote_code` and take raw text and PIL
    # images rather than a processor's tensors.
    api: str = "hf"
    trust_remote_code: bool = False
    # How many tokens the text tower accepts. Anything past this is dropped by
    # `truncation=True` with no exception and no log, and the result still
    # looks like a normal ranking — which is why the number belongs here,
    # where `retrieval/rewrite.py` can read it, rather than hard-coded in a
    # prompt string that no longer matches when the profile changes.
    #
    # Count tokens, not words. SigLIP2's multilingual tokenizer splits an
    # accented Vietnamese word into two or three tokens, so a 40-word rewrite
    # can overrun 64 while a 40-word English one does not.
    max_text_tokens: int = 64


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
        max_text_tokens=77,
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
        # Two orders of magnitude more room than SigLIP2's 64. A rewriter
        # targeting this profile does not need to budget at all.
        max_text_tokens=8192,
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
