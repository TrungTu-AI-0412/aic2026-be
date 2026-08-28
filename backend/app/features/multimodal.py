import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from app.features.errors import FeatureExtractionError
from app.features.profiles import FeatureProfile, get_profile

logger = logging.getLogger(__name__)


def embed_text(feature_profile: str, text: str) -> list[float]:
    """Create a query vector in the same space used for image ingestion."""
    if not text.strip():
        raise ValueError("text query must not be empty")

    profile = get_profile(feature_profile)
    runtime = _load_runtime(profile)

    if profile.api == "jina":
        # Takes the string itself and returns an array already, so there is no
        # processor step and nothing to move to the device by hand. Truncation
        # is the model's own, at 8192 tokens, which nothing here approaches.
        with runtime.torch.inference_mode():
            features = runtime.model.encode_text([text])
        return pool_features(np.asarray(features), profile.dimension)

    _warn_if_truncated(runtime, profile, text)
    inputs = runtime.processor(
        text=[text],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = _move_inputs(inputs, runtime)

    with runtime.torch.inference_mode():
        features = runtime.model.get_text_features(**inputs)

    return pool_features(_as_numpy(features), profile.dimension)


def encode_images(
    profile: FeatureProfile,
    images: Sequence[np.ndarray],
    batch_size: int | None = None,
) -> np.ndarray:
    """Raw image features, one row per image, in input order.

    Preprocessing is handed straight to the accelerator: the fast processor
    accepts a `device`, which keeps the 384x384 resize off the four CPU cores
    that are also feeding it JPEGs.
    """
    if not images:
        raise FeatureExtractionError("no images to embed")

    runtime = _load_runtime(profile)
    step = batch_size or profile.image_batch_size
    chunks: list[np.ndarray] = []

    for start in range(0, len(images), step):
        batch = images[start : start + step]
        if profile.api == "jina":
            with runtime.torch.inference_mode():
                features = runtime.model.encode_image(_as_pil(batch))
            chunks.append(np.asarray(features))
            continue
        if runtime.image_processor is not None:
            inputs = runtime.image_processor(
                images=list(batch),
                return_tensors="pt",
                device=str(runtime.device),
            )
        else:
            inputs = runtime.processor(images=list(batch), return_tensors="pt")
        inputs = _move_inputs(inputs, runtime)
        with runtime.torch.inference_mode():
            features = runtime.model.get_image_features(**inputs)
        chunks.append(_as_numpy(features))

    return np.concatenate(chunks, axis=0)


def embed_images(
    profile: FeatureProfile, images: Sequence[np.ndarray]
) -> list[float]:
    """Encode and mean-pool one or more images into a unit vector.

    Used for clips, where several sampled frames stand for one shot.
    """
    return pool_features(encode_images(profile, images), profile.dimension)


def embed_images_each(
    profile: FeatureProfile,
    images: Sequence[np.ndarray],
    batch_size: int | None = None,
) -> list[list[float]]:
    """One unit vector per image, rather than one for the whole batch.

    This is what lets keyframes share a forward pass. `embed_images` cannot:
    it mean-pools everything handed to it, so batching independent keyframes
    through it would average unrelated frames into a single vector.
    """
    features = encode_images(profile, images, batch_size or profile.embed_batch_size)
    return [
        normalize_one(row, profile.dimension) for row in np.asarray(features, np.float32)
    ]


def normalize_one(values: np.ndarray, expected_dimension: int) -> list[float]:
    """L2-normalise a single feature row, with the same guards as pooling."""
    if values.ndim != 1 or values.shape[0] != expected_dimension:
        raise FeatureExtractionError(
            f"embedding dimension mismatch: expected {expected_dimension}, "
            f"got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise FeatureExtractionError("embedding contains non-finite values")

    norm = np.linalg.norm(values)
    if norm == 0:
        raise FeatureExtractionError("embedding model returned a zero vector")
    return (values / norm).tolist()


def _warn_if_truncated(
    runtime: "_ModelRuntime", profile: FeatureProfile, text: str
) -> None:
    """Say something when the text tower is about to drop the tail of a query.

    `truncation=True` in `embed_text` is silent: a query longer than the
    tower's context comes back as a perfectly ordinary ranking computed from
    its first N tokens. SigLIP2 gives 64 of them, and a Vietnamese word with
    diacritics often costs two or three, so a rewritten query overruns that far
    sooner than its word count suggests.

    This does not raise. A truncated query still returns useful results, and
    failing a competition query outright would be worse than answering it from
    a prefix. It just has to stop being invisible.
    """
    tokenizer = getattr(runtime.processor, "tokenizer", None)
    if tokenizer is None:
        return
    length = len(tokenizer(text, truncation=False)["input_ids"])
    if length > profile.max_text_tokens:
        logger.warning(
            "query truncated: %d tokens for '%s', which accepts %d; "
            "the last %d tokens do not reach the encoder",
            length,
            profile.model_id,
            profile.max_text_tokens,
            length - profile.max_text_tokens,
        )


def _as_pil(images: Sequence[np.ndarray]) -> list:
    """Jina's encoder takes PIL images, not the arrays the rest of this uses."""
    from PIL import Image

    return [
        image if not isinstance(image, np.ndarray) else Image.fromarray(image)
        for image in images
    ]


def pool_features(features: np.ndarray, expected_dimension: int) -> list[float]:
    """Normalize individual features, mean-pool them, then normalize again."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[0] == 0:
        raise FeatureExtractionError(
            f"embedding model returned invalid shape {values.shape}"
        )
    if values.shape[1] != expected_dimension:
        raise FeatureExtractionError(
            "embedding dimension mismatch: "
            f"expected {expected_dimension}, got {values.shape[1]}"
        )
    if not np.isfinite(values).all():
        raise FeatureExtractionError("embedding contains non-finite values")

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise FeatureExtractionError("embedding model returned a zero vector")
    pooled = (values / norms).mean(axis=0)
    pooled_norm = np.linalg.norm(pooled)
    if pooled_norm == 0:
        raise FeatureExtractionError("pooled embedding is a zero vector")
    return (pooled / pooled_norm).tolist()


def _as_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


@dataclass(frozen=True)
class _ModelRuntime:
    processor: Any
    model: Any
    torch: Any
    device: Any
    dtype: Any
    # The fast (torchvision) image processor, kept separate from `processor`
    # because the text path still needs the slow one's tokenizer.
    image_processor: Any = None


@lru_cache(maxsize=None)
def _load_runtime(profile: FeatureProfile) -> _ModelRuntime:
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor
    except ImportError as exc:
        raise FeatureExtractionError(
            "embedding dependencies are missing; install requirements.txt"
        ) from exc

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
    elif (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    try:
        if profile.api == "jina":
            # No AutoProcessor and no image processor: the remote code bundles
            # its own tokenizer and image transform behind `encode_text` and
            # `encode_image`.
            model = AutoModel.from_pretrained(
                profile.model_id,
                trust_remote_code=profile.trust_remote_code,
                dtype=dtype,
            )
            return _ModelRuntime(None, model.to(device).eval(), torch, device, dtype)
        processor = AutoProcessor.from_pretrained(profile.model_id)
        # The default (slow, PIL-based) image processor resizes on the CPU and
        # measured 126 img/s against a 174 img/s GPU forward pass — it, not the
        # model, was the ingestion bottleneck. The fast processor is
        # torchvision-backed and can resize on the GPU, at ~1231 img/s.
        #
        # Loaded via AutoImageProcessor rather than passing use_fast to
        # AutoProcessor: on this model the latter routes the flag into the
        # tokenizer and fails on a missing SentencePiece backend.
        image_processor = AutoImageProcessor.from_pretrained(
            profile.model_id, use_fast=True
        )
        model = AutoModel.from_pretrained(profile.model_id, dtype=dtype)
        model = model.to(device).eval()
    except (OSError, ValueError) as exc:
        raise FeatureExtractionError(
            f"cannot load embedding model '{profile.model_id}': {exc}"
        ) from exc

    return _ModelRuntime(
        processor, model, torch, device, dtype, image_processor=image_processor
    )


def _move_inputs(inputs: Any, runtime: _ModelRuntime) -> dict[str, Any]:
    moved = {}
    for name, value in inputs.items():
        value = value.to(runtime.device)
        if name == "pixel_values":
            value = value.to(dtype=runtime.dtype)
        moved[name] = value
    return moved
