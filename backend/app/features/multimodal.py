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
        # Takes the string itself and returns a numpy array already, so there
        # is no processor step and nothing to move to the device by hand.
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


def embed_images(
    profile: FeatureProfile, images: Sequence[np.ndarray]
) -> list[float]:
    """Encode and mean-pool one or more images into a unit vector."""
    runtime = _load_runtime(profile)
    chunks: list[np.ndarray] = []

    for start in range(0, len(images), profile.image_batch_size):
        batch = images[start : start + profile.image_batch_size]
        if profile.api == "jina":
            with runtime.torch.inference_mode():
                features = runtime.model.encode_image(_as_pil(batch))
            chunks.append(np.asarray(features))
            continue
        inputs = runtime.processor(images=list(batch), return_tensors="pt")
        inputs = _move_inputs(inputs, runtime)
        with runtime.torch.inference_mode():
            features = runtime.model.get_image_features(**inputs)
        chunks.append(_as_numpy(features))

    return pool_features(np.concatenate(chunks, axis=0), profile.dimension)


def _warn_if_truncated(
    runtime: "_ModelRuntime", profile: FeatureProfile, text: str
) -> None:
    """Say something when the text tower is about to drop the tail of a query.

    `truncation=True` below is silent: a query longer than the tower's context
    comes back as a perfectly ordinary ranking computed from its first N
    tokens. SigLIP2 gives 64 of them, and a Vietnamese word with diacritics
    often costs two or three, so a rewritten query overruns that far sooner
    than its word count suggests.

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


@lru_cache(maxsize=None)
def _load_runtime(profile: FeatureProfile) -> _ModelRuntime:
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
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
            # No AutoProcessor: the remote code bundles its own tokenizer and
            # image transform behind `encode_text` / `encode_image`.
            processor = None
            model = AutoModel.from_pretrained(
                profile.model_id,
                trust_remote_code=profile.trust_remote_code,
                torch_dtype=dtype,
            )
        else:
            processor = AutoProcessor.from_pretrained(profile.model_id)
            model = AutoModel.from_pretrained(profile.model_id, torch_dtype=dtype)
        model = model.to(device).eval()
    except (OSError, ValueError) as exc:
        raise FeatureExtractionError(
            f"cannot load embedding model '{profile.model_id}': {exc}"
        ) from exc

    return _ModelRuntime(processor, model, torch, device, dtype)


def _move_inputs(inputs: Any, runtime: _ModelRuntime) -> dict[str, Any]:
    moved = {}
    for name, value in inputs.items():
        value = value.to(runtime.device)
        if name == "pixel_values":
            value = value.to(dtype=runtime.dtype)
        moved[name] = value
    return moved
