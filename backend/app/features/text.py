"""Dense text embeddings for the speech collection.

Separate from `app.features.multimodal` because it answers a different
question. SigLIP2 puts images and short captions in one space, which is what
finding a picture from a description needs. ASR retrieval is text against text:
a Vietnamese query against fluent Vietnamese speech, often a couple of hundred
characters of it. SigLIP2's text tower is trained on caption-length input and
loses that; a multilingual text encoder does not.

The output is a plain list of unit vectors, so `app/vector_store/` decides what
to do with them.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.features.errors import FeatureExtractionError
from app.features.profiles import FeatureProfile, get_profile

# Speech segments run to a few hundred characters; 512 tokens covers the
# longest in this corpus without truncating anything that matters.
DEFAULT_MAX_LENGTH = 512

# Chosen from measurement, not guessed: batch-64 of short strings is ~0.04s on
# an L40S, so the 36k segments in this corpus embed in well under a minute and
# there is nothing to gain from tuning it further.
DEFAULT_BATCH_SIZE = 64


@dataclass(frozen=True)
class _TextRuntime:
    tokenizer: Any
    model: Any
    torch: Any
    device: Any
    dtype: Any


@lru_cache(maxsize=None)
def _load_runtime(profile: FeatureProfile) -> _TextRuntime:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
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
        # Left padding is load-bearing. Qwen3-Embedding pools the *last* token
        # rather than the mean, so with right padding the final position of a
        # short sequence is a pad token and its embedding is garbage.
        tokenizer = AutoTokenizer.from_pretrained(
            profile.model_id, padding_side="left"
        )
        model = AutoModel.from_pretrained(profile.model_id, dtype=dtype)
        model = model.to(device).eval()
    except (OSError, ValueError) as exc:
        raise FeatureExtractionError(
            f"cannot load text embedding model '{profile.model_id}': {exc}"
        ) from exc

    return _TextRuntime(tokenizer, model, torch, device, dtype)


def embed_texts(
    feature_profile: str,
    texts: Sequence[str],
    max_length: int = DEFAULT_MAX_LENGTH,
) -> list[list[float]]:
    """Encode each text into its own unit vector, in input order.

    One vector per input, unlike `multimodal.embed_images`, which mean-pools a
    clip's frames into a single vector. Segments are independent documents.
    """
    if not texts:
        return []

    profile = get_profile(feature_profile)
    runtime = _load_runtime(profile)
    torch = runtime.torch

    # A blank string tokenises to nothing but must still yield a vector, or the
    # returned list would silently stop lining up with the input rows.
    prepared = [text if text.strip() else " " for text in texts]

    inputs = runtime.tokenizer(
        prepared,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {name: value.to(runtime.device) for name, value in inputs.items()}

    with torch.inference_mode():
        hidden = runtime.model(**inputs).last_hidden_state[:, -1]

    return normalize_rows(hidden.float(), profile.dimension, torch)


def embed_query(feature_profile: str, text: str) -> list[float]:
    """Encode one query into the speech collection's text space."""
    if not text.strip():
        raise ValueError("text query must not be empty")
    return embed_texts(feature_profile, [text])[0]


def normalize_rows(values: Any, expected_dimension: int, torch: Any) -> list[list[float]]:
    """L2-normalise each row, so cosine similarity is a plain dot product."""
    if values.ndim != 2 or values.shape[0] == 0:
        raise FeatureExtractionError(
            f"text model returned invalid shape {tuple(values.shape)}"
        )
    if values.shape[1] != expected_dimension:
        raise FeatureExtractionError(
            f"text model returned dimension {values.shape[1]}, "
            f"expected {expected_dimension}"
        )
    if not bool(torch.isfinite(values).all()):
        raise FeatureExtractionError("text model returned non-finite values")

    norms = values.norm(dim=-1)
    if bool((norms == 0).any()):
        raise FeatureExtractionError("text model returned a zero vector")

    return (values / norms.unsqueeze(-1)).cpu().tolist()
