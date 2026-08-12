"""Second-stage cross-encoder rerank over the head of the retrieved list.

First-stage scores come from a dual encoder: the query and the image are
embedded separately and never meet, so the model cannot tell "a man holding a
red umbrella" from "a red man holding an umbrella". BLIP's image-text-matching
head cross-attends the caption over the image patches and can, at the cost of
one forward pass per candidate - far too slow for a collection, fine for the
top N.

The reranked head keeps positions 1..N as a block: its scores are matching
probabilities and are not comparable with the cosine scores of the tail below
it, so the two are never sorted against each other.
"""

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

import numpy as np

from app.features import media
from app.features.errors import FeatureExtractionError
from app.vector_store.search import ScoredFrame

DEFAULT_MODEL = "Salesforce/blip-itm-large-coco"
DEFAULT_TOP_N = 30

# A clip hit is judged on a few frames spread over the shot, scored by its best
# one: the moment the query describes rarely lasts the whole shot.
CLIP_FRAMES = 3

# Pairs per forward pass. Cross-attention over 384px images is the memory cost
# here, not the text.
BATCH_SIZE = 8


def rerank(
    text: str,
    hits: list[ScoredFrame],
    top_n: int = DEFAULT_TOP_N,
    model_id: str = DEFAULT_MODEL,
) -> list[ScoredFrame]:
    """Re-score the first `top_n` hits with the cross-encoder."""
    head, tail = hits[:top_n], hits[top_n:]
    if len(head) < 2:
        return list(hits)

    scores = score_hits(text, head, model_id)
    reranked = [replace(hit, score=score) for hit, score in zip(head, scores)]
    reranked.sort(key=lambda hit: hit.score, reverse=True)
    return reranked + tail


def score_hits(
    text: str, hits: list[ScoredFrame], model_id: str = DEFAULT_MODEL
) -> list[float]:
    """Matching probability per hit, the best of its images."""
    runtime = _load_runtime(model_id)

    images: list[np.ndarray] = []
    owners: list[int] = []
    for index, hit in enumerate(hits):
        for image in _hit_images(hit):
            images.append(image)
            owners.append(index)

    # A hit whose keyframe or source video went missing scores 0: it sinks
    # inside the reranked block instead of failing the whole query.
    scores = [0.0] * len(hits)
    for start in range(0, len(images), BATCH_SIZE):
        batch = images[start : start + BATCH_SIZE]
        for offset, value in enumerate(_match_probabilities(text, batch, runtime)):
            owner = owners[start + offset]
            scores[owner] = max(scores[owner], value)

    return scores


def _hit_images(hit: ScoredFrame) -> list[np.ndarray]:
    try:
        if hit.original_frame_id is not None:
            return [media.read_image(hit.path)] if hit.path else []
        if hit.path is None or hit.start_sec is None or hit.end_sec is None:
            return []
        segment = media.ClipSegment(
            path=hit.path,
            video_id=hit.video_id,
            shot_id=hit.shot_id,
            start_frame=hit.start_frame or 0,
            end_frame=hit.end_frame or 0,
            start_sec=hit.start_sec,
            end_sec=hit.end_sec,
        )
        return media.sample_clip_frames(segment, CLIP_FRAMES)
    except FeatureExtractionError:
        return []


def _match_probabilities(
    text: str, images: list[np.ndarray], runtime: "_RerankRuntime"
) -> list[float]:
    inputs = runtime.processor(
        images=images,
        text=[text] * len(images),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {name: value.to(runtime.device) for name, value in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=runtime.dtype)

    with runtime.torch.inference_mode():
        output = runtime.model(**inputs, use_itm_head=True)

    # itm_score is a 2-way logit per pair; column 1 is "matches".
    probabilities = output.itm_score.float().softmax(dim=1)[:, 1]
    return probabilities.detach().cpu().tolist()


@dataclass(frozen=True)
class _RerankRuntime:
    processor: Any
    model: Any
    torch: Any
    device: Any
    dtype: Any


@lru_cache(maxsize=1)
def _load_runtime(model_id: str) -> _RerankRuntime:
    try:
        import torch
        from transformers import AutoProcessor, BlipForImageTextRetrieval
    except ImportError as exc:
        raise FeatureExtractionError(
            "reranking dependencies are missing; install requirements.txt"
        ) from exc

    if torch.cuda.is_available():
        device, dtype = torch.device("cuda"), torch.float16
    else:
        device, dtype = torch.device("cpu"), torch.float32

    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = BlipForImageTextRetrieval.from_pretrained(model_id, torch_dtype=dtype)
        model = model.to(device).eval()
    except (OSError, ValueError) as exc:
        raise FeatureExtractionError(
            f"cannot load rerank model '{model_id}': {exc}"
        ) from exc

    return _RerankRuntime(processor, model, torch, device, dtype)
