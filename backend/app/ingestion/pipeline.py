from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import av
import numpy as np

from app.ingestion import manifest as manifest_module
from app.schemas.ingestions import IngestionEntity
from app.vector_store import collections, payload_indexes, upsert
from app.vector_store.client import get_qdrant_client


@dataclass(frozen=True)
class FeatureProfile:
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

FEATURE_PROFILE_DIMENSIONS = {
    name: profile.dimension for name, profile in FEATURE_PROFILES.items()
}


class ManifestNotFoundError(Exception):
    pass


class UnknownFeatureProfileError(Exception):
    pass


class FeatureExtractionError(Exception):
    pass


def validate_manifest(manifest_path: str, entity: IngestionEntity) -> int:
    if not Path(manifest_path).is_file():
        raise ManifestNotFoundError(f"manifest not found: {manifest_path}")

    manifest_module.validate_columns(manifest_path, entity)
    return manifest_module.count_rows(manifest_path)


def create_collection(collection_name: str, feature_profile: str) -> None:
    vector_size = _feature_profile_dimension(feature_profile)
    client = get_qdrant_client()
    collections.create_collection(client, collection_name, vector_size)


def create_payload_indexes(collection_name: str, entity: IngestionEntity) -> None:
    client = get_qdrant_client()
    payload_indexes.create_payload_indexes(client, collection_name, entity)


def upsert_points(
    collection_name: str,
    manifest_path: str,
    entity: IngestionEntity,
    feature_profile: str,
    on_progress: Callable[[int], None],
) -> None:
    client = get_qdrant_client()
    completed = 0
    batch = []

    for row in manifest_module.iter_rows(manifest_path, entity):
        vector = _embed(feature_profile, row)
        point_id = upsert.deterministic_point_id(*row.point_parts())
        batch.append(
            upsert.make_point(
                point_id=point_id,
                vector=vector,
                payload=row.payload(),
            )
        )

        if len(batch) >= upsert.DEFAULT_BATCH_SIZE:
            completed += upsert.upsert_points(client, collection_name, batch)
            on_progress(completed)
            batch = []

    if batch:
        completed += upsert.upsert_points(client, collection_name, batch)
        on_progress(completed)


def optimize_collection(collection_name: str) -> None:
    raise NotImplementedError("collection optimization is not implemented yet")


def _feature_profile_dimension(feature_profile: str) -> int:
    return _feature_profile(feature_profile).dimension


def _embed(feature_profile: str, row: manifest_module.ManifestRow) -> list[float]:
    """Embed one keyframe or one shot into the profile's image-text space.

    A clip is represented by uniformly sampled frames. Each frame is L2
    normalised before mean pooling so no high-magnitude frame dominates, and
    the pooled vector is normalised again for Qdrant cosine distance.
    """
    profile = _feature_profile(feature_profile)

    if isinstance(row, manifest_module.KeyframeManifestRow):
        images = [_read_image(row.path)]
    elif isinstance(row, manifest_module.ClipManifestRow):
        images = _sample_clip_frames(row, profile.clip_frame_count)
    else:
        raise TypeError(f"unsupported manifest row type: {type(row).__name__}")

    return _embed_images(profile, images)


def embed_text(feature_profile: str, text: str) -> list[float]:
    """Create a query vector in exactly the same space as `_embed`."""
    if not text.strip():
        raise ValueError("text query must not be empty")

    profile = _feature_profile(feature_profile)
    runtime = _load_runtime(profile)
    inputs = runtime.processor(
        text=[text],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = _move_inputs(inputs, runtime)

    with runtime.torch.inference_mode():
        features = runtime.model.get_text_features(**inputs)

    return _pool_features(_as_numpy(features), profile.dimension)


def _feature_profile(feature_profile: str) -> FeatureProfile:
    try:
        return FEATURE_PROFILES[feature_profile]
    except KeyError as exc:
        supported = ", ".join(sorted(FEATURE_PROFILES))
        raise UnknownFeatureProfileError(
            f"unknown feature_profile '{feature_profile}'; supported: {supported}"
        ) from exc


def _read_image(path: str) -> np.ndarray:
    source = Path(path)
    if not source.is_file():
        raise FeatureExtractionError(f"keyframe not found: {path}")

    try:
        with av.open(str(source)) as container:
            if not container.streams.video:
                raise FeatureExtractionError(f"no image stream in {path}")
            frame = next(container.decode(video=0), None)
            if frame is None:
                raise FeatureExtractionError(f"cannot decode keyframe: {path}")
            return frame.to_ndarray(format="rgb24")
    except av.FFmpegError as exc:
        raise FeatureExtractionError(f"cannot decode keyframe '{path}': {exc}") from exc


def _sample_clip_frames(
    row: manifest_module.ClipManifestRow, frame_count: int
) -> list[np.ndarray]:
    """Decode representative RGB frames from one inclusive shot range."""
    source = Path(row.path)
    if not source.is_file():
        raise FeatureExtractionError(f"clip source video not found: {row.path}")

    available = row.end_frame - row.start_frame + 1
    count = min(frame_count, available)
    target_times = np.linspace(row.start_sec, row.end_sec, count).tolist()
    images: list[np.ndarray] = []
    last_image: np.ndarray | None = None

    if row.end_frame > row.start_frame:
        half_frame = (row.end_sec - row.start_sec) / (
            2 * (row.end_frame - row.start_frame)
        )
    else:
        half_frame = 0.0

    try:
        with av.open(str(source)) as container:
            if not container.streams.video:
                raise FeatureExtractionError(f"no video stream in {row.path}")

            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            if row.start_sec > 0 and stream.time_base is not None:
                offset = max(0, int(row.start_sec / float(stream.time_base)))
                container.seek(offset, stream=stream, backward=True)

            target_index = 0
            for frame in container.decode(stream):
                if frame.time is None:
                    continue

                timestamp = float(frame.time)
                if timestamp + half_frame < row.start_sec:
                    continue
                if timestamp - half_frame > row.end_sec and target_index < count:
                    break

                current = frame.to_ndarray(format="rgb24")
                last_image = current
                while (
                    target_index < count
                    and timestamp + half_frame >= target_times[target_index]
                ):
                    images.append(current)
                    target_index += 1

                if target_index == count:
                    break
    except av.FFmpegError as exc:
        raise FeatureExtractionError(
            f"cannot decode clip '{row.video_id}:{row.shot_id}': {exc}"
        ) from exc

    # Timestamp rounding near the inclusive end frame can leave the last
    # target unmatched. Reusing the last decoded frame is preferable to
    # dropping the whole shot and is bounded to at most half a frame.
    if last_image is not None:
        images.extend([last_image] * (count - len(images)))
    if not images:
        raise FeatureExtractionError(
            f"no decodable frames for clip '{row.video_id}:{row.shot_id}'"
        )
    return images


def _embed_images(
    profile: FeatureProfile, images: Sequence[np.ndarray]
) -> list[float]:
    runtime = _load_runtime(profile)
    chunks: list[np.ndarray] = []

    for start in range(0, len(images), profile.image_batch_size):
        batch = images[start : start + profile.image_batch_size]
        inputs = runtime.processor(images=list(batch), return_tensors="pt")
        inputs = _move_inputs(inputs, runtime)
        with runtime.torch.inference_mode():
            features = runtime.model.get_image_features(**inputs)
        chunks.append(_as_numpy(features))

    return _pool_features(np.concatenate(chunks, axis=0), profile.dimension)


def _pool_features(features: np.ndarray, expected_dimension: int) -> list[float]:
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
