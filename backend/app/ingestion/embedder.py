"""Adapt ingestion manifest rows to the generic feature-extraction layer."""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from app.features import media, multimodal, text
from app.features.profiles import get_profile
from app.ingestion import manifest as manifest_module

# JPEG decode releases the GIL, so threads genuinely overlap here. Serial reads
# measured 225 img/s against a 154 img/s GPU consumer, and running them in
# sequence would cap the pipeline near 91 img/s; four readers take the decode
# cost off the critical path. More than four buys nothing on a 4-vCPU box.
_READ_WORKERS = 4


@lru_cache(maxsize=1)
def _readers() -> ThreadPoolExecutor:
    """One pool for the process, not one per batch."""
    return ThreadPoolExecutor(max_workers=_READ_WORKERS)


def embed_row(
    feature_profile: str, row: manifest_module.ManifestRow
) -> list[float]:
    """Embed a single row. Prefer `embed_rows`, which shares a forward pass."""
    return embed_rows(feature_profile, [row])[0]


def embed_rows(
    feature_profile: str, rows: Sequence[manifest_module.ManifestRow]
) -> list[list[float]]:
    """Embed a batch of same-typed rows, one vector per row in input order.

    Batching is the whole point. A keyframe is one image, so embedding rows one
    at a time ran SigLIP2 at batch size 1 and left the GPU idle between
    forward passes; a batch of keyframes shares one pass. Clips are the
    exception — each already mean-pools its own sampled frames into a single
    vector, so they stay one row at a time.
    """
    if not rows:
        return []

    profile = get_profile(feature_profile)
    first = rows[0]

    if isinstance(first, manifest_module.KeyframeManifestRow):
        images = list(_readers().map(media.read_image, [row.path for row in rows]))
        return multimodal.embed_images_each(profile, images)

    if isinstance(first, manifest_module.AsrSegmentManifestRow):
        return text.embed_texts(
            feature_profile, [row.text_corrected for row in rows]
        )

    if isinstance(first, manifest_module.ClipManifestRow):
        return [_embed_clip(profile, row) for row in rows]

    raise TypeError(f"unsupported manifest row type: {type(first).__name__}")


def _embed_clip(profile, row: manifest_module.ClipManifestRow) -> list[float]:
    segment = media.ClipSegment(
        path=row.path,
        video_id=row.video_id,
        shot_id=row.shot_id,
        start_frame=row.start_frame,
        end_frame=row.end_frame,
        start_sec=row.start_sec,
        end_sec=row.end_sec,
    )
    images = media.sample_clip_frames(segment, profile.clip_frame_count)
    return multimodal.embed_images(profile, images)
