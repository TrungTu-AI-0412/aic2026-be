"""Adapt ingestion manifest rows to the generic feature-extraction layer."""

from app.features import media, multimodal
from app.features.profiles import get_profile
from app.ingestion import manifest as manifest_module


def embed_row(
    feature_profile: str, row: manifest_module.ManifestRow
) -> list[float]:
    profile = get_profile(feature_profile)

    if isinstance(row, manifest_module.KeyframeManifestRow):
        images = [media.read_image(row.path)]
    elif isinstance(row, manifest_module.ClipManifestRow):
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
    else:
        raise TypeError(f"unsupported manifest row type: {type(row).__name__}")

    return multimodal.embed_images(profile, images)
