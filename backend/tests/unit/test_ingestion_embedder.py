import numpy as np

from app.features import multimodal
from app.ingestion import embedder, manifest


def keyframe(path: str) -> manifest.KeyframeManifestRow:
    return manifest.KeyframeManifestRow(
        video_id="L01_V001",
        shot_id=2,
        original_frame_id=12,
        pts_sec=0.48,
        path=path,
    )


def clip(path: str) -> manifest.ClipManifestRow:
    return manifest.ClipManifestRow(
        video_id="L01_V001",
        shot_id=2,
        start_frame=5,
        end_frame=14,
        start_sec=0.5,
        end_sec=1.4,
        path=path,
    )


def test_keyframe_row_is_adapted_to_one_image(monkeypatch):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    seen = {}

    monkeypatch.setattr(embedder.media, "read_image", lambda path: image)

    def fake_embed(profile, images):
        seen["profile"] = profile
        seen["images"] = images
        return [0.25]

    monkeypatch.setattr(multimodal, "embed_images", fake_embed)

    assert embedder.embed_row("clip-b32-v1", keyframe("frame.jpg")) == [0.25]
    assert seen["profile"].model_id == "openai/clip-vit-base-patch32"
    assert seen["images"] == [image]


def test_clip_row_is_adapted_to_a_media_segment(monkeypatch):
    images = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]
    seen = {}

    def fake_sample(segment, frame_count):
        seen["segment"] = segment
        seen["frame_count"] = frame_count
        return images

    monkeypatch.setattr(embedder.media, "sample_clip_frames", fake_sample)
    monkeypatch.setattr(
        multimodal, "embed_images", lambda profile, value: [float(len(value))]
    )

    result = embedder.embed_row(
        "siglip2-so400m-patch14-384-v1", clip("video.mp4")
    )

    assert result == [3.0]
    assert seen["segment"].path == "video.mp4"
    assert seen["segment"].start_frame == 5
    assert seen["segment"].end_frame == 14
    assert seen["frame_count"] == 8
