import numpy as np

from app.features import multimodal
from app.ingestion import embedder, manifest


def keyframe(path: str) -> manifest.KeyframeManifestRow:
    return manifest.KeyframeManifestRow(
        video_id="L01_V001",
        shot_id=2,
        keyframe_n=1,
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

    def fake_embed_each(profile, images, batch_size=None):
        seen["profile"] = profile
        seen["images"] = images
        return [[0.25]]

    monkeypatch.setattr(multimodal, "embed_images_each", fake_embed_each)

    assert embedder.embed_row("clip-b32-v1", keyframe("frame.jpg")) == [0.25]
    assert seen["profile"].model_id == "openai/clip-vit-base-patch32"
    assert seen["images"] == [image]


def test_keyframes_share_one_forward_pass(monkeypatch):
    """Keyframes must reach the model as a batch, not one call per row.

    Embedding row by row ran the image model at batch size 1, which was the
    dominant cost of the upsert stage. `embed_images_each` is used rather than
    `embed_images` because the latter mean-pools whatever it is given, so a
    batch of unrelated keyframes would collapse into a single vector.
    """
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    calls = []

    monkeypatch.setattr(embedder.media, "read_image", lambda path: image)

    def fake_embed_each(profile, images, batch_size=None):
        calls.append(len(images))
        return [[float(index)] for index in range(len(images))]

    monkeypatch.setattr(multimodal, "embed_images_each", fake_embed_each)

    rows = [keyframe(f"frame{index}.jpg") for index in range(3)]
    vectors = embedder.embed_rows("clip-b32-v1", rows)

    assert calls == [3], "expected one batched call, not one per row"
    assert vectors == [[0.0], [1.0], [2.0]]


def test_asr_rows_are_embedded_as_text(monkeypatch):
    """Speech is matched text-to-text, so it never touches the image model."""
    seen = {}

    def fake_embed_texts(profile, texts):
        seen["profile"] = profile
        seen["texts"] = list(texts)
        return [[0.5] for _ in texts]

    monkeypatch.setattr(embedder.text, "embed_texts", fake_embed_texts)

    rows = [
        manifest.AsrSegmentManifestRow(
            video_id="L01_V001",
            segment=index + 1,
            start_sec=float(index),
            end_sec=float(index) + 1.0,
            text_corrected=f"câu {index}",
        )
        for index in range(2)
    ]

    assert embedder.embed_rows("qwen3-embed-0.6b-v1", rows) == [[0.5], [0.5]]
    assert seen["profile"] == "qwen3-embed-0.6b-v1"
    assert seen["texts"] == ["câu 0", "câu 1"]


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
