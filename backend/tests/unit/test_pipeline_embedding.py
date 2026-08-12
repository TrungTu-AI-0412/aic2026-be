from contextlib import nullcontext
from pathlib import Path

import av
import numpy as np
import pytest

from app.ingestion import manifest, pipeline


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


class TestFeatureProfiles:
    def test_siglip2_giant_collection_dimension_matches_model_projection(self):
        assert (
            pipeline._feature_profile_dimension(
                "siglip2-giant-opt-patch16-384-v1"
            )
            == 1536
        )

    def test_siglip2_collection_dimension_matches_model_projection(self):
        assert (
            pipeline._feature_profile_dimension(
                "siglip2-so400m-patch14-384-v1"
            )
            == 1152
        )

    def test_unknown_profile_is_rejected_before_loading_media(self):
        with pytest.raises(pipeline.UnknownFeatureProfileError, match="supported"):
            pipeline._embed("does-not-exist", keyframe("missing.jpg"))


class TestEmbeddingDispatch:
    def test_keyframe_reads_its_image(self, monkeypatch):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        seen = {}

        monkeypatch.setattr(pipeline, "_read_image", lambda path: image)

        def fake_embed(profile, images):
            seen["profile"] = profile
            seen["images"] = images
            return [0.25]

        monkeypatch.setattr(pipeline, "_embed_images", fake_embed)

        assert pipeline._embed("clip-b32-v1", keyframe("frame.jpg")) == [0.25]
        assert seen["profile"].model_id == "openai/clip-vit-base-patch32"
        assert seen["images"] == [image]

    def test_clip_samples_multiple_video_frames(self, monkeypatch):
        images = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]
        seen = {}

        def fake_sample(row, frame_count):
            seen["row"] = row
            seen["frame_count"] = frame_count
            return images

        monkeypatch.setattr(pipeline, "_sample_clip_frames", fake_sample)
        monkeypatch.setattr(
            pipeline, "_embed_images", lambda profile, value: [float(len(value))]
        )

        row = clip("video.mp4")
        assert pipeline._embed("siglip2-so400m-patch14-384-v1", row) == [3.0]
        assert seen == {"row": row, "frame_count": 8}


class TestPooling:
    def test_normalises_frames_before_mean_pooling(self):
        result = pipeline._pool_features(
            np.array([[10.0, 0.0], [0.0, 2.0]], dtype=np.float32), 2
        )

        assert result == pytest.approx([2**-0.5, 2**-0.5])
        assert np.linalg.norm(result) == pytest.approx(1.0)

    def test_rejects_wrong_model_dimension(self):
        with pytest.raises(pipeline.FeatureExtractionError, match="dimension mismatch"):
            pipeline._pool_features(np.ones((1, 3), dtype=np.float32), 2)


class _FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def to(self, *args, **kwargs):
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class TestTextEmbedding:
    def test_uses_the_same_profile_and_returns_a_unit_vector(self, monkeypatch):
        calls = {}

        class Processor:
            def __call__(self, **kwargs):
                calls["processor"] = kwargs
                return {"input_ids": _FakeTensor([[1, 2]])}

        class Model:
            def get_text_features(self, **inputs):
                calls["inputs"] = inputs
                return _FakeTensor([[3.0, 4.0]])

        class Torch:
            @staticmethod
            def inference_mode():
                return nullcontext()

        runtime = pipeline._ModelRuntime(
            processor=Processor(),
            model=Model(),
            torch=Torch(),
            device="cpu",
            dtype="float32",
        )
        profile = pipeline.FeatureProfile("fake", dimension=2)
        monkeypatch.setitem(pipeline.FEATURE_PROFILES, "fake-v1", profile)
        monkeypatch.setattr(pipeline, "_load_runtime", lambda value: runtime)

        result = pipeline.embed_text("fake-v1", "người đang chạy")

        assert result == pytest.approx([0.6, 0.8])
        assert calls["processor"]["text"] == ["người đang chạy"]
        assert calls["processor"]["padding"] == "max_length"


def _write_video(path: Path, frame_count: int = 20, rate: int = 10) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "0"}
        for index in range(frame_count):
            image = np.full((32, 32, 3), index * 10, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_clip_sampling_covers_the_shot(tmp_path):
    source = tmp_path / "L01_V001.mp4"
    _write_video(source)

    images = pipeline._sample_clip_frames(clip(str(source)), frame_count=4)

    assert len(images) == 4
    means = [float(image.mean()) for image in images]
    assert means == sorted(means)
    assert means[0] == pytest.approx(50, abs=5)
    assert means[-1] == pytest.approx(140, abs=5)
