from contextlib import nullcontext

import numpy as np
import pytest

from app.features import multimodal, profiles
from app.features.errors import FeatureExtractionError


class TestPooling:
    def test_normalises_frames_before_mean_pooling(self):
        result = multimodal.pool_features(
            np.array([[10.0, 0.0], [0.0, 2.0]], dtype=np.float32), 2
        )

        assert result == pytest.approx([2**-0.5, 2**-0.5])
        assert np.linalg.norm(result) == pytest.approx(1.0)

    def test_rejects_wrong_model_dimension(self):
        with pytest.raises(FeatureExtractionError, match="dimension mismatch"):
            multimodal.pool_features(np.ones((1, 3), dtype=np.float32), 2)


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


def test_text_embedding_uses_profile_and_returns_unit_vector(monkeypatch):
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

    runtime = multimodal._ModelRuntime(
        processor=Processor(),
        model=Model(),
        torch=Torch(),
        device="cpu",
        dtype="float32",
    )
    profile = profiles.FeatureProfile("fake", dimension=2)
    monkeypatch.setitem(profiles.FEATURE_PROFILES, "fake-v1", profile)
    monkeypatch.setattr(multimodal, "_load_runtime", lambda value: runtime)

    result = multimodal.embed_text("fake-v1", "người đang chạy")

    assert result == pytest.approx([0.6, 0.8])
    assert calls["processor"]["text"] == ["người đang chạy"]
    assert calls["processor"]["padding"] == "max_length"
