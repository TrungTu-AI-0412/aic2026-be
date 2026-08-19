"""Dense text embeddings for the speech collection.

The runtime is mocked throughout: these assert the pooling and normalisation
contract, not the model's weights, and no test should download 600M parameters.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from app.features import text
from app.features.errors import FeatureExtractionError


def fake_runtime(hidden: torch.Tensor):
    """A runtime whose model returns `hidden` as its last hidden state."""
    runtime = MagicMock()
    runtime.torch = torch
    runtime.device = torch.device("cpu")
    runtime.dtype = torch.float32
    runtime.tokenizer.return_value = {
        "input_ids": torch.ones((hidden.shape[0], 4), dtype=torch.long),
        "attention_mask": torch.ones((hidden.shape[0], 4), dtype=torch.long),
    }
    runtime.model.return_value = MagicMock(last_hidden_state=hidden)
    return runtime


class TestEmbedTexts:
    def test_one_unit_vector_per_input(self):
        hidden = torch.zeros((3, 4, 1024))
        hidden[:, -1, 0] = 5.0

        with patch.object(text, "_load_runtime", return_value=fake_runtime(hidden)):
            vectors = text.embed_texts("qwen3-embed-0.6b-v1", ["a", "b", "c"])

        assert len(vectors) == 3
        for vector in vectors:
            assert len(vector) == 1024
            assert pytest.approx(sum(value**2 for value in vector), abs=1e-5) == 1.0

    def test_the_last_token_is_pooled_not_the_mean(self):
        """Qwen3-Embedding pools the final position; averaging would mix in the
        earlier tokens and produce a different vector entirely."""
        hidden = torch.zeros((1, 3, 1024))
        hidden[0, 0, 0] = 9.0   # first token, must be ignored
        hidden[0, -1, 1] = 1.0  # last token, must be used

        with patch.object(text, "_load_runtime", return_value=fake_runtime(hidden)):
            vector = text.embed_texts("qwen3-embed-0.6b-v1", ["a"])[0]

        assert vector[0] == pytest.approx(0.0)
        assert vector[1] == pytest.approx(1.0)

    def test_tokenizer_is_asked_to_pad_and_truncate(self):
        hidden = torch.zeros((1, 2, 1024))
        hidden[0, -1, 0] = 1.0
        runtime = fake_runtime(hidden)

        with patch.object(text, "_load_runtime", return_value=runtime):
            text.embed_texts("qwen3-embed-0.6b-v1", ["a"], max_length=128)

        kwargs = runtime.tokenizer.call_args.kwargs
        assert kwargs["padding"] is True
        assert kwargs["truncation"] is True
        assert kwargs["max_length"] == 128

    def test_a_blank_string_still_yields_a_vector(self):
        """Otherwise the returned list stops lining up with the input rows and
        every vector after the blank is attached to the wrong segment."""
        hidden = torch.zeros((2, 2, 1024))
        hidden[:, -1, 0] = 1.0
        runtime = fake_runtime(hidden)

        with patch.object(text, "_load_runtime", return_value=runtime):
            vectors = text.embed_texts("qwen3-embed-0.6b-v1", ["", "b"])

        assert len(vectors) == 2
        assert runtime.tokenizer.call_args.args[0] == [" ", "b"]

    def test_no_inputs_means_no_model_call(self):
        with patch.object(text, "_load_runtime") as load:
            assert text.embed_texts("qwen3-embed-0.6b-v1", []) == []
        load.assert_not_called()

    def test_a_wrong_dimension_is_rejected(self):
        hidden = torch.zeros((1, 2, 512))
        hidden[0, -1, 0] = 1.0

        with patch.object(text, "_load_runtime", return_value=fake_runtime(hidden)):
            with pytest.raises(FeatureExtractionError, match="dimension"):
                text.embed_texts("qwen3-embed-0.6b-v1", ["a"])

    def test_a_zero_vector_is_rejected(self):
        """A zero vector has no direction, so cosine similarity against it is
        undefined and it would silently match everything or nothing."""
        with patch.object(
            text, "_load_runtime", return_value=fake_runtime(torch.zeros((1, 2, 1024)))
        ):
            with pytest.raises(FeatureExtractionError, match="zero"):
                text.embed_texts("qwen3-embed-0.6b-v1", ["a"])

    def test_non_finite_values_are_rejected(self):
        hidden = torch.zeros((1, 2, 1024))
        hidden[0, -1, 0] = float("nan")

        with patch.object(text, "_load_runtime", return_value=fake_runtime(hidden)):
            with pytest.raises(FeatureExtractionError, match="finite"):
                text.embed_texts("qwen3-embed-0.6b-v1", ["a"])


class TestEmbedQuery:
    def test_a_query_returns_one_flat_vector(self):
        hidden = torch.zeros((1, 2, 1024))
        hidden[0, -1, 0] = 1.0

        with patch.object(text, "_load_runtime", return_value=fake_runtime(hidden)):
            vector = text.embed_query("qwen3-embed-0.6b-v1", "Xuân Sơn")

        assert isinstance(vector[0], float)
        assert len(vector) == 1024

    def test_an_empty_query_is_refused(self):
        with pytest.raises(ValueError):
            text.embed_query("qwen3-embed-0.6b-v1", "   ")
