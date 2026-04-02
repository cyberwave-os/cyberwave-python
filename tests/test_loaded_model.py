"""Tests for cyberwave.models.loaded_model — LoadedModel wrapper."""

from unittest.mock import MagicMock

from cyberwave.models.loaded_model import LoadedModel
from cyberwave.models.types import Detection, BoundingBox, PredictionResult


class TestLoadedModel:
    def _make_model(self, runtime_name="test_rt"):
        runtime = MagicMock()
        runtime.name = runtime_name
        return LoadedModel(
            name="test-model",
            runtime=runtime,
            model_handle="handle",
            device="cpu",
            model_path="/tmp/model.pt",
        )

    def test_properties(self):
        m = self._make_model()
        assert m.name == "test-model"
        assert m.runtime == "test_rt"
        assert m.device == "cpu"

    def test_predict_delegates(self):
        m = self._make_model()
        expected = PredictionResult(
            detections=[Detection(label="a", confidence=0.9, bbox=BoundingBox(0, 0, 1, 1))]
        )
        m._runtime.predict.return_value = expected
        result = m.predict("input", confidence=0.7, classes=["a"])
        m._runtime.predict.assert_called_once_with(
            "handle", "input", confidence=0.7, classes=["a"]
        )
        assert result is expected

    def test_repr(self):
        m = self._make_model()
        r = repr(m)
        assert "test-model" in r
        assert "test_rt" in r
        assert "cpu" in r
