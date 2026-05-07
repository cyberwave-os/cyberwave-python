"""Tests for cyberwave.models.loaded_model — LoadedModel wrapper."""

import json
from unittest.mock import MagicMock

import numpy as np

from cyberwave.models.loaded_model import LoadedModel
from cyberwave.models.types import Detection, BoundingBox, PredictionResult


class TestLoadedModel:
    def _make_model(self, runtime_name="test_rt", data_bus=None):
        runtime = MagicMock()
        runtime.name = runtime_name
        return LoadedModel(
            name="test-model",
            runtime=runtime,
            model_handle="handle",
            device="cpu",
            model_path="/tmp/model.pt",
            data_bus=data_bus,
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

    def test_predict_publishes_detection_overlays_via_raw_data_bus(self):
        data_bus = MagicMock()
        m = self._make_model(data_bus=data_bus)
        expected = PredictionResult(
            detections=[
                Detection(
                    label="person",
                    confidence=0.8764,
                    bbox=BoundingBox(10.2, 20.6, 30.9, 40.4),
                )
            ]
        )
        m._runtime.predict.return_value = expected

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = m.predict(frame)

        assert result is expected
        data_bus.publish_raw.assert_called_once()
        channel, payload = data_bus.publish_raw.call_args.args
        assert channel == f"detections/{m.runtime}"

        published = json.loads(payload.decode())
        assert published == {
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.876,
                    "x1": 10,
                    "y1": 21,
                    "x2": 31,
                    "y2": 40,
                }
            ],
            "frame_width": 640,
            "frame_height": 480,
            "timestamp": published["timestamp"],
        }
        assert isinstance(published["timestamp"], float)

        stats = m.inference_stats()
        assert stats["name"] == "test-model"
        assert stats["device"] == "cpu"
        assert stats["count"] == 1
        assert "avg_ms" in stats
        assert "p95_ms" in stats
        assert "p99_ms" in stats

    def test_predict_without_detections_publishes_empty_heartbeat(self):
        # Empty batches must still be published so overlay consumers
        # (e.g. the camera driver's detection cache) see a heartbeat at
        # the worker's inference cadence and don't fall into their
        # staleness cutoff when the scene transiently has nothing to
        # detect. The payload serialises ``detections: []`` which the
        # driver already renders as "no box".
        data_bus = MagicMock()
        m = self._make_model(data_bus=data_bus)
        expected = PredictionResult(detections=[])
        m._runtime.predict.return_value = expected

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = m.predict(frame)

        assert result is expected
        data_bus.publish_raw.assert_called_once()
        channel, payload = data_bus.publish_raw.call_args.args
        assert channel == f"detections/{m.runtime}"

        published = json.loads(payload.decode())
        assert published["detections"] == []
        assert published["frame_width"] == 320
        assert published["frame_height"] == 240
        assert isinstance(published["timestamp"], float)
