"""Tests for cyberwave.models.runtimes.onnxruntime_rt — ONNX Runtime backend."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cyberwave.models.runtimes.onnxruntime_rt import OnnxRuntime, _postprocess
from cyberwave.models.types import PredictionResult
from tests.test_runtime_conformance import RuntimeConformanceMixin


class TestOnnxRuntimeConformance(RuntimeConformanceMixin):
    runtime_class = OnnxRuntime


class TestOnnxRuntimeIsAvailable:
    def test_available_when_installed(self):
        mock_ort = MagicMock()
        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            assert OnnxRuntime().is_available() is True

    def test_unavailable_when_missing(self):
        with patch.dict("sys.modules", {"onnxruntime": None}):
            assert OnnxRuntime().is_available() is False


class TestOnnxRuntimeLoad:
    def test_load_creates_session_with_cpu(self):
        mock_ort = MagicMock()
        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            rt = OnnxRuntime()
            rt.load("/path/model.onnx", device="cpu")
            mock_ort.InferenceSession.assert_called_once_with(
                "/path/model.onnx",
                providers=["CPUExecutionProvider"],
            )

    def test_load_creates_session_with_cuda(self):
        mock_ort = MagicMock()
        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            rt = OnnxRuntime()
            rt.load("/path/model.onnx", device="cuda:0")
            mock_ort.InferenceSession.assert_called_once_with(
                "/path/model.onnx",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )


class TestOnnxRuntimePredict:
    def _make_session(self, *, output: np.ndarray, class_names: str = ""):
        session = MagicMock()
        session.get_inputs.return_value = [
            MagicMock(name="images", shape=[1, 3, 640, 640]),
        ]
        session.get_inputs.return_value[0].name = "images"
        meta = MagicMock()
        meta.custom_metadata_map = {"names": class_names} if class_names else {}
        session.get_modelmeta.return_value = meta
        session.run.return_value = [output]
        return session

    def test_predict_returns_prediction_result(self):
        # [1, 4+2classes, 3detections] — Ultralytics ONNX layout
        # After transpose: det0=(cat 0.9), det1=(dog 0.7), det2=(cat 0.8)
        raw = np.array(
            [
                [
                    [320, 320, 320],  # cx
                    [320, 320, 320],  # cy
                    [100, 50, 200],  # w
                    [100, 50, 200],  # h
                    [0.9, 0.3, 0.8],  # class0 score
                    [0.1, 0.7, 0.2],  # class1 score
                ]
            ],
            dtype=np.float32,
        )
        session = self._make_session(
            output=raw,
            class_names="{0: 'cat', 1: 'dog'}",
        )
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        rt = OnnxRuntime()
        result = rt.predict(session, img, confidence=0.5)

        assert isinstance(result, PredictionResult)
        assert len(result.detections) == 3
        labels = {d.label for d in result.detections}
        assert labels == {"cat", "dog"}

    def test_predict_filters_by_confidence(self):
        raw = np.array(
            [
                [
                    [320],  # cx
                    [320],  # cy
                    [100],  # w
                    [100],  # h
                    [0.3],  # class0 score (below threshold)
                    [0.1],  # class1 score
                ]
            ],
            dtype=np.float32,
        )
        session = self._make_session(output=raw)
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        rt = OnnxRuntime()
        result = rt.predict(session, img, confidence=0.5)
        assert len(result.detections) == 0

    def test_predict_filters_by_class(self):
        raw = np.array(
            [
                [
                    [320],
                    [320],
                    [100],
                    [100],
                    [0.9],
                    [0.1],
                ]
            ],
            dtype=np.float32,
        )
        session = self._make_session(
            output=raw,
            class_names="{0: 'cat', 1: 'dog'}",
        )
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        rt = OnnxRuntime()
        result = rt.predict(session, img, confidence=0.5, classes=["dog"])
        assert len(result.detections) == 0


class TestOnnxPostprocess:
    def test_empty_when_no_detections_above_threshold(self):
        raw = np.zeros((1, 6, 0), dtype=np.float32)
        result = _postprocess(
            raw,
            confidence=0.5,
            classes=None,
            class_names={},
            img_w=640,
            img_h=640,
            input_shape=[1, 3, 640, 640],
        )
        assert result == []

    def test_returns_detections_for_valid_output(self):
        # Single detection, 2 classes, [1, 4+2, 1] layout
        raw = np.array(
            [
                [
                    [320],
                    [320],
                    [100],
                    [100],
                    [0.85],
                    [0.15],
                ]
            ],
            dtype=np.float32,
        )
        result = _postprocess(
            raw,
            confidence=0.5,
            classes=None,
            class_names={0: "person", 1: "car"},
            img_w=640,
            img_h=640,
            input_shape=[1, 3, 640, 640],
        )
        assert len(result) == 1
        assert result[0].label == "person"
        assert result[0].confidence == pytest.approx(0.85)
