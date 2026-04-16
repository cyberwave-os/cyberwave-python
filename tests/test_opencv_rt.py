"""Tests for cyberwave.models.runtimes.opencv_rt — OpenCV DNN backend."""

from unittest.mock import MagicMock, patch

import numpy as np

from cyberwave.models.runtimes.opencv_rt import OpenCVRuntime, _parse_detections
from cyberwave.models.types import PredictionResult
from tests.test_runtime_conformance import RuntimeConformanceMixin


class TestOpenCVRuntimeConformance(RuntimeConformanceMixin):
    runtime_class = OpenCVRuntime


class TestOpenCVRuntimeIsAvailable:
    def test_available_when_cv2_has_dnn(self):
        mock_cv2 = MagicMock()
        mock_cv2.dnn = MagicMock()
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            assert OpenCVRuntime().is_available() is True

    def test_unavailable_when_missing(self):
        with patch.dict("sys.modules", {"cv2": None}):
            assert OpenCVRuntime().is_available() is False


class TestOpenCVRuntimeLoad:
    def test_load_creates_net_with_cpu(self):
        mock_cv2 = MagicMock()
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            rt = OpenCVRuntime()
            handle = rt.load("/path/model.xml", device="cpu")
            mock_cv2.dnn.readNet.assert_called_once_with("/path/model.xml", "")
            assert handle.net is mock_cv2.dnn.readNet.return_value

    def test_load_with_cuda_sets_backend(self):
        mock_cv2 = MagicMock()
        mock_cv2.dnn.DNN_BACKEND_CUDA = 5
        mock_cv2.dnn.DNN_TARGET_CUDA = 6
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            rt = OpenCVRuntime()
            handle = rt.load("/path/model.xml", device="cuda:0")
            handle.net.setPreferableBackend.assert_called_with(5)
            handle.net.setPreferableTarget.assert_called_with(6)

    def test_load_passes_class_names(self):
        mock_cv2 = MagicMock()
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            rt = OpenCVRuntime()
            handle = rt.load(
                "/path/model.xml",
                device="cpu",
                class_names=["person", "car"],
            )
            assert handle.class_names == ["person", "car"]


class TestOpenCVRuntimePredict:
    def test_predict_returns_prediction_result(self):
        mock_cv2 = MagicMock()
        mock_cv2.dnn.blobFromImage.return_value = np.zeros((1, 3, 416, 416))

        net = MagicMock()
        net.getUnconnectedOutLayersNames.return_value = ["out"]
        # SSD-style output: (1, 1, 1, 7)
        ssd_output = np.array([[[[0, 0, 0.9, 0.1, 0.1, 0.5, 0.5]]]])
        net.forward.return_value = [ssd_output]

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            from cyberwave.models.runtimes.opencv_rt import _OpenCVHandle

            handle = _OpenCVHandle(
                net=net,
                class_names=["person"],
                input_size=(416, 416),
            )
            rt = OpenCVRuntime()
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            result = rt.predict(handle, img, confidence=0.5)

        assert isinstance(result, PredictionResult)
        assert len(result.detections) == 1
        assert result.detections[0].label == "person"


class TestParseDetections:
    def test_ssd_style(self):
        # (1, 1, 2, 7) — two detections
        raw = np.array(
            [
                [
                    [
                        [0, 1, 0.95, 0.1, 0.1, 0.5, 0.5],
                        [0, 0, 0.30, 0.2, 0.2, 0.4, 0.4],
                    ]
                ]
            ]
        )
        dets = _parse_detections(
            [raw],
            confidence=0.5,
            classes=None,
            class_names=["bg", "person"],
            img_w=640,
            img_h=480,
        )
        assert len(dets) == 1
        assert dets[0].label == "person"

    def test_yolo_style(self):
        # (1, 5+2) — one detection, 2 classes
        raw = np.array(
            [[0.5, 0.5, 0.2, 0.2, 0.9, 0.85, 0.15]],
            dtype=np.float32,
        )
        dets = _parse_detections(
            [raw],
            confidence=0.5,
            classes=None,
            class_names=["cat", "dog"],
            img_w=640,
            img_h=480,
        )
        assert len(dets) == 1
        assert dets[0].label == "cat"

    def test_class_filter(self):
        raw = np.array(
            [[0.5, 0.5, 0.2, 0.2, 0.9, 0.85, 0.15]],
            dtype=np.float32,
        )
        dets = _parse_detections(
            [raw],
            confidence=0.5,
            classes=["dog"],
            class_names=["cat", "dog"],
            img_w=640,
            img_h=480,
        )
        assert len(dets) == 0

    def test_empty_when_below_threshold(self):
        raw = np.array(
            [[0.5, 0.5, 0.2, 0.2, 0.3, 0.2, 0.1]],
            dtype=np.float32,
        )
        dets = _parse_detections(
            [raw],
            confidence=0.5,
            classes=None,
            class_names=["cat", "dog"],
            img_w=640,
            img_h=480,
        )
        assert len(dets) == 0
