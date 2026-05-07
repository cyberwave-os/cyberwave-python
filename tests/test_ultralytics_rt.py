"""Tests for cyberwave.models.runtimes.ultralytics_rt — Ultralytics backend."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from cyberwave.models.runtimes.ultralytics_rt import UltralyticsRuntime
from cyberwave.models.types import PredictionResult
from tests.test_runtime_conformance import RuntimeConformanceMixin


class TestUltralyticsRuntimeConformance(RuntimeConformanceMixin):
    runtime_class = UltralyticsRuntime


def _box_mock(xyxy: list[float], cls: int, conf: float) -> MagicMock:
    box = MagicMock()
    box.xyxy = [MagicMock()]
    box.xyxy[0].tolist.return_value = xyxy
    box.cls = [cls]
    box.conf = [conf]
    return box


def _result_mock(
    *,
    boxes: list[MagicMock],
    names: dict[int, str],
    orig_shape: tuple[int, int] = (480, 640),
    keypoints_data: np.ndarray | None = None,
) -> MagicMock:
    result = MagicMock()
    result.orig_shape = orig_shape
    result.boxes = boxes
    result.names = names
    if keypoints_data is None:
        result.keypoints = None
    else:
        result.keypoints = MagicMock()
        tensor_mock = MagicMock()
        tensor_mock.cpu.return_value.numpy.return_value = keypoints_data
        result.keypoints.data = tensor_mock
    return result


class TestUltralyticsPredictDetection:
    def test_returns_prediction_result(self):
        rt = UltralyticsRuntime()
        result_obj = _result_mock(
            boxes=[
                _box_mock([10.0, 20.0, 110.0, 220.0], cls=0, conf=0.9),
                _box_mock([200.0, 300.0, 250.0, 350.0], cls=1, conf=0.8),
            ],
            names={0: "person", 1: "car"},
        )
        model = MagicMock(return_value=[result_obj])

        pred = rt.predict(
            model, np.zeros((480, 640, 3), dtype=np.uint8), confidence=0.5
        )

        assert isinstance(pred, PredictionResult)
        assert len(pred.detections) == 2
        labels = [d.label for d in pred.detections]
        assert labels == ["person", "car"]
        # No pose data → keypoints should be None.
        assert all(d.keypoints is None for d in pred.detections)

    def test_filters_by_class(self):
        rt = UltralyticsRuntime()
        result_obj = _result_mock(
            boxes=[
                _box_mock([10.0, 20.0, 110.0, 220.0], cls=0, conf=0.9),
                _box_mock([200.0, 300.0, 250.0, 350.0], cls=1, conf=0.8),
            ],
            names={0: "person", 1: "car"},
        )
        model = MagicMock(return_value=[result_obj])

        pred = rt.predict(
            model, np.zeros((480, 640, 3), dtype=np.uint8), classes=["person"]
        )

        assert len(pred.detections) == 1
        assert pred.detections[0].label == "person"


class TestUltralyticsPredictPose:
    def test_keypoints_attached_to_detections(self):
        rt = UltralyticsRuntime()
        # 2 detections × 17 keypoints × 3 (x, y, vis).
        kp = np.arange(2 * 17 * 3, dtype=np.float32).reshape(2, 17, 3)
        result_obj = _result_mock(
            boxes=[
                _box_mock([10.0, 20.0, 110.0, 220.0], cls=0, conf=0.9),
                _box_mock([200.0, 300.0, 250.0, 350.0], cls=0, conf=0.8),
            ],
            names={0: "person"},
            keypoints_data=kp,
        )
        model = MagicMock(return_value=[result_obj])

        pred = rt.predict(model, np.zeros((480, 640, 3), dtype=np.uint8))

        assert len(pred.detections) == 2
        assert pred.detections[0].keypoints is not None
        assert pred.detections[0].keypoints.shape == (17, 3)
        np.testing.assert_array_equal(pred.detections[0].keypoints, kp[0])
        np.testing.assert_array_equal(pred.detections[1].keypoints, kp[1])

    def test_keypoints_omitted_when_class_filter_skips_box(self):
        # Two pose detections; user filters to "dog" only — none survive.
        rt = UltralyticsRuntime()
        kp = np.arange(2 * 17 * 3, dtype=np.float32).reshape(2, 17, 3)
        result_obj = _result_mock(
            boxes=[
                _box_mock([10.0, 20.0, 110.0, 220.0], cls=0, conf=0.9),
                _box_mock([200.0, 300.0, 250.0, 350.0], cls=0, conf=0.8),
            ],
            names={0: "person"},
            keypoints_data=kp,
        )
        model = MagicMock(return_value=[result_obj])

        pred = rt.predict(
            model, np.zeros((480, 640, 3), dtype=np.uint8), classes=["dog"]
        )
        assert pred.detections == []

    def test_handles_missing_keypoints_gracefully(self):
        # `result.keypoints.data` raises AttributeError → we return no keypoints.
        rt = UltralyticsRuntime()

        class _NoData:
            """Stand-in for a result.keypoints object without ``.data``."""

        result = MagicMock()
        result.orig_shape = (480, 640)
        result.boxes = [_box_mock([10.0, 20.0, 110.0, 220.0], cls=0, conf=0.9)]
        result.names = {0: "person"}
        result.keypoints = _NoData()
        model = MagicMock(return_value=[result])

        pred = rt.predict(model, np.zeros((480, 640, 3), dtype=np.uint8))

        assert len(pred.detections) == 1
        assert pred.detections[0].keypoints is None


class TestUltralyticsAvailable:
    def test_is_available_returns_bool(self):
        assert isinstance(UltralyticsRuntime().is_available(), bool)


class TestUltralyticsLoadDeviceCompat:
    """``load()`` must tolerate the ``TypeError`` Ultralytics raises from
    ``model.to(device)`` for non-PyTorch backends (ONNX, TensorRT, …).

    The wrapped model still produces predictions through Ultralytics'
    own ``__call__`` path — losing only the early device move — so the
    SDK should swallow the format-mismatch error and return the handle
    instead of crashing the worker at module import time.
    """

    def _install_fake_ultralytics(
        self, monkeypatch: pytest.MonkeyPatch, *, yolo_factory
    ) -> None:
        """Install a stub ``ultralytics`` module exposing *yolo_factory* as ``YOLO``."""
        fake = types.ModuleType("ultralytics")
        fake.YOLO = yolo_factory  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "ultralytics", fake)

    def test_load_swallows_typeerror_from_to_for_onnx_handle(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        model_path = tmp_path / "yolov8n.onnx"
        model_path.write_bytes(b"")  # exists() must be True so load() skips chdir-download

        handle = MagicMock(name="onnx_yolo_handle")
        handle.to.side_effect = TypeError(
            "model='yolov8n.onnx' should be a *.pt PyTorch model"
        )
        self._install_fake_ultralytics(monkeypatch, yolo_factory=lambda _: handle)

        rt = UltralyticsRuntime()
        returned = rt.load(str(model_path), device="cpu")

        assert returned is handle
        handle.to.assert_called_once_with("cpu")

    def test_load_still_calls_to_for_pt_handle(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The compatibility shim must not regress the happy path: a real
        PyTorch handle should still receive ``model.to(device)``."""
        model_path = tmp_path / "yolov8n.pt"
        model_path.write_bytes(b"")

        handle = MagicMock(name="pt_yolo_handle")
        self._install_fake_ultralytics(monkeypatch, yolo_factory=lambda _: handle)

        rt = UltralyticsRuntime()
        returned = rt.load(str(model_path), device="cpu")

        assert returned is handle
        handle.to.assert_called_once_with("cpu")

    def test_load_does_not_swallow_unrelated_errors_from_to(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Only the ``TypeError`` raised by Ultralytics' format guard is
        absorbed — any other failure (e.g. CUDA OOM, invalid device string)
        must surface so the worker can log a real diagnostic instead of
        silently running on the wrong device."""
        model_path = tmp_path / "yolov8n.pt"
        model_path.write_bytes(b"")

        handle = MagicMock(name="pt_yolo_handle")
        handle.to.side_effect = RuntimeError("CUDA out of memory")
        self._install_fake_ultralytics(monkeypatch, yolo_factory=lambda _: handle)

        rt = UltralyticsRuntime()
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            rt.load(str(model_path), device="cuda:0")
