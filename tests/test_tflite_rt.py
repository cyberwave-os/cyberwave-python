"""Tests for cyberwave.models.runtimes.tflite_rt — TFLite backend."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cyberwave.models.runtimes.tflite_rt import (
    TFLiteRuntime,
    _dequantize,
    _is_ssd_style,
    _postprocess_ssd,
    _postprocess_yolo,
    _preprocess,
)
from cyberwave.models.types import PredictionResult
from tests.test_runtime_conformance import RuntimeConformanceMixin


class TestTFLiteRuntimeConformance(RuntimeConformanceMixin):
    runtime_class = TFLiteRuntime


class TestTFLiteRuntimeIsAvailable:
    def test_available_with_tflite_runtime(self):
        mock_interp = MagicMock()
        with patch.dict(
            "sys.modules",
            {"tflite_runtime": MagicMock(), "tflite_runtime.interpreter": mock_interp},
        ):
            assert TFLiteRuntime().is_available() is True

    def test_available_with_tensorflow(self):
        mock_tf = MagicMock()
        mock_tf.lite = MagicMock()
        with patch.dict(
            "sys.modules",
            {"tflite_runtime": None, "tflite_runtime.interpreter": None, "tensorflow": mock_tf},
        ):
            assert TFLiteRuntime().is_available() is True

    def test_unavailable_when_missing(self):
        with patch.dict(
            "sys.modules",
            {
                "tflite_runtime": None,
                "tflite_runtime.interpreter": None,
                "tensorflow": None,
            },
        ):
            assert TFLiteRuntime().is_available() is False


class TestTFLiteRuntimeLoad:
    def test_load_creates_interpreter(self, tmp_path):
        model_file = tmp_path / "model.tflite"
        model_file.write_bytes(b"\x00")

        mock_interp_cls = MagicMock()
        mock_mod = MagicMock()
        mock_mod.Interpreter = mock_interp_cls
        with patch.dict(
            "sys.modules",
            {"tflite_runtime": MagicMock(), "tflite_runtime.interpreter": mock_mod},
        ):
            rt = TFLiteRuntime()
            handle = rt.load(str(model_file))
            mock_interp_cls.assert_called_once_with(
                model_path=str(model_file), num_threads=4,
            )
            handle.allocate_tensors.assert_called_once()

    def test_load_passes_num_threads(self, tmp_path):
        model_file = tmp_path / "model.tflite"
        model_file.write_bytes(b"\x00")

        mock_interp_cls = MagicMock()
        mock_mod = MagicMock()
        mock_mod.Interpreter = mock_interp_cls
        with patch.dict(
            "sys.modules",
            {"tflite_runtime": MagicMock(), "tflite_runtime.interpreter": mock_mod},
        ):
            rt = TFLiteRuntime()
            rt.load(str(model_file), num_threads=2)
            mock_interp_cls.assert_called_once_with(
                model_path=str(model_file), num_threads=2,
            )


class TestTFLiteSupportsPredictFlag:
    def test_supports_predict_is_true(self):
        assert TFLiteRuntime().supports_predict is True


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------


class TestPreprocess:
    def _detail(self, *, shape, dtype, quant_scales=None, quant_zps=None):
        d = {"shape": np.array(shape), "dtype": dtype}
        if quant_scales is not None:
            d["quantization_parameters"] = {
                "scales": np.array(quant_scales),
                "zero_points": np.array(quant_zps or [0]),
            }
        return d

    def test_float32_normalised(self):
        detail = self._detail(shape=[1, 4, 4, 3], dtype=np.float32)
        img = np.full((4, 4, 3), 255, dtype=np.uint8)
        t = _preprocess(img, detail)
        assert t.dtype == np.float32
        assert t.shape == (1, 4, 4, 3)
        np.testing.assert_allclose(t, 1.0, atol=1e-6)

    def test_uint8_passthrough(self):
        detail = self._detail(shape=[1, 4, 4, 3], dtype=np.uint8)
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        t = _preprocess(img, detail)
        assert t.dtype == np.uint8
        assert np.all(t == 128)

    def test_int8_quantised(self):
        detail = self._detail(
            shape=[1, 4, 4, 3], dtype=np.int8,
            quant_scales=[0.00392157], quant_zps=[-128],
        )
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        t = _preprocess(img, detail)
        assert t.dtype == np.int8
        assert t.shape == (1, 4, 4, 3)
        # 0/255 / 0.00392157 + (-128) ≈ -128
        assert t.min() == -128

    def test_resize(self):
        detail = self._detail(shape=[1, 8, 8, 3], dtype=np.float32)
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        t = _preprocess(img, detail)
        assert t.shape == (1, 8, 8, 3)

    def test_grayscale_expanded(self):
        detail = self._detail(shape=[1, 4, 4, 3], dtype=np.float32)
        img = np.zeros((4, 4), dtype=np.uint8)
        t = _preprocess(img, detail)
        assert t.shape == (1, 4, 4, 3)


# ------------------------------------------------------------------
# Dequantization
# ------------------------------------------------------------------


class TestDequantize:
    def test_float_passthrough(self):
        tensor = np.array([1.0, 2.0], dtype=np.float32)
        detail: dict = {}
        result = _dequantize(tensor, detail)
        np.testing.assert_array_equal(result, tensor)

    def test_uint8_dequantize(self):
        tensor = np.array([128, 0], dtype=np.uint8)
        detail = {
            "quantization_parameters": {
                "scales": np.array([0.1]),
                "zero_points": np.array([100]),
            },
        }
        result = _dequantize(tensor, detail)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [2.8, -10.0], atol=1e-5)

    def test_no_quant_params_casts_to_float(self):
        tensor = np.array([1, 2], dtype=np.int32)
        result = _dequantize(tensor, {})
        assert result.dtype == np.float32


# ------------------------------------------------------------------
# SSD-style postprocessing
# ------------------------------------------------------------------


class TestIsSsdStyle:
    def test_four_outputs(self):
        assert _is_ssd_style([{}, {}, {}, {}]) is True

    def test_one_output(self):
        assert _is_ssd_style([{}]) is False


class TestPostprocessSsd:
    def _make_outputs(self, *, n_dets=2, scores=None, class_ids=None):
        """Build mock SSD output: boxes, classes, scores, num_dets."""
        if scores is None:
            scores = [0.95, 0.80]
        if class_ids is None:
            class_ids = [1.0, 0.0]
        n = len(scores)
        boxes = np.array(
            [[[0.1, 0.2, 0.5, 0.6]] * n], dtype=np.float32,
        )  # [1, N, 4] — [y1, x1, y2, x2]
        classes = np.array([class_ids], dtype=np.float32)
        scores_arr = np.array([scores], dtype=np.float32)
        count = np.array([n_dets], dtype=np.float32)
        return [boxes, classes, scores_arr, count]

    def _details(self):
        return [{"quantization_parameters": {}} for _ in range(4)]

    def test_basic_ssd(self):
        outputs = self._make_outputs()
        dets = _postprocess_ssd(
            outputs,
            output_details=self._details(),
            confidence=0.5,
            classes=None,
            class_names={0: "bg", 1: "person"},
            img_w=640,
            img_h=480,
        )
        assert len(dets) == 2
        assert dets[0].label == "person"
        assert dets[0].confidence == pytest.approx(0.95)
        # x1 = 0.2 * 640 = 128, y1 = 0.1 * 480 = 48
        assert dets[0].bbox.x1 == pytest.approx(128.0)
        assert dets[0].bbox.y1 == pytest.approx(48.0)

    def test_confidence_filter(self):
        outputs = self._make_outputs(scores=[0.9, 0.3])
        dets = _postprocess_ssd(
            outputs,
            output_details=self._details(),
            confidence=0.5,
            classes=None,
            class_names={},
            img_w=640,
            img_h=480,
        )
        assert len(dets) == 1

    def test_class_filter(self):
        outputs = self._make_outputs(scores=[0.9], class_ids=[1.0], n_dets=1)
        dets = _postprocess_ssd(
            outputs,
            output_details=self._details(),
            confidence=0.5,
            classes=["cat"],
            class_names={1: "person"},
            img_w=640,
            img_h=480,
        )
        assert len(dets) == 0

    def test_quantised_uint8(self):
        """Verify dequantization works end-to-end for SSD outputs."""
        # Quantised boxes: scale=0.01, zp=0
        boxes = np.array([[[10, 20, 50, 60]]], dtype=np.uint8)
        classes = np.array([[1]], dtype=np.uint8)
        scores = np.array([[230]], dtype=np.uint8)  # 230 * 0.004 ≈ 0.92
        count = np.array([1], dtype=np.float32)

        details = [
            {"quantization_parameters": {"scales": np.array([0.01]), "zero_points": np.array([0])}},
            {"quantization_parameters": {"scales": np.array([1.0]), "zero_points": np.array([0])}},
            {"quantization_parameters": {"scales": np.array([0.004]), "zero_points": np.array([0])}},
            {"quantization_parameters": {}},
        ]
        dets = _postprocess_ssd(
            [boxes, classes, scores, count],
            output_details=details,
            confidence=0.5,
            classes=None,
            class_names={1: "person"},
            img_w=640,
            img_h=480,
        )
        assert len(dets) == 1
        assert dets[0].label == "person"
        assert dets[0].confidence == pytest.approx(0.92, abs=0.01)


# ------------------------------------------------------------------
# YOLO-style postprocessing
# ------------------------------------------------------------------


class TestPostprocessYolo:
    def test_basic_yolo(self):
        # [1, N=1, 4+2classes] — single detection, 2 classes
        raw = np.array(
            [[[160, 160, 100, 100, 0.9, 0.1]]],
            dtype=np.float32,
        )
        dets = _postprocess_yolo(
            raw,
            confidence=0.5,
            classes=None,
            class_names={0: "cat", 1: "dog"},
            img_w=320,
            img_h=320,
            input_shape=[1, 320, 320, 3],
        )
        assert len(dets) == 1
        assert dets[0].label == "cat"
        assert dets[0].confidence == pytest.approx(0.9)

    def test_transposed_layout(self):
        # [4+2, N=1] — transposed
        raw = np.array(
            [[[160], [160], [100], [100], [0.85], [0.15]]],
            dtype=np.float32,
        )
        dets = _postprocess_yolo(
            raw,
            confidence=0.5,
            classes=None,
            class_names={0: "person"},
            img_w=320,
            img_h=320,
            input_shape=[1, 320, 320, 3],
        )
        assert len(dets) == 1
        assert dets[0].label == "person"

    def test_below_threshold(self):
        raw = np.array(
            [[[160, 160, 100, 100, 0.2, 0.1]]],
            dtype=np.float32,
        )
        dets = _postprocess_yolo(
            raw,
            confidence=0.5,
            classes=None,
            class_names={},
            img_w=320,
            img_h=320,
            input_shape=[1, 320, 320, 3],
        )
        assert len(dets) == 0

    def test_class_filter(self):
        raw = np.array(
            [[[160, 160, 100, 100, 0.9, 0.1]]],
            dtype=np.float32,
        )
        dets = _postprocess_yolo(
            raw,
            confidence=0.5,
            classes=["dog"],
            class_names={0: "cat", 1: "dog"},
            img_w=320,
            img_h=320,
            input_shape=[1, 320, 320, 3],
        )
        assert len(dets) == 0

    def test_scaling_to_original_image(self):
        # Model input 160x160, original image 640x480
        raw = np.array(
            [[[80, 80, 40, 40, 0.9, 0.1]]],
            dtype=np.float32,
        )
        dets = _postprocess_yolo(
            raw,
            confidence=0.5,
            classes=None,
            class_names={0: "obj"},
            img_w=640,
            img_h=480,
            input_shape=[1, 160, 160, 3],
        )
        assert len(dets) == 1
        # cx=80 → scaled by 640/160=4x and 480/160=3x
        # x1 = (80-20)*4 = 240, x2 = (80+20)*4 = 400
        # y1 = (80-20)*3 = 180, y2 = (80+20)*3 = 300
        assert dets[0].bbox.x1 == pytest.approx(240.0)
        assert dets[0].bbox.y1 == pytest.approx(180.0)
        assert dets[0].bbox.x2 == pytest.approx(400.0)
        assert dets[0].bbox.y2 == pytest.approx(300.0)


# ------------------------------------------------------------------
# Full predict() integration (mocked interpreter)
# ------------------------------------------------------------------


class TestTFLitePredictIntegration:
    def _make_interpreter(self, *, ssd=True, dtype=np.float32):
        """Build a mock interpreter that returns canned detection data."""
        interp = MagicMock()

        input_detail = {
            "index": 0,
            "shape": np.array([1, 320, 320, 3]),
            "dtype": dtype,
        }
        if dtype == np.int8:
            input_detail["quantization_parameters"] = {
                "scales": np.array([0.00392157]),
                "zero_points": np.array([-128]),
            }
        interp.get_input_details.return_value = [input_detail]

        if ssd:
            boxes = np.array([[[0.1, 0.2, 0.5, 0.6]]], dtype=np.float32)
            class_ids = np.array([[1.0]], dtype=np.float32)
            scores = np.array([[0.95]], dtype=np.float32)
            count = np.array([1], dtype=np.float32)
            outputs = [boxes, class_ids, scores, count]
            out_details = [
                {"index": i, "quantization_parameters": {}} for i in range(4)
            ]
        else:
            yolo_out = np.array(
                [[[160, 160, 80, 80, 0.9, 0.1]]], dtype=np.float32,
            )
            outputs = [yolo_out]
            out_details = [{"index": 0, "quantization_parameters": {}}]

        interp.get_output_details.return_value = out_details

        def get_tensor(idx):
            for od, arr in zip(out_details, outputs):
                if od["index"] == idx:
                    return arr
            return np.array([])

        interp.get_tensor.side_effect = get_tensor
        return interp

    def test_ssd_predict(self):
        interp = self._make_interpreter(ssd=True)
        rt = TFLiteRuntime()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = rt.predict(
            interp, img, confidence=0.5,
            class_names={0: "bg", 1: "person"},
        )
        assert isinstance(result, PredictionResult)
        assert len(result.detections) == 1
        assert result.detections[0].label == "person"
        assert result.detections[0].confidence == pytest.approx(0.95)

    def test_yolo_predict(self):
        interp = self._make_interpreter(ssd=False)
        rt = TFLiteRuntime()
        img = np.zeros((320, 320, 3), dtype=np.uint8)
        result = rt.predict(
            interp, img, confidence=0.5,
            class_names={0: "cat", 1: "dog"},
        )
        assert isinstance(result, PredictionResult)
        assert len(result.detections) == 1
        assert result.detections[0].label == "cat"

    def test_int8_input_ssd_predict(self):
        interp = self._make_interpreter(ssd=True, dtype=np.int8)
        rt = TFLiteRuntime()
        img = np.zeros((320, 320, 3), dtype=np.uint8)
        result = rt.predict(
            interp, img, confidence=0.5,
            class_names={1: "person"},
        )
        assert len(result.detections) == 1
        interp.set_tensor.assert_called_once()
        tensor_arg = interp.set_tensor.call_args[0][1]
        assert tensor_arg.dtype == np.int8

    def test_detection_schema_matches_other_runtimes(self):
        """Verify the detection output uses the same fields as ONNX/Ultralytics."""
        interp = self._make_interpreter(ssd=True)
        rt = TFLiteRuntime()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = rt.predict(
            interp, img, confidence=0.5,
            class_names={1: "person"},
        )
        det = result.detections[0]
        assert hasattr(det, "label")
        assert hasattr(det, "confidence")
        assert hasattr(det, "bbox")
        assert hasattr(det.bbox, "x1")
        assert hasattr(det.bbox, "y1")
        assert hasattr(det.bbox, "x2")
        assert hasattr(det.bbox, "y2")
        assert hasattr(det, "area_ratio")
        assert det.area_ratio > 0
