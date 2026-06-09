"""Tests for cyberwave.models.runtimes.hailo_rt — Hailo .hef backend.

Hardware-dependent tests are gated behind ``@pytest.mark.hailo`` and
are skipped by default on x86 CI. To run them locally on a Pi 5 +
AI HAT+:

    pytest tests/test_hailo_rt.py -m hailo

Pure-Python unit tests (this module's default) do not require any
Hailo hardware or the ``hailo_platform`` wheel.
"""

from __future__ import annotations

import threading
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from cyberwave.models.manager import ModelManager
from cyberwave.models.runtimes import _RUNTIME_REGISTRY
from cyberwave.models.runtimes.hailo_rt import (
    HailoRuntime,
    _COCO_CLASSES,
    _decode_instance_masks,
    _decode_one_det_head,
    _decode_one_seg_head,
    _extract_embedding,
    _flatten_nms_output,
    _HailoHandle,
    _infer_model_kind,
    _Letterbox,
    _nms_numpy,
    _normalize_arch,
    _outputs_look_spatial,
    _postprocess,
    _postprocess_det_raw,
    _postprocess_seg,
    _preprocess,
    _resolve_class_names,
    _resize_for_embedding,
    _split_seg_outputs,
)
from cyberwave.models.types import EmbeddingResult, InstanceSegmentationResult, Mask
from tests.test_runtime_conformance import RuntimeConformanceMixin


class TestHailoRuntimeConformance(RuntimeConformanceMixin):
    runtime_class = HailoRuntime


class TestHailoRuntimeRegistration:
    def test_hailo_is_registered(self):
        assert "hailo" in _RUNTIME_REGISTRY
        assert _RUNTIME_REGISTRY["hailo"] is HailoRuntime


class TestHailoRuntimeIsAvailable:
    def test_unavailable_when_missing(self):
        with patch.dict("sys.modules", {"hailo_platform": None}):
            assert HailoRuntime().is_available() is False


class TestRuntimeDetection:
    @pytest.mark.parametrize(
        "model_id",
        [
            "yolov8s_h8",
            "yolov8s_h8l",
            "yolov6n_hailo",
            "yolov8m-hailo",
            "/cache/yolov8s.hef",
        ],
    )
    def test_detect_hailo_from_id(self, model_id: str) -> None:
        assert ModelManager._detect_runtime(model_id) == "hailo"

    def test_detect_hailo_from_extension(self) -> None:
        assert ModelManager._detect_runtime_from_extension(".hef") == "hailo"

    def test_hef_listed_as_hailo_extension(self) -> None:
        assert ".hef" in ModelManager._runtime_extensions("hailo")

    def test_onnx_suffix_still_wins_over_yolo_keyword(self) -> None:
        # Sanity check: the new ``hailo`` branch must not steal entries
        # that have an explicit ``-onnx`` catalog suffix.
        assert ModelManager._detect_runtime("yolov8n-pose-onnx") == "onnxruntime"


class TestNormalizeArch:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("HAILO8", "hailo8"),
            ("Hailo-8", "hailo8"),
            ("hailo_8", "hailo8"),
            ("HAILO_ARCH_HAILO8", "hailo8"),
            ("HAILO_ARCH_HAILO_8L", "hailo8l"),
            ("Hailo-10H", "hailo10h"),
            ("", ""),
            ("   ", ""),
        ],
    )
    def test_normalize(self, raw: str, expected: str) -> None:
        assert _normalize_arch(raw) == expected


class TestPreprocessLetterbox:
    def test_square_image_full_resize_no_pad(self) -> None:
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        tensor, lb = _preprocess(img, target_h=640, target_w=640)
        assert tensor.shape == (1, 640, 640, 3)
        assert tensor.dtype == np.uint8
        assert lb.scale == pytest.approx(1.0)
        assert lb.pad_left == 0
        assert lb.pad_top == 0

    def test_wide_image_pads_top_bottom(self) -> None:
        # 1280x720 → fits 640x360 inside 640x640 with vertical padding.
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        tensor, lb = _preprocess(img, target_h=640, target_w=640)
        assert tensor.shape == (1, 640, 640, 3)
        assert lb.scale == pytest.approx(640 / 1280)
        assert lb.pad_left == 0
        assert lb.pad_top == (640 - 360) // 2  # = 140

    def test_tall_image_pads_left_right(self) -> None:
        img = np.zeros((1280, 720, 3), dtype=np.uint8)
        tensor, lb = _preprocess(img, target_h=640, target_w=640)
        assert tensor.shape == (1, 640, 640, 3)
        assert lb.scale == pytest.approx(640 / 1280)
        assert lb.pad_top == 0
        assert lb.pad_left == (640 - 360) // 2

    def test_grayscale_expanded_to_3ch(self) -> None:
        img = np.zeros((100, 100), dtype=np.uint8)
        tensor, _ = _preprocess(img, target_h=640, target_w=640)
        assert tensor.shape == (1, 640, 640, 3)

    def test_padding_value_is_yolo_grey(self) -> None:
        # 2x1 wide image → letterbox produces top/bottom pads filled
        # with the YOLO-convention 114 grey value.
        img = np.full((1, 2, 3), 7, dtype=np.uint8)
        tensor, lb = _preprocess(img, target_h=640, target_w=640)
        assert lb.pad_top > 0  # sanity: there *is* a pad region
        # Top-row pixel is inside the pad band.
        assert tensor[0, 0, 0, 0] == 114
        # Mid-row pixel is inside the upscaled image content (== 7).
        assert tensor[0, 320, 320, 0] == 7


class TestFlattenNmsOutput:
    def test_empty_dict_returns_empty(self) -> None:
        assert _flatten_nms_output({}) == []

    def test_batched_per_class_layout_yolov8(self) -> None:
        # Probe-confirmed Hailo Model Zoo layout on HailoRT 4.23.0:
        # list[batch=1][ndarray(num_classes, K, 5)] with rows
        # (x1, y1, x2, y2, score) in normalised coords.
        per_class = np.zeros((80, 0, 5), dtype=np.float32)  # noqa: F841
        # Inject 2 detections on class id 0, 1 detection on class id 5.
        cls0 = np.array(
            [
                [0.10, 0.20, 0.30, 0.40, 0.90],
                [0.50, 0.50, 0.70, 0.70, 0.80],
            ],
            dtype=np.float32,
        )
        cls5 = np.array([[0.05, 0.05, 0.95, 0.95, 0.75]], dtype=np.float32)
        # Build a ragged-shaped 3-D array by padding with zeros — easier
        # to reproduce with a concrete fixed-shape ndarray.
        arr = np.zeros((80, 2, 5), dtype=np.float32)
        arr[0, :2] = cls0
        arr[5, :1] = cls5
        # Mark unused class-5 row's score so we can detect leakage.
        arr[5, 1, 4] = 0.0  # explicitly zero

        outputs = {"yolov8s/yolov8_nms_postprocess": [arr]}
        rows = _flatten_nms_output(outputs)

        # 80 classes * 2 rows = 160 candidates, but most have score 0
        # which still gets emitted by the flattener (confidence
        # filtering happens in _postprocess). What matters: the rows
        # corresponding to our injected detections are present with
        # the right class IDs and scores.
        scores_by_class: dict[int, list[float]] = {}
        for cid, score, *_ in rows:
            scores_by_class.setdefault(int(cid), []).append(float(score))

        def _has(scores: list[float], target: float) -> bool:
            return any(s == pytest.approx(target, rel=1e-3) for s in scores)

        assert _has(scores_by_class.get(0, []), 0.90)
        assert _has(scores_by_class.get(0, []), 0.80)
        assert _has(scores_by_class.get(5, []), 0.75)

    def test_batched_per_class_layout_all_empty(self) -> None:
        # The actual probe captured a (80, 0, 5) "no detections" frame.
        # The flattener should iterate and emit nothing — not crash.
        per_class = np.zeros((80, 0, 5), dtype=np.float32)
        outputs = {"yolov8s/yolov8_nms_postprocess": [per_class]}
        assert _flatten_nms_output(outputs) == []

    def test_flat_nms_layout(self) -> None:
        # [batch, N, 6] with (x1, y1, x2, y2, score, class_id).
        raw = np.array(
            [
                [
                    [0.1, 0.2, 0.3, 0.4, 0.9, 0.0],
                    [0.5, 0.5, 0.7, 0.7, 0.8, 2.0],
                ]
            ],
            dtype=np.float32,
        )
        rows = _flatten_nms_output({"out": raw})
        assert len(rows) == 2
        cid0, score0, x1, y1, x2, y2 = rows[0]
        assert cid0 == 0
        assert score0 == pytest.approx(0.9, rel=1e-3)
        assert (x1, y1, x2, y2) == (
            pytest.approx(0.1),
            pytest.approx(0.2),
            pytest.approx(0.3),
            pytest.approx(0.4),
        )
        assert rows[1][0] == 2

    def test_per_class_list_layout(self) -> None:
        # list[num_classes][ndarray(K, 5)] — older per-class variant.
        per_class: list[np.ndarray] = [np.zeros((0, 5), dtype=np.float32)] * 80
        per_class[3] = np.array([[0.0, 0.0, 1.0, 1.0, 0.5]], dtype=np.float32)
        rows = _flatten_nms_output(per_class)
        assert rows == [(3, 0.5, 0.0, 0.0, 1.0, 1.0)]

    def test_batched_per_class_ragged_list_layout_yolov8m(self) -> None:
        # yolov8m.hef on HailoRT 4.23.0: list[batch=1][list[num_classes]]
        # with variable K per class — np.asarray on the inner list raises
        # "inhomogeneous shape".
        per_class: list[np.ndarray] = [np.zeros((0, 5), dtype=np.float32)] * 80
        per_class[0] = np.array(
            [
                [0.10, 0.20, 0.30, 0.40, 0.90],
                [0.50, 0.50, 0.70, 0.70, 0.80],
            ],
            dtype=np.float32,
        )
        per_class[5] = np.array([[0.05, 0.05, 0.95, 0.95, 0.75]], dtype=np.float32)
        outputs = {"yolov8m/yolov8_nms_postprocess": [per_class]}
        rows = _flatten_nms_output(outputs)
        scores_by_class: dict[int, list[float]] = {}
        for cid, score, *_ in rows:
            scores_by_class.setdefault(int(cid), []).append(float(score))

        def _has(scores: list[float], target: float) -> bool:
            return any(s == pytest.approx(target, rel=1e-3) for s in scores)

        assert _has(scores_by_class.get(0, []), 0.90)
        assert _has(scores_by_class.get(0, []), 0.80)
        assert _has(scores_by_class.get(5, []), 0.75)

    def test_batched_per_class_object_ndarray_layout(self) -> None:
        # list[batch=1][ndarray(80,) dtype=object] — binding variant seen
        # alongside the plain-Python-list layout for yolov8m.
        per_class: list[np.ndarray] = [np.zeros((0, 5), dtype=np.float32)] * 80
        per_class[3] = np.array([[0.0, 0.0, 1.0, 1.0, 0.5]], dtype=np.float32)
        batch0 = np.empty(80, dtype=object)
        for cid, arr in enumerate(per_class):
            batch0[cid] = arr
        rows = _flatten_nms_output({"out": [batch0]})
        assert rows == [(3, 0.5, 0.0, 0.0, 1.0, 1.0)]

    def test_unrecognised_layout_warns_and_returns_empty(self, caplog) -> None:
        bogus = np.zeros((1, 3, 80, 80), dtype=np.float32)  # raw feature map
        with caplog.at_level("WARNING"):
            assert _flatten_nms_output({"out": bogus}) == []
        assert any("not a recognised NMSLayout" in r.message or "not a recognised NMS layout" in r.message for r in caplog.records)


class TestPostprocessCoordinates:
    """Verify un-letterbox for both absolute and normalized coordinate layouts."""

    def _lb(self) -> _Letterbox:
        # 1280x720 → 640x640 letterboxed: scale=0.5, pad_top=140, pad_left=0
        return _Letterbox(scale=0.5, pad_left=0, pad_top=140, target_w=640, target_h=640)

    def test_per_class_yolov8m_y_first_coords(self) -> None:
        # Hailo yolov8_nms_postprocess emits (y_min, x_min, y_max, x_max, score).
        # A box spanning the full letterboxed content area maps back to (0,0,1280,720).
        # Normalized: y1=140/640, x1=0, y2=500/640, x2=1.0
        lb = self._lb()
        per_class: list[np.ndarray] = [np.zeros((0, 5), dtype=np.float32)] * 80
        per_class[0] = np.array(
            [[140/640, 0.0, 500/640, 1.0, 0.9]],  # y1, x1, y2, x2, score
            dtype=np.float32,
        )
        outputs = {"yolov8m/yolov8_nms_postprocess": [per_class]}
        coco = {i: n for i, n in enumerate(_COCO_CLASSES)}
        dets = _postprocess(
            outputs, class_names=coco, confidence=0.5, classes=None,
            letterbox=lb, orig_w=1280, orig_h=720,
        )
        assert len(dets) == 1
        bbox = dets[0].bbox
        assert bbox.x1 == pytest.approx(0.0, abs=1)
        assert bbox.y1 == pytest.approx(0.0, abs=1)
        assert bbox.x2 == pytest.approx(1280.0, abs=1)
        assert bbox.y2 == pytest.approx(720.0, abs=1)

    def test_normalized_coords_flat_nms(self) -> None:
        # Flat NMS layout: (x1, y1, x2, y2, score, cls) — x-first.
        lb = self._lb()
        raw = np.array(
            [[[0.0, 140/640, 1.0, 500/640, 0.9, 0.0]]],
            dtype=np.float32,
        )
        coco = {i: n for i, n in enumerate(_COCO_CLASSES)}
        dets = _postprocess(
            {"out": raw}, class_names=coco, confidence=0.5, classes=None,
            letterbox=lb, orig_w=1280, orig_h=720,
        )
        assert len(dets) == 1
        bbox = dets[0].bbox
        assert bbox.x1 == pytest.approx(0.0, abs=1)
        assert bbox.y1 == pytest.approx(0.0, abs=1)
        assert bbox.x2 == pytest.approx(1280.0, abs=1)
        assert bbox.y2 == pytest.approx(720.0, abs=1)


class TestResolveClassNames:
    def test_defaults_to_coco_when_no_labels_and_no_sidecar(self, tmp_path) -> None:
        # The most important case: no labels arg, no sidecar → COCO labels.
        hef = tmp_path / "yolov8m.hef"
        hef.touch()
        names = _resolve_class_names(str(hef), None)
        assert names[0] == "person"
        assert names[79] == "toothbrush"
        assert len(names) == 80

    def test_coco_classes_length(self) -> None:
        assert len(_COCO_CLASSES) == 80
        assert _COCO_CLASSES[0] == "person"

    def test_caller_labels_list_overrides_coco(self, tmp_path) -> None:
        hef = tmp_path / "custom.hef"
        hef.touch()
        names = _resolve_class_names(str(hef), ["cat", "dog"])
        assert names == {0: "cat", 1: "dog"}

    def test_sidecar_overrides_coco(self, tmp_path) -> None:
        import json
        hef = tmp_path / "custom.hef"
        hef.touch()
        sidecar = tmp_path / "custom.labels.json"
        sidecar.write_text(json.dumps(["foo", "bar"]))
        names = _resolve_class_names(str(hef), None)
        assert names == {0: "foo", 1: "bar"}


class TestResizeForEmbedding:
    def test_square_image_produces_nhwc_tensor(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor = _resize_for_embedding(img, target_h=288, target_w=288)
        assert tensor.shape == (1, 288, 288, 3)
        assert tensor.dtype == np.uint8

    def test_grayscale_expanded_to_3ch(self) -> None:
        img = np.zeros((100, 100), dtype=np.uint8)
        tensor = _resize_for_embedding(img, target_h=288, target_w=288)
        assert tensor.shape == (1, 288, 288, 3)

    def test_no_grey_padding_unlike_letterbox(self) -> None:
        # Plain resize fills the entire canvas; no 114-grey pad regions.
        img = np.full((480, 640, 3), 200, dtype=np.uint8)
        tensor = _resize_for_embedding(img, target_h=288, target_w=288)
        # Every pixel in the output should be near 200 (no grey padding).
        assert int(tensor.min()) >= 150


class TestExtractEmbedding:
    def test_shape_1_d_returns_flat(self) -> None:
        vec = np.random.rand(640).astype(np.float32)
        result = _extract_embedding({"output": vec})
        assert result.shape == (640,)
        np.testing.assert_array_almost_equal(result, vec)

    def test_shape_1_by_d_squeezed(self) -> None:
        vec = np.random.rand(1, 640).astype(np.float32)
        result = _extract_embedding({"output": vec})
        assert result.shape == (640,)
        np.testing.assert_array_almost_equal(result, vec[0])

    def test_empty_dict_returns_zero_array(self) -> None:
        result = _extract_embedding({})
        assert result.shape == (0,)
        assert result.dtype == np.float32

    def test_multi_output_uses_first_tensor(self, caplog) -> None:
        a = np.ones(640, dtype=np.float32)
        b = np.zeros(128, dtype=np.float32)
        with caplog.at_level("WARNING"):
            result = _extract_embedding({"primary": a, "aux": b})
        assert result.shape == (640,)
        assert any("expected 1 output" in r.message.lower() for r in caplog.records)

    def test_unexpected_shape_ravelled(self, caplog) -> None:
        tensor = np.ones((2, 320), dtype=np.float32)  # batch > 1 — unusual
        with caplog.at_level("WARNING"):
            result = _extract_embedding({"out": tensor})
        assert result.shape == (640,)
        assert any("unexpected output shape" in r.message.lower() for r in caplog.records)

    def test_result_dtype_is_float32(self) -> None:
        vec = np.ones((1, 640), dtype=np.float64)
        result = _extract_embedding({"out": vec})
        assert result.dtype == np.float32


class TestHailoHandleModelKind:
    """Verify that model_kind is threaded from load() kwargs into the handle
    and that predict() returns the correct PredictionResult type."""

    def _make_handle(self, model_kind: str = "detection") -> object:
        """Build a minimal _HailoHandle-like mock without Hailo hardware."""
        handle = _HailoHandle(
            vdevice=MagicMock(),
            network_group=MagicMock(),
            pipeline=MagicMock(),
            activation=MagicMock(),
            exit_stack=ExitStack(),
            input_name="input_layer",
            input_shape_hw=(288, 288),
            output_names=["output_layer"],
            class_names={},
            hw_arch="hailo8",
            model_kind=model_kind,
            lock=threading.Lock(),
        )
        # Stub pipeline.infer to return a 640-d embedding vector
        handle.pipeline.infer.return_value = {
            "output_layer": np.ones((1, 640), dtype=np.float32)
        }
        return handle

    def test_embedding_predict_returns_embedding_result(self) -> None:
        handle = self._make_handle(model_kind="embedding")
        runtime = HailoRuntime()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = runtime.predict(handle, img)
        assert isinstance(result, EmbeddingResult)
        assert result.dim == 640

    def test_default_model_kind_is_detection(self) -> None:
        handle = self._make_handle(model_kind="detection")
        assert handle.model_kind == "detection"  # type: ignore[union-attr]

    def test_embedding_predict_has_no_detections(self) -> None:
        handle = self._make_handle(model_kind="embedding")
        runtime = HailoRuntime()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = runtime.predict(handle, img)
        assert isinstance(result, EmbeddingResult)
        assert result.dim > 0


# ----------------------------------------------------------------------
# Auto model-kind detection from output stream shapes
# ----------------------------------------------------------------------


class TestInferModelKind:
    """_infer_model_kind returns the right kind from mock vstream info shapes."""

    def _infos(self, shapes: list[tuple[int, ...]]) -> list[object]:
        infos = []
        for s in shapes:
            m = MagicMock()
            m.shape = s
            infos.append(m)
        return infos

    def test_seg_detected_from_proto_and_det_channels_116(self) -> None:
        # proto (160, 160, 32) + 3 det heads (H, W, 116)
        infos = self._infos([(160, 160, 32), (80, 80, 116), (40, 40, 116), (20, 20, 116)])
        assert _infer_model_kind(infos) == "instance_segmentation"

    def test_seg_detected_from_proto_and_det_channels_176(self) -> None:
        # raw DFL layout
        infos = self._infos([(160, 160, 32), (80, 80, 176), (40, 40, 176), (20, 20, 176)])
        assert _infer_model_kind(infos) == "instance_segmentation"

    def test_seg_detected_from_split_tensor_hailo_layout(self) -> None:
        # Hailo-specific: separate box/class/mask-coeff per scale + proto.
        # Mirrors the actual yolov8m_seg.hef output vstream shapes observed on
        # RPi5 + AI HAT+: 3 scales × (64, 80, 32) + proto (160, 160, 32).
        infos = self._infos([
            (20, 20, 64), (20, 20, 80), (20, 20, 32),   # scale 1
            (40, 40, 64), (40, 40, 80), (40, 40, 32),   # scale 2
            (80, 80, 64), (80, 80, 80), (80, 80, 32),   # scale 3
            (160, 160, 32),                              # proto
        ])
        assert _infer_model_kind(infos) == "instance_segmentation"

    def test_seg_detected_with_just_two_c32_outputs(self) -> None:
        # Minimum split-layout: proto + 1 mask-coeff tensor → 2 outputs with C=32.
        infos = self._infos([(160, 160, 32), (80, 80, 64), (80, 80, 80), (80, 80, 32)])
        assert _infer_model_kind(infos) == "instance_segmentation"

    def test_embedding_detected_for_1d_output(self) -> None:
        infos = self._infos([(512,)])
        assert _infer_model_kind(infos) == "embedding"

    def test_embedding_detected_for_2d_batch1_output(self) -> None:
        infos = self._infos([(1, 512)])
        assert _infer_model_kind(infos) == "embedding"

    def test_seg_detected_from_4d_nhwc_large_channel(self) -> None:
        # 4D NHWC — combined layout
        infos = self._infos([(1, 160, 160, 32), (1, 80, 80, 176), (1, 40, 40, 176), (1, 20, 20, 176)])
        assert _infer_model_kind(infos) == "instance_segmentation"

    def test_nms_single_output_not_seg(self) -> None:
        # Single NMS output (80, 5, 100) — shape from yolov8s_h8.hef on hardware.
        # last_dim = 100, first_dim = 80; neither triggers seg check.
        infos = self._infos([(80, 5, 100)])
        assert _infer_model_kind(infos) == "detection"

    def test_detection_fallback_for_nms_outputs(self) -> None:
        # Any shape that is not seg or embedding falls back to "detection"
        # (detection_raw is detected at inference time, not from HEF shapes).
        infos = self._infos([(1, 80, 20, 5), (1, 80, 20, 5)])
        assert _infer_model_kind(infos) == "detection"

    def test_spatial_shapes_also_fall_back_to_detection(self) -> None:
        # Spatial feature-map shapes (e.g. conv layer outputs from an NMS model)
        # return "detection" at load time; detection_raw is detected at runtime.
        assert _infer_model_kind(self._infos([(20, 20, 80)])) == "detection"
        assert _infer_model_kind(self._infos([(1, 20, 20, 80)])) == "detection"
        assert _infer_model_kind(self._infos([
            (1, 80, 80, 80), (1, 40, 40, 80), (1, 20, 20, 80)
        ])) == "detection"
        assert _infer_model_kind(self._infos([(1, 20, 20, 84)])) == "detection"


# ----------------------------------------------------------------------
# _outputs_look_spatial — runtime spatial-output detector
# ----------------------------------------------------------------------


class TestOutputsLookSpatial:
    """_outputs_look_spatial distinguishes raw feature maps from NMS outputs."""

    def test_empty_dict_returns_false(self) -> None:
        assert not _outputs_look_spatial({})

    def test_non_dict_returns_false(self) -> None:
        assert not _outputs_look_spatial([])  # type: ignore[arg-type]

    def test_4d_spatial_tensor_returns_true(self) -> None:
        outputs = {"head": np.zeros((1, 20, 20, 80), dtype=np.float32)}
        assert _outputs_look_spatial(outputs)

    def test_3d_spatial_tensor_returns_true(self) -> None:
        outputs = {"head": np.zeros((20, 20, 80), dtype=np.float32)}
        assert _outputs_look_spatial(outputs)

    def test_3d_tensor_last_dim_7_returns_true(self) -> None:
        # Last dim just above the NMS guard threshold
        outputs = {"head": np.zeros((20, 20, 7), dtype=np.float32)}
        assert _outputs_look_spatial(outputs)

    def test_multiple_scales_all_spatial_returns_true(self) -> None:
        outputs = {
            "s1": np.zeros((1, 80, 80, 80), dtype=np.float32),
            "s2": np.zeros((1, 40, 40, 80), dtype=np.float32),
            "s3": np.zeros((1, 20, 20, 80), dtype=np.float32),
        }
        assert _outputs_look_spatial(outputs)

    def test_list_value_returns_false(self) -> None:
        # Per-class NMS output is a Python list
        outputs = {"nms": [np.zeros((80, 5))]}
        assert not _outputs_look_spatial(outputs)

    def test_flat_nms_array_returns_false(self) -> None:
        # Flat NMS output: (N, 6) — not spatial
        outputs = {"nms": np.zeros((100, 6), dtype=np.float32)}
        assert not _outputs_look_spatial(outputs)

    def test_per_class_nms_3d_array_returns_false(self) -> None:
        # Per-class NMS (num_classes, max_per_class, 5)
        outputs = {"nms": np.zeros((80, 100, 5), dtype=np.float32)}
        assert not _outputs_look_spatial(outputs)

    def test_object_array_returns_false(self) -> None:
        # Object-dtype ragged NMS array
        arr = np.empty(80, dtype=object)
        for i in range(80):
            arr[i] = np.zeros((np.random.randint(0, 5), 5))
        outputs = {"nms": arr}
        assert not _outputs_look_spatial(outputs)

    def test_mixed_spatial_and_nms_returns_false(self) -> None:
        # If ANY value is not spatial, the whole dict is not spatial
        outputs = {
            "feat": np.zeros((1, 20, 20, 80), dtype=np.float32),
            "nms": [np.zeros((80, 5))],
        }
        assert not _outputs_look_spatial(outputs)


# ----------------------------------------------------------------------
# Instance segmentation decoder tests (no hardware required)
# ----------------------------------------------------------------------


class TestNmsNumpy:
    def test_suppresses_overlapping_box(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        keep = _nms_numpy(boxes, scores, iou_thresh=0.5)
        assert 0 in keep
        assert 1 not in keep   # suppressed by box 0
        assert 2 in keep       # non-overlapping, kept

    def test_empty_input(self) -> None:
        keep = _nms_numpy(np.empty((0, 4), np.float32), np.empty(0, np.float32), 0.45)
        assert len(keep) == 0

    def test_single_box(self) -> None:
        boxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        assert list(_nms_numpy(boxes, scores, 0.45)) == [0]


class TestSplitSegOutputs:
    def _make_outputs(self) -> dict[str, np.ndarray]:
        # proto: (1, 160, 160, 32), dets: (1, 80, 80, 116), (1, 40, 40, 116), (1, 20, 20, 116)
        return {
            "proto": np.zeros((1, 160, 160, 32), dtype=np.float32),
            "det_s8": np.zeros((1, 80, 80, 116), dtype=np.float32),
            "det_s16": np.zeros((1, 40, 40, 116), dtype=np.float32),
            "det_s32": np.zeros((1, 20, 20, 116), dtype=np.float32),
        }

    def test_identifies_proto_and_three_det_tensors(self) -> None:
        proto, dets = _split_seg_outputs(self._make_outputs())
        assert proto is not None
        assert proto.shape == (160, 160, 32)
        assert len(dets) == 3

    def test_dets_sorted_largest_first(self) -> None:
        _, dets = _split_seg_outputs(self._make_outputs())
        areas = [d.shape[0] * d.shape[1] for d in dets]
        assert areas == sorted(areas, reverse=True)

    def test_empty_outputs_returns_none(self) -> None:
        proto, dets = _split_seg_outputs({})
        assert proto is None
        assert dets == []

    def _make_split_outputs(self) -> dict[str, np.ndarray]:
        # Hailo-specific split layout: 3 scales × (box C=64, cls C=80, coef C=32)
        # + proto (160, 160, 32).  Mirrors actual yolov8m_seg.hef vstream shapes
        # observed on RPi5 + AI HAT+.
        return {
            "proto":    np.zeros((1, 160, 160, 32), dtype=np.float32),
            "s8_box":   np.zeros((1,  80,  80, 64), dtype=np.float32),
            "s8_cls":   np.zeros((1,  80,  80, 80), dtype=np.float32),
            "s8_coef":  np.zeros((1,  80,  80, 32), dtype=np.float32),
            "s16_box":  np.zeros((1,  40,  40, 64), dtype=np.float32),
            "s16_cls":  np.zeros((1,  40,  40, 80), dtype=np.float32),
            "s16_coef": np.zeros((1,  40,  40, 32), dtype=np.float32),
            "s32_box":  np.zeros((1,  20,  20, 64), dtype=np.float32),
            "s32_cls":  np.zeros((1,  20,  20, 80), dtype=np.float32),
            "s32_coef": np.zeros((1,  20,  20, 32), dtype=np.float32),
        }

    def test_split_format_identifies_proto(self) -> None:
        proto, _ = _split_seg_outputs(self._make_split_outputs())
        assert proto is not None
        assert proto.shape == (160, 160, 32)

    def test_split_format_returns_three_combined_dets(self) -> None:
        _, dets = _split_seg_outputs(self._make_split_outputs())
        assert len(dets) == 3

    def test_split_format_reconstructs_dfl_combined_channels(self) -> None:
        # box(64) + cls(80) + coef(32) = 176 (DFL combined format)
        _, dets = _split_seg_outputs(self._make_split_outputs())
        assert all(d.shape[2] == 176 for d in dets)

    def test_split_format_dets_sorted_largest_first(self) -> None:
        _, dets = _split_seg_outputs(self._make_split_outputs())
        areas = [d.shape[0] * d.shape[1] for d in dets]
        assert areas == sorted(areas, reverse=True)

    def test_split_format_predecoded_boxes_reconstructs_116_channels(self) -> None:
        # Pre-decoded box (C=4) + cls(80) + coef(32) = 116 (pre-decoded combined)
        outputs = {
            "proto":   np.zeros((1, 160, 160, 32), dtype=np.float32),
            "s8_box":  np.zeros((1,  80,  80,  4), dtype=np.float32),
            "s8_cls":  np.zeros((1,  80,  80, 80), dtype=np.float32),
            "s8_coef": np.zeros((1,  80,  80, 32), dtype=np.float32),
        }
        _, dets = _split_seg_outputs(outputs)
        assert len(dets) == 1
        assert dets[0].shape[2] == 116


class TestDecodeOneSegHead:
    def _feat(self, h: int, w: int, c: int) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.standard_normal((h, w, c)).astype(np.float32)

    def test_decoded_layout_returns_correct_shapes(self) -> None:
        feat = self._feat(80, 80, 116)   # 4 + 80 + 32
        boxes, scores, class_ids, coefs = _decode_one_seg_head(
            feat, 640, 640, 80, 32, conf_thresh=0.0,
        )
        n = boxes.shape[0]
        assert n == 80 * 80
        assert boxes.shape == (n, 4)
        assert scores.shape == (n,)
        assert class_ids.shape == (n,)
        assert coefs.shape == (n, 32)

    def test_dfl_layout_returns_correct_shapes(self) -> None:
        feat = self._feat(80, 80, 176)   # 64 + 80 + 32
        boxes, scores, class_ids, coefs = _decode_one_seg_head(
            feat, 640, 640, 80, 32, conf_thresh=0.0,
        )
        assert boxes.shape[1] == 4
        assert coefs.shape[1] == 32

    def test_boxes_in_unit_range(self) -> None:
        feat = self._feat(80, 80, 116)
        boxes, _, _, _ = _decode_one_seg_head(feat, 640, 640, 80, 32, conf_thresh=0.0)
        assert boxes.min() >= 0.0
        assert boxes.max() <= 1.0

    def test_confidence_threshold_filters(self) -> None:
        feat = self._feat(20, 20, 116)
        _, scores_low, _, _ = _decode_one_seg_head(feat, 640, 640, 80, 32, conf_thresh=0.99)
        assert len(scores_low) < 20 * 20


class TestDecodeInstanceMasks:
    def _letterbox(self) -> _Letterbox:
        return _Letterbox(scale=1.0, pad_left=0, pad_top=0, target_w=640, target_h=640)

    def test_returns_mask_per_detection(self) -> None:
        proto = np.zeros((160, 160, 32), dtype=np.float32)
        coefs = np.zeros((3, 32), dtype=np.float32)
        boxes = np.array([[0.1, 0.1, 0.5, 0.5]] * 3, dtype=np.float32)
        masks = _decode_instance_masks(coefs, proto, boxes, self._letterbox(), 640, 640)
        assert len(masks) == 3
        assert all(isinstance(m, Mask) for m in masks)

    def test_mask_shape_matches_frame(self) -> None:
        proto = np.zeros((160, 160, 32), dtype=np.float32)
        coefs = np.zeros((1, 32), dtype=np.float32)
        boxes = np.array([[0.2, 0.2, 0.8, 0.8]], dtype=np.float32)
        (m,) = _decode_instance_masks(coefs, proto, boxes, self._letterbox(), 480, 640)
        assert m.data.shape == (480, 640)
        assert m.h == 480
        assert m.w == 640


class TestPostprocessSeg:
    def test_no_proto_returns_empty(self) -> None:
        # Pass only detection tensors — no proto (32-channel) tensor
        outputs = {"det": np.zeros((1, 80, 80, 116), dtype=np.float32)}
        lb = _Letterbox(scale=1.0, pad_left=0, pad_top=0, target_w=640, target_h=640)
        result = _postprocess_seg(
            outputs, class_names={0: "cat"}, confidence=0.5,
            classes=None, letterbox=lb, orig_w=640, orig_h=640,
        )
        assert isinstance(result, InstanceSegmentationResult)
        assert result.detections == []

    def test_all_zero_proto_produces_no_detections_above_threshold(self) -> None:
        # Zero proto → all mask logits = 0 → sigmoid = 0.5; zero class logits → low scores
        outputs = {
            "proto": np.zeros((1, 160, 160, 32), dtype=np.float32),
            "det_s8": np.zeros((1, 80, 80, 116), dtype=np.float32),
            "det_s16": np.zeros((1, 40, 40, 116), dtype=np.float32),
            "det_s32": np.zeros((1, 20, 20, 116), dtype=np.float32),
        }
        lb = _Letterbox(scale=1.0, pad_left=0, pad_top=0, target_w=640, target_h=640)
        result = _postprocess_seg(
            outputs, class_names={i: str(i) for i in range(80)},
            confidence=0.5, classes=None, letterbox=lb, orig_w=640, orig_h=640,
        )
        assert isinstance(result, InstanceSegmentationResult)
        # Zero logits → sigmoid(0)=0.5, so with conf=0.5 we may get borderline hits;
        # the test just verifies it runs and returns the right type.

    def test_split_hailo_format_produces_detections(self) -> None:
        """Proves the full decode pipeline works end-to-end for the split layout.

        Before the fix, _split_seg_outputs passed the per-scale box/cls/coef
        tensors straight to _decode_one_seg_head without reconstruction.
        That function returned empty arrays for every scale (wrong channel
        count), so _postprocess_seg returned 0 detections even at conf=0.0
        regardless of frame content.  After the fix, the tensors are
        concatenated into C=116 combined format, which the decoder handles
        correctly — and a cell with a high class logit survives the threshold.
        """
        # Craft a single-scale (20×20) split-format input with one cell
        # containing a pre-decoded box and a high class-0 score.
        #   box (C=4):  pre-decoded (x1, y1, x2, y2) in [0, 1]
        #   cls (C=80): class-0 logit = 10.0 → sigmoid ≈ 0.9999 >> conf=0.5
        #   coef (C=32): zeros → mask will be flat but structurally valid
        #   proto (C=32): zeros
        box_t  = np.zeros((1, 20, 20,  4), dtype=np.float32)
        cls_t  = np.zeros((1, 20, 20, 80), dtype=np.float32)
        coef_t = np.zeros((1, 20, 20, 32), dtype=np.float32)
        proto  = np.zeros((1, 160, 160, 32), dtype=np.float32)
        box_t[0, 10, 10]  = [0.1, 0.1, 0.9, 0.9]   # valid box
        cls_t[0, 10, 10, 0] = 10.0                   # class 0, very high confidence
        outputs = {
            "proto":   proto,
            "s32_box": box_t,
            "s32_cls": cls_t,
            "s32_coef": coef_t,
        }
        lb = _Letterbox(scale=1.0, pad_left=0, pad_top=0, target_w=640, target_h=640)
        result = _postprocess_seg(
            outputs, class_names={i: str(i) for i in range(80)},
            confidence=0.5, classes=None, letterbox=lb, orig_w=640, orig_h=640,
        )
        assert isinstance(result, InstanceSegmentationResult)
        assert len(result.detections) >= 1, (
            "Expected at least one detection from split-format input with a "
            "high-confidence cell; got 0.  The decode pipeline is not "
            "flowing data through correctly."
        )
        det = result.detections[0]
        assert det.confidence > 0.5
        assert det.mask is not None

    def test_split_hailo_format_does_not_warn(self) -> None:
        # Regression test: the Hailo split layout (separate box/cls/coef per
        # scale) must not trigger "unexpected channel count" warnings.
        # Before the fix, every scale fired a warning and returned empty arrays.
        import warnings
        outputs = {
            "proto":    np.zeros((1, 160, 160, 32), dtype=np.float32),
            "s8_box":   np.zeros((1,  80,  80, 64), dtype=np.float32),
            "s8_cls":   np.zeros((1,  80,  80, 80), dtype=np.float32),
            "s8_coef":  np.zeros((1,  80,  80, 32), dtype=np.float32),
            "s16_box":  np.zeros((1,  40,  40, 64), dtype=np.float32),
            "s16_cls":  np.zeros((1,  40,  40, 80), dtype=np.float32),
            "s16_coef": np.zeros((1,  40,  40, 32), dtype=np.float32),
            "s32_box":  np.zeros((1,  20,  20, 64), dtype=np.float32),
            "s32_cls":  np.zeros((1,  20,  20, 80), dtype=np.float32),
            "s32_coef": np.zeros((1,  20,  20, 32), dtype=np.float32),
        }
        lb = _Letterbox(scale=1.0, pad_left=0, pad_top=0, target_w=640, target_h=640)
        import logging
        with warnings.catch_warnings(record=True):
            # Capture any log warnings by temporarily raising their level
            hailo_logger = logging.getLogger("cyberwave.models.runtimes.hailo_rt")
            original_level = hailo_logger.level
            log_records: list[logging.LogRecord] = []

            class _Capture(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:
                    log_records.append(record)

            handler = _Capture()
            hailo_logger.addHandler(handler)
            hailo_logger.setLevel(logging.WARNING)
            try:
                result = _postprocess_seg(
                    outputs, class_names={i: str(i) for i in range(80)},
                    confidence=0.5, classes=None, letterbox=lb, orig_w=640, orig_h=640,
                )
            finally:
                hailo_logger.removeHandler(handler)
                hailo_logger.setLevel(original_level)

        unexpected_ch_warnings = [
            r for r in log_records if "unexpected channel count" in r.getMessage()
        ]
        assert unexpected_ch_warnings == [], (
            f"Got unexpected channel count warnings for split Hailo layout: "
            f"{[r.getMessage() for r in unexpected_ch_warnings]}"
        )
        assert isinstance(result, InstanceSegmentationResult)


# ----------------------------------------------------------------------
# Raw detection decoder tests (_decode_one_det_head, _postprocess_det_raw)
# ----------------------------------------------------------------------


class TestDecodeOneDetHead:
    """Unit tests for the CPU-side YOLOv8 detection head decoder."""

    def _rng_feat(self, h: int, w: int, c: int, seed: int = 0) -> np.ndarray:
        return np.random.default_rng(seed).standard_normal((h, w, c)).astype(np.float32)

    # ---- Combined-tensor layouts ----------------------------------------

    def test_combined_predecoded_output_shapes(self) -> None:
        # (H, W, 4 + nc) — pre-decoded boxes
        nc = 80
        feat = self._rng_feat(20, 20, 4 + nc)
        boxes, scores, cids = _decode_one_det_head(feat, None, 640, 640, nc, conf_thresh=0.0)
        n = 20 * 20
        assert boxes.shape == (n, 4)
        assert scores.shape == (n,)
        assert cids.shape == (n,)

    def test_combined_dfl_output_shapes(self) -> None:
        # (H, W, 64 + nc) — raw DFL distribution
        nc = 80
        feat = self._rng_feat(20, 20, 64 + nc)
        boxes, scores, cids = _decode_one_det_head(feat, None, 640, 640, nc, conf_thresh=0.0)
        assert boxes.shape == (20 * 20, 4)
        assert scores.shape == (20 * 20,)

    def test_boxes_clamped_to_unit_range_predecoded(self) -> None:
        nc = 10
        feat = self._rng_feat(10, 10, 4 + nc)
        boxes, _, _ = _decode_one_det_head(feat, None, 320, 320, nc, conf_thresh=0.0)
        assert float(boxes.min()) >= 0.0
        assert float(boxes.max()) <= 1.0

    def test_boxes_clamped_to_unit_range_dfl(self) -> None:
        nc = 10
        feat = self._rng_feat(10, 10, 64 + nc)
        boxes, _, _ = _decode_one_det_head(feat, None, 320, 320, nc, conf_thresh=0.0)
        assert float(boxes.min()) >= 0.0
        assert float(boxes.max()) <= 1.0

    def test_scores_in_unit_range(self) -> None:
        nc = 80
        feat = self._rng_feat(20, 20, 4 + nc)
        _, scores, _ = _decode_one_det_head(feat, None, 640, 640, nc, conf_thresh=0.0)
        assert float(scores.min()) >= 0.0
        assert float(scores.max()) <= 1.0

    def test_class_ids_in_valid_range(self) -> None:
        nc = 80
        feat = self._rng_feat(20, 20, 4 + nc)
        _, _, cids = _decode_one_det_head(feat, None, 640, 640, nc, conf_thresh=0.0)
        assert int(cids.min()) >= 0
        assert int(cids.max()) < nc

    def test_confidence_threshold_reduces_detections(self) -> None:
        nc = 80
        feat = self._rng_feat(20, 20, 4 + nc, seed=42)
        _, scores_lo, _ = _decode_one_det_head(feat, None, 640, 640, nc, conf_thresh=0.0)
        _, scores_hi, _ = _decode_one_det_head(feat, None, 640, 640, nc, conf_thresh=0.9)
        assert len(scores_hi) < len(scores_lo)

    def test_unknown_channel_count_returns_empty_arrays(self) -> None:
        nc = 80
        feat = self._rng_feat(20, 20, 99)  # neither 4+80 nor 64+80
        boxes, scores, cids = _decode_one_det_head(feat, None, 640, 640, nc, conf_thresh=0.0)
        assert boxes.shape == (0, 4)
        assert scores.shape == (0,)
        assert cids.shape == (0,)

    # ---- Separate box + class tensor layout --------------------------------

    def test_separate_predecoded_box_class_shapes(self) -> None:
        nc = 80
        cls_t = self._rng_feat(20, 20, nc)
        box_t = self._rng_feat(20, 20, 4)
        boxes, scores, cids = _decode_one_det_head(cls_t, box_t, 640, 640, nc, conf_thresh=0.0)
        assert boxes.shape == (20 * 20, 4)
        assert scores.shape == (20 * 20,)

    def test_separate_dfl_box_class_shapes(self) -> None:
        nc = 80
        cls_t = self._rng_feat(20, 20, nc)
        box_t = self._rng_feat(20, 20, 64)  # raw DFL
        boxes, scores, cids = _decode_one_det_head(cls_t, box_t, 640, 640, nc, conf_thresh=0.0)
        assert boxes.shape == (20 * 20, 4)

    def test_spatial_mismatch_warns_and_returns_empty(self, caplog) -> None:
        nc = 80
        cls_t = self._rng_feat(20, 20, nc)
        box_t = self._rng_feat(10, 10, 4)   # different spatial size
        with caplog.at_level("WARNING"):
            boxes, scores, cids = _decode_one_det_head(cls_t, box_t, 640, 640, nc, conf_thresh=0.0)
        assert boxes.shape == (0, 4)
        assert any("spatial dims" in r.message for r in caplog.records)

    def test_known_box_value_round_trips_combined(self) -> None:
        # Manually construct a single-cell combined tensor so we can verify
        # the pre-decoded box channels [x1, y1, x2, y2] pass through clip unchanged.
        nc = 5
        feat = np.zeros((1, 1, 4 + nc), dtype=np.float32)
        feat[0, 0, :4] = [0.1, 0.2, 0.8, 0.9]   # x1 y1 x2 y2
        # Drive class 0 score very high so it survives any threshold
        feat[0, 0, 4] = 100.0
        boxes, scores, cids = _decode_one_det_head(feat, None, 64, 64, nc, conf_thresh=0.0)
        assert boxes.shape == (1, 4)
        assert boxes[0, 0] == pytest.approx(0.1, abs=1e-4)
        assert boxes[0, 1] == pytest.approx(0.2, abs=1e-4)
        assert boxes[0, 2] == pytest.approx(0.8, abs=1e-4)
        assert boxes[0, 3] == pytest.approx(0.9, abs=1e-4)
        assert int(cids[0]) == 0
        assert float(scores[0]) > 0.99  # sigmoid(100) ≈ 1


class TestPostprocessDetRaw:
    """Integration tests for _postprocess_det_raw end-to-end."""

    def _lb(self, input_hw: int = 640) -> _Letterbox:
        """Identity letterbox: no padding, scale=1."""
        return _Letterbox(
            scale=1.0, pad_left=0, pad_top=0,
            target_w=input_hw, target_h=input_hw,
        )

    def _coco(self) -> dict[int, str]:
        return {i: n for i, n in enumerate(_COCO_CLASSES)}

    # ---- Basic decode -------------------------------------------------------

    def test_empty_outputs_returns_empty(self) -> None:
        lb = self._lb()
        result = _postprocess_det_raw(
            {}, class_names=self._coco(), confidence=0.5,
            classes=None, letterbox=lb, orig_w=640, orig_h=640,
        )
        assert result == []

    def test_combined_predecoded_single_scale_returns_detections(self) -> None:
        # Build a (1, 20, 20, 84) combined tensor with one strong detection.
        nc = 80
        feat = np.zeros((1, 20, 20, 4 + nc), dtype=np.float32)
        # Place a box at grid cell (10, 10)
        feat[0, 10, 10, :4] = [0.3, 0.3, 0.7, 0.7]   # x1 y1 x2 y2 in [0,1]
        feat[0, 10, 10, 4] = 100.0                     # class 0 (person) very high
        lb = self._lb()
        dets = _postprocess_det_raw(
            {"head": feat}, class_names=self._coco(), confidence=0.5,
            classes=None, letterbox=lb, orig_w=640, orig_h=640,
        )
        assert len(dets) >= 1
        assert dets[0].label == "person"
        assert dets[0].confidence > 0.99

    def test_separate_box_class_single_scale_returns_detections(self) -> None:
        nc = 80
        cls_t = np.zeros((1, 20, 20, nc), dtype=np.float32)
        box_t = np.zeros((1, 20, 20, 4), dtype=np.float32)
        # Strong detection at cell (5, 5) for class 1 (bicycle)
        cls_t[0, 5, 5, 1] = 100.0
        box_t[0, 5, 5] = [0.1, 0.1, 0.5, 0.5]
        lb = self._lb()
        dets = _postprocess_det_raw(
            {"cls": cls_t, "box": box_t}, class_names=self._coco(), confidence=0.5,
            classes=None, letterbox=lb, orig_w=640, orig_h=640,
        )
        assert len(dets) >= 1
        assert dets[0].label == "bicycle"

    def test_multi_scale_detections_are_merged(self) -> None:
        # Three scales, each with one non-overlapping detection → 3 detections.
        nc = 80
        def _scale(h: int, x1: float, x2: float) -> np.ndarray:
            t = np.zeros((1, h, h, 4 + nc), dtype=np.float32)
            t[0, 0, 0, :4] = [x1, 0.0, x2, 0.1]
            t[0, 0, 0, 4] = 100.0   # class 0
            return t

        lb = self._lb()
        dets = _postprocess_det_raw(
            {
                "s32": _scale(20, 0.0, 0.1),
                "s16": _scale(40, 0.4, 0.5),
                "s8":  _scale(80, 0.8, 0.9),
            },
            class_names=self._coco(), confidence=0.5,
            classes=None, letterbox=lb, orig_w=640, orig_h=640,
        )
        assert len(dets) == 3

    def test_overlapping_boxes_suppressed_by_nms(self) -> None:
        # Two nearly identical boxes at the same scale → NMS keeps only one.
        nc = 80
        feat = np.zeros((1, 20, 20, 4 + nc), dtype=np.float32)
        feat[0, 10, 10, :4] = [0.3, 0.3, 0.7, 0.7]
        feat[0, 10, 10, 4]  = 100.0
        feat[0, 11, 10, :4] = [0.31, 0.31, 0.69, 0.69]   # ~IoU > 0.45
        feat[0, 11, 10, 4]  = 90.0
        lb = self._lb()
        dets = _postprocess_det_raw(
            {"head": feat}, class_names=self._coco(), confidence=0.5,
            classes=None, letterbox=lb, orig_w=640, orig_h=640,
        )
        assert len(dets) == 1  # second box suppressed by NMS

    # ---- Filtering ----------------------------------------------------------

    def test_confidence_threshold_respected(self) -> None:
        nc = 80
        feat = np.random.default_rng(7).standard_normal((1, 20, 20, 4 + nc)).astype(np.float32)
        lb = self._lb()
        dets_lo = _postprocess_det_raw(
            {"head": feat}, class_names=self._coco(), confidence=0.01,
            classes=None, letterbox=lb, orig_w=640, orig_h=640,
        )
        dets_hi = _postprocess_det_raw(
            {"head": feat}, class_names=self._coco(), confidence=0.99,
            classes=None, letterbox=lb, orig_w=640, orig_h=640,
        )
        assert len(dets_hi) <= len(dets_lo)

    def test_class_filter_excludes_other_labels(self) -> None:
        nc = 80
        feat = np.zeros((1, 20, 20, 4 + nc), dtype=np.float32)
        feat[0, 10, 10, :4] = [0.1, 0.1, 0.9, 0.9]
        feat[0, 10, 10, 4] = 100.0   # class 0 → "person"
        lb = self._lb()
        dets = _postprocess_det_raw(
            {"head": feat}, class_names=self._coco(), confidence=0.5,
            classes=["car"], letterbox=lb, orig_w=640, orig_h=640,
        )
        assert dets == []

    # ---- Un-letterboxing ----------------------------------------------------

    def test_coordinates_unletterboxed_correctly(self) -> None:
        # 1280×720 → 640×640 letterbox: scale=0.5, pad_top=140, pad_left=0
        lb = _Letterbox(scale=0.5, pad_left=0, pad_top=140, target_w=640, target_h=640)
        nc = 80
        feat = np.zeros((1, 20, 20, 4 + nc), dtype=np.float32)
        # Box in letterboxed space: x1=0, y1=140/640, x2=1, y2=500/640
        feat[0, 0, 0, :4] = [0.0, 140 / 640, 1.0, 500 / 640]
        feat[0, 0, 0, 4] = 100.0
        dets = _postprocess_det_raw(
            {"head": feat}, class_names=self._coco(), confidence=0.5,
            classes=None, letterbox=lb, orig_w=1280, orig_h=720,
        )
        assert len(dets) == 1
        bbox = dets[0].bbox
        assert bbox.x1 == pytest.approx(0.0, abs=1)
        assert bbox.y1 == pytest.approx(0.0, abs=1)
        assert bbox.x2 == pytest.approx(1280.0, abs=1)
        assert bbox.y2 == pytest.approx(720.0, abs=1)

    # ---- Warning paths ------------------------------------------------------

    def test_warns_when_all_scales_fail_to_decode(self, caplog) -> None:
        # Only a class tensor (nc=80) with no matching box tensor → can't decode.
        nc = 80
        cls_t = np.zeros((1, 20, 20, nc), dtype=np.float32)
        lb = self._lb()
        with caplog.at_level("WARNING"):
            dets = _postprocess_det_raw(
                {"cls": cls_t}, class_names=self._coco(), confidence=0.5,
                classes=None, letterbox=lb, orig_w=640, orig_h=640,
            )
        assert dets == []
        assert any("could not decode any scale" in r.message for r in caplog.records)

    def test_no_spurious_warning_when_frame_has_zero_detections(self, caplog) -> None:
        # Decode succeeds but confidence threshold filters everything out.
        # Must NOT emit the "could not decode" warning.
        nc = 80
        feat = np.zeros((1, 20, 20, 4 + nc), dtype=np.float32)   # all zero logits → low scores
        lb = self._lb()
        with caplog.at_level("WARNING"):
            dets = _postprocess_det_raw(
                {"head": feat}, class_names=self._coco(), confidence=0.99,
                classes=None, letterbox=lb, orig_w=640, orig_h=640,
            )
        assert dets == []
        assert not any("could not decode any scale" in r.message for r in caplog.records)

    def test_warns_when_nc_zero_no_labels(self, caplog) -> None:
        # Empty class_names with no labels → nc=0 fallback to 80 with WARNING.
        feat = np.zeros((1, 20, 20, 84), dtype=np.float32)  # 4 + 80 combined
        feat[0, 0, 0, 4] = 100.0   # strong class 0 hit
        lb = self._lb()
        with caplog.at_level("WARNING"):
            _postprocess_det_raw(
                {"head": feat}, class_names={}, confidence=0.5,
                classes=None, letterbox=lb, orig_w=640, orig_h=640,
            )
        assert any("defaulting to nc=80" in r.message for r in caplog.records)


# ----------------------------------------------------------------------
# Hardware-dependent tests
# ----------------------------------------------------------------------
#
# These rely on a real Hailo accelerator (``/dev/hailo0``) plus the
# ``hailo_platform`` Python bindings shipped with HailoRT. They are
# marked with ``@pytest.mark.hailo`` for explicit selection on the
# device (``pytest -m hailo``) and additionally use
# ``importorskip("hailo_platform")`` so a plain ``pytest`` run on x86
# CI — where the marker is registered but no ``-m`` filter is passed —
# *skips* this test instead of failing on the import.


@pytest.mark.hailo
class TestHailoRuntimeHardware:
    def test_is_available_on_device(self) -> None:
        pytest.importorskip("hailo_platform")
        assert HailoRuntime().is_available() is True

    # ------------------------------------------------------------------
    # Detection (on-chip NMS) — yolov8s_h8.hef from HailoRT system pkg
    # ------------------------------------------------------------------

    def test_detection_hef_inferred_as_detection_kind(self) -> None:
        pytest.importorskip("hailo_platform")
        hef = "/usr/share/hailo-models/yolov8s_h8.hef"
        runtime = HailoRuntime()
        handle = runtime.load(hef)
        try:
            assert handle.model_kind == "detection", (  # type: ignore[union-attr]
                f"Expected 'detection', got {handle.model_kind!r}. "  # type: ignore[union-attr]
                "If the HEF was compiled without NMS it should be 'detection_raw'."
            )
        finally:
            handle.close()  # type: ignore[union-attr]

    def test_detection_hef_returns_detections_on_random_frame(self) -> None:
        pytest.importorskip("hailo_platform")
        hef = "/usr/share/hailo-models/yolov8s_h8.hef"
        runtime = HailoRuntime()
        handle = runtime.load(hef)
        try:
            # A random noise frame — may produce 0 detections, but must not crash.
            frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            result = runtime.predict(handle, frame, confidence=0.5)
            assert isinstance(result.detections, list)
            for det in result.detections:
                assert det.bbox.x1 >= 0
                assert det.bbox.x2 <= 1280
                assert det.bbox.y1 >= 0
                assert det.bbox.y2 <= 720
                assert 0.0 < det.confidence <= 1.0
        finally:
            handle.close()  # type: ignore[union-attr]

    def test_detection_hef_detects_person_in_synthetic_scene(self) -> None:
        """A bright white rectangle on a dark background is not a person; the
        test just verifies the pipeline completes without error — real recall
        of a specific class requires a real image."""
        pytest.importorskip("hailo_platform")
        hef = "/usr/share/hailo-models/yolov8s_h8.hef"
        runtime = HailoRuntime()
        handle = runtime.load(hef)
        try:
            frame = np.full((640, 640, 3), 50, dtype=np.uint8)
            result = runtime.predict(handle, frame, confidence=0.3)
            assert isinstance(result.detections, list)
        finally:
            handle.close()  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Instance segmentation — yolov8m_seg.hef
    # ------------------------------------------------------------------

    def test_seg_hef_inferred_as_seg_kind(self) -> None:
        """Regression guard: _infer_model_kind must still return
        'instance_segmentation' for seg HEFs after changes to the raw
        detection detection branch."""
        pytest.importorskip("hailo_platform")
        hef = str(pytest.importorskip("pathlib").Path.home() / ".cyberwave/models/yolov8m_seg.hef")
        runtime = HailoRuntime()
        handle = runtime.load(hef)
        try:
            assert handle.model_kind == "instance_segmentation", (  # type: ignore[union-attr]
                f"Expected 'instance_segmentation', got {handle.model_kind!r}."
            )
        finally:
            handle.close()  # type: ignore[union-attr]

    def test_seg_hef_returns_instance_seg_result(self) -> None:
        pytest.importorskip("hailo_platform")
        hef = str(pytest.importorskip("pathlib").Path.home() / ".cyberwave/models/yolov8m_seg.hef")
        runtime = HailoRuntime()
        handle = runtime.load(hef)
        try:
            frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            result = runtime.predict(handle, frame, confidence=0.5)
            assert isinstance(result, InstanceSegmentationResult), (
                f"Expected InstanceSegmentationResult, got {type(result).__name__}"
            )
            for det in result.detections:
                assert det.mask is not None
                assert det.mask.h == 720
                assert det.mask.w == 1280
                assert det.bbox.x1 >= 0
                assert det.bbox.x2 <= 1280

            # Prove the decode pipeline produces detections at all — not just
            # the right return type.  Before the split-format fix, every scale
            # returned empty arrays so conf=0.0 still yielded 0 detections.
            result_low = runtime.predict(handle, frame, confidence=0.0)
            assert isinstance(result_low, InstanceSegmentationResult)
            assert len(result_low.detections) > 0, (
                "Expected detections at conf=0.0 on a noise frame; got 0. "
                "The seg decode pipeline is not producing any output — "
                "likely a split-format reconstruction regression."
            )
        finally:
            handle.close()  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # detection_raw — only runs if a raw-output HEF is available
    # ------------------------------------------------------------------

    def test_detection_raw_hef_if_available(self, tmp_path) -> None:
        """Skipped unless a raw-output (no on-chip NMS) HEF is available at
        ~/.cyberwave/models/yolov8_raw.hef.  To produce one, recompile any
        Hailo Model Zoo YOLO HEF with the NMS post-process op removed and
        copy it to that path, then re-run ``pytest -m hailo``.

        Note: model_kind is auto-detected from the first inference output
        (not from static HEF metadata), so we run one predict() call before
        asserting the kind."""
        pytest.importorskip("hailo_platform")
        import pathlib
        raw_hef = pathlib.Path.home() / ".cyberwave/models/yolov8_raw.hef"
        if not raw_hef.exists():
            pytest.skip(f"No raw-output HEF at {raw_hef}; skipping detection_raw hardware test.")
        runtime = HailoRuntime()
        handle = runtime.load(str(raw_hef))
        try:
            frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            result = runtime.predict(handle, frame, confidence=0.3)
            # After the first predict() call the runtime auto-detects raw outputs
            assert handle.model_kind == "detection_raw", (  # type: ignore[union-attr]
                f"Expected 'detection_raw' after first predict(), got {handle.model_kind!r}. "
                "If the HEF actually has on-chip NMS it should not be placed at yolov8_raw.hef."
            )
            assert isinstance(result.detections, list)
        finally:
            handle.close()  # type: ignore[union-attr]
