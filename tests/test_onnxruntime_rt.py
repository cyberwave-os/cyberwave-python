"""Tests for cyberwave.models.runtimes.onnxruntime_rt — ONNX Runtime backend."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cyberwave.models.runtimes.onnxruntime_rt import (
    OnnxRuntime,
    _nms_per_class,
    _parse_kpt_shape,
    _postprocess,
)
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


class TestOnnxPostprocessPose:
    """Pose-model output: ``[1, 4 + 1 + K*3, N]`` where K=17 for COCO pose."""

    def _build_pose_output(
        self, *, num_keypoints: int = 17, score: float = 0.9
    ) -> np.ndarray:
        # One detection. cx,cy,w,h,person_score, then K*(x,y,vis).
        feat = 4 + 1 + num_keypoints * 3
        col = np.zeros((feat, 1), dtype=np.float32)
        col[0, 0] = 320  # cx
        col[1, 0] = 320  # cy
        col[2, 0] = 200  # w
        col[3, 0] = 400  # h
        col[4, 0] = score  # person score
        # Fill keypoints with a recognisable pattern: x=10*i, y=20*i, vis=0.5
        for k in range(num_keypoints):
            col[5 + k * 3, 0] = 10 * k
            col[5 + k * 3 + 1, 0] = 20 * k
            col[5 + k * 3 + 2, 0] = 0.5
        return col[np.newaxis, :, :]  # [1, feat, 1]

    def test_pose_layout_populates_keypoints(self):
        raw = self._build_pose_output(num_keypoints=17)
        result = _postprocess(
            raw,
            confidence=0.5,
            classes=None,
            class_names={0: "person"},
            img_w=640,
            img_h=640,
            input_shape=[1, 3, 640, 640],
            num_keypoints=17,
        )
        assert len(result) == 1
        det = result[0]
        assert det.label == "person"
        assert det.keypoints is not None
        assert det.keypoints.shape == (17, 3)
        # Sanity check the recognisable pattern.
        assert det.keypoints[5, 0] == pytest.approx(50.0)  # x = 10*5
        assert det.keypoints[5, 1] == pytest.approx(100.0)  # y = 20*5
        assert det.keypoints[5, 2] == pytest.approx(0.5)  # visibility

    def test_pose_keypoints_scaled_to_original_image(self):
        raw = self._build_pose_output(num_keypoints=17)
        # Half-height image (640 -> 320 in y), full width.
        result = _postprocess(
            raw,
            confidence=0.5,
            classes=None,
            class_names={0: "person"},
            img_w=640,
            img_h=320,
            input_shape=[1, 3, 640, 640],
            num_keypoints=17,
        )
        det = result[0]
        # x unscaled (sx=1.0), y scaled by 0.5 (sy=320/640).
        assert det.keypoints[5, 0] == pytest.approx(50.0)
        assert det.keypoints[5, 1] == pytest.approx(50.0)  # 100 * 0.5
        # Visibility is unchanged.
        assert det.keypoints[5, 2] == pytest.approx(0.5)

    def test_detection_layout_has_no_keypoints(self):
        raw = np.array(
            [[[320], [320], [100], [100], [0.85], [0.15]]],
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
            num_keypoints=0,
        )
        assert len(result) == 1
        assert result[0].keypoints is None

    def test_pose_filters_by_confidence(self):
        raw = self._build_pose_output(num_keypoints=17, score=0.3)
        result = _postprocess(
            raw,
            confidence=0.5,
            classes=None,
            class_names={0: "person"},
            img_w=640,
            img_h=640,
            input_shape=[1, 3, 640, 640],
            num_keypoints=17,
        )
        assert result == []


class TestParseKptShape:
    def test_returns_zero_when_metadata_missing(self):
        meta = MagicMock()
        meta.custom_metadata_map = {}
        assert _parse_kpt_shape(meta) == (0, 0)

    def test_parses_list_literal(self):
        meta = MagicMock()
        meta.custom_metadata_map = {"kpt_shape": "[17, 3]"}
        assert _parse_kpt_shape(meta) == (17, 3)

    def test_parses_tuple_literal(self):
        meta = MagicMock()
        meta.custom_metadata_map = {"kpt_shape": "(11, 3)"}
        assert _parse_kpt_shape(meta) == (11, 3)

    def test_parses_visibility_less_export(self):
        """Some exports drop the visibility column (``[K, 2]``)."""
        meta = MagicMock()
        meta.custom_metadata_map = {"kpt_shape": "[17, 2]"}
        assert _parse_kpt_shape(meta) == (17, 2)

    def test_returns_zero_on_malformed(self):
        meta = MagicMock()
        meta.custom_metadata_map = {"kpt_shape": "garbage"}
        assert _parse_kpt_shape(meta) == (0, 0)


class TestPostprocessDynamicAxes:
    """Dynamic-axis exports declare ``input_shape = [None, 3, None, None]``;
    bbox / keypoint scaling MUST NOT divide by 1 in that case (which would
    blow coordinates up by ``img_w`` / ``img_h``).
    """

    def test_dynamic_input_shape_skips_scaling(self):
        # Single detection at cx=200, cy=300 in a 640x480 frame the ONNX
        # session was fed at native resolution (dynamic axes).
        raw = np.array(
            [[[200], [300], [40], [60], [0.9], [0.1]]],
            dtype=np.float32,
        )
        result = _postprocess(
            raw,
            confidence=0.5,
            classes=None,
            class_names={0: "person", 1: "car"},
            img_w=640,
            img_h=480,
            input_shape=[None, 3, None, None],
        )
        assert len(result) == 1
        # Coords stay in input-space (model and image are the same here).
        assert result[0].bbox.x1 == pytest.approx(180.0)
        assert result[0].bbox.x2 == pytest.approx(220.0)
        assert result[0].bbox.y1 == pytest.approx(270.0)
        assert result[0].bbox.y2 == pytest.approx(330.0)


class TestPostprocessVisibilityLessKeypoints:
    """``kpt_shape = [K, 2]`` — no visibility column."""

    def test_kp_dim_2_returns_xy_only(self):
        num_keypoints = 17
        kp_dim = 2
        feat = 4 + 1 + num_keypoints * kp_dim
        col = np.zeros((feat, 1), dtype=np.float32)
        col[0, 0] = 320  # cx
        col[1, 0] = 320  # cy
        col[2, 0] = 200
        col[3, 0] = 400
        col[4, 0] = 0.9
        for k in range(num_keypoints):
            col[5 + k * 2, 0] = 10 * k
            col[5 + k * 2 + 1, 0] = 20 * k
        raw = col[np.newaxis, :, :]

        result = _postprocess(
            raw,
            confidence=0.5,
            classes=None,
            class_names={0: "person"},
            img_w=640,
            img_h=640,
            input_shape=[1, 3, 640, 640],
            num_keypoints=17,
            kp_dim=2,
        )
        assert len(result) == 1
        assert result[0].keypoints is not None
        assert result[0].keypoints.shape == (17, 2)
        assert result[0].keypoints[5, 0] == pytest.approx(50.0)
        assert result[0].keypoints[5, 1] == pytest.approx(100.0)


class TestPostprocessNMS:
    """YOLO ONNX exports emit ~8400 anchor predictions; without NMS each
    real object becomes a cluster of overlapping boxes around the same
    location.  The user-visible symptom that motivated this layer:
    swapping ``yolov8s.pt`` for ``yolov8s.onnx`` produced overlapping
    bounding boxes on the WebRTC overlay because the ``.pt`` Ultralytics
    path applied NMS internally while the raw onnxruntime adapter did
    not.
    """

    @staticmethod
    def _cluster_around(
        cx: float, cy: float, w: float, h: float, scores: list[float], cls_id: int = 0
    ) -> np.ndarray:
        """Build a [1, 4+1, N] tensor of N near-identical boxes for one class.

        All boxes share the same centre/size; only the per-anchor
        confidence varies.  This mimics what YOLO does: many anchors
        light up around a real object, all with very high IoU between
        each other.
        """
        n = len(scores)
        feat = np.zeros((4 + 1, n), dtype=np.float32)
        feat[0] = cx
        feat[1] = cy
        feat[2] = w
        feat[3] = h
        feat[4 + cls_id] = np.asarray(scores, dtype=np.float32)
        return feat[np.newaxis, :, :]

    def test_clustered_anchors_collapse_to_single_detection(self):
        # 6 near-identical "person" boxes around (320, 320) — IoU between
        # any two of them is ≈1.0, so NMS should pick the highest-scoring
        # one and suppress the other 5.  N=6 keeps the (feat=5, N=6)
        # tensor non-square so ``_postprocess``'s auto-transpose does
        # the right thing.
        raw = self._cluster_around(
            cx=320, cy=320, w=100, h=200,
            scores=[0.92, 0.88, 0.85, 0.81, 0.77, 0.72],
        )
        result = _postprocess(
            raw,
            confidence=0.5,
            classes=None,
            class_names={0: "person"},
            img_w=640,
            img_h=640,
            input_shape=[1, 3, 640, 640],
        )
        assert len(result) == 1
        assert result[0].label == "person"
        # Highest-scoring anchor wins.
        assert result[0].confidence == pytest.approx(0.92)

    def test_per_class_nms_does_not_suppress_overlapping_other_class(self):
        # Same box, two classes — a high-confidence person at (320,320)
        # and a slightly lower-confidence handbag at the same spot.
        # Per-class NMS must keep both.
        cx, cy, w, h = 320.0, 320.0, 100.0, 100.0
        feat = np.zeros((4 + 2, 2), dtype=np.float32)
        feat[0] = cx
        feat[1] = cy
        feat[2] = w
        feat[3] = h
        # Anchor 0: person 0.95 / handbag 0.0
        # Anchor 1: person 0.0  / handbag 0.85
        feat[4, 0] = 0.95
        feat[5, 1] = 0.85
        raw = feat[np.newaxis, :, :]

        result = _postprocess(
            raw,
            confidence=0.5,
            classes=None,
            class_names={0: "person", 1: "handbag"},
            img_w=640,
            img_h=640,
            input_shape=[1, 3, 640, 640],
        )
        labels = sorted(d.label for d in result)
        assert labels == ["handbag", "person"]

    def test_iou_one_disables_nms(self):
        # Same clustered input as above; with iou=1.0 the suppression
        # is short-circuited and every anchor crossing confidence
        # survives — the documented escape hatch for callers that want
        # raw output (custom tracker, ensemble, debugging).
        raw = self._cluster_around(
            cx=320, cy=320, w=100, h=200,
            scores=[0.92, 0.88, 0.85, 0.81, 0.77, 0.72],
        )
        result = _postprocess(
            raw,
            confidence=0.5,
            classes=None,
            class_names={0: "person"},
            img_w=640,
            img_h=640,
            input_shape=[1, 3, 640, 640],
            iou=1.0,
        )
        assert len(result) == 6

    def test_disjoint_boxes_all_kept(self):
        # Three boxes far apart; IoU between any pair is 0, so NMS
        # mustn't suppress anything regardless of how strict iou is.
        feat = np.zeros((4 + 1, 3), dtype=np.float32)
        feat[0] = np.array([100, 320, 540], dtype=np.float32)  # cx
        feat[1] = np.array([100, 320, 540], dtype=np.float32)  # cy
        feat[2] = 50  # w
        feat[3] = 50  # h
        feat[4] = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        raw = feat[np.newaxis, :, :]

        result = _postprocess(
            raw,
            confidence=0.5,
            classes=None,
            class_names={0: "person"},
            img_w=640,
            img_h=640,
            input_shape=[1, 3, 640, 640],
            iou=0.5,
        )
        assert len(result) == 3

    def test_keypoints_stay_aligned_after_nms(self):
        # Two near-identical pose detections (IoU≈1.0).  NMS keeps the
        # higher-confidence one; the surviving keypoints must be the
        # ones that were originally attached to that anchor — not a
        # mismatched row from the suppressed anchor.
        num_keypoints = 17
        feat_dim = 4 + 1 + num_keypoints * 3
        feat = np.zeros((feat_dim, 2), dtype=np.float32)
        feat[0] = 320  # cx
        feat[1] = 320  # cy
        feat[2] = 200  # w
        feat[3] = 400  # h
        feat[4] = np.array([0.95, 0.85], dtype=np.float32)  # person score
        # Anchor 0 (winner) has keypoint x = 11*k for distinguishability;
        # anchor 1 (loser) has keypoint x = 99*k.
        for k in range(num_keypoints):
            feat[5 + k * 3, 0] = 11 * k
            feat[5 + k * 3 + 1, 0] = 22 * k
            feat[5 + k * 3 + 2, 0] = 0.9
            feat[5 + k * 3, 1] = 99 * k
            feat[5 + k * 3 + 1, 1] = 88 * k
            feat[5 + k * 3 + 2, 1] = 0.1
        raw = feat[np.newaxis, :, :]

        result = _postprocess(
            raw,
            confidence=0.5,
            classes=None,
            class_names={0: "person"},
            img_w=640,
            img_h=640,
            input_shape=[1, 3, 640, 640],
            num_keypoints=17,
        )
        assert len(result) == 1
        det = result[0]
        assert det.confidence == pytest.approx(0.95)
        # Must be the winner's pattern (11*k, 22*k, 0.9), not the loser's.
        assert det.keypoints[5, 0] == pytest.approx(55.0)  # 11 * 5
        assert det.keypoints[5, 1] == pytest.approx(110.0)  # 22 * 5
        assert det.keypoints[5, 2] == pytest.approx(0.9)


class TestNMSPerClass:
    """Direct unit tests for ``_nms_per_class`` — the IoU math itself."""

    def test_empty_input_returns_empty_array(self):
        empty = np.empty(0, dtype=np.float32)
        keep = _nms_per_class(
            empty, empty, empty, empty, empty, empty.astype(np.int64),
            iou_threshold=0.7,
        )
        assert keep.shape == (0,)

    def test_single_box_kept(self):
        keep = _nms_per_class(
            np.array([10.0]), np.array([10.0]),
            np.array([20.0]), np.array([20.0]),
            np.array([0.9]),
            np.array([0]),
            iou_threshold=0.7,
        )
        assert keep.tolist() == [0]

    def test_returns_indices_sorted_by_descending_score(self):
        # Three disjoint boxes with shuffled scores; NMS keeps all three
        # but the returned indices should rank by score (high → low).
        x1 = np.array([0.0, 100.0, 200.0])
        y1 = np.array([0.0, 0.0, 0.0])
        x2 = np.array([50.0, 150.0, 250.0])
        y2 = np.array([50.0, 50.0, 50.0])
        scores = np.array([0.5, 0.9, 0.7])  # idx 1 highest, then 2, then 0
        class_ids = np.array([0, 0, 0])

        keep = _nms_per_class(x1, y1, x2, y2, scores, class_ids, iou_threshold=0.5)
        assert keep.tolist() == [1, 2, 0]

    def test_iou_above_threshold_suppresses(self):
        # Box A: (0,0,100,100), Box B: (10,10,110,110).
        # Intersection = 90*90 = 8100; union = 10000 + 10000 - 8100 = 11900.
        # IoU = 8100 / 11900 ≈ 0.68.
        x1 = np.array([0.0, 10.0])
        y1 = np.array([0.0, 10.0])
        x2 = np.array([100.0, 110.0])
        y2 = np.array([100.0, 110.0])
        scores = np.array([0.9, 0.8])
        class_ids = np.array([0, 0])

        # Threshold 0.5 → IoU 0.68 exceeds → suppress B.
        keep_strict = _nms_per_class(
            x1, y1, x2, y2, scores, class_ids, iou_threshold=0.5
        )
        assert keep_strict.tolist() == [0]

        # Threshold 0.7 → IoU 0.68 below → keep both.
        keep_loose = _nms_per_class(
            x1, y1, x2, y2, scores, class_ids, iou_threshold=0.7
        )
        assert sorted(keep_loose.tolist()) == [0, 1]
