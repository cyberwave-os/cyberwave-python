"""ONNX Runtime inference backend.

Supports ``.onnx`` models exported from frameworks like Ultralytics,
PyTorch, or TensorFlow.  Automatically selects CUDA or CPU execution
providers based on the requested device.

Expected ONNX output tensor layout (YOLO-style object detection):
  ``[batch, num_detections, 4 + num_classes]``
where the first four values per detection are ``cx, cy, w, h`` and
the remaining columns are per-class confidence scores.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from cyberwave.models.runtimes.base import ModelRuntime
from cyberwave.models.types import BoundingBox, Detection, PredictionResult

logger = logging.getLogger(__name__)


class OnnxRuntime(ModelRuntime):
    """Runtime backend for ONNX models via ``onnxruntime``."""

    name = "onnxruntime"

    def is_available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401

            return True
        except ImportError:
            return False

    def load(
        self,
        model_path: str,
        *,
        device: str | None = None,
        **kwargs: Any,
    ) -> Any:
        import onnxruntime as ort

        providers: list[str] = []
        if device and device.startswith("cuda"):
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        session = ort.InferenceSession(model_path, providers=providers)
        logger.info(
            "ONNX session created with providers %s",
            session.get_providers(),
        )
        return session

    def predict(
        self,
        model_handle: Any,
        input_data: Any,
        *,
        confidence: float = 0.5,
        classes: list[str] | None = None,
        iou: float = 0.7,
        **kwargs: Any,
    ) -> PredictionResult:
        """Run inference and return post-processed detections.

        ``iou`` is the IoU threshold used by per-class non-max suppression.
        The default (``0.7``) matches Ultralytics' :py:meth:`YOLO.predict`
        default so swapping ``yolov8s.pt`` for ``yolov8s.onnx`` produces
        the same number of boxes per object.  Pass ``iou=1.0`` to disable
        NMS and return every anchor that crosses ``confidence``.
        """
        session = model_handle
        meta = session.get_modelmeta()
        class_names: dict[int, str] = _parse_class_names(meta)
        num_keypoints, kp_dim = _parse_kpt_shape(meta)

        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape  # e.g. [1, 3, 640, 640]

        img_h, img_w = 0, 0
        if isinstance(input_data, np.ndarray) and input_data.ndim >= 2:
            img_h, img_w = input_data.shape[:2]

        tensor = _preprocess(input_data, input_shape)
        outputs = session.run(None, {input_name: tensor})
        raw_output = outputs[0]  # [batch, num_detections, 4+num_classes(+kp_dim*K)]

        detections = _postprocess(
            raw_output,
            confidence=confidence,
            classes=classes,
            class_names=class_names,
            img_w=img_w,
            img_h=img_h,
            input_shape=input_shape,
            num_keypoints=num_keypoints,
            kp_dim=kp_dim,
            iou=iou,
        )
        return PredictionResult(detections=detections, raw=outputs)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_class_names(meta: Any) -> dict[int, str]:
    """Extract ``{idx: name}`` map from ONNX model metadata.

    Ultralytics-exported ONNX models store a ``names`` key in custom
    metadata with a Python dict literal like ``{0: 'person', 1: 'car'}``.
    """
    props = meta.custom_metadata_map if meta else {}
    raw = props.get("names", "")
    if not raw:
        return {}
    # Tight catch — ast.literal_eval raises ValueError/SyntaxError on garbage,
    # and the dict comprehension can raise TypeError on non-int keys.
    try:
        import ast

        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return {int(k): str(v) for k, v in parsed.items()}
    except (ValueError, SyntaxError, TypeError) as exc:
        logger.warning("Could not parse ONNX 'names' metadata %r: %s", raw, exc)
    return {}


def _parse_kpt_shape(meta: Any) -> tuple[int, int]:
    """Extract ``(K, dim)`` from Ultralytics pose model metadata.

    Pose exports include ``kpt_shape`` (e.g. ``[17, 3]``) in
    ``custom_metadata_map``; ``dim`` is ``3`` for ``(x, y, visibility)``
    and ``2`` for visibility-less variants. Returns ``(0, 0)`` for
    non-pose models so callers can short-circuit on ``num_keypoints == 0``.
    """
    props = meta.custom_metadata_map if meta else {}
    raw = props.get("kpt_shape", "")
    if not raw:
        return 0, 0
    try:
        import ast

        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            return int(parsed[0]), int(parsed[1])
        if isinstance(parsed, (list, tuple)) and len(parsed) == 1:
            # Old/odd export with just K — assume the standard (x, y, vis).
            return int(parsed[0]), 3
    except (ValueError, SyntaxError, TypeError) as exc:
        logger.warning("Could not parse ONNX 'kpt_shape' metadata %r: %s", raw, exc)
    return 0, 0


def _nms_per_class(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    *,
    iou_threshold: float,
) -> np.ndarray:
    """Per-class greedy non-max suppression in pure NumPy.

    Returns the indices (into the input arrays) of the boxes that survive
    suppression, ordered by descending score.  "Per-class" means boxes
    of different classes never suppress each other — this matches
    Ultralytics' default and prevents a high-confidence ``person`` box
    from eating an overlapping ``handbag`` detection.

    Standard greedy NMS:
      1. Sort candidates by descending score.
      2. Take the highest-scoring box, mark it "kept".
      3. Drop every remaining box of the same class whose IoU with the
         kept box is ``>= iou_threshold``.
      4. Repeat with the next surviving box.

    Pure NumPy keeps the runtime free of torch / torchvision; for the
    typical post-confidence-threshold count (a few hundred boxes) this
    is plenty fast.
    """
    if len(x1) == 0:
        return np.empty(0, dtype=np.int64)

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    keep: list[int] = []

    for cls in np.unique(class_ids):
        cls_idx = np.where(class_ids == cls)[0]
        if len(cls_idx) == 1:
            keep.append(int(cls_idx[0]))
            continue

        order = cls_idx[np.argsort(-scores[cls_idx])]
        while len(order) > 0:
            i = int(order[0])
            keep.append(i)
            if len(order) == 1:
                break

            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h

            union = areas[i] + areas[rest] - inter
            iou = np.where(union > 0, inter / union, 0.0)

            order = rest[iou < iou_threshold]

    keep.sort(key=lambda i: -float(scores[i]))
    return np.asarray(keep, dtype=np.int64)


def _preprocess(input_data: Any, input_shape: list[Any]) -> np.ndarray:
    """Convert an image (HWC uint8) to the NCHW float32 tensor ONNX expects."""
    img = np.asarray(input_data)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    target_h = input_shape[2] if len(input_shape) >= 4 else img.shape[0]
    target_w = input_shape[3] if len(input_shape) >= 4 else img.shape[1]

    if isinstance(target_h, int) and isinstance(target_w, int):
        if img.shape[0] != target_h or img.shape[1] != target_w:
            try:
                import cv2

                img = cv2.resize(img, (target_w, target_h))
            except ImportError:
                from PIL import Image

                pil = Image.fromarray(img).resize((target_w, target_h))
                img = np.asarray(pil)

    tensor = img.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)  # HWC -> CHW
    tensor = np.expand_dims(tensor, 0)  # add batch dim
    return np.ascontiguousarray(tensor)


def _postprocess(
    raw_output: np.ndarray,
    *,
    confidence: float,
    classes: list[str] | None,
    class_names: dict[int, str],
    img_w: int,
    img_h: int,
    input_shape: list[Any],
    num_keypoints: int = 0,
    kp_dim: int = 3,
    iou: float = 0.7,
) -> list[Detection]:
    """Parse YOLO-style ONNX output into ``Detection`` objects.

    Handles the common ``[1, 4+num_classes, num_detections]`` layout
    produced by ``ultralytics`` ONNX exports (transposed relative to
    the ``[1, num_detections, 4+num_classes]`` convention).

    Pose models append ``K * kp_dim`` keypoint values per detection after
    the class scores; layout becomes
    ``[1, 4 + num_classes + kp_dim*K, num_detections]``. Standard exports
    use ``kp_dim=3`` (``x, y, visibility``); some variants drop the
    visibility column (``kp_dim=2``). Pass ``num_keypoints=K`` to enable
    parsing.

    YOLO ONNX exports emit one prediction per anchor (≈8400 for
    ``yolov8`` at 640x640).  Without NMS each real object is reported as
    a cluster of overlapping boxes — the symptom that motivated this
    helper to apply per-class non-max suppression with a default IoU
    threshold of ``0.7`` (Ultralytics' :py:meth:`YOLO.predict` default).
    Pass ``iou=1.0`` to keep every surviving anchor.
    """
    preds = raw_output[0]  # drop batch dim

    # Ultralytics ONNX exports use [4+C, N] layout; transpose to [N, 4+C].
    # Transpose when the feature dim (4+C, always >= 5) is on axis 0:
    #   - shape[0] < shape[1]: common case (e.g. 84 features, 8400 dets)
    #   - shape[1] < 5: second dim is too small to be the feature dim
    if preds.ndim == 2 and (preds.shape[0] < preds.shape[1] or preds.shape[1] < 5):
        preds = preds.T  # -> [N, 4+C]

    if preds.ndim != 2 or preds.shape[1] < 5:
        return []

    feat_dim = preds.shape[1]
    # Pose models have known keypoint count; remaining columns are class scores.
    if num_keypoints > 0 and kp_dim > 0:
        kp_width = num_keypoints * kp_dim
        n_classes = max(1, feat_dim - 4 - kp_width)
    else:
        # Use class_names length when present (most accurate); else assume all
        # trailing columns are class scores (legacy detection-only behaviour).
        n_classes = len(class_names) if class_names else feat_dim - 4
        n_classes = max(1, min(n_classes, feat_dim - 4))

    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    class_scores = preds[:, 4 : 4 + n_classes]

    class_ids = np.argmax(class_scores, axis=1)
    max_scores = class_scores[np.arange(len(class_ids)), class_ids]

    mask = max_scores >= confidence
    if not np.any(mask):
        return []

    cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
    class_ids = class_ids[mask]
    max_scores = max_scores[mask]

    keypoints_per_det: np.ndarray | None = None
    if (
        num_keypoints > 0
        and kp_dim > 0
        and feat_dim >= 4 + n_classes + num_keypoints * kp_dim
    ):
        kp_flat = preds[:, 4 + n_classes : 4 + n_classes + num_keypoints * kp_dim]
        kp_flat = kp_flat[mask]
        # Reshape to (N, K, kp_dim). Last axis is (x, y[, visibility]).
        keypoints_per_det = kp_flat.reshape(-1, num_keypoints, kp_dim)

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Per-class non-max suppression. Done after the confidence filter so
    # we only sort the few hundred candidates that crossed the threshold,
    # not all 8400 anchors.  ``iou >= 1.0`` is the documented escape
    # hatch — it short-circuits the suppression and returns every
    # surviving anchor (useful for callers that want raw output to feed
    # into a custom tracker / NMS variant).
    if iou < 1.0 and len(x1) > 1:
        keep = _nms_per_class(x1, y1, x2, y2, max_scores, class_ids, iou_threshold=iou)
        if len(keep) != len(x1):
            x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
            class_ids = class_ids[keep]
            max_scores = max_scores[keep]
            if keypoints_per_det is not None:
                keypoints_per_det = keypoints_per_det[keep]

    # Scale from model input size back to original image size. Dynamic-axis
    # exports declare ``input_shape = [None, 3, None, None]``: in that case
    # ``_preprocess`` skipped its resize and fed the model the raw image, so
    # there is nothing to scale and we must keep sx=sy=1.0. Falling back to
    # ``model_w/h = 1`` here would have produced nonsense (sx = img_w / 1).
    static_h = (
        input_shape[2]
        if len(input_shape) >= 4
        and isinstance(input_shape[2], int)
        and input_shape[2] > 0
        else None
    )
    static_w = (
        input_shape[3]
        if len(input_shape) >= 4
        and isinstance(input_shape[3], int)
        and input_shape[3] > 0
        else None
    )
    if static_h is not None and static_w is not None and img_w > 0 and img_h > 0:
        sx = img_w / static_w
        sy = img_h / static_h
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy
        if keypoints_per_det is not None:
            keypoints_per_det = keypoints_per_det.copy()
            keypoints_per_det[..., 0] *= sx
            keypoints_per_det[..., 1] *= sy

    frame_area = img_w * img_h if img_w > 0 and img_h > 0 else 1

    detections: list[Detection] = []
    for i in range(len(x1)):
        label = class_names.get(int(class_ids[i]), str(int(class_ids[i])))
        if classes and label not in classes:
            continue
        bbox = BoundingBox(
            x1=float(x1[i]),
            y1=float(y1[i]),
            x2=float(x2[i]),
            y2=float(y2[i]),
        )
        kps = keypoints_per_det[i] if keypoints_per_det is not None else None
        detections.append(
            Detection(
                label=label,
                confidence=float(max_scores[i]),
                bbox=bbox,
                area_ratio=bbox.area / frame_area if frame_area else 0.0,
                keypoints=kps,
            )
        )
    return detections
