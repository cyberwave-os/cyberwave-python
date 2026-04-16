"""TensorFlow Lite inference backend.

Loads ``.tflite`` models via ``tflite_runtime`` (preferred) or the
full ``tensorflow`` package.  Supports both SSD-style (multi-tensor)
and YOLO-style (single-tensor) detection outputs, including quantized
INT8 models common on Raspberry Pi and other constrained devices.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from cyberwave.models.runtimes.base import ModelRuntime
from cyberwave.models.types import BoundingBox, Detection, PredictionResult

logger = logging.getLogger(__name__)


class TFLiteRuntime(ModelRuntime):
    """Runtime backend for TensorFlow Lite models."""

    name = "tflite"

    def is_available(self) -> bool:
        try:
            import tflite_runtime.interpreter  # noqa: F401

            return True
        except ImportError:
            pass
        try:
            import tensorflow as tf  # noqa: F401

            return hasattr(tf, "lite")
        except ImportError:
            return False

    def load(
        self,
        model_path: str,
        *,
        device: str | None = None,
        **kwargs: Any,
    ) -> Any:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter  # type: ignore[import-untyped]

        num_threads: int = kwargs.get("num_threads", 4)
        interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        dtype_name = "?"
        if input_details:
            dtype = input_details[0].get("dtype")
            dtype_name = getattr(dtype, "__name__", str(dtype))
        logger.info(
            "TFLite model loaded: %d input(s), %d output(s), dtype=%s",
            len(input_details),
            len(output_details),
            dtype_name,
        )
        return interpreter

    def predict(
        self,
        model_handle: Any,
        input_data: Any,
        *,
        confidence: float = 0.5,
        classes: list[str] | None = None,
        **kwargs: Any,
    ) -> PredictionResult:
        interpreter = model_handle
        class_names: dict[int, str] = kwargs.get("class_names", {})

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        img = np.asarray(input_data)
        img_h, img_w = img.shape[:2] if img.ndim >= 2 else (0, 0)

        tensor = _preprocess(img, input_details[0])
        interpreter.set_tensor(input_details[0]["index"], tensor)
        interpreter.invoke()

        raw_outputs = [
            interpreter.get_tensor(od["index"]) for od in output_details
        ]

        if _is_ssd_style(output_details):
            detections = _postprocess_ssd(
                raw_outputs,
                output_details=output_details,
                confidence=confidence,
                classes=classes,
                class_names=class_names,
                img_w=img_w,
                img_h=img_h,
            )
        else:
            input_shape = list(input_details[0]["shape"])
            detections = _postprocess_yolo(
                raw_outputs[0],
                confidence=confidence,
                classes=classes,
                class_names=class_names,
                img_w=img_w,
                img_h=img_h,
                input_shape=input_shape,
            )

        return PredictionResult(detections=detections, raw=raw_outputs)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _preprocess(img: np.ndarray, input_detail: dict[str, Any]) -> np.ndarray:
    """Resize and format *img* to match the interpreter's input tensor."""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    shape = input_detail["shape"]  # e.g. [1, 320, 320, 3]
    target_h, target_w = int(shape[1]), int(shape[2])

    if img.shape[0] != target_h or img.shape[1] != target_w:
        try:
            import cv2

            img = cv2.resize(img, (target_w, target_h))
        except ImportError:
            img = _numpy_resize(img, target_h, target_w)

    dtype = input_detail["dtype"]

    if dtype == np.float32:
        tensor = img.astype(np.float32) / 255.0
    elif dtype == np.int8:
        quant = input_detail.get("quantization_parameters", {})
        scale = quant.get("scales", np.array([1.0]))[0]
        zp = quant.get("zero_points", np.array([0]))[0]
        tensor = (img.astype(np.float32) / 255.0 / scale + zp).astype(np.int8)
    elif dtype == np.uint8:
        tensor = img.astype(np.uint8)
    else:
        tensor = img.astype(dtype)

    return np.expand_dims(tensor, 0)


def _numpy_resize(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Nearest-neighbour resize using only NumPy (no cv2/PIL dependency)."""
    src_h, src_w = img.shape[:2]
    row_idx = (np.arange(target_h) * src_h // target_h).astype(int)
    col_idx = (np.arange(target_w) * src_w // target_w).astype(int)
    return img[np.ix_(row_idx, col_idx)]


def _dequantize(tensor: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    """Dequantize an integer tensor to float32 if quantisation params exist."""
    if tensor.dtype in (np.float32, np.float64):
        return tensor.astype(np.float32)
    quant = detail.get("quantization_parameters", {})
    scales = quant.get("scales", np.array([]))
    zps = quant.get("zero_points", np.array([]))
    if len(scales) > 0 and scales[0] != 0:
        return ((tensor.astype(np.float32) - float(zps[0])) * float(scales[0])).astype(np.float32)
    return tensor.astype(np.float32)


def _is_ssd_style(output_details: list[dict[str, Any]]) -> bool:
    """Heuristic: SSD-style models have 4 output tensors (boxes, classes, scores, count)."""
    return len(output_details) >= 4


def _postprocess_ssd(
    raw_outputs: list[np.ndarray],
    *,
    output_details: list[dict[str, Any]],
    confidence: float,
    classes: list[str] | None,
    class_names: dict[int, str],
    img_w: int,
    img_h: int,
) -> list[Detection]:
    """Parse SSD MobileNet-style TFLite output (4 tensors).

    Standard TF Object Detection API output order:
      0: boxes      — [1, N, 4] normalised [y1, x1, y2, x2]
      1: classes     — [1, N]
      2: scores      — [1, N]
      3: num_dets    — [1]
    """
    boxes = np.atleast_2d(_dequantize(raw_outputs[0], output_details[0]).squeeze())
    class_ids = np.atleast_1d(_dequantize(raw_outputs[1], output_details[1]).squeeze())
    scores = np.atleast_1d(_dequantize(raw_outputs[2], output_details[2]).squeeze())
    num_dets = int(_dequantize(raw_outputs[3], output_details[3]).flatten()[0])

    frame_area = img_w * img_h if img_w > 0 and img_h > 0 else 1
    detections: list[Detection] = []

    for i in range(min(num_dets, len(scores))):
        score = float(scores[i])
        if score < confidence:
            continue
        cls_id = int(class_ids[i])
        label = class_names.get(cls_id, str(cls_id))
        if classes and label not in classes:
            continue

        # SSD boxes are normalised [y1, x1, y2, x2]
        y1_n, x1_n, y2_n, x2_n = boxes[i]
        bbox = BoundingBox(
            x1=float(x1_n) * img_w,
            y1=float(y1_n) * img_h,
            x2=float(x2_n) * img_w,
            y2=float(y2_n) * img_h,
        )
        detections.append(
            Detection(
                label=label,
                confidence=score,
                bbox=bbox,
                area_ratio=bbox.area / frame_area if frame_area else 0.0,
            )
        )

    return detections


def _postprocess_yolo(
    raw_output: np.ndarray,
    *,
    confidence: float,
    classes: list[str] | None,
    class_names: dict[int, str],
    img_w: int,
    img_h: int,
    input_shape: list[int],
) -> list[Detection]:
    """Parse YOLO-style single-tensor TFLite output.

    Expected layout: ``[1, num_detections, 4 + num_classes]`` (or transposed).
    """
    # Remove batch dim only; avoid squeeze() which collapses [1, 1, 6] to [6].
    preds = raw_output
    while preds.ndim > 2 and preds.shape[0] == 1:
        preds = preds[0]
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)
    if preds.ndim != 2:
        return []

    # Transpose [features, N] → [N, features] only when axis 1 is definitely
    # too small to hold 4 box coords + at least 1 class.  TFLite YOLO models
    # normally output [N, 4+C] already, unlike Ultralytics ONNX exports which
    # use [4+C, N].
    if preds.shape[1] < 5:
        preds = preds.T

    if preds.shape[1] < 5:
        return []

    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    class_scores = preds[:, 4:]

    cls_ids = np.argmax(class_scores, axis=1)
    max_scores = class_scores[np.arange(len(cls_ids)), cls_ids]

    mask = max_scores >= confidence
    if not np.any(mask):
        return []

    cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
    cls_ids = cls_ids[mask]
    max_scores = max_scores[mask]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    model_h = input_shape[1] if len(input_shape) >= 3 and isinstance(input_shape[1], int) else 1
    model_w = input_shape[2] if len(input_shape) >= 3 and isinstance(input_shape[2], int) else 1
    if img_w > 0 and img_h > 0 and model_w > 0 and model_h > 0:
        sx = img_w / model_w
        sy = img_h / model_h
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy

    frame_area = img_w * img_h if img_w > 0 and img_h > 0 else 1

    detections: list[Detection] = []
    for i in range(len(x1)):
        label = class_names.get(int(cls_ids[i]), str(int(cls_ids[i])))
        if classes and label not in classes:
            continue
        bbox = BoundingBox(
            x1=float(x1[i]),
            y1=float(y1[i]),
            x2=float(x2[i]),
            y2=float(y2[i]),
        )
        detections.append(
            Detection(
                label=label,
                confidence=float(max_scores[i]),
                bbox=bbox,
                area_ratio=bbox.area / frame_area if frame_area else 0.0,
            )
        )
    return detections
