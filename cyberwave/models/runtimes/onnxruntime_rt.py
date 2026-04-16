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
        **kwargs: Any,
    ) -> PredictionResult:
        session = model_handle
        meta = session.get_modelmeta()
        class_names: dict[int, str] = _parse_class_names(meta)

        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape  # e.g. [1, 3, 640, 640]

        img_h, img_w = 0, 0
        if isinstance(input_data, np.ndarray) and input_data.ndim >= 2:
            img_h, img_w = input_data.shape[:2]

        tensor = _preprocess(input_data, input_shape)
        outputs = session.run(None, {input_name: tensor})
        raw_output = outputs[0]  # [batch, num_detections, 4+num_classes]

        detections = _postprocess(
            raw_output,
            confidence=confidence,
            classes=classes,
            class_names=class_names,
            img_w=img_w,
            img_h=img_h,
            input_shape=input_shape,
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
    try:
        import ast

        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return {int(k): str(v) for k, v in parsed.items()}
    except Exception:
        pass
    return {}


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
) -> list[Detection]:
    """Parse YOLO-style ONNX output into ``Detection`` objects.

    Handles the common ``[1, 4+num_classes, num_detections]`` layout
    produced by ``ultralytics`` ONNX exports (transposed relative to
    the ``[1, num_detections, 4+num_classes]`` convention).
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

    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    class_scores = preds[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    max_scores = class_scores[np.arange(len(class_ids)), class_ids]

    mask = max_scores >= confidence
    if not np.any(mask):
        return []

    cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
    class_ids = class_ids[mask]
    max_scores = max_scores[mask]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Scale from model input size back to original image size.
    model_h = input_shape[2] if len(input_shape) >= 4 and isinstance(input_shape[2], int) else 1
    model_w = input_shape[3] if len(input_shape) >= 4 and isinstance(input_shape[3], int) else 1
    if img_w > 0 and img_h > 0 and model_w > 0 and model_h > 0:
        sx = img_w / model_w
        sy = img_h / model_h
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy

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
        detections.append(
            Detection(
                label=label,
                confidence=float(max_scores[i]),
                bbox=bbox,
                area_ratio=bbox.area / frame_area if frame_area else 0.0,
            )
        )
    return detections
