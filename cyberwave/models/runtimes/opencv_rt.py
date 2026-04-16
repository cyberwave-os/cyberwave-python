"""OpenCV DNN inference backend.

Supports models loadable by ``cv2.dnn.readNet``, including Caffe
(``.prototxt`` / ``.caffemodel``), Darknet (``.cfg`` / ``.weights``),
ONNX (``.onnx``), and TensorFlow (``.pb``) formats.

This runtime is intentionally lightweight and avoids heavy framework
dependencies — only ``opencv-python`` is required.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from cyberwave.models.runtimes.base import ModelRuntime
from cyberwave.models.types import BoundingBox, Detection, PredictionResult

logger = logging.getLogger(__name__)

_INPUT_SIZE = (416, 416)
_SCALE_FACTOR = 1.0 / 255.0


class OpenCVRuntime(ModelRuntime):
    """Runtime backend for models via ``cv2.dnn``."""

    name = "opencv"

    def is_available(self) -> bool:
        try:
            import cv2  # noqa: F401

            return hasattr(cv2, "dnn")
        except ImportError:
            return False

    def load(
        self,
        model_path: str,
        *,
        device: str | None = None,
        **kwargs: Any,
    ) -> Any:
        import cv2

        config_path: str = kwargs.get("config", "")
        net = cv2.dnn.readNet(model_path, config_path)

        if device and device.startswith("cuda"):
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        class_names: list[str] = list(kwargs.get("class_names", []))
        input_size: tuple[int, int] = kwargs.get("input_size", _INPUT_SIZE)

        return _OpenCVHandle(net=net, class_names=class_names, input_size=input_size)

    def predict(
        self,
        model_handle: Any,
        input_data: Any,
        *,
        confidence: float = 0.5,
        classes: list[str] | None = None,
        **kwargs: Any,
    ) -> PredictionResult:
        import cv2

        handle: _OpenCVHandle = model_handle
        net = handle.net
        class_names = handle.class_names
        input_size = handle.input_size

        img = np.asarray(input_data)
        img_h, img_w = img.shape[:2]

        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=_SCALE_FACTOR,
            size=input_size,
            swapRB=True,
            crop=False,
        )
        net.setInput(blob)

        layer_names = net.getUnconnectedOutLayersNames()
        outputs = net.forward(layer_names)

        detections = _parse_detections(
            outputs,
            confidence=confidence,
            classes=classes,
            class_names=class_names,
            img_w=img_w,
            img_h=img_h,
        )
        return PredictionResult(detections=detections, raw=outputs)


class _OpenCVHandle:
    """Bundles a ``cv2.dnn.Net`` with its class names and input size."""

    __slots__ = ("net", "class_names", "input_size")

    def __init__(
        self,
        *,
        net: Any,
        class_names: list[str],
        input_size: tuple[int, int],
    ) -> None:
        self.net = net
        self.class_names = class_names
        self.input_size = input_size


def _parse_detections(
    outputs: list[np.ndarray],
    *,
    confidence: float,
    classes: list[str] | None,
    class_names: list[str],
    img_w: int,
    img_h: int,
) -> list[Detection]:
    """Parse OpenCV DNN forward outputs into ``Detection`` objects.

    Supports two common output layouts:
    1. **SSD-style**: shape ``(1, 1, N, 7)`` — each row is
       ``[batch, class_id, conf, x1, y1, x2, y2]`` with coords normalised to [0,1].
    2. **YOLO-style**: shape ``(N, 5 + C)`` per layer — each row is
       ``[cx, cy, w, h, obj_conf, cls1, cls2, ...]`` with coords relative to
       the input blob size.
    """
    frame_area = img_w * img_h if img_w > 0 and img_h > 0 else 1
    detections: list[Detection] = []

    if len(outputs) == 1 and outputs[0].ndim == 4 and outputs[0].shape[3] == 7:
        raw = outputs[0][0, 0]  # (N, 7)
        for row in raw:
            conf = float(row[2])
            if conf < confidence:
                continue
            cls_id = int(row[1])
            label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            if classes and label not in classes:
                continue
            bbox = BoundingBox(
                x1=float(row[3]) * img_w,
                y1=float(row[4]) * img_h,
                x2=float(row[5]) * img_w,
                y2=float(row[6]) * img_h,
            )
            detections.append(
                Detection(
                    label=label,
                    confidence=conf,
                    bbox=bbox,
                    area_ratio=bbox.area / frame_area if frame_area else 0.0,
                )
            )
        return detections

    # YOLO-style: iterate over each detection layer output.
    for output in outputs:
        if output.ndim == 3:
            output = output[0]
        if output.ndim != 2 or output.shape[1] < 5:
            continue
        for row in output:
            obj_conf = float(row[4])
            class_scores = row[5:]
            if len(class_scores) == 0:
                continue
            cls_id = int(np.argmax(class_scores))
            score = obj_conf * float(class_scores[cls_id])
            if score < confidence:
                continue
            label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            if classes and label not in classes:
                continue
            cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            bbox = BoundingBox(
                x1=(cx - w / 2) * img_w,
                y1=(cy - h / 2) * img_h,
                x2=(cx + w / 2) * img_w,
                y2=(cy + h / 2) * img_h,
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
