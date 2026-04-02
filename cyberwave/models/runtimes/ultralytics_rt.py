"""Ultralytics (YOLOv8 / YOLOv11) runtime adapter."""

from __future__ import annotations

from typing import Any

from cyberwave.models.runtimes.base import ModelRuntime
from cyberwave.models.types import BoundingBox, Detection, PredictionResult


class UltralyticsRuntime(ModelRuntime):
    """Runtime backend for Ultralytics YOLO models."""

    name = "ultralytics"

    def is_available(self) -> bool:
        try:
            import ultralytics  # noqa: F401

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
        from ultralytics import YOLO

        model = YOLO(model_path)
        if device:
            model.to(device)
        return model

    def predict(
        self,
        model_handle: Any,
        input_data: Any,
        *,
        confidence: float = 0.5,
        classes: list[str] | None = None,
        **kwargs: Any,
    ) -> PredictionResult:
        results = model_handle(input_data, conf=confidence, verbose=False)
        detections: list[Detection] = []

        for result in results:
            frame_area = (
                result.orig_shape[0] * result.orig_shape[1]
                if result.orig_shape
                else 1
            )
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = result.names[int(box.cls[0])]
                conf = float(box.conf[0])
                if classes and label not in classes:
                    continue
                bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                detections.append(
                    Detection(
                        label=label,
                        confidence=conf,
                        bbox=bbox,
                        area_ratio=bbox.area / frame_area if frame_area else 0.0,
                    )
                )

        return PredictionResult(detections=detections, raw=results)
