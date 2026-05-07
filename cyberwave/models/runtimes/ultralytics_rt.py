"""Ultralytics (YOLOv8 / YOLOv11) runtime adapter."""

from __future__ import annotations

import os
from pathlib import Path
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

        p = Path(model_path)

        # Ultralytics strips the directory from missing weights and downloads
        # to CWD.  When the file doesn't exist yet, chdir into a writable
        # model directory so the auto-downloaded weights don't land in an
        # unwritable CWD (common in containers).
        #
        # NOTE: os.chdir() is process-global.  Concurrent load() calls for
        # missing models will race on CWD.  In practice model loading is a
        # one-time startup operation so this is acceptable.
        download_dir = self._writable_model_dir(p) if not p.exists() else None
        old_cwd = os.getcwd() if download_dir else None
        try:
            if download_dir:
                os.chdir(download_dir)
            model = YOLO(p.name if download_dir else model_path)
        finally:
            if old_cwd is not None:
                os.chdir(old_cwd)

        if device:
            # ``model.to(device)`` raises ``TypeError`` for any non-PyTorch
            # backend (ONNX, TensorRT, OpenVINO, …) loaded through the
            # Ultralytics ``YOLO`` wrapper — those formats are inference-
            # only and pin the device at export time. We still want
            # ``cw.models.load('foo.onnx', runtime='ultralytics')`` to
            # succeed (Ultralytics provides letterboxing + NMS that the
            # raw onnxruntime adapter does not), so swallow the format
            # mismatch and rely on the per-call ``device=`` kwarg in
            # ``predict()`` instead.
            try:
                model.to(device)
            except TypeError:
                pass
        return model

    @staticmethod
    def _writable_model_dir(model_path: Path) -> Path:
        """Find a writable directory for Ultralytics auto-downloads."""
        candidates = []

        if model_path.is_absolute() and model_path.parent != Path("/"):
            candidates.append(model_path.parent)

        env_dir = os.environ.get("CYBERWAVE_MODELS_DIR") or os.environ.get(
            "CYBERWAVE_MODEL_DIR"
        )
        if env_dir:
            candidates.append(Path(env_dir))

        candidates.extend([Path("/app/models"), Path.home() / ".cyberwave" / "models"])

        for d in candidates:
            try:
                d.mkdir(parents=True, exist_ok=True)
                if os.access(d, os.W_OK):
                    return d
            except OSError:
                continue

        return Path("/tmp")

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
                result.orig_shape[0] * result.orig_shape[1] if result.orig_shape else 1
            )
            if result.boxes is None:
                continue
            # Pose models (e.g. yolov8n-pose) attach keypoints aligned to boxes.
            kp_data = None
            if getattr(result, "keypoints", None) is not None:
                try:
                    kp_data = result.keypoints.data.cpu().numpy()
                except AttributeError:
                    kp_data = None
            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = result.names[int(box.cls[0])]
                conf = float(box.conf[0])
                if classes and label not in classes:
                    continue
                bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                # Ultralytics aligns ``boxes[i]`` with ``keypoints.data[i]``
                # by index, so this lookup should always succeed when
                # ``kp_data`` is present. The ``i < len(kp_data)`` guard is
                # defensive against future Ultralytics releases that might
                # decouple the two arrays (e.g. NMS dropping a few rows from
                # one but not the other).
                kps = kp_data[i] if kp_data is not None and i < len(kp_data) else None
                detections.append(
                    Detection(
                        label=label,
                        confidence=conf,
                        bbox=bbox,
                        area_ratio=bbox.area / frame_area if frame_area else 0.0,
                        keypoints=kps,
                    )
                )

        return PredictionResult(detections=detections, raw=results)
