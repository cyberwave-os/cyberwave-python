"""Ultralytics (YOLOv8 / YOLOv11) runtime adapter.

Supports all Ultralytics task types:

* **detect** — :class:`DetectionResult` with plain :class:`Detection` objects.
* **segment** — each :class:`Detection` carries a :class:`Mask` in ``.mask``.
* **pose** — each :class:`Detection` carries a :class:`KeypointSet` in
  ``.keypoint_set`` (raw numpy array also kept in ``.keypoints`` for
  backward compatibility).
* **obb** — each :class:`Detection` carries an :class:`OrientedBoundingBox`
  in ``.obb``; ``.bbox`` is the tightest axis-aligned bounding box.
* **classify** — :class:`ClassificationPrediction` in ``PredictionResult.classification``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from cyberwave.models.runtimes.base import ModelRuntime
from cyberwave.models.types import (
    BoundingBox,
    ClassificationCandidate,
    ClassificationPrediction,
    Detection,
    DetectionResult,
    InstanceSegmentationResult,
    KeypointSet,
    Mask,
    OBBResult,
    OrientedBoundingBox,
    PoseResult,
    PredictionResult,
)


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
        classification: ClassificationPrediction | None = None
        # Resolved during the loop; last result wins (all frames share one task).
        task = "detect"

        for result in results:
            frame_h, frame_w = result.orig_shape or (0, 0)
            frame_area = frame_h * frame_w

            if getattr(result, "probs", None) is not None:
                task = "classify"
                classification = _parse_classification(result)
                continue

            if getattr(result, "obb", None) is not None:
                task = "obb"
                detections.extend(_parse_obb(result, frame_area, classes))
                continue

            if result.boxes is None:
                continue

            # Detect / segment / pose — all share result.boxes
            names = _names_dict(result)
            kp_data = _tensor_attr(result, "keypoints", "data")
            mask_data = _tensor_attr(result, "masks", "data")

            if kp_data is not None:
                task = "pose"
            elif mask_data is not None:
                task = "segment"

            for i, box in enumerate(result.boxes):
                label = names.get(_box_cls(box), str(_box_cls(box)))
                if classes and label not in classes:
                    continue

                bbox = _box_bbox(box)

                msk: Mask | None = None
                if mask_data is not None and i < len(mask_data):
                    msk = Mask(data=mask_data[i], h=frame_h, w=frame_w)

                raw_kps = None
                kp_set: KeypointSet | None = None
                if kp_data is not None and i < len(kp_data):
                    raw_kps = kp_data[i]
                    kp_set = KeypointSet.from_array(raw_kps)

                detections.append(
                    Detection(
                        label=label,
                        confidence=_box_conf(box),
                        bbox=bbox,
                        area_ratio=bbox.area / frame_area if frame_area else 0.0,
                        mask=msk,
                        keypoints=raw_kps,
                        keypoint_set=kp_set,
                    )
                )

        # Return the most specific result type matching the detected task.
        if task == "classify":
            return PredictionResult(output=classification, raw=results)
        if task == "obb":
            return PredictionResult(output=OBBResult(detections), raw=results)
        if task == "pose":
            return PredictionResult(output=PoseResult(detections), raw=results)
        if task == "segment":
            return PredictionResult(output=InstanceSegmentationResult(detections), raw=results)
        return PredictionResult(output=DetectionResult(detections), raw=results)


# ---------------------------------------------------------------------------
# Module-level helpers (same convention as onnxruntime_rt.py)
# ---------------------------------------------------------------------------


def _tensor_attr(obj: Any, attr: str, sub: str) -> Any | None:
    """Return ``getattr(obj, attr).<sub>.cpu().numpy()``, or ``None``."""
    container = getattr(obj, attr, None)
    if container is None:
        return None
    try:
        return getattr(container, sub).cpu().numpy()
    except AttributeError:
        return None


def _names_dict(result: Any) -> dict[int, str]:
    """Normalize ``result.names`` to ``dict[int, str]`` across YOLO versions."""
    raw = getattr(result, "names", None) or {}
    try:
        return {int(k): str(v) for k, v in raw.items()}
    except (AttributeError, TypeError, ValueError):
        return {}


def _box_cls(box: Any) -> int:
    """Extract class index from a box tensor — robust to v8/v11/v26 layouts."""
    try:
        return int(box.cls[0])
    except (IndexError, TypeError):
        # Scalar tensor (no batch dim) introduced in some export variants
        return int(box.cls)


def _box_conf(box: Any) -> float:
    """Extract confidence score from a box tensor."""
    try:
        return float(box.conf[0])
    except (IndexError, TypeError):
        return float(box.conf)


def _box_bbox(box: Any) -> BoundingBox:
    """Extract axis-aligned bbox from a box tensor as a :class:`BoundingBox`."""
    try:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
    except (IndexError, TypeError):
        # Newer versions may expose xyxy already as a 1-D tensor
        try:
            x1, y1, x2, y2 = box.xyxy.tolist()
        except TypeError:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy]
    return BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))


def _parse_classification(result: Any) -> ClassificationPrediction:
    probs = result.probs
    names = _names_dict(result)
    top_indices: list[int] = []
    top_confs: list[float] = []
    try:
        top_indices = [int(i) for i in probs.top5]
        top_confs = [float(c) for c in probs.top5conf]
    except (AttributeError, TypeError):
        try:
            top_indices = [int(probs.top1)]
            top_confs = [float(probs.top1conf)]
        except (AttributeError, TypeError):
            pass
    candidates = [
        ClassificationCandidate(
            label=names.get(idx, str(idx)),
            confidence=min(max(float(conf), 0.0), 1.0),
            index=idx,
        )
        for idx, conf in zip(top_indices, top_confs)
    ]
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return ClassificationPrediction(top=candidates)


def _parse_obb(
    result: Any,
    frame_area: int,
    classes: list[str] | None,
) -> list[Detection]:
    obb = result.obb
    names = _names_dict(result)
    detections: list[Detection] = []
    try:
        # ``xywhr`` is the canonical Ultralytics name (r = angle in radians).
        # Older checkpoints may expose ``xywha`` instead.
        # Use explicit `is None` — never `or` on a tensor (ambiguous bool).
        _xywhr = getattr(obb, "xywhr", None)
        xywhr = (_xywhr if _xywhr is not None else obb.xywha).cpu().numpy()
        confs = obb.conf.cpu().numpy()
        clss = obb.cls.cpu().numpy()
        xyxy = obb.xyxy.cpu().numpy()    # (N, 4) axis-aligned bbox
    except AttributeError:
        return detections
    for i in range(len(xywhr)):
        label = names.get(int(clss[i]), str(int(clss[i])))
        if classes and label not in classes:
            continue
        cx, cy, w, h, angle = (float(v) for v in xywhr[i])
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        aabb = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        detections.append(
            Detection(
                label=label,
                confidence=float(confs[i]),
                bbox=aabb,
                area_ratio=aabb.area / frame_area if frame_area else 0.0,
                obb=OrientedBoundingBox(cx=cx, cy=cy, w=w, h=h, angle_rad=angle),
            )
        )
    return detections
