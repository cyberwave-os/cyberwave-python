"""Internal: normalize heterogeneous detection inputs to ``Detection``.

The public SDK vision helpers (:mod:`cyberwave.vision.annotate`,
:mod:`cyberwave.vision.anonymize`) document their inputs as
:class:`~cyberwave.models.types.Detection` dataclasses, but in
practice detections flow in from many producers in the workflow
runtime:

- SDK runtime adapters (``cyberwave.models.runtimes.ultralytics_rt``,
  ``tflite_rt``, ``opencv_rt``, ``hailo_rt``, ``onnxruntime_rt`` and
  the cloud adapter in ``cyberwave.models.cloud``) return
  ``Detection`` instances directly.
- Workflow nodes compiled into generated worker code (e.g.
  ``call_model``, ``barcode_reader``) hand downstream perception nodes
  plain dicts instead of typed instances: a list of
  ``{"label", "class", "confidence", "bbox", "bbox_pixels", ...}`` dicts.
- Operator-authored payloads (webhook nodes, data warehouse rows) can
  hand in either shape.

Coercing once at the SDK boundary keeps every drawing / redaction loop
attribute-typed and avoids each helper having to branch on
``isinstance(det, dict)``. Other parts of the pipeline that produce
these dicts implement the same polymorphism — this helper mirrors
that pattern at the SDK layer.
"""

from __future__ import annotations

from typing import Any

from cyberwave.models.types import BoundingBox, Detection

# Keys consumed by the typed ``Detection`` fields; everything else on a
# dict-shaped detection is preserved under ``Detection.metadata`` so
# downstream callers that read e.g. ``det.metadata["text"]`` (barcode
# reader output) keep working.
_RESERVED_DICT_KEYS = frozenset(
    {"label", "class", "confidence", "bbox", "bbox_pixels", "mask", "keypoints"}
)


def _coerce_bbox(value: Any) -> BoundingBox | None:
    """Accept list / tuple / dict / duck-typed bbox shapes; return ``BoundingBox``.

    Mirrors the same bbox-shape polymorphism used elsewhere in the
    pipeline so producers that emit either shape land at the same
    typed view here.
    """
    if value is None:
        return None
    if isinstance(value, BoundingBox):
        return value
    if isinstance(value, list | tuple) and len(value) >= 4:
        try:
            return BoundingBox(*(float(v) for v in value[:4]))
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict) and {"x1", "y1", "x2", "y2"}.issubset(value):
        try:
            return BoundingBox(
                x1=float(value["x1"]),
                y1=float(value["y1"]),
                x2=float(value["x2"]),
                y2=float(value["y2"]),
            )
        except (TypeError, ValueError):
            return None
    if all(hasattr(value, attr) for attr in ("x1", "y1", "x2", "y2")):
        try:
            return BoundingBox(
                x1=float(value.x1),
                y1=float(value.y1),
                x2=float(value.x2),
                y2=float(value.y2),
            )
        except (TypeError, ValueError):
            return None
    return None


def as_detection(det: Any) -> Detection:
    """Coerce ``det`` to a :class:`Detection`.

    Already-typed ``Detection`` instances pass through unchanged.
    Plain dicts (the canonical workflow-runtime detection shape) are
    projected by reading the documented field names — ``bbox_pixels``
    is preferred over ``bbox`` because the former is the canonical
    pixel-xyxy list, while the dict-``bbox`` form can be either xyxy or
    xywh depending on the producer.

    Duck-typed objects exposing the ``Detection`` attribute surface
    (``.label / .confidence / .bbox``) are returned as-is; downstream
    attribute access will raise loudly if a required attribute is
    missing — preferable to silent drops here.

    Raises ``ValueError`` when a dict detection has no usable bbox.
    """
    if isinstance(det, Detection):
        return det
    if isinstance(det, dict):
        bbox = _coerce_bbox(det.get("bbox_pixels")) or _coerce_bbox(det.get("bbox"))
        if bbox is None:
            raise ValueError(f"detection has no usable bbox: {det!r}")
        return Detection(
            label=str(det.get("label") or det.get("class") or ""),
            confidence=float(det.get("confidence", 1.0)),
            bbox=bbox,
            mask=det.get("mask"),
            keypoints=det.get("keypoints"),
            metadata={
                k: v for k, v in det.items() if k not in _RESERVED_DICT_KEYS
            },
        )
    return det
