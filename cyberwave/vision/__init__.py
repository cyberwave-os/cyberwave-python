"""Vision utilities for working with model predictions.

Two families of helpers, both designed to be safe to chain and cheap
enough to run on a per-frame basis on edge workers:

- :func:`blank_persons` obscures the bounding boxes of detected people —
  ``"pixelate"`` by default (mosaic), with ``"redact"`` / ``"blur"`` /
  ``"bbox"`` as alternatives.
- :func:`draw_skeleton` overlays a pose skeleton on a frame. Optional —
  use it when you want to surface posture on top of an obscured region.
- :func:`anonymize_frame` composes the two into a single call — useful
  when your model emits pose keypoints and you want both the bounding
  box obscured and the skeleton preserved on top. If you just want a
  clean mosaic without any skeleton overlay, call :func:`blank_persons`
  directly.
- :func:`annotate_detections` draws bounding boxes and ``label conf``
  captions — the numpy-frame analogue of the cloud-side ``annotate``
  workflow node, used by the edge-filter worker template when
  ``annotate`` is wired into a ``camera_frame`` chain.
"""

from cyberwave.vision.annotate import (
    OVERLAY_PAYLOAD_VERSION,
    annotate_detections,
    build_overlay_payload,
)
from cyberwave.vision.anonymize import (
    COCO17_EDGE_GROUPS,
    COCO17_SEGMENT_COLORS,
    COCO17_SKELETON,
    anonymize_frame,
    blank_persons,
    draw_skeleton,
)

__all__ = [
    "COCO17_EDGE_GROUPS",
    "COCO17_SEGMENT_COLORS",
    "COCO17_SKELETON",
    "OVERLAY_PAYLOAD_VERSION",
    "annotate_detections",
    "anonymize_frame",
    "blank_persons",
    "build_overlay_payload",
    "draw_skeleton",
]
