"""Detection and workflow overlays drawn onto camera frames before encoding.

Two sources, both read from the twin's Zenoh data bus:

* ``detections/<runtime>``: raw detection batches published by an ML worker
  (``{"detections": [{"label", "confidence", "x1".."y2"}], "frame_width", ...}``),
  drawn by :func:`draw_detections` in a fixed style.
* :data:`cyberwave.data.FRAME_OVERLAY_CHANNEL` (``frames/overlay``): the styled
  spec built by :func:`cyberwave.vision.build_overlay_payload` (boxes, polygons,
  per-box colours, line width, font scale), drawn by :func:`draw_overlay`. A
  fresh overlay preempts the raw detections.

Overlays are composited on the encoded copy only: Zenoh frame subscribers
(including the worker that produced the detections) always see clean pixels.
They are BGR-only, so depth cameras skip them.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ── Detection overlay ──────────────────────────────────────────────
#
# Mirrors the C++ OBSBOT driver: an ML worker publishes JSON detection
# payloads on ``cw/<twin_uuid>/data/detections/<runtime>``; the driver caches
# the latest batch and draws bounding boxes on frames before WebRTC encoding.
# The capture thread copies the frame before drawing, so Zenoh subscribers on
# ``frames/*`` (including the ML worker itself) only ever see clean pixels.
#
# Overlays are RGB-only — depth cameras (RealSense) skip this path.

DETECTIONS_STALE_MS = 2000  # matches the OBSBOT C++ driver's `> 2000` check


@dataclass(slots=True, frozen=True)
class DetectionBox:
    """A single bounding box emitted by an ML worker."""

    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionCache:
    """Thread-safe holder for the most recent detection batch.

    Updated by the Zenoh subscriber thread, read by the capture thread.
    Returns ``None`` from :meth:`snapshot` when the cache is empty or the
    batch is older than :data:`DETECTIONS_STALE_MS`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._boxes: list[DetectionBox] = []
        self._frame_w = 0
        self._frame_h = 0
        self._updated_at = 0.0

    def update(self, boxes: list[DetectionBox], frame_w: int, frame_h: int) -> None:
        with self._lock:
            self._boxes = boxes
            self._frame_w = frame_w
            self._frame_h = frame_h
            self._updated_at = time.monotonic()

    def snapshot(self) -> tuple[list[DetectionBox], int, int] | None:
        with self._lock:
            if not self._boxes:
                return None
            if (time.monotonic() - self._updated_at) * 1000.0 > DETECTIONS_STALE_MS:
                return None
            # Return a shallow copy so callers can't mutate the cache's list
            # out from under the subscriber thread.  ``DetectionBox`` is frozen,
            # so the shallow copy is sufficient.
            return list(self._boxes), self._frame_w, self._frame_h


def parse_detections_payload(payload: bytes) -> tuple[list[DetectionBox], int, int] | None:
    """Decode a JSON detection payload. Returns ``None`` on parse error."""
    try:
        data = json.loads(payload)
        dets_raw = data.get("detections") or []
        frame_w = int(data.get("frame_width", 0) or 0)
        frame_h = int(data.get("frame_height", 0) or 0)
        boxes: list[DetectionBox] = []
        for det in dets_raw:
            if not isinstance(det, dict):
                continue
            boxes.append(
                DetectionBox(
                    label=str(det.get("label", "?")),
                    confidence=float(det.get("confidence", 0.0) or 0.0),
                    x1=int(det.get("x1", 0) or 0),
                    y1=int(det.get("y1", 0) or 0),
                    x2=int(det.get("x2", 0) or 0),
                    y2=int(det.get("y2", 0) or 0),
                )
            )
    except (ValueError, TypeError, AttributeError):
        logger.debug("Failed to parse detection payload", exc_info=True)
        return None
    return boxes, frame_w, frame_h


def draw_detections(
    frame: np.ndarray,
    boxes: list[DetectionBox],
    det_w: int,
    det_h: int,
) -> None:
    """Draw ``boxes`` in-place on ``frame``.

    Matches the OBSBOT C++ draw routine (same colors, font, and label
    background style) so the frontend renders both driver families
    identically.
    """
    import cv2

    h, w = frame.shape[:2]
    sx = (w / det_w) if det_w > 0 else 1.0
    sy = (h / det_h) if det_h > 0 else 1.0

    for box in boxes:
        x1 = int(box.x1 * sx)
        y1 = int(box.y1 * sy)
        x2 = int(box.x2 * sx)
        y2 = int(box.y2 * sy)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{box.label} {box.confidence * 100.0:.0f}%"
        (text_w, text_h), _baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        # Place the label above the bbox when there's room, otherwise drop it
        # inside the top of the bbox so near-edge detections don't render with
        # the label clipped off the frame.
        if y1 - text_h - 6 >= 0:
            bg_top, bg_bottom, text_y = y1 - text_h - 6, y1, y1 - 4
        else:
            bg_top, bg_bottom, text_y = y1, y1 + text_h + 6, y1 + text_h + 2
        cv2.rectangle(
            frame,
            (x1, bg_top),
            (x1 + text_w + 4, bg_bottom),
            (0, 255, 0),
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )



# ── Workflow overlay (annotate-node-driven) ─────────────────────────
#
# The ``annotate`` workflow node publishes a JSON spec to
# ``FRAME_OVERLAY_CHANNEL`` (``frames/overlay``) that carries the
# user's styling parameters (``line_width``, ``font_scale``, label
# filter) on top of the raw detection list. The driver caches the
# latest payload per twin and composites it onto every frame
# pre-encode, *unconditionally* — overlay is additive, not pixel
# substitution, so it doesn't ride the privacy-fail-closed
# ``frame_filter_enabled`` gate that ``anonymize`` uses.
#
# When a fresh overlay payload is present it preempts the legacy
# raw-``detections/<runtime>`` overlay so the user's styling wins.
# When stale or absent, the driver falls back to ``draw_detections``.


class OverlayCache:
    """Thread-safe holder for the most recent overlay payload.

    ``payload`` is the dict returned by
    :func:`cyberwave.vision.build_overlay_payload`. Returns ``None``
    from :meth:`snapshot` when the cache is empty or the last update
    is older than :data:`DETECTIONS_STALE_MS` (same TTL as the raw
    detection cache, keeping the two paths feel-equivalent to the
    operator).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._payload: dict | None = None
        self._updated_at = 0.0

    def update(self, payload: dict) -> None:
        with self._lock:
            self._payload = payload
            self._updated_at = time.monotonic()

    def snapshot(self) -> dict | None:
        with self._lock:
            if self._payload is None:
                return None
            if (time.monotonic() - self._updated_at) * 1000.0 > DETECTIONS_STALE_MS:
                return None
            # Defensive shallow copy — mirrors ``DetectionCache.snapshot``
            # so a misbehaving caller can't mutate the cached dict (e.g.
            # by sorting the boxes list in place) and corrupt subsequent
            # reads on the encoder thread.
            return dict(self._payload)


def validate_overlay_payload(data: object) -> dict | None:
    """Schema-check an already-decoded overlay payload.

    The Zenoh subscriber gets the decoded dict directly from the data
    bus (``cw.data.publish`` writes a CONTENT_TYPE_JSON envelope and
    ``cw.data.subscribe`` strips the header + ``json.loads`` it), so
    the validation step doesn't need to know about wire bytes at all.
    Kept separate from :func:`parse_overlay_payload` so the bytes-in
    test entry point stays usable.
    """
    if not isinstance(data, dict):
        return None
    # ``v == 1`` is the only schema this driver knows; anything else
    # is silently dropped so future bumps remain forward-compatible
    # (the worker keeps publishing, just nothing renders here).
    if data.get("v") != 1:
        return None
    if not isinstance(data.get("boxes"), list):
        return None
    return data


def parse_overlay_payload(payload: bytes) -> dict | None:
    """Decode + validate an overlay JSON payload from raw bytes.

    Used by tests that want to exercise the full bytes-in / dict-out
    round-trip without spinning up a data bus. The driver subscriber
    callback uses :func:`validate_overlay_payload` directly because
    ``cw.data.subscribe`` already hands back a decoded dict.
    """
    try:
        data = json.loads(payload)
    except (ValueError, TypeError):
        logger.debug("Failed to parse overlay payload", exc_info=True)
        return None
    return validate_overlay_payload(data)


# Defensive cap on incoming polygon vertices per detection. The SDK
# helper caps at ~64 via approxPolyDP, so anything over this is either
# a misbehaving worker or a future schema change we don't know about.
# Worst-case 64 KB of int32 points per detection — bounded enough to
# composite at frame rate without DoSing the encoder.
POLY_MAX_POINTS: int = 1024


def overlay_color_from_box(box: dict) -> tuple[int, int, int]:
    """Return the pre-resolved BGR color from a box entry, or fall back to auto.

    ``build_overlay_payload`` in the SDK resolves the palette and includes
    ``color: [r, g, b]`` on each box so the driver never needs its own palette
    copy. For payloads from older SDK versions that lack the field we delegate
    to ``cyberwave.vision.annotate.label_color``; if the installed SDK predates
    that helper we degrade to a constant green so the frame callback never
    raises ``ImportError`` on every frame.
    """
    raw = box.get("color")
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        try:
            return (int(raw[0]), int(raw[1]), int(raw[2]))
        except (TypeError, ValueError):
            pass
    try:
        from cyberwave.vision.annotate import label_color
    except ImportError:
        return (0, 200, 0)
    return label_color(str(box.get("label", "")))


def draw_overlay_masks(
    frame: np.ndarray,
    boxes: list,
    *,
    mask_alpha: float,
    mask_outline: bool,
    line_width: int,
) -> None:
    """Composite per-detection polygons onto ``frame`` in place.

    Polygons live in original-frame coordinates. Pixels outside any
    polygon are untouched. Boxes without a ``polygon`` field skip the
    mask path so detection-only models cost nothing here.
    """
    import cv2

    h, w = frame.shape[:2]
    fill_overlay: np.ndarray | None = None
    mask_acc_u8: np.ndarray | None = None
    outlines: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    for box in boxes:
        poly = box.get("polygon")
        if not isinstance(poly, list) or not (3 <= len(poly) <= POLY_MAX_POINTS):
            continue
        try:
            pts = np.array(
                [[int(p[0]), int(p[1])] for p in poly if len(p) >= 2],
                dtype=np.int32,
            )
        except (TypeError, ValueError):
            continue
        if pts.shape[0] < 3:
            continue
        np.clip(pts[:, 0], 0, w - 1, out=pts[:, 0])
        np.clip(pts[:, 1], 0, h - 1, out=pts[:, 1])
        color = overlay_color_from_box(box)
        if mask_alpha > 0:
            if fill_overlay is None or mask_acc_u8 is None:
                fill_overlay = frame.copy()
                mask_acc_u8 = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(fill_overlay, [pts], color)
            cv2.fillPoly(mask_acc_u8, [pts], 1)
        if mask_outline:
            outlines.append((pts, color))
    if fill_overlay is not None and mask_acc_u8 is not None and mask_alpha > 0:
        mask_acc = mask_acc_u8.astype(bool)
        if mask_acc.any():
            blended = (
                mask_alpha * fill_overlay.astype(np.float32)
                + (1.0 - mask_alpha) * frame.astype(np.float32)
            ).astype(np.uint8)
            frame[mask_acc] = blended[mask_acc]
    if mask_outline and line_width > 0:
        for pts, color in outlines:
            cv2.polylines(frame, [pts], True, color, max(1, line_width))


def draw_overlay(frame: np.ndarray, payload: dict) -> None:
    """Draw the overlay payload's boxes + captions on ``frame`` in place.

    Mirrors :func:`draw_detections` visually but reads styling from the
    payload's ``style`` block. Per-box colour comes from the ``color`` field
    that ``build_overlay_payload`` in the SDK pre-resolves from the chosen
    palette — the driver has no palette knowledge and stays in sync with the
    SDK automatically.
    """
    import cv2

    boxes = payload.get("boxes") or []
    if not boxes:
        return
    style = payload.get("style") or {}
    line_width = max(0, int(style.get("line_width", 2) or 0))
    font_scale = float(style.get("font_scale", 0.5) or 0.0)
    show_confidence = bool(style.get("show_confidence", True))
    mask_alpha = max(0.0, min(1.0, float(style.get("mask_alpha", 0.0) or 0.0)))
    mask_outline = bool(style.get("mask_outline", True))

    h, w = frame.shape[:2]

    if mask_alpha > 0 or mask_outline:
        draw_overlay_masks(
            frame,
            boxes,
            mask_alpha=mask_alpha,
            mask_outline=mask_outline,
            line_width=line_width,
        )

    for box in boxes:
        coords = box.get("box_2d")
        if not isinstance(coords, list | tuple) or len(coords) != 4:
            continue
        try:
            x1 = max(0, min(int(coords[0]), w - 1))
            y1 = max(0, min(int(coords[1]), h - 1))
            x2 = max(0, min(int(coords[2]), w - 1))
            y2 = max(0, min(int(coords[3]), h - 1))
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue

        color = overlay_color_from_box(box)

        if line_width > 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_width)

        if font_scale <= 0:
            continue

        label_text = str(box.get("label", "?"))
        if show_confidence:
            try:
                conf = float(box.get("conf", 0.0) or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            label_text = f"{label_text} {conf * 100.0:.0f}%"
        text_thickness = max(1, line_width // 2 if line_width > 0 else 1)
        (text_w, text_h), _baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        if y1 - text_h - 6 >= 0:
            bg_top, bg_bottom, text_y = y1 - text_h - 6, y1, y1 - 4
        else:
            bg_top, bg_bottom, text_y = y1, y1 + text_h + 6, y1 + text_h + 2
        cv2.rectangle(
            frame,
            (x1, bg_top),
            (x1 + text_w + 4, bg_bottom),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            label_text,
            (x1 + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA,
        )
