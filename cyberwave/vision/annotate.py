"""Detection annotation overlay — the numpy-frame analogue of the
cloud-side ``ANNOTATE_IMAGE`` workflow node.

Used by edge workers that want to publish a "human-readable" version of
their model output back through the camera driver's frame-filter, e.g.

    from cyberwave.vision import annotate_detections
    result = model.predict(frame)
    debug_frame = annotate_detections(frame, result.detections)
    cw.data.publish(FILTERED_FRAME_CHANNEL, debug_frame, twin_uuid=...)

The helper draws an axis-aligned bounding box around each matching
detection and a small ``label conf`` caption above (or just below, if
the box hugs the top of the frame). It is intentionally a thin,
predictable utility — just rectangle + caption — so the codegen
template stays trivial.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Iterable

import numpy as np

from cyberwave.models.types import Detection
from cyberwave.vision.anonymize import _bbox_int, _require_cv2

# Pleasant, distinguishable BGR palette.  Picked so adjacent indices stay
# visually distinct (no two reds in a row); long enough that small
# class-vocabulary models almost never reuse a colour.
_DEFAULT_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 200, 0),  # green
    (0, 165, 255),  # orange
    (255, 0, 0),  # blue
    (0, 0, 255),  # red
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
    (128, 0, 128),  # purple
    (255, 128, 0),  # azure
    (128, 128, 0),  # teal
    (0, 128, 255),  # amber
    (203, 192, 255),  # pink
)


def _default_color_for(label: str) -> tuple[int, int, int]:
    """Deterministic per-label colour from the default palette.

    Hashing the label keeps the same class on the same colour across
    frames (visual continuity) without forcing the caller to maintain a
    label→colour map. ``hashlib.md5`` is used purely as a stable hash;
    nothing here is cryptographic.
    """
    digest = hashlib.md5(label.encode("utf-8")).digest()
    return _DEFAULT_PALETTE[digest[0] % len(_DEFAULT_PALETTE)]


def annotate_detections(
    frame: np.ndarray,
    detections: Iterable[Detection],
    *,
    labels: Iterable[str] | None = None,
    line_width: int = 2,
    font_scale: float = 0.5,
    color_fn: Callable[[Detection], tuple[int, int, int]] | None = None,
    show_confidence: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """Draw bounding boxes + ``label conf`` captions onto ``frame``.

    Args:
        frame: HxWxC uint8 image (BGR).
        detections: Iterable of :class:`~cyberwave.models.types.Detection`.
        labels: Optional iterable restricting which classes are drawn.
            ``None`` (default) draws every detection. An empty iterable
            draws nothing — useful when codegen passes a dynamically
            computed target-class list and may end up with zero classes.
        line_width: Bounding-box line thickness in pixels. ``0`` skips
            the box (caption-only mode).
        font_scale: Caption font scale (cv2 units). ``0`` skips the
            caption (box-only mode).
        color_fn: Callable mapping a :class:`Detection` to a BGR colour
            tuple. ``None`` (default) uses a deterministic per-label
            palette so the same class keeps the same colour across
            frames without the caller maintaining a map.
        show_confidence: When ``True`` (default) the caption reads
            ``"label 0.87"``. ``False`` drops the score and leaves just
            the label, useful for documentation-style frames.
        inplace: When ``True``, mutate ``frame`` directly. Default
            ``False`` returns a fresh array (input untouched), matching
            the rest of :mod:`cyberwave.vision`.

    Returns:
        Either a new array (default) or the mutated input.

    Raises:
        ValueError: if ``frame`` is not at least 2-D.
    """
    if frame.ndim < 2:
        raise ValueError(f"frame must be at least 2-D, got shape {frame.shape}")

    cv2 = _require_cv2()
    out = frame if inplace else frame.copy()
    h, w = out.shape[:2]

    active_labels: frozenset[str] | None = (
        frozenset(labels) if labels is not None else None
    )

    for det in detections:
        if active_labels is not None and det.label not in active_labels:
            continue
        x1, y1, x2, y2 = _bbox_int(det, out.shape)
        if x2 <= x1 or y2 <= y1:
            continue

        # ``color_fn`` receives the full Detection so callers can vary
        # colour by confidence, class index, etc.  The built-in palette
        # only needs the label string, so the two paths have different
        # argument types — kept explicit rather than merged under one name.
        colour = color_fn(det) if color_fn is not None else _default_color_for(det.label)

        if line_width > 0:
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, line_width)

        if font_scale <= 0:
            continue

        caption = f"{det.label} {det.confidence:.2f}" if show_confidence else det.label
        # ``FONT_HERSHEY_SIMPLEX`` matches Ultralytics' default annotator
        # so worker output and cloud-rendered annotations stay visually
        # consistent. cv2 caption thickness must be >= 1.
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_thickness = max(1, line_width // 2 if line_width > 0 else 1)
        (text_w, text_h), baseline = cv2.getTextSize(
            caption, font, font_scale, text_thickness
        )

        # Place the caption above the box if there's room, otherwise
        # tuck it just inside the top edge so it doesn't fall off-frame.
        pad = 2
        bg_y2 = y1
        bg_y1 = bg_y2 - text_h - baseline - 2 * pad
        if bg_y1 < 0:
            bg_y1 = y1
            bg_y2 = bg_y1 + text_h + baseline + 2 * pad
        bg_x1 = max(0, x1)
        bg_x2 = min(w, bg_x1 + text_w + 2 * pad)

        # Filled background rectangle in the box colour, with the text
        # drawn in the contrasting colour (white on dark, black on light).
        cv2.rectangle(out, (bg_x1, bg_y1), (bg_x2, bg_y2), colour, -1)
        text_colour = _contrast_text_colour(colour)
        text_x = bg_x1 + pad
        text_y = bg_y2 - baseline - pad
        cv2.putText(
            out,
            caption,
            (text_x, text_y),
            font,
            font_scale,
            text_colour,
            text_thickness,
            lineType=cv2.LINE_AA,
        )

    return out


def _contrast_text_colour(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    """Pick black or white text for the given BGR background colour.

    Uses the standard ITU-R BT.601 luma approximation, biased so that
    mid-tones lean toward white text (which reads better against the
    saturated palette colours used by :func:`_default_color_for`).
    """
    b, g, r = bg
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luma > 160 else (255, 255, 255)


# ── Overlay payload (driver-side draw, no pixel substitution) ───────

# Schema version for ``build_overlay_payload``. The camera driver
# checks ``payload["v"]`` so a future breaking change can be rolled
# out by bumping this and teaching the driver to accept both versions
# during the transition.
OVERLAY_PAYLOAD_VERSION: int = 1


def build_overlay_payload(
    detections: Iterable[Detection],
    *,
    labels: Iterable[str] | None = None,
    line_width: int = 2,
    font_size: int = 14,
    font_scale: float | None = None,
    show_confidence: bool = True,
) -> dict:
    """Build a JSON-serialisable overlay spec for the camera driver.

    The ``annotate`` workflow node publishes the result of this helper
    on :data:`cyberwave.data.FRAME_OVERLAY_CHANNEL`. The driver caches
    the latest payload per twin and composites the boxes/captions onto
    the frame just before WebRTC encode — independently of the
    per-twin ``frame_filter_enabled`` flag (which gates pixel
    *substitution*, an anonymise-only concern).

    Args:
        detections: Iterable of :class:`~cyberwave.models.types.Detection`
            from the upstream ``call_model`` node.
        labels: Optional iterable restricting which classes are drawn.
            ``None`` (default) draws every detection. An empty iterable
            draws nothing — useful when codegen passes a dynamically
            computed target-class list and may end up with zero classes.
        line_width: Bounding-box line thickness in pixels. Mirrors the
            same parameter on :func:`annotate_detections`.
        font_size: Caption font size in points (the workflow node's
            user-facing unit). Converted to ``font_scale`` using the
            same divisor (``28``) the codegen uses today so the visual
            output matches the in-process numpy helper.
        font_scale: Direct cv2 font-scale override. When set, takes
            precedence over ``font_size``. Lets callers that already
            have a scale keep using it.
        show_confidence: Whether captions include the score. Defaults
            to ``True``; set ``False`` for cleaner documentation-style
            overlays.

    Returns:
        A JSON-serialisable dict::

            {
                "v": 1,
                "boxes": [
                    {"box_2d": [x1, y1, x2, y2], "label": "person", "conf": 0.92},
                    ...
                ],
                "style": {
                    "line_width": 2,
                    "font_scale": 0.5,
                    "show_confidence": True,
                },
            }

        Coordinates are in the **original frame's pixel space** —
        the driver clamps them to the encode resolution.
    """
    active_labels: frozenset[str] | None = (
        frozenset(labels) if labels is not None else None
    )
    boxes: list[dict] = []
    for det in detections:
        if active_labels is not None and det.label not in active_labels:
            continue
        bbox = det.bbox
        # ``BoundingBox`` already validates non-inverted coordinates;
        # we just float-cast so the payload survives ``json.dumps``
        # (numpy floats from inference backends would otherwise break).
        boxes.append(
            {
                "box_2d": [
                    float(bbox.x1),
                    float(bbox.y1),
                    float(bbox.x2),
                    float(bbox.y2),
                ],
                "label": det.label,
                "conf": float(det.confidence),
            }
        )

    if font_scale is None:
        # Same divisor codegen used historically when translating the
        # node's user-facing ``font_size`` (points) to cv2 ``font_scale``.
        # Kept here so the conversion lives next to the schema and the
        # in-process helper stays a pure cv2 wrapper.
        font_scale = font_size / 28.0

    return {
        "v": OVERLAY_PAYLOAD_VERSION,
        "boxes": boxes,
        "style": {
            "line_width": int(line_width),
            "font_scale": float(font_scale),
            "show_confidence": bool(show_confidence),
        },
    }


__all__ = [
    "OVERLAY_PAYLOAD_VERSION",
    "annotate_detections",
    "build_overlay_payload",
]
