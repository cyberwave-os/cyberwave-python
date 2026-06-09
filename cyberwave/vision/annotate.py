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

import base64
import hashlib
import logging
from typing import Any, Callable, Iterable, Literal

import numpy as np

from cyberwave.models.types import Detection, Mask
from cyberwave.vision._detection_view import as_detection
from cyberwave.vision.anonymize import _bbox_int, _require_cv2

logger = logging.getLogger(__name__)

MaskFormat = Literal["polygon", "png", "polygon+png"]

BoxColor = Literal["auto", "cool", "warm", "neon", "pastel"]
"""Named palette for bounding-box colours.

``"auto"`` (default) picks a deterministic per-label colour from a
diverse 12-hue palette so the same class always gets the same shade.
The other four names are thematic palette sets suited to different
camera environments.
"""

_PALETTES: dict[str, tuple[tuple[int, int, int], ...]] = {
    "auto": (
        (0, 200, 0),
        (0, 165, 255),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 128),
        (255, 128, 0),
        (128, 128, 0),
        (0, 128, 255),
        (203, 192, 255),
    ),
    "cool": (
        (255, 144, 30),
        (209, 206, 0),
        (208, 224, 64),
        (226, 43, 138),
        (180, 130, 70),
        (170, 178, 32),
    ),
    "warm": (
        (0, 69, 255),
        (0, 165, 255),
        (0, 215, 255),
        (60, 20, 220),
        (180, 105, 255),
        (0, 140, 255),
    ),
    "neon": (
        (0, 255, 0),
        (255, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (100, 0, 255),
        (0, 100, 255),
    ),
    "pastel": (
        (241, 214, 174),
        (191, 223, 169),
        (145, 176, 245),
        (226, 189, 215),
        (159, 231, 249),
        (160, 215, 250),
    ),
}

_VALID_BOX_COLORS: frozenset[str] = frozenset(_PALETTES)

# Keep the old name as an alias so existing callers that import it directly
# still work.
_DEFAULT_PALETTE = _PALETTES["auto"]


def _default_color_for(label: str, box_color: BoxColor = "auto") -> tuple[int, int, int]:
    """Deterministic per-label colour from the named palette.

    ``hashlib.md5`` is used purely as a stable hash; nothing here is
    cryptographic.
    """
    palette = _PALETTES.get(box_color, _PALETTES["auto"])
    digest = hashlib.md5(label.encode("utf-8")).digest()
    return palette[digest[0] % len(palette)]


def label_color(label: str, box_color: BoxColor = "auto") -> tuple[int, int, int]:
    """Return a stable BGR colour for *label* from the named palette.

    Drivers and other consumers should call this instead of reimplementing
    the palette lookup, so the colour assignment stays byte-identical to what
    ``build_overlay_payload`` embeds in every box entry.
    """
    return _default_color_for(label, box_color)


def annotate_detections(
    frame: np.ndarray,
    detections: Iterable[Detection],
    *,
    labels: Iterable[str] | None = None,
    line_width: int = 2,
    font_scale: float = 0.5,
    color_fn: Callable[[Detection], tuple[int, int, int]] | None = None,
    box_color: BoxColor = "auto",
    show_confidence: bool = True,
    mask_alpha: float = 0.35,
    mask_outline: bool = True,
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
            tuple. ``None`` (default) uses the palette selected by
            ``box_color``. Takes priority over ``box_color`` when set.
        box_color: Named palette — ``"auto"`` (default), ``"cool"``,
            ``"warm"``, ``"neon"``, or ``"pastel"``. Ignored when
            ``color_fn`` is provided.
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

    mask_alpha = float(np.clip(mask_alpha, 0.0, 1.0))
    visible: list[tuple[Detection, int, int, int, int, tuple[int, int, int]]] = []
    for raw_det in detections:
        det = as_detection(raw_det)
        if active_labels is not None and det.label not in active_labels:
            continue
        x1, y1, x2, y2 = _bbox_int(det, out.shape)
        if x2 <= x1 or y2 <= y1:
            continue
        colour = (
            color_fn(det)
            if color_fn is not None
            else _default_color_for(det.label, box_color)
        )
        visible.append((det, x1, y1, x2, y2, colour))

    if mask_alpha > 0 or mask_outline:
        fill_overlay: np.ndarray | None = None
        mask_acc_u8: np.ndarray | None = None
        outlines: list[tuple[np.ndarray, tuple[int, int, int]]] = []
        for det, *_, colour in visible:
            det_mask = getattr(det, "mask", None)
            if det_mask is None:
                continue
            poly = mask_to_polygon(det_mask)
            if poly is None:
                continue
            pts = np.array(poly, dtype=np.int32)
            if mask_alpha > 0:
                if fill_overlay is None:
                    fill_overlay = out.copy()
                    mask_acc_u8 = np.zeros(out.shape[:2], dtype=np.uint8)
                cv2.fillPoly(fill_overlay, [pts], colour)
                cv2.fillPoly(mask_acc_u8, [pts], 1)
            if mask_outline:
                outlines.append((pts, colour))
        if fill_overlay is not None and mask_acc_u8 is not None:
            mask_acc = mask_acc_u8.astype(bool)
            if mask_acc.any():
                blended = (
                    mask_alpha * fill_overlay.astype(np.float32)
                    + (1.0 - mask_alpha) * out.astype(np.float32)
                ).astype(np.uint8)
                out[mask_acc] = blended[mask_acc]
        if mask_outline and line_width > 0:
            for pts, colour in outlines:
                cv2.polylines(out, [pts], True, colour, max(1, line_width))

    for det, x1, y1, x2, y2, colour in visible:
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


# ── Segmentation helpers ────────────────────────────────────────────

# Max polygon vertices per detection before we stop tightening
# approxPolyDP and accept the result. 64 is plenty for YOLO-seg blobs
# in practice; a curved shoreline still reads well at this density.
_MASK_POLY_MAX_POINTS: int = 64
# Skip masks whose largest contour covers fewer pixels than this —
# noise from low-confidence segments doesn't deserve a polygon.
_MASK_POLY_MIN_AREA_PX: int = 16


def mask_to_polygon(
    mask: Mask,
    *,
    max_points: int = _MASK_POLY_MAX_POINTS,
    min_area_px: int = _MASK_POLY_MIN_AREA_PX,
) -> list[list[int]] | None:
    """Extract the largest external contour from ``mask`` as an (x, y) polygon.

    Returns ``None`` when cv2 is unavailable, the mask is empty, or the
    largest contour fails the area filter. v1 silently drops holes and
    secondary blobs (single-instance YOLO-seg case is the common one).

    Public so the backend's ``spatial_filter`` codegen can import it
    without reaching across packages for a private symbol — the worker
    inlines :func:`workflow_utils._detection_polygon` via
    ``inspect.getsource`` and that helper imports this name at runtime.
    """
    try:
        cv2 = _require_cv2()
    except Exception:
        logger.debug("cv2 unavailable; skipping mask polygon extraction")
        return None
    binary = _mask_to_binary(mask, cv2)
    if binary is None:
        return None
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area_px:
        return None
    perimeter = cv2.arcLength(largest, True)
    epsilon = max(1.0, 0.005 * perimeter)
    poly = cv2.approxPolyDP(largest, epsilon, True)
    for _ in range(4):
        if len(poly) <= max_points:
            break
        epsilon *= 1.5
        poly = cv2.approxPolyDP(largest, epsilon, True)
    return [[int(p[0][0]), int(p[0][1])] for p in poly]


def _mask_to_binary(mask: Mask, cv2: Any) -> np.ndarray | None:
    """Convert a :class:`Mask` to a uint8 binary array sized ``(mask.h, mask.w)``.

    Threshold is ``> 0.5`` so Ultralytics' float sigmoid output binarizes
    the same way its own annotator does. ``bool`` and integer masks pass
    through the same threshold (both ``0`` and ``False`` fail the test).
    Returns ``None`` for empty / missing data.
    """
    data = mask.data
    if data is None:
        return None
    arr = np.asarray(data)
    if arr.size == 0:
        return None
    binary = (arr > 0.5).astype(np.uint8) * 255
    if mask.h > 0 and mask.w > 0 and binary.shape[:2] != (mask.h, mask.w):
        binary = cv2.resize(binary, (mask.w, mask.h), interpolation=cv2.INTER_NEAREST)
    return binary


def _mask_to_png_b64(mask: Mask, bbox: tuple[int, int, int, int]) -> str | None:
    """Encode the bbox-cropped binary mask as a base64 PNG (cloud parity)."""
    try:
        cv2 = _require_cv2()
    except Exception:
        return None
    binary = _mask_to_binary(mask, cv2)
    if binary is None:
        return None
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(binary.shape[1], x2)
    y2 = min(binary.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = binary[y1:y2, x1:x2]
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ── Overlay payload (driver-side draw, no pixel substitution) ───────

# Schema version for ``build_overlay_payload``. The camera driver
# checks ``payload["v"]`` so a future breaking change can be rolled
# out by bumping this and teaching the driver to accept both versions
# during the transition.
OVERLAY_PAYLOAD_VERSION: int = 1

# Per-detection mask payload cannot exceed this many bytes (base64).
# YOLO-seg masks on a 1080p frame routinely compress to under 4 KB; a
# 16 KB cap leaves headroom for messy scenes without ballooning the
# overlay bus when one model emits dozens of instances.
_MASK_B64_MAX_BYTES: int = 16384


def build_overlay_payload(
    detections: Iterable[Detection],
    *,
    labels: Iterable[str] | None = None,
    line_width: int = 2,
    font_size: int = 14,
    font_scale: float | None = None,
    show_confidence: bool = True,
    mask_format: MaskFormat = "polygon",
    mask_alpha: float = 0.35,
    mask_outline: bool = True,
    box_color: BoxColor = "auto",
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
        box_color: Named palette — ``"auto"`` (default), ``"cool"``,
            ``"warm"``, ``"neon"``, or ``"pastel"``. Each box entry in the
            payload receives a pre-resolved ``color`` field so the camera
            driver can draw the correct colour without any palette knowledge.
            ``style.box_color`` is also included for reference. Unknown values
            silently fall back to ``"auto"``.

    Returns:
        A JSON-serialisable dict::

            {
                "v": 1,
                "boxes": [
                    {
                        "box_2d": [x1, y1, x2, y2],
                        "label": "person",
                        "conf": 0.92,
                        "color": [r, g, b],
                    },
                    ...
                ],
                "style": {
                    "line_width": 2,
                    "font_scale": 0.5,
                    "show_confidence": True,
                    "mask_alpha": 0.35,
                    "mask_outline": True,
                    "box_color": "auto",
                },
            }

        Coordinates are in the **original frame's pixel space** —
        the driver clamps them to the encode resolution.
    """
    active_labels: frozenset[str] | None = (
        frozenset(labels) if labels is not None else None
    )
    resolved_box_color = box_color if box_color in _VALID_BOX_COLORS else "auto"
    want_polygon = mask_format in ("polygon", "polygon+png")
    want_png = mask_format in ("png", "polygon+png")
    boxes: list[dict] = []
    for raw_det in detections:
        det = as_detection(raw_det)
        if active_labels is not None and det.label not in active_labels:
            continue
        bbox = det.bbox
        # ``BoundingBox`` already validates non-inverted coordinates;
        # we just float-cast so the payload survives ``json.dumps``
        # (numpy floats from inference backends would otherwise break).
        r, g, b = _default_color_for(det.label, resolved_box_color)
        entry: dict[str, Any] = {
            "box_2d": [
                float(bbox.x1),
                float(bbox.y1),
                float(bbox.x2),
                float(bbox.y2),
            ],
            "label": det.label,
            "conf": float(det.confidence),
            "color": [r, g, b],
        }
        det_mask = getattr(det, "mask", None)
        if det_mask is not None and (want_polygon or want_png):
            if want_polygon:
                poly = mask_to_polygon(det_mask)
                if poly is not None:
                    entry["polygon"] = poly
            if want_png:
                ibox = (
                    int(bbox.x1),
                    int(bbox.y1),
                    int(bbox.x2),
                    int(bbox.y2),
                )
                b64 = _mask_to_png_b64(det_mask, ibox)
                if b64 is not None and len(b64) <= _MASK_B64_MAX_BYTES:
                    entry["mask_b64"] = b64
        boxes.append(entry)

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
            "mask_alpha": float(np.clip(mask_alpha, 0.0, 1.0)),
            "mask_outline": bool(mask_outline),
            "box_color": resolved_box_color,
        },
    }


__all__ = [
    "OVERLAY_PAYLOAD_VERSION",
    "BoxColor",
    "MaskFormat",
    "annotate_detections",
    "build_overlay_payload",
    "label_color",
    "mask_to_polygon",
]
