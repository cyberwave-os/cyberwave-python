"""Pose-aware visual obscuring helpers.

These functions transform a frame so that detected people are visually
obscured while the useful structural information (presence, posture)
is preserved as a skeleton overlay.

Typical use is from a worker that runs a pose model:

    from cyberwave.vision import anonymize_frame
    result = model.predict(frame, classes=["person"])
    out = anonymize_frame(frame, result.detections)  # mode="pixelate" by default

Four obscuring modes are supported:

- ``"pixelate"`` (default) applies a mosaic effect — the industry-standard
  treatment for "this is a person but you can't tell who". Recognisable
  silhouette and motion are preserved.
- ``"redact"`` paints a grid of solid ``color`` (default black) blocks
  with thin visible separators — the "censored document" look. Destroys
  all underlying pixel information like ``"bbox"`` but keeps a visible
  mosaic structure, which reads as deliberate redaction rather than a
  rendering glitch.
- ``"blur"`` applies a heavy Gaussian blur to each person bbox.
- ``"bbox"`` fills each person bbox with a solid colour. Bluntest option;
  mainly useful when the downstream consumer needs a clean mask region.

In all modes the detected pose skeleton is drawn on top so the frame
still conveys "someone is here, in this posture" — useful for security
review, occupancy heatmaps, and motion analytics.

Privacy caveat
--------------
These helpers are designed for casual visual obscuring, not as a
cryptographic anonymisation primitive. In particular:

- ``"pixelate"`` is reversible. Public depixelation models can recover
  recognisable faces from low-density mosaics, especially at our default
  block size (~24 blocks across the short side, tuned for visibility).
- ``"blur"`` with the default kernel (99) is much harder to invert but
  not impossible — strong blur is *not* a substitute for redaction when
  legal de-identification is required.
- ``"bbox"`` and ``"redact"`` both destroy the underlying pixel
  information — they paint solid ``color`` blocks with no original
  pixels surviving. Prefer ``"redact"`` when you want the destruction
  to *look* deliberate (audit trails, public release), ``"bbox"`` when
  you want a clean uniform mask. Either way verify the encoded stream
  if you need a hard guarantee that no source pixels leave the device.

For tighter requirements (e.g. GDPR-grade de-identification), combine
the bbox mask with format-shifting (publish only detection events, not
frames), or run the obscured frame through a second irreversible
transform.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from cyberwave.models.types import Detection

# COCO-17 keypoint connectivity. Each tuple is a pair of keypoint indices
# that should be connected with a line segment to form the skeleton.
# Index reference (Ultralytics / COCO):
#   0  nose            5  left shoulder    11 left hip       15 left ankle
#   1  left eye        6  right shoulder   12 right hip      16 right ankle
#   2  right eye       7  left elbow       13 left knee
#   3  left ear        8  right elbow      14 right knee
#   4  right ear       9  left wrist
#                     10  right wrist
COCO17_SKELETON: Sequence[tuple[int, int]] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # head
    (5, 6),  # shoulders
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    (5, 11),
    (6, 12),
    (11, 12),  # torso
    (11, 13),
    (13, 15),  # left leg
    (12, 14),
    (14, 16),  # right leg
)


# Colour palette per body part (BGR — OpenCV native).  These colours are
# distinguishable on both light and dark backgrounds and follow the broadcast
# convention of warm colours on the subject's right side, cool on the left.
COCO17_SEGMENT_COLORS: dict[str, tuple[int, int, int]] = {
    "head": (255, 191, 0),  # cyan
    "torso": (0, 255, 255),  # yellow
    "right_arm": (0, 128, 255),  # orange  (subject's right = image left)
    "left_arm": (0, 255, 128),  # spring green
    "right_leg": (0, 0, 255),  # red
    "left_leg": (255, 128, 0),  # azure
}


# Map every COCO-17 edge to a body-part group so we can colour-code the
# skeleton.  Symmetric to :data:`COCO17_SKELETON`.
COCO17_EDGE_GROUPS: dict[tuple[int, int], str] = {
    (0, 1): "head",
    (0, 2): "head",
    (1, 3): "head",
    (2, 4): "head",
    (5, 6): "torso",
    (5, 11): "torso",
    (6, 12): "torso",
    (11, 12): "torso",
    (5, 7): "left_arm",
    (7, 9): "left_arm",
    (6, 8): "right_arm",
    (8, 10): "right_arm",
    (11, 13): "left_leg",
    (13, 15): "left_leg",
    (12, 14): "right_leg",
    (14, 16): "right_leg",
}


_JOINT_DEFAULT_COLOR: tuple[int, int, int] = (255, 255, 255)  # white — high contrast


def _require_cv2() -> Any:
    """Import cv2 lazily so the SDK can be imported without OpenCV installed."""
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - exercised when cv2 missing
        raise ImportError(
            "cyberwave.vision requires OpenCV. Install with: pip install opencv-python"
        ) from exc
    return cv2


def _bbox_int(
    det: Detection, frame_shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    """Clamp a Detection's bbox to integer pixel coords inside the frame."""
    h, w = frame_shape[:2]
    x1 = max(0, min(int(det.bbox.x1), w))
    y1 = max(0, min(int(det.bbox.y1), h))
    x2 = max(0, min(int(det.bbox.x2), w))
    y2 = max(0, min(int(det.bbox.y2), h))
    return x1, y1, x2, y2


def _redact_roi(
    roi: np.ndarray,
    pixel_size: int,
    color: tuple[int, int, int],
    cv2: Any,
) -> np.ndarray:
    """Tile the ROI with solid ``color`` blocks separated by thin grid lines.

    The visible grid is what distinguishes this from ``mode="bbox"`` —
    grid lines are drawn 40 BGR units brighter than ``color`` so the
    mosaic structure is visible against any base colour. For very light
    ``color`` (>=215 in any channel), the offset clamps at 255 and the
    grid silently vanishes; that's a deliberate trade-off (a configurable
    grid colour would just push the API surface up).
    """
    h, w = roi.shape[:2]
    block = max(2, min(pixel_size, h, w))
    out = np.empty_like(roi)
    out[:] = color
    grid = tuple(int(min(255, c + 40)) for c in color)
    for x in range(block, w, block):
        cv2.line(out, (x, 0), (x, h - 1), grid, 1)
    for y in range(block, h, block):
        cv2.line(out, (0, y), (w - 1, y), grid, 1)
    return out


def _pixelate_roi(roi: np.ndarray, pixel_size: int, cv2: Any) -> np.ndarray:
    """Mosaic an ROI by area-averaging downscale then nearest-neighbour upscale.

    ``pixel_size`` is the size of one mosaic block in pixels. It is clamped
    to the ROI dimensions so a 1×1 ROI doesn't crash ``cv2.resize``. When
    ``pixel_size`` ≥ ROI side, the result collapses to a single block of
    the average ROI colour (effectively a same-shape ``mode="bbox"`` fill
    with the mean pixel value).
    """
    h, w = roi.shape[:2]
    block = max(2, min(pixel_size, h, w))
    # INTER_AREA is OpenCV's correct primitive for mosaicking — it averages
    # the source pixels in each block. INTER_LINEAR works but bleeds
    # slight gradients across block boundaries, especially on hair/clothing.
    small = cv2.resize(
        roi,
        (max(1, w // block), max(1, h // block)),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def blank_persons(
    frame: np.ndarray,
    detections: Iterable[Detection],
    *,
    mode: str = "pixelate",
    color: tuple[int, int, int] = (0, 0, 0),
    blur_kernel: int = 99,
    pixel_size: int | None = None,
    label: str = "person",
    labels: Iterable[str] | None = None,
    inplace: bool = False,
) -> np.ndarray:
    """Obscure every matching detection in ``frame``.

    Args:
        frame: HxWxC uint8 image (BGR).
        detections: Iterable of :class:`~cyberwave.models.types.Detection`.
            Detections whose label is not in the active label set are
            skipped silently — check your model's class names if nothing
            seems to be obscured.
        mode: ``"pixelate"`` (mosaic, default), ``"redact"`` (solid blocks
            with visible grid), ``"blur"`` (Gaussian blur), or ``"bbox"``
            (solid colour fill).
        color: Fill colour for ``mode="bbox"`` and ``mode="redact"``. Also
            used as the fallback fill for ROIs too small (<3 px on a side)
            to meaningfully pixelate or blur, regardless of ``mode`` —
            pass a colour you'd be happy seeing in the corners of the
            frame. BGR.
        blur_kernel: Odd integer kernel size for ``mode="blur"``. Clamped
            to the bbox size. Default ``99`` is intentionally aggressive
            and roughly doubles the per-pixel cost vs. a smaller kernel
            (OpenCV uses a separable Gaussian, so cost scales linearly
            with the kernel size). Lower it on CPU-bound edge devices.
        pixel_size: Mosaic block size for ``mode="pixelate"`` and
            ``mode="redact"``. When ``None`` (default), scales with the
            bbox so a person at the back of the room and one filling the
            frame both get ~24 blocks across the short side. Pass an int
            to force a constant block size.
        label: Single detection label to obscure. Backwards-compat shim
            for the original single-class API. Ignored when ``labels`` is
            given. Default ``"person"`` is kept for source-compat with
            callers that don't pass either parameter.
        labels: Iterable of detection labels to obscure. When provided
            (even as an empty iterable), takes precedence over ``label``.
            Pass an empty iterable to obscure nothing — useful when the
            caller computes the target set dynamically and may end up
            with zero classes (returns an unchanged copy without raising).
        inplace: When ``True``, mutate ``frame`` directly and return it.
            Saves an HxWxC copy per call — useful for hot-path workers.
            Default ``False`` returns a fresh array (input untouched).

    Returns:
        Either a new array (``inplace=False``) or the mutated input.

    Raises:
        ValueError: if ``mode`` is not one of the supported values.
    """
    if mode not in {"bbox", "blur", "pixelate", "redact"}:
        raise ValueError(
            f"mode must be one of 'pixelate', 'redact', 'blur', 'bbox', got {mode!r}"
        )
    if frame.ndim < 2:
        raise ValueError(f"frame must be at least 2-D, got shape {frame.shape}")

    # ``labels=`` takes precedence; fall back to the single ``label=`` when
    # the caller is using the legacy form. Materialise to a set so the
    # per-detection membership check is O(1) regardless of iterable type.
    active_labels: frozenset[str] = (
        frozenset(labels) if labels is not None else frozenset({label})
    )

    # Always import — cv2 caches after the first call so the cost is zero,
    # and unconditional import keeps the type narrow (no Optional). The bbox
    # path doesn't actually use cv2 here; that's fine, the import is cheap.
    cv2 = _require_cv2()
    out = frame if inplace else frame.copy()

    for det in detections:
        if det.label not in active_labels:
            continue
        x1, y1, x2, y2 = _bbox_int(det, frame.shape)
        if x2 <= x1 or y2 <= y1:
            continue
        if mode == "bbox":
            out[y1:y2, x1:x2] = color
            continue

        roi = out[y1:y2, x1:x2]
        min_side = min(roi.shape[0], roi.shape[1])
        # Tiny ROIs (1-2 px) cannot be meaningfully pixelated or blurred —
        # cv2.GaussianBlur needs both dims >= the kernel size, and resize
        # to a 0-pixel image is undefined. Fall back to a solid fill.
        if min_side < 3:
            out[y1:y2, x1:x2] = color
            continue

        if mode == "blur":
            # Gaussian kernel size must be odd and <= each ROI side.
            k = max(3, blur_kernel | 1)
            k = min(k, min_side if min_side % 2 == 1 else min_side - 1)
            out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
        else:
            # pixelate / redact share the same adaptive block-size cascade:
            # ~24 blocks across the short side so the mosaic stays
            # fine-grained but still quantised. Smaller blocks → more
            # recognisable silhouette, larger blocks → more privacy.
            ps = pixel_size if pixel_size is not None else max(2, min_side // 24)
            if mode == "pixelate":
                out[y1:y2, x1:x2] = _pixelate_roi(roi, ps, cv2)
            else:  # mode == "redact"
                out[y1:y2, x1:x2] = _redact_roi(roi, ps, color, cv2)
    return out


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    *,
    edges: Sequence[tuple[int, int]] = COCO17_SKELETON,
    conf_threshold: float = 0.3,
    color: tuple[int, int, int] | None = None,
    joint_color: tuple[int, int, int] | None = None,
    thickness: int | None = None,
    radius: int | None = None,
    inplace: bool = False,
) -> np.ndarray:
    """Overlay a stick-figure skeleton onto ``frame``.

    Args:
        frame: HxWxC uint8 image (BGR).
        keypoints: ``(K, 3)`` array of ``(x, y, visibility)`` per keypoint —
            or ``(K, 2)`` for models without a visibility column. Pixel
            coordinates in the original image space.
        edges: Sequence of ``(i, j)`` keypoint index pairs to connect.
            Defaults to COCO-17. Edges referencing missing keypoints are
            skipped.
        conf_threshold: Visibility/confidence below this hides the keypoint
            and any edge that depends on it. Ignored for ``(K, 2)`` arrays.
        color: Line colour. BGR. When ``None`` (default), each edge is
            coloured by its body part (head/torso/arms/legs) using
            :data:`COCO17_SEGMENT_COLORS`. Pass an explicit tuple to draw
            every limb in a single colour (e.g. for legacy UIs).
        joint_color: Colour for the joint dots. When ``None``, defaults to
            white when ``color`` is ``None`` (per-segment mode) so joints
            pop against the coloured limbs, otherwise mirrors ``color``.
        thickness: Line thickness in pixels. When ``None`` (default), scales
            with the frame size: ``max(2, min(h, w) // 240)``.
        radius: Joint dot radius in pixels. When ``None``, defaults to
            ``thickness + 1`` — or ``0`` (no joints) when ``thickness=0``,
            so passing ``thickness=0`` cleanly disables the entire overlay
            without also having to pass ``radius=0``.
        inplace: ``False`` (default) returns a new array, matching
            :func:`blank_persons` semantics so the two helpers compose
            safely. ``True`` mutates ``frame`` directly — used internally
            by :func:`anonymize_frame` to avoid an extra copy.

    Returns:
        Either a new array (default) or the mutated input (``inplace=True``).
        Non-finite or out-of-frame keypoint coordinates are silently skipped.
    """
    if keypoints is None or keypoints.size == 0:
        return frame if inplace else frame.copy()

    cv2 = _require_cv2()
    kp = np.asarray(keypoints)
    if kp.ndim != 2 or kp.shape[1] < 2:
        return frame if inplace else frame.copy()

    out = frame if inplace else frame.copy()
    has_conf = kp.shape[1] >= 3
    n_kp = kp.shape[0]
    h, w = out.shape[:2]

    if thickness is None:
        thickness = max(2, min(h, w) // 240)
    if radius is None:
        # thickness=0 → caller wants no overlay at all; honour that for joints
        # too instead of silently drawing 2-pixel dots.
        radius = 0 if thickness == 0 else max(2, thickness + 1)
    if joint_color is None:
        joint_color = _JOINT_DEFAULT_COLOR if color is None else color

    def _edge_color(edge: tuple[int, int]) -> tuple[int, int, int]:
        if color is not None:
            return color
        # Edge groups are stored canonical (lower index first); look up either way.
        group = COCO17_EDGE_GROUPS.get(edge) or COCO17_EDGE_GROUPS.get(
            (edge[1], edge[0])
        )
        return COCO17_SEGMENT_COLORS.get(group, _JOINT_DEFAULT_COLOR)  # type: ignore[arg-type]

    def _to_pixel(idx: int) -> tuple[int, int] | None:
        x, y = kp[idx, 0], kp[idx, 1]
        # NaN/inf guard — int() of either raises, breaking the whole frame.
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        ix, iy = int(x), int(y)
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return None
        return ix, iy

    # cv2.line asserts thickness > 0, so skip the loop entirely when the
    # caller asked for no lines (mirrors the radius=0 short-circuit below).
    if thickness > 0:
        for edge in edges:
            i, j = edge
            if i >= n_kp or j >= n_kp:
                continue
            if has_conf and (kp[i, 2] < conf_threshold or kp[j, 2] < conf_threshold):
                continue
            p1 = _to_pixel(i)
            p2 = _to_pixel(j)
            if p1 is None or p2 is None:
                continue
            # cv2.line with LINE_AA gives antialiased edges — much nicer
            # than the default jagged 8-connected lines.
            cv2.line(out, p1, p2, _edge_color(edge), thickness, lineType=cv2.LINE_AA)

    if radius > 0:
        for k in range(n_kp):
            if has_conf and kp[k, 2] < conf_threshold:
                continue
            p = _to_pixel(k)
            if p is None:
                continue
            cv2.circle(out, p, radius, joint_color, thickness=-1, lineType=cv2.LINE_AA)
    return out


def anonymize_frame(
    frame: np.ndarray,
    detections: Iterable[Detection],
    *,
    mode: str = "pixelate",
    color: tuple[int, int, int] = (0, 0, 0),
    blur_kernel: int = 99,
    pixel_size: int | None = None,
    draw_skeleton: bool = True,
    skeleton_color: tuple[int, int, int] | None = None,
    skeleton_threshold: float = 0.3,
    label: str = "person",
    labels: Iterable[str] | None = None,
    inplace: bool = False,
) -> np.ndarray:
    """Obscure matching detections in ``frame`` and optionally overlay pose skeletons.

    High-level entry point: pass in the model's detections (optionally with
    keypoints from a pose model) and get back a frame safe to publish to
    downstream consumers. See the module docstring for the privacy caveat.

    Args:
        frame: HxWxC uint8 image (BGR).
        detections: Detections. Keypoints are optional; when using a plain
            detector (no pose head) the skeleton overlay is implicitly a
            no-op and ``draw_skeleton`` has no effect.
        mode: Obscuring mode for the bbox region — ``"pixelate"`` (default),
            ``"redact"``, ``"blur"``, or ``"bbox"``.
        color: Fill colour for ``mode="bbox"`` and ``mode="redact"`` (and
            the small-ROI fallback; see :func:`blank_persons`).
        blur_kernel: Kernel size for ``mode="blur"``.
        pixel_size: Mosaic block size for ``mode="pixelate"``. ``None`` =
            adaptive (see :func:`blank_persons`).
        draw_skeleton: When ``True`` (default), overlay the pose skeleton
            on top of the obscured region for any detection that carries
            keypoints. Set to ``False`` to suppress the overlay even when
            keypoints are present — useful when the obscuring mode itself
            (e.g. ``"pixelate"``) already conveys the silhouette and you
            want a clean mosaic with no drawn primitives.
        skeleton_color: Colour for the overlaid skeleton. ``None`` (default)
            uses the per-body-part palette so head/torso/arms/legs are
            distinguishable. Pass a tuple to force a single colour. Ignored
            when ``draw_skeleton=False``.
        skeleton_threshold: Visibility threshold for keypoints. Ignored
            when ``draw_skeleton=False``.
        label: Single detection label to obscure. Backwards-compat shim
            for the original single-class API. Ignored when ``labels`` is
            given. The skeleton overlay is also limited to this label
            in the legacy single-class path.
        labels: Iterable of detection labels to obscure. When provided,
            takes precedence over ``label``. The skeleton overlay covers
            every detection whose label is in this set.
        inplace: When ``True``, mutate ``frame`` directly. Default ``False``
            returns a fresh array (input untouched).

    Returns:
        Either a new array or the mutated input.
    """
    # The module-level ``draw_skeleton`` function is shadowed here by the
    # kwarg of the same name (kept for API ergonomics). Grab an alias
    # before the parameter assignment takes effect.
    _draw_skeleton_fn = globals()["draw_skeleton"]

    active_labels: frozenset[str] = (
        frozenset(labels) if labels is not None else frozenset({label})
    )

    dets = list(detections)
    out = blank_persons(
        frame,
        dets,
        mode=mode,
        color=color,
        blur_kernel=blur_kernel,
        pixel_size=pixel_size,
        labels=active_labels,
        inplace=inplace,
    )
    if not draw_skeleton:
        return out
    # ``out`` is either the caller's frame (inplace=True) or a fresh copy
    # from blank_persons. Either way we can draw in place — when
    # inplace=False the caller already opted into a copy, when inplace=True
    # they explicitly want mutation.
    for det in dets:
        if det.label not in active_labels or det.keypoints is None:
            continue
        _draw_skeleton_fn(
            out,
            det.keypoints,
            color=skeleton_color,
            conf_threshold=skeleton_threshold,
            inplace=True,
        )
    return out
