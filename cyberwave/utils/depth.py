"""Depth map encoding helpers used by every ``twin/*/depth`` publisher.

Depth is always **metres** — every distance in this module and on the wire
(``min_depth``, ``max_depth``, and the decoded result) is metres, and
``depth_scale`` is metres per stored unit. The wire ``dtype`` is only how a
producer transports it:

- a float dtype carries metres directly;
- ``uint16`` carries integer units whose size is given by ``depth_scale``.
  Absent a declared ``depth_scale`` this falls back to **millimetres**
  (``0.001``), which is what a stock RealSense emits — but it is a fallback,
  not a guarantee. ``depth_units`` is configurable and differs across devices,
  so a producer that knows its real value must publish it, and one that does
  not must publish nothing and let the twin's schema calibration stand.

:func:`depth_to_uint16` quantises a float depth map into ``uint16``; the two
supported modes are ``normalized_uint16`` (min/max rescale, unitless) and
``metric_mm`` (metres * ``scale_factor``, clipped). In both, ``0`` means **no
reading** and nothing else — ``normalized_uint16`` places valid samples on
``[1, 65535]`` precisely so the sentinel stays unambiguous, since consumers drop
it on the raw sample rather than on decoded metres. :func:`build_depth_mqtt_payload`
bakes a ``uint16`` millimetre **or** a ``float32`` metre array into the canonical
self-describing ``{depth_binary, width, height, dtype}`` wire dict — the ``dtype``
tag always matches the encoded bytes, so consumers decode it unambiguously. It
also optionally carries ``output_mode``/``min_depth``/``max_depth``/``depth_scale``
so a ``uint16``-quantised payload can be decoded back to real-world distances.

:func:`depth_to_colored_pointcloud` reconstructs a sparse coloured 3-D point
cloud from a ``uint16`` depth map using camera pinhole intrinsics and a
jet-colourmap depth colouring. The output is a flat ``float32`` array with
``[x, y, z, r, g, b]`` interleaved per point and is ready for
``publish_pointcloud`` in the MQTT client.
"""

from __future__ import annotations

import base64
from typing import Any

DEPTH_OUTPUT_MODE_NORMALIZED_UINT16 = "normalized_uint16"
DEPTH_OUTPUT_MODE_METRIC_MM = "metric_mm"


def depth_to_uint16(
    depth: Any,
    *,
    output_mode: str = DEPTH_OUTPUT_MODE_NORMALIZED_UINT16,
    scale_factor: float = 1000.0,
    min_depth: float | None = None,
    max_depth: float | None = None,
    input_is_disparity: bool = False,
) -> Any:
    """Convert a floating-point depth map to a ``uint16 H x W`` payload.

    Args:
        depth:        ``np.ndarray`` (typically ``float32``) with the raw depth
                      map. NaN / infinity values are replaced with zero.
        output_mode:  ``"normalized_uint16"`` remaps valid samples onto
                      ``[1, 65535]``, reserving ``0`` for "no reading";
                      ``"metric_mm"`` multiplies by ``scale_factor`` (default
                      ``1000`` = mm) and clips to ``[0, 65535]``.
        scale_factor: Multiplier applied in ``metric_mm`` mode.
        min_depth:    Near clip in metres. In ``normalized_uint16`` mode this and
                      ``max_depth`` become the **encode** window, so the
                      published ``min_depth``/``max_depth``/``depth_scale`` are
                      the actual decode affine and the scale is stable frame to
                      frame. Omit both to keep the legacy per-frame auto-range.
                      Metric input only: with ``input_is_disparity`` the window
                      is still auto-ranged per frame, so the affine describes a
                      display range rather than a fixed mapping.
        max_depth:    Far clip in metres. See ``min_depth``.
        input_is_disparity:
                      ``True`` when ``depth`` is *inverse* depth (relative
                      Depth-Anything / Video-Depth-Anything checkpoints, i.e.
                      ``DepthResult.metric is False``), where a larger value
                      means *nearer*. Such a map has no metric scale at all, so
                      it is auto-ranged **per frame** and inverted onto the
                      ``[min_depth, max_depth]`` display window — without the
                      inversion the reconstructed scene comes out inside-out.
                      The per-frame ranging is inherent, so a stationary surface
                      can still shift value as the rest of the scene changes.

    Returns:
        ``np.ndarray`` with the same shape as ``depth`` and dtype ``np.uint16``.

    Raises:
        ValueError: If ``output_mode`` is not one of the two supported values,
            or if ``metric_mm`` is requested for a disparity input (there is no
            millimetre interpretation of inverse depth).
    """
    import numpy as np

    clean = np.nan_to_num(np.asarray(depth), nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )

    if output_mode == DEPTH_OUTPUT_MODE_METRIC_MM:
        if input_is_disparity:
            raise ValueError(
                f"{DEPTH_OUTPUT_MODE_METRIC_MM!r} cannot encode a disparity map: "
                "relative depth checkpoints carry no metric scale. Use "
                f"{DEPTH_OUTPUT_MODE_NORMALIZED_UINT16!r}, or switch to a metric "
                "checkpoint."
            )
        metric = np.clip(clean * scale_factor, 0, 65535)
        return metric.astype(np.uint16)

    if output_mode != DEPTH_OUTPUT_MODE_NORMALIZED_UINT16:
        raise ValueError(
            f"Unsupported output_mode '{output_mode}'. "
            f"Use {DEPTH_OUTPUT_MODE_NORMALIZED_UINT16!r} or "
            f"{DEPTH_OUTPUT_MODE_METRIC_MM!r}."
        )

    # Use only valid (non-zero) pixels for the range so that padding zeros from
    # nan_to_num do not collapse the minimum to 0 and corrupt the near-field.
    valid_mask = clean > 0
    if not np.any(valid_mask):
        return np.zeros_like(clean, dtype=np.uint16)

    has_window = (
        min_depth is not None
        and max_depth is not None
        and float(max_depth) > float(min_depth)
    )

    if input_is_disparity:
        # No metric scale exists, so auto-range the disparity and flip it: the
        # largest disparity (nearest surface) has to land on u16=0, which the
        # decoder maps back to min_depth.
        d_min = float(np.min(clean[valid_mask]))
        d_max = float(np.max(clean[valid_mask]))
        if d_max - d_min < 1e-8:
            return np.zeros_like(clean, dtype=np.uint16)
        unit = 1.0 - (clean - d_min) / (d_max - d_min)
    elif has_window:
        # Encode against the *declared* window so the published affine is true
        # and the scale does not breathe as the scene's own range shifts.
        lo, hi = float(min_depth), float(max_depth)  # type: ignore[arg-type]
        unit = (clean - lo) / (hi - lo)
    else:
        # Legacy per-frame auto-range (RealSense and other callers that publish
        # no window). Decoders fall back to their own depth_scale for these.
        d_min = float(np.min(clean[valid_mask]))
        d_max = float(np.max(clean[valid_mask]))
        if d_max - d_min < 1e-8:
            return np.zeros_like(clean, dtype=np.uint16)
        unit = (clean - d_min) / (d_max - d_min)

    # Valid samples occupy [1, 65535] so u16=0 is the *exclusive* "no reading"
    # sentinel, which is the only thing that makes the consumers' guards
    # correct — the raw-sample guard in depth_to_colored_pointcloud and the
    # matching `raw === 0` gate in the frontend's depthToPoints().
    #
    # Without the offset the affine's own zero collides with the sentinel, and
    # both guards then delete real geometry: the inverted branch below puts the
    # *nearest* surface at unit 0 deliberately, and the windowed branch puts
    # everything at or below min_depth there. A flat wall filling 30% of the
    # frame loses all 30%.
    #
    # Cost of the offset is 1/65535 of the window on decode — 0.15 mm over 10 m
    # — so every existing decoder stays correct without a matching change, and
    # the top of the range still saturates exactly at max_depth.
    normalized = 1.0 + np.clip(unit, 0.0, 1.0) * 65534.0
    # Invalid pixels land on 0 regardless of what the affine did with them (the
    # inverted branch in particular maps 0.0 to unit 1.0).
    normalized = np.where(valid_mask, normalized, 0.0)
    return normalized.astype(np.uint16)


def build_depth_mqtt_payload(
    depth: Any,
    *,
    wire_dtype: str = "uint16",
    output_mode: str | None = None,
    min_depth: float | None = None,
    max_depth: float | None = None,
    depth_scale: float | None = None,
) -> dict[str, Any]:
    """Return the canonical ``{depth_binary, width, height, dtype}`` dict.

    Single point of truth for the depth wire format — every publisher routes
    through here. The payload is self-describing: the ``dtype`` tag always
    matches the encoded bytes, and consumers decode by it (a float dtype is
    absolute **metres**, ``uint16`` is **millimetres**).

    ``wire_dtype`` selects the transport:

    - ``"uint16"`` (default): millimetres. Coerces to ``uint16`` when the caller
      forgets the cast — the legacy RealSense / point-cloud path.
    - ``"float32"``: absolute **metres**, carried verbatim (no quantisation).
      Use when the producer already holds a metric float depth map and wants to
      preserve sub-millimetre precision or ranges beyond ``uint16`` mm (65.535 m).

    ``output_mode`` and the optional ``min_depth``/``max_depth``/``depth_scale``
    are depth-range metadata forwarded from the Send Depth node parameters so
    the frontend can reconstruct real-world distances without re-computing them
    on every frame.

    ``output_mode`` is only stamped when the caller passes it. Leaving it unset
    means "legacy uint16 millimetres" and consumers keep their own default
    ``depth_scale`` — publishers that have not opted in (e.g. the RealSense
    driver before it declared ``metric_mm``) must not be labelled as normalised,
    or the viewer rescales their clouds by (max-min)/65535 instead of 0.001.
    """
    import numpy as np

    wire = str(wire_dtype).lower()
    arr = np.asarray(depth)
    if wire == "float32":
        arr = arr.astype(np.float32, copy=False)
    elif wire == "uint16":
        if arr.dtype != np.uint16:
            arr = arr.astype(np.uint16)
    else:
        raise ValueError(
            f"Unsupported wire_dtype {wire_dtype!r}; use 'uint16' or 'float32'."
        )
    arr = np.ascontiguousarray(arr)
    height, width = (arr.shape + (0, 0))[:2]
    payload: dict[str, Any] = {
        "depth_binary": base64.b64encode(arr.tobytes()).decode("utf-8"),
        "width": int(width),
        "height": int(height),
        "dtype": arr.dtype.name,
    }
    if output_mode is not None:
        payload["output_mode"] = output_mode
    if min_depth is not None:
        payload["min_depth"] = float(min_depth)
    if max_depth is not None:
        payload["max_depth"] = float(max_depth)
    if depth_scale is not None:
        payload["depth_scale"] = float(depth_scale)
    return payload


def _grid_points(h: int, w: int, step: int) -> int:
    """Points a ``step``-strided sample of an ``h x w`` frame yields."""
    return (-(-h // step)) * (-(-w // step))


def depth_to_colored_pointcloud(
    depth_uint16: Any,
    *,
    output_mode: str = DEPTH_OUTPUT_MODE_NORMALIZED_UINT16,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    depth_scale: float = 0.001,
    fx: float | None = None,
    fy: float | None = None,
    fx_normalized: float | None = None,
    fy_normalized: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    step: int = 4,
    max_points: int | None = None,
) -> Any:
    """Reconstruct a sparse coloured 3-D point cloud from a ``uint16`` depth map.

    Converts a ``uint16`` depth image (produced by :func:`depth_to_uint16`) into
    a flat ``float32`` array with ``[x, y, z, r, g, b]`` interleaved per point,
    suitable for publishing on the ``twin/{uuid}/pointcloud`` MQTT topic via
    ``client.mqtt.publish_pointcloud``.

    Coordinates are in the **depth-camera optical frame** (X-right, Y-down,
    Z-forward), which matches what the ``ColoredPointCloud`` frontend component
    expects when ``coordinateMode="depth_camera"``.

    Args:
        depth_uint16: ``np.ndarray`` of shape ``(H, W)`` and dtype ``uint16``.
        output_mode:  Encoding used by :func:`depth_to_uint16`:
                      ``"normalized_uint16"`` reconstructs via
                      ``d = min_depth + u16/65535*(max_depth-min_depth)``;
                      ``"metric_mm"`` via ``d = u16 * depth_scale``.
        min_depth:    Minimum depth in metres (``normalized_uint16`` only).
        max_depth:    Maximum depth in metres (``normalized_uint16`` only).
        depth_scale:  Metres-per-uint16-unit (``metric_mm`` only).
        fx:           Focal length in **pixels** (x axis). Only correct when the
                      caller knows the frame resolution. Defaults to a 69° HFOV
                      estimate derived from the image width.
        fy:           Focal length in pixels (y axis).  Defaults to ``fx``.
        fx_normalized:
                      Focal length as a *fraction of image width*, multiplied by
                      the actual width at call time. Prefer this over ``fx``
                      whenever the frame resolution is not known up front — e.g.
                      generated workflow code, where a depth model may resample
                      the frame away from the twin's declared resolution.
                      Ignored when ``fx`` is given.
        fy_normalized:
                      Focal length as a fraction of image height. Ignored when
                      ``fy`` is given.
        cx:           Principal point x.  Defaults to ``width / 2``.
        cy:           Principal point y.  Defaults to ``height / 2``.
        step:         Pixel stride for subsampling.  ``step=4`` gives one point
                      per 4×4 block. At 640x480 that is 19 200 points ≈ 450 KiB
                      of float32 (≈600 KiB base64) per frame. Resolution
                      *dependent* — prefer ``max_points`` unless you know the
                      frame size.
        max_points:   Ceiling on the sampling grid, which coarsens ``step`` as
                      needed to stay under it. Resolution-independent, so the
                      same value costs the same at 480p and 1080p — the stride
                      alone does not (19 200 points vs 129 600 for ``step=4``).
                      Applied per frame because only the caller's frame knows
                      its true size: a depth model may resample away from the
                      twin's declared resolution, which is the same reason
                      focals arrive here normalized. ``None`` disables the cap.

    Returns:
        ``np.ndarray`` of shape ``(N*6,)`` and dtype ``float32``, or ``None``
        when no valid depth pixels remain after filtering.
    """
    import math

    import numpy as np

    # Kept in its native dtype: the decode below runs on the *sampled* points
    # only, so converting the whole frame to float32 here would be work (and a
    # full-frame allocation) thrown away for every pixel the stride skips.
    arr = np.asarray(depth_uint16)
    if arr.ndim < 2:
        return None
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        return None

    # --- camera intrinsics ---
    # Precedence: explicit pixel focals > resolution-independent normalized
    # focals (scaled by this frame) > 69° HFOV guess.
    _cx = float(cx) if cx is not None else w * 0.5
    _cy = float(cy) if cy is not None else h * 0.5
    _fx: float
    if fx is not None:
        _fx = float(fx)
    elif fx_normalized is not None and float(fx_normalized) > 0:
        _fx = float(fx_normalized) * w
    else:
        _fx = w / (2.0 * math.tan(math.radians(34.5)))
    if fy is not None:
        _fy = float(fy)
    elif fy_normalized is not None and float(fy_normalized) > 0:
        _fy = float(fy_normalized) * h
    else:
        _fy = _fx

    # --- density cap -> effective stride ---
    # ``max`` rather than ``=``: the budget is a ceiling, so an explicitly
    # coarser ``step`` still wins. Bounded on the sampling grid (before the
    # validity filter), since that is what sizes the allocation and the payload.
    if max_points is not None and max_points > 0:
        step = max(step, math.ceil(math.sqrt((h * w) / max_points)))
        # ``sqrt`` sizes the grid only in the continuous case; rounding each
        # axis up independently can still land over the budget (160x120 at 100
        # points gives 9x12 = 108). Tighten until it genuinely fits, so the cap
        # is a guarantee rather than an estimate. Terminates: a step past the
        # larger axis always yields a 1x1 grid.
        while step < max(h, w) and _grid_points(h, w, step) > max_points:
            step += 1

    # --- subsampled pixel grid ---
    us = np.arange(0, w, step, dtype=np.float32)
    vs = np.arange(0, h, step, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    rows = vv.astype(np.int32)
    cols = uu.astype(np.int32)
    raw = arr[rows, cols].astype(np.float32)

    # --- decode uint16 → metric depth (metres) ---
    # Deliberately after the subsample. Decoding the full frame first and then
    # indexing it made the cost scale with frame area while the output is capped
    # by ``max_points`` — 1080p spent ~1.1 ms and two 8 MB transient allocations
    # to produce ~17 000 points. Decoding only the sampled values makes this
    # flat in frame size (~0.12 ms at any resolution), which is the invariant
    # ``max_points`` is meant to buy.
    if output_mode == DEPTH_OUTPUT_MODE_METRIC_MM:
        d = raw * float(depth_scale)
    else:
        d = float(min_depth) + raw / 65535.0 * (float(max_depth) - float(min_depth))

    # u16 == 0 is the "no reading" sentinel in both encodings, so test the raw
    # sample rather than the decoded metres. Under normalized_uint16 the affine
    # puts a 0 sample at min_depth (0.1 m by default), so a decoded-depth guard
    # passes every invalid pixel and paints the sky as a wall in front of the
    # camera. Mirrors the depthToPoints() gate in PointCloud.tsx.
    valid = (raw > 0) & (d > 0.01)
    d = d[valid]
    if d.size == 0:
        return None
    u = uu[valid]
    v = vv[valid]

    # --- 3-D back-projection (optical frame: X-right, Y-down, Z-forward) ---
    x = (u - _cx) * d / _fx
    y = (v - _cy) * d / _fy
    z = d

    # --- jet-colourmap colouring based on depth ---
    t = np.clip(
        (d - float(min_depth)) / max(float(max_depth) - float(min_depth), 1e-8),
        0.0,
        1.0,
    )
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)

    n = int(x.size)
    out = np.empty(n * 6, dtype=np.float32)
    out[0::6] = x
    out[1::6] = y
    out[2::6] = z
    out[3::6] = r
    out[4::6] = g
    out[5::6] = b
    return out
