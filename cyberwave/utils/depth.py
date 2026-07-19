"""Depth map encoding helpers used by every ``twin/*/depth`` publisher.

Depth is always **metres**; the wire ``dtype`` is only how a producer transports
it — a float dtype carries metres directly, ``uint16`` carries millimetres
(``/1000`` → metres). Consumers decode per the declared dtype.

:func:`depth_to_uint16` quantises a float depth map into ``uint16``; the two
supported modes are ``normalized_uint16`` (min/max rescale, unitless) and
``metric_mm`` (metres * ``scale_factor``, clipped). :func:`build_depth_mqtt_payload`
bakes a ``uint16`` millimetre **or** a ``float32`` metre array into the canonical
self-describing ``{depth_binary, width, height, dtype}`` wire dict — the ``dtype``
tag always matches the encoded bytes, so consumers decode it unambiguously.
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
) -> Any:
    """Convert a floating-point depth map to a ``uint16 H x W`` payload.

    Args:
        depth:        ``np.ndarray`` (typically ``float32``) with the raw depth
                      map. NaN / infinity values are replaced with zero.
        output_mode:  ``"normalized_uint16"`` remaps min/max onto ``[0, 65535]``;
                      ``"metric_mm"`` multiplies by ``scale_factor`` (default
                      ``1000`` = mm) and clips to ``[0, 65535]``.
        scale_factor: Multiplier applied in ``metric_mm`` mode.

    Returns:
        ``np.ndarray`` with the same shape as ``depth`` and dtype ``np.uint16``.

    Raises:
        ValueError: If ``output_mode`` is not one of the two supported values.
    """
    import numpy as np

    clean = np.nan_to_num(np.asarray(depth), nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )

    if output_mode == DEPTH_OUTPUT_MODE_METRIC_MM:
        metric = np.clip(clean * scale_factor, 0, 65535)
        return metric.astype(np.uint16)

    if output_mode != DEPTH_OUTPUT_MODE_NORMALIZED_UINT16:
        raise ValueError(
            f"Unsupported output_mode '{output_mode}'. "
            f"Use {DEPTH_OUTPUT_MODE_NORMALIZED_UINT16!r} or "
            f"{DEPTH_OUTPUT_MODE_METRIC_MM!r}."
        )

    min_depth = float(np.min(clean))
    max_depth = float(np.max(clean))
    if max_depth - min_depth < 1e-8:
        return np.zeros_like(clean, dtype=np.uint16)

    normalized = (clean - min_depth) / (max_depth - min_depth)
    normalized = np.clip(normalized * 65535.0, 0.0, 65535.0)
    return normalized.astype(np.uint16)


def build_depth_mqtt_payload(
    depth: Any, *, wire_dtype: str = "uint16"
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
    return {
        "depth_binary": base64.b64encode(arr.tobytes()).decode("utf-8"),
        "width": int(width),
        "height": int(height),
        "dtype": arr.dtype.name,
    }
