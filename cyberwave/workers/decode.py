"""Decode wire-format samples for worker hook dispatch.

Wraps :func:`cyberwave.data.header.decode` and
:func:`cyberwave.data.api._decode_sample` into a single helper
that the worker runtime uses to convert raw ``Sample.payload`` bytes
into Python objects (numpy arrays, dicts, or plain bytes) before
passing them to user hook callbacks.

Handles two wire formats:
1. SDK wire format (header + payload) from Python drivers
2. Raw JPEG bytes from native C++ drivers

Optional downscaling is applied when ``CYBERWAVE_WORKER_INPUT_RESOLUTION``
is set (e.g. ``640x480``), resizing numpy image frames before dispatch.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from cyberwave.data.backend import Sample
from cyberwave.data.header import decode

logger = logging.getLogger(__name__)

_JPEG_SOI = b"\xff\xd8"

# Parse target resolution once at import time.
_TARGET_RESOLUTION: tuple[int, int] | None = None
_raw_res = os.environ.get("CYBERWAVE_WORKER_INPUT_RESOLUTION")
if _raw_res:
    try:
        parts = _raw_res.lower().split("x")
        _TARGET_RESOLUTION = (int(parts[0]), int(parts[1]))
        logger.info("Worker input resolution scaling enabled: %s", _TARGET_RESOLUTION)
    except (ValueError, IndexError):
        logger.warning(
            "Invalid CYBERWAVE_WORKER_INPUT_RESOLUTION '%s'; expected WIDTHxHEIGHT (e.g. 640x480)",
            _raw_res,
        )


def _maybe_resize(data: Any) -> Any:
    """Downscale a numpy image array to ``_TARGET_RESOLUTION`` if configured."""
    if _TARGET_RESOLUTION is None:
        return data
    import numpy as np

    if not isinstance(data, np.ndarray) or data.ndim < 2:
        return data
    h, w = data.shape[:2]
    tw, th = _TARGET_RESOLUTION
    if w == tw and h == th:
        return data
    import cv2

    return cv2.resize(data, (tw, th), interpolation=cv2.INTER_LINEAR)


def _jpeg_to_ndarray(data: bytes) -> Any:
    """Decode JPEG bytes into a BGR numpy array via OpenCV."""
    import numpy as np

    arr = np.frombuffer(data, dtype=np.uint8)

    import cv2

    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode returned None for JPEG payload")
    return img


def _decode_payload(header: Any, payload: bytes) -> Any:
    """Decode payload bytes based on header content_type."""
    from cyberwave.data.header import CONTENT_TYPE_JSON, CONTENT_TYPE_NUMPY

    if header.content_type == CONTENT_TYPE_NUMPY:
        if header.shape is None or header.dtype is None:
            return payload
        import numpy as np

        return np.frombuffer(payload, dtype=header.dtype).reshape(header.shape).copy()
    if header.content_type == CONTENT_TYPE_JSON:
        import json

        return json.loads(payload)
    return payload


def decode_sample_payload(sample: Sample, *, content_hint: str = "") -> tuple[Any, float]:
    """Decode a raw ``Sample`` into ``(decoded_data, timestamp)``.

    When *content_hint* is ``"numpy"``, the JPEG fallback is tried
    immediately if SDK header decode fails, and the JSON decode branch
    is skipped — saving an import and parse attempt on every frame.

    Tries the SDK wire format first.  If that fails and the payload
    looks like a JPEG image (starts with ``\\xff\\xd8``), it is decoded
    to a BGR numpy array via OpenCV — this covers native C++ drivers
    that publish raw JPEG frames on Zenoh.

    Returns ``(decoded_data, timestamp)`` where *timestamp* comes from
    the wire header when available, otherwise from the Sample itself.
    """
    raw = sample.payload
    ts_fallback = getattr(sample, "timestamp", 0.0)

    try:
        header, payload = decode(raw)
        if content_hint == "numpy":
            from cyberwave.data.header import CONTENT_TYPE_NUMPY

            if header.content_type == CONTENT_TYPE_NUMPY and header.shape and header.dtype:
                import numpy as np

                arr = np.frombuffer(payload, dtype=header.dtype).reshape(header.shape).copy()
                return _maybe_resize(arr), header.ts
        decoded = _decode_payload(header, payload)
        return _maybe_resize(decoded), header.ts
    except Exception:
        pass

    if len(raw) > 2 and raw[:2] == _JPEG_SOI:
        try:
            return _maybe_resize(_jpeg_to_ndarray(raw)), ts_fallback
        except Exception:
            logger.warning("JPEG decode failed", exc_info=True)

    return raw, ts_fallback
