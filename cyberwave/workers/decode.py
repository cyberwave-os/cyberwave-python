"""Decode wire-format samples for worker hook dispatch.

Wraps :func:`cyberwave.data.header.decode` and
:func:`cyberwave.data.api._decode_sample` into a single helper
that the worker runtime uses to convert raw ``Sample.payload`` bytes
into Python objects (numpy arrays, dicts, or plain bytes) before
passing them to user hook callbacks.

Handles two wire formats:
1. SDK wire format (header + payload) from Python drivers
2. Raw JPEG bytes from native C++ drivers
"""

from __future__ import annotations

import logging
from typing import Any

from cyberwave.data.backend import Sample
from cyberwave.data.header import decode

logger = logging.getLogger(__name__)

_JPEG_SOI = b"\xff\xd8"


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


def decode_sample_payload(sample: Sample) -> tuple[Any, float]:
    """Decode a raw ``Sample`` into ``(decoded_data, timestamp)``.

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
        decoded = _decode_payload(header, payload)
        return decoded, header.ts
    except Exception:
        pass

    if len(raw) > 2 and raw[:2] == _JPEG_SOI:
        try:
            return _jpeg_to_ndarray(raw), ts_fallback
        except Exception:
            logger.warning("JPEG decode failed", exc_info=True)

    return raw, ts_fallback
