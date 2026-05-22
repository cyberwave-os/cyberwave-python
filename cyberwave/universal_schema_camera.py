"""Discover camera sensor ids from a twin universal schema document.

Used to choose ``sensor_id`` for :meth:`~cyberwave.twin.Twin.get_latest_frame` and
:meth:`~cyberwave.twin.Twin.capture_frame` on multi-camera twins.
"""

from __future__ import annotations

import logging
from typing import Any, List

logger = logging.getLogger(__name__)


def _schema_sensor_lists(schema: dict) -> list[list[Any]]:
    """Collect sensor arrays from top-level ``sensors`` and ``capabilities.sensors``."""
    blocks: list[list[Any]] = []
    top = schema.get("sensors")
    if isinstance(top, list):
        blocks.append(top)
    caps = schema.get("capabilities")
    if isinstance(caps, dict):
        nested = caps.get("sensors")
        if isinstance(nested, list):
            blocks.append(nested)
    return blocks


def _is_camera_like_sensor_type(sensor_type: str) -> bool:
    t = sensor_type.lower()
    if t in ("rgb", "depth", "depth_camera", "camera"):
        return True
    return "camera" in t


def _sensor_id_from_entry(sensor: dict) -> str | None:
    for key in ("id", "sensor_id", "name"):
        v = sensor.get(key)
        if v is not None:
            sid = str(v).strip()
            if sid:
                return sid
    return None


def camera_sensor_ids_from_schema(schema: Any, *, max_ids: int = 16) -> List[str]:
    """
    Return stable id strings for camera-like sensors in a universal schema dict.

    Inspects top-level ``sensors`` and nested ``capabilities.sensors``. Entries are
    included when their ``type`` looks like a camera (``camera``, ``depth_camera``,
    ``rgb``, ``depth``, or any type containing ``"camera"``).

    Args:
        schema: Full universal schema (e.g. from :meth:`~cyberwave.twin.Twin.get_schema`).
        max_ids: Maximum number of ids to return (default ``16``).

    Returns:
        Ordered list of unique sensor ids, suitable for ``sensor_id=`` on latest-frame APIs.
    """
    if not isinstance(schema, dict):
        return []

    out: list[str] = []
    try:
        for sensors in _schema_sensor_lists(schema):
            for s in sensors:
                if not isinstance(s, dict):
                    continue
                t = str(s.get("type", "")).lower()
                if not _is_camera_like_sensor_type(t):
                    continue
                sid = _sensor_id_from_entry(s)
                if sid is not None and sid not in out:
                    out.append(sid)
                    if len(out) >= max_ids:
                        return out
    except Exception as exc:
        logger.debug("camera_sensor_ids_from_schema: %s", exc)
    return out
