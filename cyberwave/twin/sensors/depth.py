"""Depth sensor handle (PR1: same capture/latest_frame surface as RGB)."""

from __future__ import annotations

from .base import BaseSensorHandle


class DepthSensorHandle(BaseSensorHandle):
    """Depth sensor — shares the RGB camera API in PR1."""
