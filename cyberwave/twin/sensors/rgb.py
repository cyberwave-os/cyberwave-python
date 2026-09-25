"""RGB / visible camera sensor handle."""

from __future__ import annotations

from .base import BaseSensorHandle


class RGBSensorHandle(BaseSensorHandle):
    """RGB imaging sensor — delegates to :class:`~cyberwave.twin.camera.TwinCameraHandle`."""
