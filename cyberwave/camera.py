"""Camera streaming functionality for Cyberwave SDK.

This module provides backwards compatibility imports from the new sensor module.
For new code, prefer importing directly from cyberwave.sensor.

Example (new style):
    >>> from cyberwave.sensor import CV2CameraStreamer, RealSenseStreamer

Example (legacy style - still supported):
    >>> from cyberwave.camera import CameraStreamer, RealSenseStreamer
"""

# Re-export everything from the sensor module for backwards compatibility
from .sensor import (
    # Base classes
    BaseVideoTrack,
    BaseVideoStreamer,
    # CV2 implementations
    CV2VideoTrack,
    CV2CameraStreamer,
    # RealSense implementations
    RealSenseVideoTrack,
    RealSenseStreamer,
    # Constants
    DEFAULT_TURN_SERVERS,
)

# Legacy aliases for backwards compatibility
CameraStreamer = CV2CameraStreamer

__all__ = [
    # Base classes
    "BaseVideoTrack",
    "BaseVideoStreamer",
    # CV2 implementations
    "CV2VideoTrack",
    "CV2CameraStreamer",
    "CameraStreamer",  # Legacy alias
    # RealSense implementations
    "RealSenseVideoTrack",
    "RealSenseStreamer",
    # Constants
    "DEFAULT_TURN_SERVERS",
]
