"""
RTSP Camera Alias Specification

Provides an alias specification for a generic RTSP IP camera.
Reuses the generic IP camera capabilities with a more specific ID.
"""

from dataclasses import dataclass
from .ip_camera import IPCameraSpec


@dataclass
class RTSPCameraSpec(IPCameraSpec):
    """Alias for generic IP camera under RTSP naming"""

    def __post_init__(self):
        # Initialize base IP camera spec first
        super().__post_init__()

        # Override identification to RTSP-specific alias
        self.id = "generic/rtsp-camera"
        self.name = self.name or "RTSP IP Camera"
        # Keep category/manufacturer/model and other fields inherited


