"""
Camera Device Specifications

Contains specifications for IP cameras, NVR systems, and other camera devices.
"""

from .uniview_nvr import UniviewNVRSpec
from .ip_camera import IPCameraSpec
from .generic_camera import GenericCameraSpec

__all__ = [
    "UniviewNVRSpec",
    "IPCameraSpec",
    "GenericCameraSpec"
]
