"""Legacy re-exports for twin handles moved to domain modules."""

from .sensors.camera import TwinCameraHandle
from .namespaces import CamerasNamespace

__all__ = ["TwinCameraHandle", "CamerasNamespace"]
