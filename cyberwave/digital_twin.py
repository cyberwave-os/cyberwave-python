from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class AbstractAsset(ABC):
    """Base class for any asset in Cyberwave."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.level: Optional[str] = None
        self.catalog_id: Optional[str] = None
        self.transform: Dict[str, Any] = {}

    def set_position(self, x: float, y: float, z: float) -> None:
        self.transform["position"] = [x, y, z]

    def serialize(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level,
            "catalog_id": self.catalog_id,
            "transform": self.transform,
        }


class StaticAsset(AbstractAsset):
    """A static, non-behavioral asset such as scenery or props."""

    def __init__(self, name: str, model_path: str) -> None:
        super().__init__(name)
        self.model_path = model_path


class PhysicalDevice(ABC):
    """Abstract representation of a physical hardware device."""

    def __init__(self, device_id: str, device_type: str) -> None:
        self.device_id = device_id
        self.device_type = device_type
        self.connected = False

    @abstractmethod
    def connect(self) -> None:
        """Connect to the physical device."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the device."""

    def send_command(self, command: str, **kwargs: Any) -> None:
        raise NotImplementedError

    def get_telemetry(self) -> Dict[str, Any]:
        return {}


class RobotAsset(AbstractAsset):
    """Digital robot asset with optional physical device attachment."""

    def __init__(self, name: str, capabilities: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name)
        self.capabilities = capabilities or {}
        self.device: Optional[PhysicalDevice] = None

    def attach_device(self, device: PhysicalDevice) -> None:
        self.device = device

    def detach_device(self) -> None:
        self.device = None


class DigitalTwin(RobotAsset):
    """A RobotAsset linked to a physical device and reflecting live state."""

    def __init__(self, robot_asset: RobotAsset, device: PhysicalDevice) -> None:
        super().__init__(robot_asset.name, robot_asset.capabilities)
        self.device = device
        self.live_state: Dict[str, Any] = {}

    def update_from_device(self) -> None:
        if self.device:
            data = self.device.get_telemetry()
            if data:
                self.live_state.update(data)

    def send_command(self, command: str, **kwargs: Any) -> None:
        if self.device:
            self.device.send_command(command, **kwargs)
