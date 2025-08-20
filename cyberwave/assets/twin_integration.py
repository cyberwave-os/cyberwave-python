"""
Twin Integration - Connect assets to the Cyberwave platform

This module provides integration between the asset registry system and
the Cyberwave platform's Twin model, enabling seamless creation and
management of digital twins.
"""

from typing import Optional, Dict, Any, List, TYPE_CHECKING
from enum import Enum
import asyncio
from datetime import datetime

from .registry import BaseAsset, Robot, Sensor

if TYPE_CHECKING:
    from ..client import Client


class TwinMode(Enum):
    """Twin operational modes"""
    VIRTUAL = "virtual"      # Simulation only
    PHYSICAL = "physical"    # Hardware only
    HYBRID = "hybrid"        # True digital twin (both)


class TwinEnabledAsset(BaseAsset):
    """
    Base class for assets that can create twins on the Cyberwave platform.
    
    This provides integration between the asset registry and the platform's
    Twin model, handling creation, synchronization, and management.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._twin_id: Optional[int] = None
        self._twin_uuid: Optional[str] = None
        self._client: Optional['Client'] = None
        self._mode: TwinMode = TwinMode.VIRTUAL
        self._project_id: Optional[int] = None
        self._connected = False
    
    async def create_twin(
        self,
        client: 'Client',
        project_id: int,
        mode: TwinMode = TwinMode.VIRTUAL,
        name: Optional[str] = None,
        hardware_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a twin instance on the Cyberwave platform.
        
        Args:
            client: Cyberwave client instance
            project_id: Project to create the twin in
            mode: Twin operational mode
            name: Optional custom name for the twin
            hardware_id: Hardware identifier for physical/hybrid twins
            **kwargs: Additional twin configuration
            
        Returns:
            Created twin data from the API
        """
        self._client = client
        self._project_id = project_id
        self._mode = mode
        
        # First, ensure the asset type exists on the platform
        asset_data = await self._ensure_asset_exists(client)
        
        # Create the twin
        twin_data = {
            'asset_id': asset_data['id'],
            'name': name or f"{self.name} Twin",
            'mode': mode.value,
            'project_id': project_id,
        }
        
        if hardware_id and mode in [TwinMode.PHYSICAL, TwinMode.HYBRID]:
            twin_data['hardware_id'] = hardware_id
        
        # Add any additional configuration
        twin_data.update(kwargs)
        
        # Create via API
        response = await client._request(
            method="POST",
            url="/twins",
            json=twin_data
        )
        
        twin_response = response.json()
        self._twin_id = twin_response['id']
        self._twin_uuid = twin_response['uuid']
        
        return twin_response
    
    async def _ensure_asset_exists(self, client: 'Client') -> Dict[str, Any]:
        """
        Ensure the asset type exists on the platform.
        Creates it if necessary.
        """
        # Check if asset with our registry_id exists
        if self.registry_id:
            response = await client._request(
                method="GET",
                url=f"/assets?registry_id={self.registry_id}"
            )
            assets = response.json()
            
            if assets:
                return assets[0]
        
        # Create the asset
        asset_data = {
            'name': self.__class__.__name__,
            'description': self.__class__.__doc__ or f"{self.name} asset type",
            'asset_type': self._get_asset_type(),
            'registry_id': self.registry_id,
            'capabilities': self.capabilities,
            'specs': self.specs,
            'public': True,  # Could be configurable
        }
        
        response = await client._request(
            method="POST",
            url="/assets",
            json=asset_data
        )
        
        return response.json()
    
    def _get_asset_type(self) -> str:
        """Determine the asset type for the platform"""
        if isinstance(self, Robot):
            return "robot"
        elif isinstance(self, Sensor):
            return "sensor"
        elif hasattr(self, '_specs') and self._specs.get('shape'):
            return "static"
        else:
            return "other"
    
    async def connect(self, **kwargs) -> bool:
        """
        Connect to the physical twin (if applicable).
        
        This method should be overridden by specific implementations
        to handle hardware-specific connection logic.
        """
        if self._mode == TwinMode.VIRTUAL:
            # Virtual twins are always "connected"
            self._connected = True
            return True
        
        # Subclasses should implement actual connection logic
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement connect() for physical/hybrid mode"
        )
    
    async def disconnect(self) -> bool:
        """Disconnect from the physical twin"""
        self._connected = False
        return True
    
    async def update_state(self, state: Dict[str, Any]) -> None:
        """
        Update the twin's state on the platform.
        
        Args:
            state: State dictionary to send
        """
        if not self._twin_id or not self._client:
            raise RuntimeError("Twin not created yet")
        
        await self._client._request(
            method="PATCH",
            url=f"/twins/{self._twin_id}/state",
            json=state
        )
    
    async def update_position(
        self,
        x: float,
        y: float,
        z: float,
        rotation: Optional[List[float]] = None
    ) -> None:
        """
        Update the twin's position on the platform.
        
        Args:
            x, y, z: Position coordinates
            rotation: Optional quaternion rotation [w, x, y, z]
        """
        if not self._twin_id or not self._client:
            raise RuntimeError("Twin not created yet")
        
        position_data = {
            'position_x': x,
            'position_y': y,
            'position_z': z,
        }
        
        if rotation:
            position_data.update({
                'rotation_w': rotation[0],
                'rotation_x': rotation[1],
                'rotation_y': rotation[2],
                'rotation_z': rotation[3],
            })
        
        await self._client._request(
            method="PATCH",
            url=f"/twins/{self._twin_id}",
            json=position_data
        )
    
    async def send_telemetry(self, data: Dict[str, Any]) -> None:
        """
        Send telemetry data from the twin.
        
        Args:
            data: Telemetry data to send
        """
        if not self._twin_id or not self._client:
            raise RuntimeError("Twin not created yet")
        
        telemetry_data = {
            'twin_id': self._twin_id,
            'timestamp': datetime.utcnow().isoformat(),
            'data': data
        }
        
        await self._client._request(
            method="POST",
            url="/telemetry",
            json=telemetry_data
        )
    
    async def attach_sensor(
        self,
        sensor: 'Sensor',
        mount_transform: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Attach a sensor to this twin.
        
        Args:
            sensor: Sensor instance to attach
            mount_transform: Optional mounting transform
            
        Returns:
            Created sensor data
        """
        if not self._twin_id or not self._client:
            raise RuntimeError("Twin not created yet")
        
        # Ensure sensor has created its own twin if needed
        if not hasattr(sensor, '_twin_id'):
            await sensor.create_twin(
                self._client,
                self._project_id,
                mode=self._mode
            )
        
        sensor_data = {
            'twin_id': self._twin_id,
            'name': sensor.name,
            'sensor_type': sensor._get_sensor_type(),
            'mount_transform': mount_transform or {},
            'specs': sensor.specs,
        }
        
        response = await self._client._request(
            method="POST",
            url="/sensors",
            json=sensor_data
        )
        
        return response.json()
    
    @property
    def is_connected(self) -> bool:
        """Check if the twin is connected"""
        return self._connected
    
    @property
    def twin_id(self) -> Optional[int]:
        """Get the twin ID if created"""
        return self._twin_id
    
    @property
    def twin_uuid(self) -> Optional[str]:
        """Get the twin UUID if created"""
        return self._twin_uuid


class TwinEnabledRobot(Robot, TwinEnabledAsset):
    """Robot with twin integration capabilities"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        TwinEnabledAsset.__init__(self, **kwargs)
    
    async def send_command(self, command: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        """
        Send a command to the robot twin.
        
        Args:
            command: Command name
            payload: Optional command payload
            
        Returns:
            Command response
        """
        if not self._twin_id or not self._client:
            raise RuntimeError("Twin not created yet")
        
        return await self._client.send_command(
            target_entity_type="robot",
            target_entity_id=self._twin_id,
            command_name=command,
            command_payload=payload
        )


class TwinEnabledSensor(Sensor, TwinEnabledAsset):
    """Sensor with twin integration capabilities"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        TwinEnabledAsset.__init__(self, **kwargs)
    
    def _get_sensor_type(self) -> str:
        """Determine sensor type for the platform"""
        # Map capabilities to sensor types
        if 'measure_depth' in self.capabilities:
            return 'camera_depth'
        elif 'capture_image' in self.capabilities:
            return 'camera_rgb'
        elif '360_degree_scanning' in self.capabilities:
            return 'lidar'
        else:
            return 'other'
    
    async def stream_data(self, data: Any, timestamp: Optional[float] = None) -> None:
        """
        Stream sensor data to the platform.
        
        Args:
            data: Sensor data to stream
            timestamp: Optional timestamp (defaults to now)
        """
        if not self._twin_id or not self._client:
            raise RuntimeError("Twin not created yet")
        
        # Implementation would use WebSocket or streaming endpoint
        # For now, we'll use telemetry
        await self.send_telemetry({
            'sensor_data': data,
            'timestamp': timestamp or datetime.utcnow().timestamp()
        })


# Helper function to make any asset twin-enabled
def make_twin_enabled(asset_class: type) -> type:
    """
    Decorator/function to make any asset class twin-enabled.
    
    Usage:
        TwinEnabledDjiTello = make_twin_enabled(DjiTello)
        
        # Or as decorator:
        @make_twin_enabled
        class MyRobot(Robot):
            pass
    """
    if issubclass(asset_class, Robot):
        base_class = TwinEnabledRobot
    elif issubclass(asset_class, Sensor):
        base_class = TwinEnabledSensor
    else:
        base_class = TwinEnabledAsset
    
    class TwinEnabled(asset_class, base_class):
        """Twin-enabled version of the asset"""
        
        def __init__(self, **kwargs):
            asset_class.__init__(self, **kwargs)
            base_class.__init__(self, **kwargs)
    
    # Preserve class metadata
    TwinEnabled.__name__ = f"TwinEnabled{asset_class.__name__}"
    TwinEnabled.__doc__ = asset_class.__doc__
    TwinEnabled._registry_id = getattr(asset_class, '_registry_id', None)
    
    return TwinEnabled


__all__ = [
    'TwinMode',
    'TwinEnabledAsset',
    'TwinEnabledRobot',
    'TwinEnabledSensor',
    'make_twin_enabled',
] 