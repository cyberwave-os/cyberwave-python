"""
Cyberwave Asset System

This module provides a HuggingFace-style asset registry for robots, sensors,
and static assets. Assets can be instantiated directly or referenced by ID.

Examples:
    # Direct instantiation
    drone = DjiTello(ip="192.168.10.1")
    
    # Registry reference
    robot = Robot("boston-dynamics/spot")
    
    # Factory creation
    asset = AssetFactory.create("props/traffic-cone", color=[1, 0.5, 0])
    
    # Create a twin-enabled asset
    from cyberwave.assets import make_twin_enabled, DjiTello
    TwinDjiTello = make_twin_enabled(DjiTello)
    drone = TwinDjiTello(ip="192.168.10.1")
    await drone.create_twin(client, project_id, mode=TwinMode.HYBRID)
"""

# Core registry system
from .registry import (
    AssetRegistry,
    register_asset,
    AssetFactory,
    BaseAsset,
    Robot,
    FlyingRobot,
    GroundRobot,
    Sensor,
    CameraSensor,
    DepthSensor,
    StaticAsset,
    Prop,
    Landmark,
    Infrastructure,
)

# Pre-configured implementations
from .implementations import (
    # Drones
    DjiTello,
    DjiMavic3,
    ParrotAnafi,
    # Ground Robots
    BostonDynamicsSpot,
    UnitreeGo1,
    ClearpathHusky,
    FrankaPanda,
    # Sensors
    IntelRealSenseD435,
    VelodynePuck,
    ZED2,
    # Props
    Box,
    Sphere,
    Cylinder,
    TrafficCone,
    Pallet,
    # Landmarks
    ArucoMarker,
    QRCode,
    AprilTag,
    # Infrastructure
    Wall,
    ChargingPad,
    Conveyor,
    CustomMesh,
)

# Twin integration
from .twin_integration import (
    TwinMode,
    TwinEnabledAsset,
    TwinEnabledRobot,
    TwinEnabledSensor,
    make_twin_enabled,
)

from ..twin_capabilities import SO101Twin as SO101Robot

# Device spec integration utilities
try:
    from ..device_specs import DeviceSpecRegistry
    
    def get_device_spec(device_id: str):
        """Get device specification by ID"""
        return DeviceSpecRegistry.get(device_id)
    
    def list_supported_devices(category: str = None):
        """List all supported devices, optionally filtered by category"""
        if category:
            return DeviceSpecRegistry.get_by_category(category)
        return DeviceSpecRegistry.get_all()
    
    def search_devices(query: str):
        """Search devices by name, manufacturer, or category"""
        return DeviceSpecRegistry.search(query)
    
    DEVICE_SPECS_AVAILABLE = True
    
except ImportError:
    def get_device_spec(device_id: str):
        return None
    
    def list_supported_devices(category: str = None):
        return []
    
    def search_devices(query: str):
        return []
    
    DEVICE_SPECS_AVAILABLE = False

__version__ = "0.1.0"

__all__ = [
    # Registry system
    "AssetRegistry",
    "register_asset",
    "AssetFactory",
    "BaseAsset",
    "Robot",
    "FlyingRobot",
    "GroundRobot",
    "Sensor",
    "CameraSensor",
    "DepthSensor",
    "StaticAsset",
    "Prop",
    "Landmark",
    "Infrastructure",
    # Implementations
    "DjiTello",
    "DjiMavic3",
    "ParrotAnafi",
    "BostonDynamicsSpot",
    "UnitreeGo1",
    "ClearpathHusky",
    "FrankaPanda",
    "IntelRealSenseD435",
    "VelodynePuck",
    "ZED2",
    "Box",
    "Sphere",
    "Cylinder",
    "TrafficCone",
    "Pallet",
    "ArucoMarker",
    "QRCode",
    "AprilTag",
    "Wall",
    "ChargingPad",
    "Conveyor",
    "CustomMesh",
    # Twin integration
    "TwinMode",
    "TwinEnabledAsset",
    "TwinEnabledRobot",
    "TwinEnabledSensor",
    "make_twin_enabled",
    "SO101Robot",
    # Device spec utilities
    "get_device_spec",
    "list_supported_devices", 
    "search_devices",
    "DEVICE_SPECS_AVAILABLE",
] 
