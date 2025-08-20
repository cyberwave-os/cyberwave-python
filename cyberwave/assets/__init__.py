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
] 