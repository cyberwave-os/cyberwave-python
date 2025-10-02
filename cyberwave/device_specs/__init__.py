"""
Cyberwave Device Specifications

This module provides device specifications for all supported hardware in the Cyberwave platform.
Device specs serve as the single source of truth for device capabilities, protocols, and configuration.

Usage:
    from cyberwave.device_specs import DeviceSpecRegistry
    
    # Get device specification
    spec = DeviceSpecRegistry.get("dji/tello")
    
    # List all devices
    devices = DeviceSpecRegistry.get_all()
    
    # Get devices by category
    drones = DeviceSpecRegistry.get_by_category("drone")
"""

from .base import (
    DeviceSpec,
    Capability,
    Protocol,
    ConnectionInfo,
    SetupWizardField
)

from .registry import (
    DeviceSpecRegistry,
    get_devices_with_hardware_drivers,
    get_devices_with_digital_assets,
    get_devices_with_simulation,
    get_complete_devices,
    get_deployment_recommendations
)

from .fallback import (
    get_or_create_device_spec as find_or_create_device_spec,
    validate_deployment_requirements
)

from .specs.robots import (
    DjiTelloSpec,
    BostonDynamicsSpotSpec,
    So101Spec,
    KukaKr3Spec,
    UniversalRobotsUR5eSpec,
)

from .specs.robots.unitree_go1 import UnitreeGo1Spec

from .specs.sensors import (
    IntelRealSenseD435Spec,
    VelodynePuckSpec,
    Zed2Spec
)

from .specs.cameras import (
    UniviewNVRSpec,
    IPCameraSpec,
    GenericCameraSpec
)

from .specs.cameras.generic_webcam import GenericWebcamSpec
from .specs.cameras.generic_nvr import GenericNVRSpec
from .specs.cameras.rtsp_camera import RTSPCameraSpec

from .specs.construction import (
    ExcavatorSpec,
    CaterpillarExcavatorSpec,
    SecurityCameraSpec,
    PTZSecurityCameraSpec,
    SecurityDroneSpec,
    PerimeterGuardAISpec,
    CompanyBuildingSpec,
)

__version__ = "0.1.0"

__all__ = [
    # Base classes
    "DeviceSpec",
    "Capability", 
    "Protocol",
    "ConnectionInfo",
    "SetupWizardField",
    
    # Registry
    "DeviceSpecRegistry",
    
    # Capability-based discovery
    "get_devices_with_hardware_drivers",
    "get_devices_with_digital_assets", 
    "get_devices_with_simulation",
    "get_complete_devices",
    "find_or_create_device_spec",
    "get_deployment_recommendations",
    "validate_deployment_requirements",
    
    # Robot specs
    "DjiTelloSpec",
    "BostonDynamicsSpotSpec", 
    "So101Spec",
    "KukaKr3Spec",
    "UniversalRobotsUR5eSpec",
    "UnitreeGo1Spec",
    
    # Sensor specs
    "IntelRealSenseD435Spec",
    "VelodynePuckSpec",
    "Zed2Spec",
    
    # Camera specs
    "UniviewNVRSpec",
    "IPCameraSpec",
    "GenericCameraSpec",
    "GenericWebcamSpec",
    "GenericNVRSpec",
    "RTSPCameraSpec",
    
    # Construction specs
    "ExcavatorSpec",
    "CaterpillarExcavatorSpec",
    "SecurityCameraSpec", 
    "PTZSecurityCameraSpec",
    "SecurityDroneSpec",
    "PerimeterGuardAISpec",
    "CompanyBuildingSpec",
]

# Auto-register all specs when module is imported
def _register_all_specs():
    """Register all device specifications with the registry"""
    specs = [
        # Robots
        DjiTelloSpec(),
        BostonDynamicsSpotSpec(),
        So101Spec(),
        KukaKr3Spec(),
        UniversalRobotsUR5eSpec(),
        UnitreeGo1Spec(),
        
        # Sensors
        IntelRealSenseD435Spec(),
        VelodynePuckSpec(),
        Zed2Spec(),
        
        # Cameras
        UniviewNVRSpec(),
        IPCameraSpec(),
        GenericCameraSpec(),
        GenericWebcamSpec(),
        GenericNVRSpec(),
        RTSPCameraSpec(),
        
        # Construction Equipment
        ExcavatorSpec(),
        CaterpillarExcavatorSpec(),
        SecurityCameraSpec(),
        PTZSecurityCameraSpec(),
        SecurityDroneSpec(),
        PerimeterGuardAISpec(),
        CompanyBuildingSpec()
    ]
    
    for spec in specs:
        DeviceSpecRegistry.register(spec)

    # Register alias IDs for commonly referenced examples
    if not DeviceSpecRegistry.exists("intel/realsense_d435"):
        realsense_alias = IntelRealSenseD435Spec()
        realsense_alias.id = "intel/realsense_d435"
        DeviceSpecRegistry.register(realsense_alias)

# Auto-register on import
_register_all_specs()
