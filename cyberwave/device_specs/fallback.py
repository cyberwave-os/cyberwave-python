"""
Smart Fallback System for Device Specifications

Provides intelligent fallbacks when device specifications or implementations are missing.
Enables graceful degradation and automatic generic device creation.
"""

import logging
from typing import Optional, Dict, Any, List, Type
from .base import DeviceSpec, Capability, Protocol, ConnectionInfo, SetupWizardField
from .registry import DeviceSpecRegistry

logger = logging.getLogger(__name__)


class FallbackSystem:
    """
    Smart fallback system for missing device implementations
    
    Provides graceful degradation when hardware drivers, digital assets,
    or device specifications are not available.
    """
    
    @staticmethod
    def get_or_create_spec(device_id: str, category: str = None, **hints) -> DeviceSpec:
        """
        Get existing spec or create intelligent fallback
        
        Args:
            device_id: Device identifier (e.g., "unknown/camera")
            category: Device category hint
            **hints: Additional hints for fallback creation
        """
        # Try to get existing specification
        spec = DeviceSpecRegistry.get(device_id)
        if spec:
            return spec
        
        logger.info(f"Device spec not found for {device_id}, creating fallback")
        
        # Try to find generic fallback for category
        if category:
            fallback = DeviceSpecRegistry.find_fallback_for_category(category)
            if fallback:
                logger.info(f"Using generic fallback for category {category}: {fallback.id}")
                return fallback
        
        # Try to infer category from device_id
        inferred_category = FallbackSystem._infer_category_from_id(device_id)
        if inferred_category and inferred_category != category:
            fallback = DeviceSpecRegistry.find_fallback_for_category(inferred_category)
            if fallback:
                logger.info(f"Using inferred category fallback {inferred_category}: {fallback.id}")
                return fallback
        
        # Create unknown device spec
        logger.warning(f"Creating unknown device spec for {device_id}")
        return FallbackSystem._create_unknown_device_spec(device_id, category, **hints)
    
    @staticmethod
    def get_asset_class(spec: DeviceSpec) -> Optional[str]:
        """Get asset class with fallback logic"""
        if spec.has_digital_asset and spec.asset_class:
            return spec.asset_class
        
        if spec.fallback_asset_class:
            logger.info(f"Using fallback asset class for {spec.id}: {spec.fallback_asset_class}")
            return spec.fallback_asset_class
        
        # Category-based fallbacks
        category_fallbacks = {
            "drone": "cyberwave.assets.GenericDrone",
            "ground_robot": "cyberwave.assets.GenericGroundRobot", 
            "robotic_arm": "cyberwave.assets.GenericRoboticArm",
            "ip_camera": "cyberwave.assets.GenericIPCamera",
            "camera": "cyberwave.assets.GenericCamera",
            "sensor": "cyberwave.assets.GenericSensor"
        }
        
        fallback_class = category_fallbacks.get(spec.category)
        if fallback_class:
            logger.info(f"Using category-based asset fallback for {spec.id}: {fallback_class}")
            return fallback_class
        
        # Ultimate fallback
        logger.warning(f"No asset class available for {spec.id}, using generic device")
        return "cyberwave.assets.GenericDevice"
    
    @staticmethod
    def get_driver_class(spec: DeviceSpec) -> Optional[str]:
        """Get driver class with fallback logic"""
        if spec.has_hardware_driver and spec.driver_class:
            return spec.driver_class
        
        # No fallback for drivers - they must be device-specific
        logger.warning(f"No hardware driver available for {spec.id}")
        return None
    
    @staticmethod
    def create_deployment_plan(spec: DeviceSpec, requirements: List[str] = None) -> Dict[str, Any]:
        """
        Create deployment plan based on available implementations and requirements
        
        Args:
            spec: Device specification
            requirements: List of required capabilities (e.g., ["hardware_driver", "digital_asset"])
        """
        requirements = requirements or []
        
        plan = {
            "device_id": spec.id,
            "deployment_mode": spec.get_deployment_mode(),
            "available": spec.get_available_implementations(),
            "missing": spec.get_missing_implementations(),
            "components": {},
            "fallbacks": {},
            "warnings": [],
            "errors": []
        }
        
        # Check hardware driver
        if "hardware_driver" in requirements or spec.has_hardware_driver:
            driver_class = FallbackSystem.get_driver_class(spec)
            if driver_class:
                plan["components"]["hardware_driver"] = driver_class
            else:
                plan["errors"].append("Hardware driver required but not available")
                plan["warnings"].append("Device will operate in digital-only mode")
        
        # Check digital asset
        if "digital_asset" in requirements or spec.has_digital_asset:
            asset_class = FallbackSystem.get_asset_class(spec)
            if asset_class:
                plan["components"]["digital_asset"] = asset_class
                if asset_class != spec.asset_class:
                    plan["fallbacks"]["digital_asset"] = asset_class
                    plan["warnings"].append(f"Using fallback asset class: {asset_class}")
            else:
                plan["errors"].append("Digital asset required but not available")
        
        # Check simulation
        if "simulation_model" in requirements or spec.has_simulation_model:
            if spec.simulation_models:
                plan["components"]["simulation_models"] = spec.simulation_models
            else:
                plan["warnings"].append("Simulation models not available")
        
        # Determine final deployment mode
        if plan["components"].get("hardware_driver") and plan["components"].get("digital_asset"):
            plan["final_mode"] = "hybrid"
        elif plan["components"].get("hardware_driver"):
            plan["final_mode"] = "hardware_only"
        elif plan["components"].get("digital_asset"):
            plan["final_mode"] = "digital_only"
        else:
            plan["final_mode"] = "unavailable"
            plan["errors"].append("No usable implementations available")
        
        return plan
    
    @staticmethod
    def _infer_category_from_id(device_id: str) -> Optional[str]:
        """Infer device category from device ID"""
        device_id_lower = device_id.lower()
        
        # Category keywords
        category_keywords = {
            "drone": ["drone", "quadcopter", "uav", "tello", "mavic"],
            "camera": ["camera", "cam", "webcam", "ipcam"],
            "ip_camera": ["ipcamera", "ip-camera", "ipcam", "nvr"],
            "sensor": ["sensor", "lidar", "radar", "imu", "gps"],
            "robotic_arm": ["arm", "manipulator", "robot-arm"],
            "ground_robot": ["robot", "rover", "agv", "mobile"],
            "quadruped": ["dog", "spot", "quadruped"],
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in device_id_lower for keyword in keywords):
                return category
        
        return None
    
    @staticmethod
    def _create_unknown_device_spec(device_id: str, category: str = None, **hints) -> DeviceSpec:
        """Create a basic spec for completely unknown device"""
        from .base import DeviceSpec, Capability
        
        # Parse device ID
        parts = device_id.split("/")
        manufacturer = parts[0] if len(parts) > 1 else "Unknown"
        model = parts[1] if len(parts) > 1 else device_id
        
        # Use hints or infer category
        final_category = category or FallbackSystem._infer_category_from_id(device_id) or "unknown"
        
        class UnknownDeviceSpec(DeviceSpec):
            def __post_init__(self):
                self.id = device_id
                self.name = hints.get("name", f"{manufacturer} {model}".title())
                self.category = final_category
                self.manufacturer = manufacturer
                self.model = model
                self.description = hints.get("description", f"Unknown device: {device_id}")
                
                # No implementations available for unknown devices
                self.has_hardware_driver = False
                self.has_digital_asset = False
                self.has_simulation_model = False
                
                # Try to provide fallback asset class
                self.fallback_asset_class = FallbackSystem._get_category_fallback_asset(final_category)
                
                # Basic capabilities
                self.capabilities = [
                    Capability(
                        name="basic",
                        commands=["connect", "disconnect", "status"],
                        description="Basic device operations"
                    )
                ]
                
                # Basic setup wizard
                self.setup_wizard = [
                    SetupWizardField(
                        name="name",
                        type="string",
                        label="Device Name",
                        default=self.name,
                        help_text="Friendly name for this device"
                    )
                ]
                
                # Add category-specific setup fields
                if final_category in ["camera", "ip_camera"]:
                    self.setup_wizard.extend([
                        SetupWizardField(
                            name="ip_address",
                            type="ipv4",
                            label="IP Address",
                            default="192.168.1.100"
                        ),
                        SetupWizardField(
                            name="username",
                            type="string",
                            label="Username",
                            default="admin",
                            required=False
                        ),
                        SetupWizardField(
                            name="password",
                            type="string",
                            label="Password",
                            default="",
                            required=False
                        )
                    ])
                
                super().__post_init__()
        
        return UnknownDeviceSpec()
    
    @staticmethod
    def _get_category_fallback_asset(category: str) -> Optional[str]:
        """Get fallback asset class for category"""
        fallbacks = {
            "drone": "cyberwave.assets.GenericDrone",
            "ground_robot": "cyberwave.assets.GenericGroundRobot",
            "robotic_arm": "cyberwave.assets.GenericRoboticArm", 
            "camera": "cyberwave.assets.GenericCamera",
            "ip_camera": "cyberwave.assets.GenericIPCamera",
            "sensor": "cyberwave.assets.GenericSensor"
        }
        
        return fallbacks.get(category, "cyberwave.assets.GenericDevice")


# Convenience functions
def get_or_create_device_spec(device_id: str, category: str = None, **hints) -> DeviceSpec:
    """Get existing device spec or create intelligent fallback"""
    return FallbackSystem.get_or_create_spec(device_id, category, **hints)


def create_deployment_plan(device_id: str, requirements: List[str] = None) -> Dict[str, Any]:
    """Create deployment plan for device"""
    spec = get_or_create_device_spec(device_id)
    return FallbackSystem.create_deployment_plan(spec, requirements)


def validate_deployment_requirements(device_id: str, requirements: List[str]) -> Dict[str, Any]:
    """Validate if device can meet deployment requirements"""
    spec = get_or_create_device_spec(device_id)  # Use fallback system
    
    missing = []
    available = spec.get_available_implementations()
    
    for requirement in requirements:
        if requirement not in available:
            missing.append(requirement)
    
    if missing:
        return {
            "valid": False,
            "missing": missing,
            "available": available,
            "suggestions": [
                f"Install {req} implementation" for req in missing
            ]
        }
    
    return {"valid": True, "available": available}
