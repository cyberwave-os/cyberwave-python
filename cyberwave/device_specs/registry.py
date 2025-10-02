"""
Device Specification Registry

Central registry for all device specifications in the Cyberwave platform.
Provides discovery, validation, and management of device specs.
"""

from typing import Dict, List, Optional, Type, Any
import logging
from .base import DeviceSpec

logger = logging.getLogger(__name__)


class DeviceSpecRegistry:
    """
    Central registry for device specifications
    
    Manages registration, discovery, and retrieval of device specs.
    Thread-safe singleton pattern.
    """
    
    _instance: Optional['DeviceSpecRegistry'] = None
    _specs: Dict[str, DeviceSpec] = {}
    _specs_by_category: Dict[str, List[DeviceSpec]] = {}
    _specs_by_manufacturer: Dict[str, List[DeviceSpec]] = {}
    
    def __new__(cls) -> 'DeviceSpecRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, spec: DeviceSpec) -> None:
        """Register a device specification"""
        if not isinstance(spec, DeviceSpec):
            raise TypeError(f"Expected DeviceSpec, got {type(spec)}")
        
        if not spec.id:
            raise ValueError("Device spec must have an ID")
        
        # Validate spec has required fields
        if not spec.name:
            raise ValueError(f"Device spec {spec.id} must have a name")
        
        if not spec.category:
            raise ValueError(f"Device spec {spec.id} must have a category")
        
        # Register by ID
        cls._specs[spec.id] = spec
        
        # Register by category
        if spec.category not in cls._specs_by_category:
            cls._specs_by_category[spec.category] = []
        cls._specs_by_category[spec.category].append(spec)
        
        # Register by manufacturer
        if spec.manufacturer:
            if spec.manufacturer not in cls._specs_by_manufacturer:
                cls._specs_by_manufacturer[spec.manufacturer] = []
            cls._specs_by_manufacturer[spec.manufacturer].append(spec)
        
        logger.debug(f"Registered device spec: {spec.id} ({spec.name})")
    
    @classmethod
    def get(cls, device_id: str) -> Optional[DeviceSpec]:
        """Get device specification by ID"""
        return cls._specs.get(device_id)
    
    @classmethod
    def get_by_category(cls, category: str) -> List[DeviceSpec]:
        """Get all device specs for a category"""
        return cls._specs_by_category.get(category, []).copy()
    
    @classmethod
    def get_by_manufacturer(cls, manufacturer: str) -> List[DeviceSpec]:
        """Get all device specs for a manufacturer"""
        return cls._specs_by_manufacturer.get(manufacturer, []).copy()
    
    @classmethod
    def get_all(cls) -> List[DeviceSpec]:
        """Get all registered device specs"""
        return list(cls._specs.values())
    
    @classmethod
    def get_all_ids(cls) -> List[str]:
        """Get all registered device IDs"""
        return list(cls._specs.keys())
    
    @classmethod
    def get_categories(cls) -> List[str]:
        """Get all available categories"""
        return list(cls._specs_by_category.keys())
    
    @classmethod
    def get_manufacturers(cls) -> List[str]:
        """Get all available manufacturers"""
        return list(cls._specs_by_manufacturer.keys())
    
    @classmethod
    def exists(cls, device_id: str) -> bool:
        """Check if device spec exists"""
        return device_id in cls._specs
    
    @classmethod
    def search(cls, query: str) -> List[DeviceSpec]:
        """Search device specs by name, manufacturer, or category"""
        query_lower = query.lower()
        results = []
        
        for spec in cls._specs.values():
            if (query_lower in spec.name.lower() or
                query_lower in spec.manufacturer.lower() or
                query_lower in spec.category.lower() or
                query_lower in spec.id.lower()):
                results.append(spec)
        
        return results
    
    @classmethod
    def get_by_capability(cls, capability: str) -> List[DeviceSpec]:
        """Get all devices that support a specific capability"""
        results = []
        
        for spec in cls._specs.values():
            for cap in spec.capabilities:
                if cap.name == capability:
                    results.append(spec)
                    break
        
        return results
    
    @classmethod
    def get_by_command(cls, command: str) -> List[DeviceSpec]:
        """Get all devices that support a specific command"""
        results = []
        
        for spec in cls._specs.values():
            if spec.supports_command(command):
                results.append(spec)
        
        return results
    
    @classmethod
    def get_by_software_capability(cls, capability: str) -> List[DeviceSpec]:
        """Get all devices that have a specific software capability"""
        results = []
        
        for spec in cls._specs.values():
            if spec.has_capability(capability):
                results.append(spec)
        
        return results
    
    @classmethod
    def get_with_hardware_drivers(cls) -> List[DeviceSpec]:
        """Get all devices that have hardware drivers available"""
        return [spec for spec in cls._specs.values() if spec.has_hardware_driver]
    
    @classmethod
    def get_with_digital_assets(cls) -> List[DeviceSpec]:
        """Get all devices that have digital assets available"""
        return [spec for spec in cls._specs.values() if spec.has_digital_asset]
    
    @classmethod
    def get_with_simulation_models(cls) -> List[DeviceSpec]:
        """Get all devices that have simulation models available"""
        return [spec for spec in cls._specs.values() if spec.has_simulation_model]
    
    @classmethod
    def get_complete_devices(cls) -> List[DeviceSpec]:
        """Get devices with all implementations (driver + asset + simulation)"""
        return [spec for spec in cls._specs.values() if spec.is_complete()]
    
    @classmethod
    def get_by_deployment_mode(cls, mode: str) -> List[DeviceSpec]:
        """Get devices by deployment mode (hybrid, hardware_only, digital_only, etc.)"""
        return [spec for spec in cls._specs.values() if spec.get_deployment_mode() == mode]
    
    @classmethod
    def find_fallback_for_category(cls, category: str) -> Optional[DeviceSpec]:
        """Find a generic/fallback device spec for a category"""
        # Look for generic devices in the category
        generic_specs = [
            spec for spec in cls._specs.values() 
            if spec.category == category and "generic" in spec.id.lower()
        ]
        
        if generic_specs:
            # Prefer specs with both driver and asset
            complete_specs = [spec for spec in generic_specs if spec.has_hardware_driver and spec.has_digital_asset]
            if complete_specs:
                return complete_specs[0]
            
            # Fall back to any generic spec
            return generic_specs[0]
        
        return None
    
    @classmethod
    def create_unknown_device_spec(cls, device_id: str, category: str = "unknown") -> DeviceSpec:
        """Create a basic spec for an unknown device"""
        from .base import DeviceSpec, Capability
        
        # Try to parse manufacturer/model from ID
        parts = device_id.split("/")
        manufacturer = parts[0] if len(parts) > 1 else "Unknown"
        model = parts[1] if len(parts) > 1 else device_id
        
        # Create a minimal spec
        class UnknownDeviceSpec(DeviceSpec):
            def __post_init__(self):
                self.id = device_id
                self.name = f"{manufacturer} {model}".title()
                self.category = category
                self.manufacturer = manufacturer
                self.model = model
                self.description = f"Unknown device: {device_id}"
                
                # No implementations available for unknown devices
                self.has_hardware_driver = False
                self.has_digital_asset = False
                self.has_simulation_model = False
                
                # Basic capabilities
                self.capabilities = [
                    Capability(
                        name="basic",
                        commands=["connect", "disconnect", "status"],
                        description="Basic device operations"
                    )
                ]
                
                super().__post_init__()
        
        return UnknownDeviceSpec()
    
    @classmethod
    def validate_device_config(cls, device_id: str, config: Dict[str, Any]) -> bool:
        """Validate device configuration against spec"""
        spec = cls.get(device_id)
        if not spec:
            return False
        
        return spec.validate_config(config)
    
    @classmethod
    def get_setup_wizard(cls, device_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get setup wizard fields for device"""
        spec = cls.get(device_id)
        if not spec:
            return None
        
        return [
            {
                "name": field.name,
                "type": field.type,
                "label": field.label,
                "default": field.default,
                "required": field.required,
                "options": field.options,
                "validation": field.validation,
                "help_text": field.help_text
            }
            for field in spec.setup_wizard
        ]
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered specs (for testing)"""
        cls._specs.clear()
        cls._specs_by_category.clear()
        cls._specs_by_manufacturer.clear()
    
    @classmethod
    def stats(cls) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_specs": len(cls._specs),
            "categories": len(cls._specs_by_category),
            "manufacturers": len(cls._specs_by_manufacturer),
            "specs_by_category": {
                category: len(specs) 
                for category, specs in cls._specs_by_category.items()
            },
            "specs_by_manufacturer": {
                manufacturer: len(specs)
                for manufacturer, specs in cls._specs_by_manufacturer.items()
            }
        }


# Convenience functions for common operations
def get_device_spec(device_id: str) -> Optional[DeviceSpec]:
    """Get device specification by ID"""
    return DeviceSpecRegistry.get(device_id)


def list_devices(category: str = None) -> List[DeviceSpec]:
    """List all devices, optionally filtered by category"""
    if category:
        return DeviceSpecRegistry.get_by_category(category)
    return DeviceSpecRegistry.get_all()


def search_devices(query: str) -> List[DeviceSpec]:
    """Search devices by name, manufacturer, or category"""
    return DeviceSpecRegistry.search(query)


def get_supported_capabilities() -> List[str]:
    """Get all supported capabilities across all devices"""
    capabilities = set()
    for spec in DeviceSpecRegistry.get_all():
        for cap in spec.capabilities:
            capabilities.add(cap.name)
    return sorted(list(capabilities))


def get_supported_commands() -> List[str]:
    """Get all supported commands across all devices"""
    commands = set()
    for spec in DeviceSpecRegistry.get_all():
        commands.update(spec.get_all_commands())
    return sorted(list(commands))


# New capability-based convenience functions
def get_devices_with_hardware_drivers() -> List[DeviceSpec]:
    """Get all devices that have hardware drivers"""
    return DeviceSpecRegistry.get_with_hardware_drivers()


def get_devices_with_digital_assets() -> List[DeviceSpec]:
    """Get all devices that have digital assets"""
    return DeviceSpecRegistry.get_with_digital_assets()


def get_devices_with_simulation() -> List[DeviceSpec]:
    """Get all devices that have simulation models"""
    return DeviceSpecRegistry.get_with_simulation_models()


def get_complete_devices() -> List[DeviceSpec]:
    """Get devices with full implementation (driver + asset + simulation)"""
    return DeviceSpecRegistry.get_complete_devices()


def find_or_create_device_spec(device_id: str, category: str = None) -> DeviceSpec:
    """Find existing spec or create fallback for unknown device"""
    # Try to get existing spec
    spec = DeviceSpecRegistry.get(device_id)
    if spec:
        return spec
    
    # Try to find fallback for category
    if category:
        fallback = DeviceSpecRegistry.find_fallback_for_category(category)
        if fallback:
            return fallback
    
    # Create unknown device spec
    return DeviceSpecRegistry.create_unknown_device_spec(device_id, category or "unknown")


def get_deployment_recommendations(device_id: str) -> Dict[str, Any]:
    """Get deployment recommendations for a device"""
    spec = DeviceSpecRegistry.get(device_id)
    if not spec:
        return {"error": "Device not found", "recommendations": []}
    
    recommendations = []
    
    if spec.has_hardware_driver and spec.has_digital_asset:
        recommendations.append({
            "mode": "hybrid",
            "description": "Full physical + digital twin deployment",
            "benefits": ["Real hardware control", "Digital twin visualization", "Simulation fallback"]
        })
    
    if spec.has_hardware_driver:
        recommendations.append({
            "mode": "hardware_only", 
            "description": "Physical hardware control only",
            "benefits": ["Direct hardware control", "Minimal dependencies"],
            "limitations": ["No digital twin", "No simulation"]
        })
    
    if spec.has_digital_asset:
        recommendations.append({
            "mode": "digital_only",
            "description": "Digital twin and simulation only", 
            "benefits": ["Safe testing", "Visualization", "No hardware required"],
            "limitations": ["No real hardware control"]
        })
    
    if spec.has_simulation_model:
        recommendations.append({
            "mode": "simulation",
            "description": "Simulation environment testing",
            "benefits": ["Safe testing", "Physics simulation", "Scenario testing"],
            "simulators": spec.simulation_models
        })
    
    return {
        "device_id": device_id,
        "deployment_mode": spec.get_deployment_mode(),
        "available_implementations": spec.get_available_implementations(),
        "missing_implementations": spec.get_missing_implementations(),
        "recommendations": recommendations,
        "is_complete": spec.is_complete()
    }
