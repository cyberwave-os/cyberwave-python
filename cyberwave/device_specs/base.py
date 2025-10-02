"""
Base classes for device specifications

Provides the foundation for defining device capabilities, protocols, and configuration.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod


@dataclass
class Capability:
    """Represents a device capability with associated commands and UI metadata"""
    name: str
    commands: List[str]
    description: str = ""
    ui_metadata: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)  # For backward compatibility
    
    def __post_init__(self):
        if self.ui_metadata is None:
            self.ui_metadata = {}
        if self.metadata is None:
            self.metadata = {}
    
    def supports_command(self, command: str) -> bool:
        """Check if this capability supports a command"""
        return command in self.commands
    
    def get_ui_component_type(self) -> Optional[str]:
        """Get the UI component type for this capability"""
        return self.ui_metadata.get("component_type")
    
    def get_ui_controls(self) -> List[Dict[str, Any]]:
        """Get UI control definitions for this capability"""
        return self.ui_metadata.get("controls", [])


@dataclass 
class Protocol:
    """Represents a communication protocol"""
    type: str  # "tcp", "udp", "serial", "modbus_tcp", "ethernet_ip", etc.
    port: Optional[Union[int, str]] = None
    baudrate: Optional[int] = None
    commands: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.commands is None:
            self.commands = []
        if self.parameters is None:
            self.parameters = {}


@dataclass
class ConnectionInfo:
    """Device connection information"""
    type: str  # "wifi", "ethernet", "serial", "usb", etc.
    default_ip: Optional[str] = None
    default_port: Optional[Union[int, str]] = None
    setup_instructions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.setup_instructions is None:
            self.setup_instructions = []


@dataclass
class SetupWizardField:
    """Field definition for setup wizard"""
    name: str
    type: str  # "string", "int", "float", "boolean", "select", "ipv4"
    label: str
    default: Any = None
    required: bool = True
    options: Optional[List[str]] = None
    validation: Optional[str] = None
    help_text: str = ""


@dataclass 
class DependencySpec:
    """Specification for a device dependency"""
    package: str
    name: str
    version: Optional[str] = None
    optional: bool = False
    install_command: Optional[str] = None
    description: str = ""
    fallback_message: str = ""

@dataclass
class DeviceSpec(ABC):
    """
    Base class for device specifications
    
    Software-defined device specification with capability flags.
    Single source of truth that declares what implementations are available.
    """
    
    # Core identification
    id: str = ""
    name: str = ""
    category: str = ""  # "drone", "ground_robot", "sensor", "camera", etc.
    manufacturer: str = ""
    model: str = ""
    
    # Software-defined capabilities (core)
    has_hardware_driver: bool = False
    has_digital_asset: bool = False
    has_simulation_model: bool = False
    
    # Extended capabilities (extensible)
    extended_capabilities: Dict[str, bool] = field(default_factory=dict)
    
    # Device capabilities and protocols
    capabilities: List[Capability] = field(default_factory=list)
    protocols: List[Protocol] = field(default_factory=list)
    
    # Connection and setup
    connection: Optional[ConnectionInfo] = None
    setup_wizard: List[SetupWizardField] = field(default_factory=list)
    
    # Connectivity modes (e.g., web, node)
    connectivity_modes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Technical specifications
    specs: Dict[str, Any] = field(default_factory=dict)
    
    # Implementation details
    driver_class: Optional[str] = None
    asset_class: Optional[str] = None
    simulation_models: List[str] = field(default_factory=list)
    fallback_asset_class: Optional[str] = None
    
    # Dependencies for hardware driver
    dependencies: List[DependencySpec] = field(default_factory=list)
    
    # Metadata
    description: str = ""
    documentation_url: str = ""
    support_url: str = ""
    
    def __post_init__(self):
        """Initialize default values and validate spec"""
        if not self.capabilities:
            self.capabilities = []
        if not self.protocols:
            self.protocols = []
        if not self.setup_wizard:
            self.setup_wizard = []
        if not self.specs:
            self.specs = {}
        if not self.connectivity_modes:
            self.connectivity_modes = []
    
    def get_all_commands(self) -> List[str]:
        """Get all commands supported by this device"""
        commands = []
        for capability in self.capabilities:
            commands.extend(capability.commands)
        return list(set(commands))  # Remove duplicates
    
    def supports_command(self, command: str) -> bool:
        """Check if device supports a specific command"""
        return command in self.get_all_commands()
    
    def get_capability(self, name: str) -> Optional[Capability]:
        """Get capability by name"""
        for capability in self.capabilities:
            if capability.name == name:
                return capability
        return None
    
    def get_protocol(self, protocol_type: str) -> Optional[Protocol]:
        """Get protocol by type"""
        for protocol in self.protocols:
            if protocol.type == protocol_type:
                return protocol
        return None
    
    def has_capability(self, capability_name: str) -> bool:
        """Check if device has a specific software capability"""
        # Check core capabilities
        if capability_name in ["hardware_driver", "digital_asset", "simulation_model"]:
            return getattr(self, f"has_{capability_name}", False)
        
        # Check extended capabilities
        return self.extended_capabilities.get(capability_name, False)
    
    def get_all_capabilities(self) -> Dict[str, bool]:
        """Get all software capabilities"""
        capabilities = {
            "hardware_driver": self.has_hardware_driver,
            "digital_asset": self.has_digital_asset,
            "simulation_model": self.has_simulation_model
        }
        capabilities.update(self.extended_capabilities)
        return capabilities
    
    def get_available_implementations(self) -> List[str]:
        """Get list of available implementation types"""
        implementations = []
        if self.has_hardware_driver:
            implementations.append("hardware_driver")
        if self.has_digital_asset:
            implementations.append("digital_asset")
        if self.has_simulation_model:
            implementations.append("simulation_model")
        
        # Add extended capabilities that are True
        for cap_name, available in self.extended_capabilities.items():
            if available:
                implementations.append(cap_name)
        
        return implementations
    
    def get_missing_implementations(self) -> List[str]:
        """Get list of missing implementation types"""
        missing = []
        if not self.has_hardware_driver:
            missing.append("hardware_driver")
        if not self.has_digital_asset:
            missing.append("digital_asset")
        if not self.has_simulation_model:
            missing.append("simulation_model")
        return missing
    
    def is_complete(self) -> bool:
        """Check if device has all core implementations"""
        return self.has_hardware_driver and self.has_digital_asset and self.has_simulation_model
    
    def get_deployment_mode(self) -> str:
        """Get recommended deployment mode based on available implementations"""
        if self.has_hardware_driver and self.has_digital_asset:
            return "hybrid"  # Physical + Digital
        elif self.has_hardware_driver:
            return "hardware_only"
        elif self.has_digital_asset:
            return "digital_only"
        else:
            return "specification_only"
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate device configuration against spec"""
        # Basic validation - can be extended by subclasses
        for field in self.setup_wizard:
            if field.required and field.name not in config:
                return False
            
            if field.name in config:
                value = config[field.name]
                
                # Type validation
                if field.type == "int" and not isinstance(value, int):
                    return False
                elif field.type == "float" and not isinstance(value, (int, float)):
                    return False
                elif field.type == "boolean" and not isinstance(value, bool):
                    return False
                elif field.type == "select" and field.options and value not in field.options:
                    return False
                elif field.type == "ipv4" and not self._is_valid_ipv4(str(value)):
                    return False
        
        return True
    
    def _is_valid_ipv4(self, ip: str) -> bool:
        """Validate IPv4 address"""
        try:
            parts = ip.split('.')
            return len(parts) == 4 and all(0 <= int(part) <= 255 for part in parts)
        except (ValueError, AttributeError):
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert spec to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "manufacturer": self.manufacturer,
            "model": self.model,
            
            # Software-defined capabilities
            "has_hardware_driver": self.has_hardware_driver,
            "has_digital_asset": self.has_digital_asset,
            "has_simulation_model": self.has_simulation_model,
            "extended_capabilities": self.extended_capabilities,
            
            # Implementation details
            "driver_class": self.driver_class,
            "asset_class": self.asset_class,
            "simulation_models": self.simulation_models,
            "fallback_asset_class": self.fallback_asset_class,
            
            # Deployment info
            "available_implementations": self.get_available_implementations(),
            "missing_implementations": self.get_missing_implementations(),
            "deployment_mode": self.get_deployment_mode(),
            "is_complete": self.is_complete(),
            
            "capabilities": [
                {
                    "name": cap.name,
                    "commands": cap.commands,
                    "description": cap.description,
                    "metadata": cap.metadata,
                    "command_schemas": (cap.metadata or {}).get("command_schemas"),
                }
                for cap in self.capabilities
            ],
            "protocols": [
                {
                    "type": prot.type,
                    "port": prot.port,
                    "baudrate": prot.baudrate,
                    "commands": prot.commands,
                    "parameters": prot.parameters
                }
                for prot in self.protocols
            ],
            "connection": {
                "type": self.connection.type,
                "default_ip": self.connection.default_ip,
                "default_port": self.connection.default_port,
                "setup_instructions": self.connection.setup_instructions
            } if self.connection else None,
            "setup_wizard": [
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
                for field in self.setup_wizard
            ],
            "connectivity_modes": self.connectivity_modes,
            "specs": self.specs,
            "description": self.description,
            "documentation_url": self.documentation_url,
            "support_url": self.support_url
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeviceSpec':
        """Create spec from dictionary"""
        # This would need to be implemented by subclasses
        # for proper deserialization
        raise NotImplementedError("Subclasses must implement from_dict")
