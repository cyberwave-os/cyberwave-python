"""
Boston Dynamics Spot Specification

Quadruped robot with advanced mobility and sensing capabilities.
"""

from dataclasses import dataclass
from ...base import DeviceSpec, Capability, Protocol, ConnectionInfo, SetupWizardField


@dataclass
class BostonDynamicsSpotSpec(DeviceSpec):
    """Boston Dynamics Spot quadruped robot specification"""
    
    def __post_init__(self):
        # Core identification
        self.id = "boston-dynamics/spot"
        self.name = "Boston Dynamics Spot"
        self.category = "quadruped"
        self.manufacturer = "Boston Dynamics"
        self.model = "Spot"
        self.description = "Advanced quadruped robot with autonomous navigation"
        
        # Capabilities
        self.capabilities = [
            Capability(
                name="mobility",
                commands=["walk", "turn", "sit", "stand", "dance", "navigate_to"],
                description="Locomotion and movement"
            ),
            Capability(
                name="perception",
                commands=["get_camera_feed", "detect_objects", "map_environment"],
                description="Vision and sensing"
            )
        ]
        
        # Communication protocols
        self.protocols = [
            Protocol(
                type="grpc",
                port=443,
                commands=["walk", "turn", "sit", "stand"],
                parameters={"api_version": "3.0", "ssl": True}
            )
        ]
        
        # Connection information
        self.connection = ConnectionInfo(
            type="ethernet",
            default_ip="192.168.80.3",
            setup_instructions=[
                "Connect to Spot's WiFi network or ethernet",
                "Obtain API credentials from Boston Dynamics",
                "Configure SSL certificates"
            ]
        )
        
        # Setup wizard
        self.setup_wizard = [
            SetupWizardField(
                name="name",
                type="string", 
                label="Device Name",
                default="Spot Robot"
            ),
            SetupWizardField(
                name="ip_address",
                type="ipv4",
                label="IP Address",
                default="192.168.80.3"
            )
        ]
        
        # Technical specifications
        self.specs = {
            "weight": 32.7,  # kg
            "payload": 14,   # kg
            "speed": 1.6,    # m/s
            "runtime": 90,   # minutes
            "operating_temp": {"min": -20, "max": 45}  # celsius
        }
        
        super().__post_init__()
