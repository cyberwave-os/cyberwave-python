"""
Unitree Go1 Specification (stub)

Advanced quadruped robot with autonomous navigation.
"""

from dataclasses import dataclass
from ...base import DeviceSpec, Capability, Protocol, ConnectionInfo, SetupWizardField


@dataclass
class UnitreeGo1Spec(DeviceSpec):
    """Unitree Go1 quadruped robot spec (minimal stub)"""

    def __post_init__(self):
        # Identification
        self.id = "unitree/go1"
        self.name = "Unitree Go1 Quadruped"
        self.category = "quadruped_robot"
        self.manufacturer = "Unitree"
        self.model = "Go1"
        self.description = "Advanced quadruped robot with autonomous navigation"

        # Software-defined capabilities
        self.has_hardware_driver = True
        self.has_digital_asset = True
        self.has_simulation_model = True

        # Capabilities (high-level)
        self.capabilities = [
            Capability(name="walking", commands=["walk_forward", "walk_backward"], description="Locomotion"),
            Capability(name="navigation", commands=["navigate_to"], description="Autonomous navigation"),
            Capability(name="sensors", commands=["get_status"], description="Sensor telemetry")
        ]

        # Protocols (example)
        self.protocols = [
            Protocol(type="ethernet", port=8080, commands=["control"], parameters={"api": "control"}),
            Protocol(type="udp", port=8082, commands=["high_level"], parameters={"rate_hz": 50})
        ]

        # Connection
        self.connection = ConnectionInfo(type="ethernet", default_ip="192.168.123.161")

        # Setup wizard
        self.setup_wizard = [
            SetupWizardField(name="ip_address", type="ipv4", label="Robot IP Address", default="192.168.123.161"),
            SetupWizardField(name="control_mode", type="string", label="Control Mode", default="high_level", required=False)
        ]

        # Specs
        self.specs = {
            "max_speed": "3.5m/s",
            "battery_life": "2.5h",
            "payload": "5kg",
            "dimensions": "645×280×400mm"
        }

        # Docs
        self.documentation_url = "https://docs.cyberwave.com/devices/unitree-go1"
        self.support_url = "https://support.cyberwave.com"

        super().__post_init__()


