"""
KUKA KR3 Industrial Arm Specification

Industrial robotic arm for precision manufacturing tasks.
"""

from dataclasses import dataclass
from ...base import DeviceSpec, Capability, Protocol, ConnectionInfo, SetupWizardField


@dataclass 
class KukaKr3Spec(DeviceSpec):
    """KUKA KR3 industrial robotic arm specification"""
    
    def __post_init__(self):
        # Core identification
        self.id = "kuka/kr3"
        self.name = "KUKA KR3"
        self.category = "industrial_arm"
        self.manufacturer = "KUKA"
        self.model = "KR3"
        self.description = "Precision industrial robotic arm for manufacturing"
        
        # Capabilities
        self.capabilities = [
            Capability(
                name="manipulation",
                commands=["move_joints", "move_linear", "move_ptp", "home"],
                description="Precision movement and positioning"
            ),
            Capability(
                name="io",
                commands=["set_digital_out", "get_digital_in", "set_analog_out"],
                description="Industrial I/O control"
            )
        ]
        
        # Communication protocols
        self.protocols = [
            Protocol(
                type="ethernet_ip",
                port=44818,
                commands=["move_joints", "move_linear"],
                parameters={"plc_compatible": True}
            )
        ]
        
        # Connection information
        self.connection = ConnectionInfo(
            type="ethernet",
            default_ip="192.168.1.100",
            setup_instructions=[
                "Connect robot controller to network",
                "Configure IP address via teach pendant",
                "Enable external control mode"
            ]
        )
        
        # Setup wizard
        self.setup_wizard = [
            SetupWizardField(
                name="name",
                type="string",
                label="Device Name", 
                default="KUKA KR3"
            ),
            SetupWizardField(
                name="ip_address",
                type="ipv4",
                label="Controller IP",
                default="192.168.1.100"
            )
        ]
        
        # Technical specifications
        self.specs = {
            "dof": 6,
            "payload": 3,     # kg
            "reach": 635,     # mm
            "repeatability": 0.02,  # mm
            "max_speed": 10   # m/s
        }
        
        super().__post_init__()
