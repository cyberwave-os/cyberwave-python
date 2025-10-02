"""Universal Robots UR5e Specification"""

from dataclasses import dataclass

from ...base import DeviceSpec, Capability, ConnectionInfo, Protocol, SetupWizardField


@dataclass
class UniversalRobotsUR5eSpec(DeviceSpec):
    """Six-axis collaborative arm commonly referenced in examples."""

    def __post_init__(self):
        self.id = "ur/ur5e"
        self.name = "Universal Robots UR5e"
        self.category = "robot_arm"
        self.manufacturer = "Universal Robots"
        self.model = "UR5e"
        self.description = "Six-degree-of-freedom collaborative robot arm with standard gripper tooling"

        self.has_hardware_driver = False
        self.has_digital_asset = True
        self.has_simulation_model = True

        self.capabilities = [
            Capability(
                name="manipulation",
                commands=[
                    "arm.move_pose",
                    "arm.move_joints",
                    "manipulator.pick",
                    "manipulator.place",
                    "gripper.open",
                    "gripper.close",
                ],
                description="Core manipulation commands for pick and place workflows",
                metadata={
                    "command_schemas": {
                        "arm.move_pose": {
                            "type": "object",
                            "properties": {
                                "pose": {
                                    "type": "object",
                                    "properties": {
                                        "x": {"type": "number"},
                                        "y": {"type": "number"},
                                        "z": {"type": "number"},
                                    },
                                    "required": ["x", "y"],
                                }
                            },
                            "required": ["pose"],
                        },
                        "arm.move_joints": {
                            "type": "object",
                            "properties": {
                                "joints": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 6,
                                }
                            },
                            "required": ["joints"],
                        },
                        "manipulator.pick": {
                            "type": "object",
                            "properties": {
                                "object": {
                                    "type": "string",
                                    "description": "Identifier of the object to pick",
                                }
                            },
                            "required": ["object"],
                        },
                        "manipulator.place": {
                            "type": "object",
                            "properties": {
                                "target": {
                                    "type": "object",
                                    "properties": {
                                        "x": {"type": "number"},
                                        "y": {"type": "number"},
                                        "z": {"type": "number"},
                                    },
                                    "required": ["x", "y"],
                                }
                            },
                            "required": ["target"],
                        },
                    }
                },
            ),
            Capability(
                name="telemetry",
                commands=["robot_status", "joint_state", "effort"],
                description="Robot status introspection",
            ),
        ]

        self.protocols = [
            Protocol(
                type="ethernet",
                port=29999,
                commands=["arm.move_joints", "arm.move_pose"],
                parameters={"interface": "ur_rtde"},
            )
        ]

        self.connection = ConnectionInfo(
            type="ethernet",
            default_ip="192.168.0.10",
            setup_instructions=[
                "Connect controller to dedicated Ethernet network",
                "Configure URCap for RTDE streaming",
                "Enable external control program",
            ],
        )

        self.setup_wizard = [
            SetupWizardField(
                name="name",
                type="string",
                label="Robot Name",
                default="UR5e Arm",
            ),
            SetupWizardField(
                name="controller_ip",
                type="ipv4",
                label="Controller IP",
                default="192.168.0.10",
            ),
        ]

        self.simulation_models = ["mujoco", "gazebo"]
        self.extended_capabilities = {"collaborative": True, "force_sensing": True}

        super().__post_init__()
