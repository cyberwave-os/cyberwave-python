"""
SO-101 Robotic Arm Specification

Leader-follower teleoperation robotic arm with gripper.
"""

from dataclasses import dataclass
from ...base import DeviceSpec, Capability, Protocol, ConnectionInfo, SetupWizardField, DependencySpec


@dataclass
class So101Spec(DeviceSpec):
    """SO-101 robotic arm specification"""
    
    def __post_init__(self):
        # Core identification
        self.id = "cyberwave/so101"
        self.name = "SO-101 Robotic Arm"
        self.category = "robotic_arm"
        self.manufacturer = "Standard Robots"
        self.model = "SO-101"
        self.description = "Leader-follower teleoperation robotic arm with 6 DOF and gripper"
        
        # Capabilities with UI metadata for software-defined UI
        self.capabilities = [
            Capability(
                name="manipulation",
                commands=[
                    "move_joints", "move_pose", "move_to", "home", 
                    "calibrate", "get_position", "get_joints"
                ],
                description="Arm movement and positioning",
                ui_metadata={
                    "component_type": "robotic_arm_controls",
                    "icon": "robot",
                    "controls": [
                        {
                            "type": "button",
                            "label": "Home Position",
                            "command": "home",
                            "variant": "primary",
                            "icon": "home"
                        },
                        {
                            "type": "button",
                            "label": "Calibrate",
                            "command": "calibrate",
                            "variant": "secondary",
                            "icon": "settings"
                        },
                        {
                            "type": "joystick",
                            "label": "Manual Control",
                            "command": "move_pose",
                            "axes": ["x", "y", "z", "rx", "ry", "rz"],
                            "sensitivity": 0.1
                        }
                    ],
                    "metrics": [
                        {
                            "name": "Joint Positions",
                            "key": "joint_positions",
                            "type": "array",
                            "unit": "degrees"
                        },
                        {
                            "name": "End Effector Position",
                            "key": "end_effector_pose",
                            "type": "pose",
                            "unit": "mm"
                        }
                    ]
                },
                metadata={
                    "binding_template": {
                        "required": True,
                        "default_channel": "actuation",
                        "available_channels": ["actuation"],
                        "supports_multiple": False,
                        "requires_sensor": False,
                        "config_schema": {
                            "type": "object",
                            "required": ["device_port"],
                            "properties": {
                                "device_port": {
                                    "type": "string",
                                    "description": "Serial or USB port connected to the SO-101 controller"
                                },
                                "baudrate": {
                                    "type": "integer",
                                    "description": "Override serial baudrate",
                                    "default": 115200
                                }
                            }
                        }
                    }
                }
            ),
            Capability(
                name="gripper",
                commands=["open_gripper", "close_gripper", "set_gripper", "get_gripper_state"],
                description="Gripper control",
                ui_metadata={
                    "component_type": "gripper_controls",
                    "icon": "hand",
                    "controls": [
                        {
                            "type": "button",
                            "label": "Open Gripper",
                            "command": "open_gripper",
                            "variant": "outline",
                            "icon": "maximize"
                        },
                        {
                            "type": "button",
                            "label": "Close Gripper",
                            "command": "close_gripper",
                            "variant": "outline",
                            "icon": "minimize"
                        },
                        {
                            "type": "slider",
                            "label": "Gripper Position",
                            "command": "set_gripper",
                            "min": 0,
                            "max": 85,
                            "unit": "mm"
                        }
                    ],
                    "metrics": [
                        {
                            "name": "Gripper Position",
                            "key": "gripper_position",
                            "type": "value",
                            "unit": "mm"
                        },
                        {
                            "name": "Gripper Force",
                            "key": "gripper_force",
                            "type": "progress",
                            "unit": "N",
                            "thresholds": {"warning": 15, "critical": 18, "good": 10}
                        }
                    ]
                },
                metadata={
                    "binding_template": {
                        "required": True,
                        "default_channel": "gripper",
                        "available_channels": ["gripper"],
                        "supports_multiple": False,
                        "requires_sensor": False,
                        "config_schema": {
                            "type": "object",
                            "required": ["device_port"],
                            "properties": {
                                "device_port": {
                                    "type": "string",
                                    "description": "Serial port used for gripper control board"
                                }
                            }
                        }
                    }
                }
            ),
            Capability(
                name="teleoperation", 
                commands=[
                    "start_leader_follower", "stop_leader_follower", 
                    "calibrate_leader", "calibrate_follower", "sync_arms"
                ],
                description="Leader-follower teleoperation",
                ui_metadata={
                    "component_type": "teleoperation_controls",
                    "icon": "link",
                    "controls": [
                        {
                            "type": "button",
                            "label": "Start Teleoperation",
                            "command": "start_leader_follower",
                            "variant": "primary",
                            "icon": "play",
                            "enabled_when": ["torque_enabled", "calibrated"]
                        },
                        {
                            "type": "button",
                            "label": "Stop Teleoperation",
                            "command": "stop_leader_follower",
                            "variant": "destructive",
                            "icon": "square",
                            "enabled_when": ["teleoperation_active"]
                        },
                        {
                            "type": "button_group",
                            "label": "Calibration",
                            "controls": [
                                {
                                    "label": "Calibrate Leader",
                                    "command": "calibrate_leader",
                                    "icon": "target"
                                },
                                {
                                    "label": "Calibrate Follower", 
                                    "command": "calibrate_follower",
                                    "icon": "target"
                                }
                            ]
                        }
                    ],
                    "status_indicators": [
                        {
                            "name": "Teleoperation Status",
                            "key": "teleoperation_active",
                            "type": "boolean",
                            "labels": {"true": "Active", "false": "Inactive"}
                        },
                        {
                            "name": "Sync Quality",
                            "key": "sync_quality",
                            "type": "progress",
                            "unit": "%",
                            "thresholds": {"good": 90, "warning": 70, "critical": 50}
                        }
                    ]
                }
            ),
            Capability(
                name="safety",
                commands=["emergency_stop", "enable_torque", "disable_torque", "get_safety_status"],
                description="Safety and emergency controls",
                ui_metadata={
                    "component_type": "safety_controls",
                    "icon": "alert-triangle",
                    "controls": [
                        {
                            "type": "button",
                            "label": "EMERGENCY STOP",
                            "command": "emergency_stop",
                            "variant": "destructive",
                            "icon": "alert-triangle",
                            "size": "lg"
                        },
                        {
                            "type": "toggle",
                            "label": "Torque Enable",
                            "command": "enable_torque",
                            "off_command": "disable_torque",
                            "icon": "power"
                        }
                    ],
                    "metrics": [
                        {
                            "name": "Safety Status",
                            "key": "safety_status",
                            "type": "status",
                            "values": ["safe", "warning", "emergency"]
                        },
                        {
                            "name": "Torque Status",
                            "key": "torque_enabled",
                            "type": "boolean",
                            "labels": {"true": "Enabled", "false": "Disabled"}
                        }
                    ]
                }
            ),
            Capability(
                name="telemetry",
                commands=["get_joint_states", "get_pose", "get_forces"],
                description="Real-time telemetry and status",
                ui_metadata={
                    "component_type": "sensor_metrics",
                    "icon": "activity",
                    "metrics": [
                        {
                            "name": "Joint Temperatures",
                            "key": "joint_temperatures",
                            "type": "array",
                            "unit": "°C",
                            "thresholds": {"warning": 60, "critical": 80, "good": 40}
                        },
                        {
                            "name": "Motor Currents",
                            "key": "motor_currents",
                            "type": "array", 
                            "unit": "A",
                            "thresholds": {"warning": 2.0, "critical": 2.3, "good": 1.5}
                        },
                        {
                            "name": "Communication Latency",
                            "key": "communication_latency",
                            "type": "value",
                            "unit": "ms"
                        }
                    ]
                }
            )
        ]
        
        # Communication protocols
        self.protocols = [
            Protocol(
                type="serial",
                port="/dev/ttyACM0",
                baudrate=115200,
                commands=["move_joints", "get_position", "calibrate"],
                parameters={
                    "timeout": 1.0,
                    "motor_protocol": "feetech_scs",
                    "motor_ids": [1, 2, 3, 4, 5, 6, 7]  # 6 arm joints + gripper
                }
            ),
            Protocol(
                type="serial",
                port="/dev/ttyACM1", 
                baudrate=115200,
                commands=["move_joints", "get_position", "calibrate"],
                parameters={
                    "timeout": 1.0,
                    "motor_protocol": "feetech_scs",
                    "motor_ids": [1, 2, 3, 4, 5, 6, 7],
                    "role": "follower"
                }
            )
        ]
        
        # Connection information
        self.connection = ConnectionInfo(
            type="serial",
            default_port="/dev/ttyACM0",
            setup_instructions=[
                "Connect leader arm to USB port (usually /dev/ttyACM0)",
                "Connect follower arm to second USB port (usually /dev/ttyACM1)", 
                "Ensure both arms are powered on",
                "Check that motor IDs are properly configured",
                "Run calibration sequence for both arms"
            ]
        )
        
        # Setup wizard
        self.setup_wizard = [
            SetupWizardField(
                name="name",
                type="string",
                label="Device Name",
                default="SO-101 Arm",
                help_text="Friendly name for this robotic arm"
            ),
            SetupWizardField(
                name="leader_port",
                type="string",
                label="Leader Arm Port",
                default="/dev/ttyACM0",
                help_text="Serial port for the leader arm"
            ),
            SetupWizardField(
                name="follower_port", 
                type="string",
                label="Follower Arm Port",
                default="/dev/ttyACM1",
                help_text="Serial port for the follower arm"
            ),
            SetupWizardField(
                name="enable_teleoperation",
                type="boolean",
                label="Enable Teleoperation",
                default=True,
                required=False,
                help_text="Enable leader-follower teleoperation mode"
            ),
            SetupWizardField(
                name="safety_limits",
                type="boolean", 
                label="Enable Safety Limits",
                default=True,
                required=False,
                help_text="Enable joint angle and velocity safety limits"
            ),
            SetupWizardField(
                name="calibration_mode",
                type="select",
                label="Calibration Mode",
                options=["auto", "manual", "skip"],
                default="auto",
                help_text="Calibration method for arm setup"
            )
        ]
        
        # Technical specifications
        self.specs = {
            "dof": 6,  # degrees of freedom
            "payload": 1.0,  # kg
            "reach": 600,  # mm
            "repeatability": 1.0,  # mm
            "joint_limits": {
                "shoulder_pan": [-180, 180],      # degrees
                "shoulder_lift": [-90, 90],
                "elbow": [-150, 150],
                "wrist_1": [-180, 180],
                "wrist_2": [-90, 90], 
                "wrist_3": [-180, 180]
            },
            "joint_velocities": {
                "max_velocity": 180,  # degrees/second
                "default_velocity": 45
            },
            "gripper": {
                "type": "parallel",
                "max_opening": 85,  # mm
                "force": 20,  # N
                "precision": 0.1  # mm
            },
            "motors": {
                "type": "Feetech SCS series",
                "protocol": "Serial Control System",
                "voltage": 12,  # V
                "current_limit": 2.5  # A per motor
            },
            "communication": {
                "baudrate": 115200,
                "protocol": "Serial",
                "latency": 10  # ms
            }
        }
        
        # Implementation details
        self.driver_class = "cyberwave_cli.drivers.so101.SO101Driver"
        self.asset_class = "cyberwave.assets.SO101Robot"
        self.simulation_models = ["gazebo", "mujoco"]
        self.fallback_asset_class = "cyberwave.assets.GenericRoboticArm"
        
        # Dependencies for hardware driver
        self.dependencies = [
            DependencySpec(
                name="lerobot",
                package="lerobot",
                description="LeRobot framework for advanced robot control and learning"
            ),
            DependencySpec(
                name="pyserial",
                package="pyserial", 
                description="Serial communication library for hardware interface"
            ),
            DependencySpec(
                name="numpy",
                package="numpy",
                description="Numerical computing library for robotics calculations"
            )
        ]
        
        # Documentation
        self.documentation_url = "https://docs.cyberwave.com/devices/so101"
        self.support_url = "https://support.cyberwave.com/so101"
        
        super().__post_init__()
