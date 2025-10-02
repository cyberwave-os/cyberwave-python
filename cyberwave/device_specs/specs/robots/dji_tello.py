"""
DJI Tello Drone Specification

Educational drone with WiFi control and camera capabilities.
"""

from dataclasses import dataclass
from ...base import DeviceSpec, Capability, Protocol, ConnectionInfo, SetupWizardField, DependencySpec


@dataclass
class DjiTelloSpec(DeviceSpec):
    """DJI Tello educational drone specification"""
    
    def __post_init__(self):
        # Core identification
        self.id = "dji/tello"
        self.name = "DJI Tello"
        self.category = "drone"
        self.manufacturer = "DJI"
        self.model = "Tello"
        self.description = "Educational drone with WiFi control and HD camera"
        
        # Software-defined capabilities
        self.has_hardware_driver = True
        self.has_digital_asset = True
        self.has_simulation_model = True
        
        # Implementation details
        self.driver_class = "cyberwave_cli.drivers.tello.TelloDriver"
        self.asset_class = "cyberwave.assets.DjiTello"
        self.simulation_models = ["gazebo", "airsim"]
        self.fallback_asset_class = "cyberwave.assets.GenericDrone"
        
        # Extended capabilities for future use
        self.extended_capabilities = {
            "has_ros_driver": False,
            "has_unity_model": False,
            "has_mobile_app": True
        }
        
        # Capabilities with UI metadata
        self.capabilities = [
            Capability(
                name="flight",
                commands=[
                    "takeoff", "land", "emergency", "up", "down", 
                    "left", "right", "forward", "back", "cw", "ccw",
                    "flip", "go", "curve", "speed"
                ],
                description="Flight control and movement commands",
                ui_metadata={
                    "component_type": "flight_controller",
                    "layout": "grid",
                    "controls": [
                        {
                            "id": "takeoff",
                            "type": "button",
                            "label": "Takeoff",
                            "icon": "plane",
                            "variant": "primary",
                            "command": "takeoff",
                            "requires_confirmation": True,
                            "disabled_when": ["flying"]
                        },
                        {
                            "id": "land", 
                            "type": "button",
                            "label": "Land",
                            "icon": "plane-landing",
                            "variant": "secondary",
                            "command": "land",
                            "enabled_when": ["flying"]
                        },
                        {
                            "id": "emergency",
                            "type": "button", 
                            "label": "EMERGENCY",
                            "icon": "alert-triangle",
                            "variant": "destructive",
                            "command": "emergency",
                            "enabled_when": ["flying"]
                        },
                        {
                            "id": "movement_pad",
                            "type": "directional_pad",
                            "commands": {
                                "up": {"command": "forward", "params": {"distance": 50}},
                                "down": {"command": "back", "params": {"distance": 50}},
                                "left": {"command": "left", "params": {"distance": 50}},
                                "right": {"command": "right", "params": {"distance": 50}}
                            },
                            "enabled_when": ["flying"]
                        },
                        {
                            "id": "altitude_controls",
                            "type": "button_group",
                            "controls": [
                                {"command": "up", "params": {"distance": 30}, "label": "↑", "icon": "arrow-up"},
                                {"command": "down", "params": {"distance": 30}, "label": "↓", "icon": "arrow-down"}
                            ],
                            "enabled_when": ["flying"]
                        },
                        {
                            "id": "rotation_controls", 
                            "type": "button_group",
                            "controls": [
                                {"command": "cw", "params": {"degrees": 90}, "label": "CW", "icon": "rotate-cw"},
                                {"command": "ccw", "params": {"degrees": 90}, "label": "CCW", "icon": "rotate-ccw"}
                            ],
                            "enabled_when": ["flying"]
                        }
                    ]
                }
            ),
            Capability(
                name="video_streaming",
                commands=["streamon", "streamoff"],
                description="Real-time video streaming from drone camera",
                ui_metadata={
                    "component_type": "video_stream",
                    "stream_config": {
                        "protocol": "udp",
                        "port": 11111,
                        "resolution": "720p", 
                        "format": "h264"
                    },
                    "controls": [
                        {
                            "id": "stream_toggle",
                            "type": "toggle_button",
                            "labels": {"on": "Stop Stream", "off": "Start Stream"},
                            "icons": {"on": "square", "off": "play"},
                            "commands": {"on": "streamoff", "off": "streamon"}
                        },
                        {
                            "id": "fullscreen",
                            "type": "button",
                            "label": "Fullscreen",
                            "icon": "maximize",
                            "action": "toggle_fullscreen"
                        }
                    ]
                }
            ),
            Capability(
                name="camera",
                commands=["photo", "video_start", "video_stop"],
                description="Camera capture and recording",
                ui_metadata={
                    "component_type": "camera_controls",
                    "controls": [
                        {
                            "id": "take_photo",
                            "type": "button",
                            "label": "Photo",
                            "icon": "camera",
                            "command": "photo"
                        }
                    ]
                }
            ),
            Capability(
                name="telemetry",
                commands=["battery?", "speed?", "time?", "height?", "temp?", "attitude?", "baro?", "acceleration?", "tof?", "wifi?"],
                description="Real-time status and sensor readings",
                ui_metadata={
                    "component_type": "telemetry_display",
                    "update_frequency": 10,
                    "layout": "grid",
                    "metrics": [
                        {
                            "id": "battery",
                            "label": "Battery",
                            "icon": "battery",
                            "type": "progress",
                            "unit": "%",
                            "thresholds": {"critical": 10, "warning": 20, "good": 50},
                            "command": "battery?"
                        },
                        {
                            "id": "height",
                            "label": "Height", 
                            "icon": "arrow-up",
                            "type": "value",
                            "unit": "cm",
                            "command": "height?"
                        },
                        {
                            "id": "temperature",
                            "label": "Temperature",
                            "icon": "thermometer",
                            "type": "range",
                            "unit": "°C", 
                            "command": "temp?"
                        },
                        {
                            "id": "attitude",
                            "label": "Attitude",
                            "icon": "navigation",
                            "type": "attitude_display",
                            "fields": ["pitch", "roll", "yaw"],
                            "unit": "°",
                            "command": "attitude?"
                        },
                        {
                            "id": "wifi_signal",
                            "label": "WiFi",
                            "icon": "wifi",
                            "type": "signal_strength",
                            "command": "wifi?"
                        }
                    ]
                }
            )
        ]
        
        # Communication protocols
        self.protocols = [
            Protocol(
                type="udp",
                port=8889,
                commands=["takeoff", "land", "emergency", "up", "down", "left", "right", "forward", "back"],
                parameters={"timeout": 5.0, "retry_count": 3}
            ),
            Protocol(
                type="udp", 
                port=8890,
                commands=["streamon", "streamoff"],
                parameters={"video_stream": True}
            ),
            Protocol(
                type="udp",
                port=11111,
                commands=["video_data"],
                parameters={"stream_type": "h264", "buffer_size": 65536}
            )
        ]
        
        # Connection information
        self.connection = ConnectionInfo(
            type="wifi",
            default_ip="192.168.10.1",
            setup_instructions=[
                "Power on the Tello drone",
                "Connect to Tello WiFi network (TELLO-XXXXXX)",
                "Wait for connection to establish",
                "Test connection with 'command' message"
            ]
        )
        
        # Setup wizard
        self.setup_wizard = [
            SetupWizardField(
                name="name",
                type="string",
                label="Device Name",
                default="My Tello",
                help_text="Friendly name for this Tello drone"
            ),
            SetupWizardField(
                name="ip_address",
                type="ipv4",
                label="IP Address", 
                default="192.168.10.1",
                help_text="IP address of the Tello drone (usually 192.168.10.1)"
            ),
            SetupWizardField(
                name="enable_video",
                type="boolean",
                label="Enable Video Stream",
                default=True,
                required=False,
                help_text="Enable video streaming from drone camera"
            ),
            SetupWizardField(
                name="max_height",
                type="int",
                label="Maximum Flight Height (m)",
                default=10,
                required=False,
                help_text="Maximum allowed flight height in meters"
            )
        ]
        
        # Technical specifications
        self.specs = {
            "weight": 80,  # grams
            "dimensions": {"length": 98, "width": 92.5, "height": 41},  # mm
            "max_flight_time": 13,  # minutes
            "max_flight_distance": 100,  # meters
            "max_speed": 8,  # m/s
            "max_altitude": 10,  # meters
            "operating_temperature": {"min": 0, "max": 40},  # celsius
            "camera": {
                "resolution": "720p",
                "fov": 82.6,  # degrees
                "format": "JPEG/MP4"
            },
            "battery": {
                "capacity": 1100,  # mAh
                "voltage": 3.8,  # V
                "type": "LiPo"
            },
            "connectivity": {
                "wifi_standard": "802.11n",
                "frequency": "2.4GHz",
                "range": 100  # meters
            }
        }
        
        # Dependencies for hardware driver
        self.dependencies = [
            DependencySpec(
                package="djitellopy",
                name="DJI Tello Python Library",
                version=">=2.5.0",
                optional=False,
                install_command="pip install djitellopy",
                description="Official DJI Tello SDK wrapper for Python",
                fallback_message="Tello driver requires djitellopy for reliable communication"
            ),
            DependencySpec(
                package="opencv-python",
                name="OpenCV Python",
                version=">=4.5.0", 
                optional=True,
                install_command="pip install opencv-python",
                description="Computer vision library for video processing",
                fallback_message="Video streaming features will be limited without OpenCV"
            ),
            DependencySpec(
                package="numpy",
                name="NumPy",
                version=">=1.20.0",
                optional=True,
                install_command="pip install numpy",
                description="Numerical computing library for telemetry processing",
                fallback_message="Advanced telemetry features require NumPy"
            )
        ]
        
        # Documentation
        self.documentation_url = "https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf"
        self.support_url = "https://www.ryzerobotics.com/tello/support"
        
        super().__post_init__()
