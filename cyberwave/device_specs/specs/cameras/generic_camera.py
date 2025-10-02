"""
Generic Camera Specification

Fallback specification for unknown IP cameras.
Provides basic camera functionality with generic drivers and assets.
"""

from dataclasses import dataclass
from ...base import DeviceSpec, Capability, Protocol, ConnectionInfo, SetupWizardField


@dataclass
class GenericCameraSpec(DeviceSpec):
    """Generic IP camera specification for unknown devices"""
    
    def __post_init__(self):
        # Core identification
        self.id = "generic/ip-camera"
        self.name = "Generic IP Camera"
        self.category = "ip_camera"
        self.manufacturer = "Generic"
        self.model = "IP Camera"
        self.description = "Generic IP camera with RTSP/HTTP streaming support"
        
        # Software-defined capabilities
        self.has_hardware_driver = True  # Generic driver available
        self.has_digital_asset = True   # Generic camera asset
        self.has_simulation_model = False  # No simulation for generic
        
        # Implementation details
        self.driver_class = "cyberwave_cli.drivers.ip_camera.IPCameraDriver"
        self.asset_class = "cyberwave.assets.GenericIPCamera"
        self.simulation_models = []
        self.fallback_asset_class = "cyberwave.assets.GenericCamera"
        
        # Extended capabilities
        self.extended_capabilities = {
            "has_onvif_support": True,
            "has_ptz_control": False,  # Unknown until detected
            "has_audio": False
        }
        
        # Capabilities
        self.capabilities = [
            Capability(
                name="video_streaming",
                commands=["start_stream", "stop_stream", "get_stream_url"],
                description="Basic video streaming via RTSP/HTTP",
                metadata={
                    "binding_template": {
                        "required": True,
                        "default_channel": "video",
                        "available_channels": ["video"],
                        "supports_multiple": False,
                        "requires_sensor": True,
                        "sensor_type": "camera",
                        "sensor_name": "Primary Camera",
                        "sensor_metadata_schema": {
                            "type": "object",
                            "required": ["mount_position"],
                            "properties": {
                                "mount_position": {
                                    "type": "string",
                                    "description": "Where the camera is mounted (e.g., gimbal, nose)"
                                }
                            }
                        },
                        "config_schema": {
                            "type": "object",
                            "required": ["rtsp_url"],
                            "properties": {
                                "rtsp_url": {
                                    "type": "string",
                                    "description": "RTSP URL or network endpoint exposed by the camera"
                                },
                                "username": {
                                    "type": "string",
                                    "description": "Camera username"
                                },
                                "password": {
                                    "type": "string",
                                    "description": "Camera password",
                                    "secret": True
                                }
                            }
                        }
                    }
                }
            ),
            Capability(
                name="image_capture",
                commands=["capture_image", "get_snapshot"],
                description="Still image capture",
                metadata={
                    "binding_template": {
                        "required": False,
                        "default_channel": "snapshot",
                        "available_channels": ["snapshot"],
                        "supports_multiple": False,
                        "requires_sensor": True,
                        "sensor_type": "camera",
                        "config_schema": {
                            "type": "object",
                            "properties": {
                                "exposure": {
                                    "type": "integer",
                                    "description": "Optional override for exposure setting"
                                }
                            }
                        }
                    }
                }
            ),
            Capability(
                name="configuration",
                commands=["get_config", "set_config", "detect_capabilities"],
                description="Camera configuration and capability detection",
                metadata={
                    "binding_template": {
                        "required": False,
                        "supports_multiple": False,
                        "requires_sensor": False
                    }
                }
            )
        ]
        
        # Communication protocols
        self.protocols = [
            Protocol(
                type="rtsp",
                port=554,
                commands=["start_stream", "get_stream_url"],
                parameters={"auth_required": True}
            ),
            Protocol(
                type="http",
                port=80,
                commands=["capture_image", "get_snapshot", "get_config"],
                parameters={"api_type": "rest"}
            ),
            Protocol(
                type="onvif",
                port=80,
                commands=["detect_capabilities", "get_device_info"],
                parameters={"discovery": True}
            )
        ]
        
        # Connection information
        self.connection = ConnectionInfo(
            type="ethernet",
            default_ip="192.168.1.100",
            setup_instructions=[
                "Connect camera to network",
                "Find camera IP address (use network scanner if needed)",
                "Test RTSP stream: rtsp://ip:554/stream1",
                "Configure authentication if required"
            ]
        )
        
        # Setup wizard for generic cameras
        self.setup_wizard = [
            SetupWizardField(
                name="name",
                type="string",
                label="Camera Name",
                default="Generic Camera",
                help_text="Friendly name for this camera"
            ),
            SetupWizardField(
                name="ip_address",
                type="ipv4",
                label="Camera IP Address",
                default="192.168.1.100",
                help_text="IP address of the camera"
            ),
            SetupWizardField(
                name="username",
                type="string",
                label="Username",
                default="admin",
                required=False,
                help_text="Camera authentication username"
            ),
            SetupWizardField(
                name="password",
                type="string",
                label="Password",
                default="",
                required=False,
                help_text="Camera authentication password"
            ),
            SetupWizardField(
                name="rtsp_path",
                type="string",
                label="RTSP Stream Path",
                default="/stream1",
                help_text="RTSP stream path (e.g., /stream1, /live/main)"
            ),
            SetupWizardField(
                name="resolution",
                type="select",
                label="Resolution",
                options=["640x480", "1280x720", "1920x1080"],
                default="1920x1080",
                required=False,
                help_text="Video resolution (will be auto-detected if possible)"
            ),
            SetupWizardField(
                name="auto_detect",
                type="boolean",
                label="Auto-detect Capabilities",
                default=True,
                required=False,
                help_text="Automatically detect camera capabilities via ONVIF"
            )
        ]
        
        # Technical specifications (generic)
        self.specs = {
            "resolution": ["720p", "1080p"],
            "framerate": {"min": 15, "max": 30},
            "protocols": ["RTSP", "HTTP", "ONVIF"],
            "compression": ["H.264", "MJPEG"],
            "features": {
                "night_vision": "unknown",
                "ptz": "unknown",
                "audio": "unknown",
                "motion_detection": "unknown"
            }
        }
        
        # Documentation
        self.documentation_url = "https://docs.cyberwave.com/devices/generic-camera"
        self.support_url = "https://support.cyberwave.com/generic-camera"
        
        super().__post_init__()
