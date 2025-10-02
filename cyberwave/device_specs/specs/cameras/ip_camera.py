"""
Generic IP Camera Specification

Generic specification for IP cameras with RTSP/HTTP streaming.
"""

from dataclasses import dataclass
from ...base import DeviceSpec, Capability, Protocol, ConnectionInfo, SetupWizardField


@dataclass
class IPCameraSpec(DeviceSpec):
    """Generic IP camera specification"""
    
    def __post_init__(self):
        # Core identification
        self.id = "generic/ip-camera"
        self.name = "Generic IP Camera"
        self.category = "ip_camera"
        self.manufacturer = "Generic"
        self.model = "IP Camera"
        self.description = "Generic IP camera with RTSP/HTTP streaming support"
        
        # Capabilities
        self.capabilities = [
            Capability(
                name="video_streaming",
                commands=["start_stream", "stop_stream", "get_stream_url"],
                description="Video streaming via RTSP/HTTP"
            ),
            Capability(
                name="ptz_control",
                commands=["pan", "tilt", "zoom", "preset_goto", "preset_set"],
                description="Pan-Tilt-Zoom control (if supported)"
            ),
            Capability(
                name="image_capture",
                commands=["capture_image", "get_snapshot"],
                description="Still image capture"
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
                commands=["capture_image", "get_snapshot"],
                parameters={"api_type": "rest"}
            )
        ]
        
        # Connection information
        self.connection = ConnectionInfo(
            type="ethernet",
            default_ip="192.168.1.100",
            setup_instructions=[
                "Connect camera to network",
                "Access camera web interface",
                "Configure network settings",
                "Test RTSP stream"
            ]
        )
        
        # Setup wizard
        self.setup_wizard = [
            SetupWizardField(
                name="name",
                type="string",
                label="Camera Name",
                default="IP Camera"
            ),
            SetupWizardField(
                name="ip_address",
                type="ipv4",
                label="Camera IP Address",
                default="192.168.1.100"
            ),
            SetupWizardField(
                name="username",
                type="string",
                label="Username",
                default="admin"
            ),
            SetupWizardField(
                name="password",
                type="string",
                label="Password",
                default=""
            ),
            SetupWizardField(
                name="rtsp_path",
                type="string",
                label="RTSP Path",
                default="/stream1",
                help_text="RTSP stream path (e.g., /stream1, /live/main)"
            ),
            SetupWizardField(
                name="resolution",
                type="select",
                label="Resolution",
                options=["640x480", "1280x720", "1920x1080", "3840x2160"],
                default="1920x1080"
            ),
            SetupWizardField(
                name="has_ptz",
                type="boolean",
                label="PTZ Support",
                default=False,
                required=False,
                help_text="Does this camera support Pan-Tilt-Zoom?"
            )
        ]
        
        # Technical specifications
        self.specs = {
            "resolution": ["720p", "1080p", "4K"],
            "framerate": {"min": 15, "max": 30},
            "protocols": ["RTSP", "HTTP", "ONVIF"],
            "compression": ["H.264", "H.265"],
            "night_vision": True,
            "weatherproof": "IP66",
            "power": {"poe": True, "dc12v": True}
        }
        
        super().__post_init__()
