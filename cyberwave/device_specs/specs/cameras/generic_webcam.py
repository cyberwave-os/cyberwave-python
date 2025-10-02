"""
Generic Webcam Specification

Webcam that can connect via browser (web) or via node on the same machine.
Demonstrates connectivity_modes with web and node options.
"""

from dataclasses import dataclass
from ...base import DeviceSpec, Capability, Protocol, ConnectionInfo, SetupWizardField


@dataclass
class GenericWebcamSpec(DeviceSpec):
    """Generic Laptop/USB Webcam specification"""

    def __post_init__(self):
        # Core identification
        self.id = "generic/webcam"
        self.name = "Laptop / USB Webcam"
        self.category = "camera"
        self.manufacturer = "Generic"
        self.model = "Webcam"
        self.description = "Webcam that can connect via browser (web) or via node on the same machine"

        # Software-defined capabilities
        self.has_hardware_driver = True
        self.has_digital_asset = False
        self.has_simulation_model = False

        # Capabilities
        self.capabilities = [
            Capability(name="video", commands=["start_stream", "stop_stream"], description="Video streaming"),
            Capability(name="snapshot", commands=["capture_snapshot"], description="Capture still image")
        ]

        # Protocols
        self.protocols = [
            Protocol(type="webrtc"),
            Protocol(type="mjpeg")
        ]

        # Connection info (generic)
        self.connection = ConnectionInfo(type="network")

        # Connectivity modes
        self.connectivity_modes = [
            {
                "mode": "web",
                "transport": "webrtc",
                "description": "Capture via browser getUserMedia and stream to backend",
                "requires_node": False,
                "config_fields": [
                    {"name": "resolution", "type": "string", "placeholder": "1280x720", "required": False},
                    {"name": "fps", "type": "number", "placeholder": 30, "required": False}
                ]
            },
            {
                "mode": "node",
                "transport": "mjpeg_proxy",
                "description": "Capture via node (ffmpeg/opencv) and expose streams over edge video proxy",
                "requires_node": True,
                "config_fields": [
                    {"name": "device_path", "type": "string", "placeholder": "/dev/video0", "required": False},
                    {"name": "camera_index", "type": "number", "placeholder": 0, "required": False}
                ]
            }
        ]

        # Setup wizard
        self.setup_wizard = [
            SetupWizardField(
                name="connectivity_mode",
                type="string",
                label="Connectivity Mode",
                default="web",
                required=True,
                help_text="Choose 'web' for browser capture or 'node' for edge node proxy"
            )
        ]

        # Specs and docs
        self.specs = {"max_resolution": "1920x1080"}
        self.documentation_url = "https://docs.cyberwave.com/devices/webcam"
        self.support_url = "https://support.cyberwave.com"

        super().__post_init__()


