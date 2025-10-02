"""
Generic NVR Specification

Network Video Recorder supporting multiple camera streams.
"""

from dataclasses import dataclass
from ...base import DeviceSpec, Capability, Protocol, ConnectionInfo, SetupWizardField


@dataclass
class GenericNVRSpec(DeviceSpec):
    """Generic NVR with multi-stream support"""

    def __post_init__(self):
        # Core identification
        self.id = "generic/nvr"
        self.name = "Generic NVR (Multi-Stream)"
        self.category = "nvr"
        self.manufacturer = "Generic"
        self.model = "NVR"
        self.description = "Network Video Recorder supporting multiple camera streams"

        # Software-defined capabilities
        self.has_hardware_driver = True
        self.has_digital_asset = False
        self.has_simulation_model = False

        # Capabilities
        self.capabilities = [
            Capability(name="video_multistream", commands=["refresh_streams"], description="Multiple channels"),
            Capability(name="video", commands=["start_stream", "stop_stream"], description="Video streaming"),
            Capability(name="snapshot", commands=["capture_snapshot"], description="Still image capture")
        ]

        # Protocols
        self.protocols = [
            Protocol(type="http", port=80, commands=["list_channels"], parameters={"api": "nvr"}),
            Protocol(type="rtsp", port=554, commands=["get_stream"], parameters={"per_channel": True})
        ]

        # Connection
        self.connection = ConnectionInfo(type="network", default_ip="192.168.1.20")

        # Specs
        self.specs = {"max_channels": 16, "stream_format": "H.264/H.265"}

        # Setup wizard
        self.setup_wizard = [
            SetupWizardField(name="host", type="string", label="NVR Host/IP", default="192.168.1.20"),
            SetupWizardField(name="username", type="string", label="Username", default="admin"),
            SetupWizardField(name="password", type="string", label="Password", default=""),
            SetupWizardField(name="channels", type="int", label="Channel Count", default=8, required=False)
        ]

        # Docs
        self.documentation_url = "https://docs.cyberwave.com/devices/generic-nvr"
        self.support_url = "https://support.cyberwave.com"

        super().__post_init__()


