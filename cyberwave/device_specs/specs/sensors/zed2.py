"""
Stereolabs ZED 2 Specification

Stereo camera with AI-powered depth sensing and object detection.
"""

from dataclasses import dataclass
from ...base import DeviceSpec, Capability, Protocol, ConnectionInfo, SetupWizardField


@dataclass
class Zed2Spec(DeviceSpec):
    """Stereolabs ZED 2 stereo camera specification"""
    
    def __post_init__(self):
        # Core identification
        self.id = "stereolabs/zed2"
        self.name = "Stereolabs ZED 2"
        self.category = "stereo_camera"
        self.manufacturer = "Stereolabs"
        self.model = "ZED 2"
        self.description = "AI-powered stereo camera with depth sensing and object detection"
        
        # Capabilities
        self.capabilities = [
            Capability(
                name="stereo_vision",
                commands=["capture_stereo", "start_stereo_stream", "stop_stereo_stream"],
                description="Stereo imaging and depth estimation"
            ),
            Capability(
                name="object_detection",
                commands=["detect_objects", "track_objects", "detect_humans"],
                description="AI-powered object detection and tracking"
            ),
            Capability(
                name="spatial_mapping",
                commands=["start_mapping", "stop_mapping", "get_mesh"],
                description="Real-time 3D mapping"
            )
        ]
        
        # Communication protocols
        self.protocols = [
            Protocol(
                type="usb",
                port="USB 3.0",
                commands=["capture_stereo", "detect_objects"],
                parameters={"interface": "zed_sdk"}
            )
        ]
        
        # Connection information
        self.connection = ConnectionInfo(
            type="usb",
            setup_instructions=[
                "Connect ZED 2 to USB 3.0 port",
                "Install ZED SDK from Stereolabs",
                "Calibrate camera if needed"
            ]
        )
        
        # Setup wizard
        self.setup_wizard = [
            SetupWizardField(
                name="name",
                type="string",
                label="Camera Name",
                default="ZED 2"
            ),
            SetupWizardField(
                name="resolution",
                type="select",
                label="Resolution",
                options=["672x376", "1280x720", "1920x1080", "2208x1242"],
                default="1280x720"
            ),
            SetupWizardField(
                name="enable_object_detection",
                type="boolean",
                label="Enable Object Detection",
                default=True,
                required=False
            ),
            SetupWizardField(
                name="enable_spatial_mapping",
                type="boolean", 
                label="Enable Spatial Mapping",
                default=False,
                required=False
            )
        ]
        
        # Technical specifications
        self.specs = {
            "resolution": "2208x1242",
            "framerate": 15,
            "baseline": 120,  # mm
            "depth_range": {"min": 0.2, "max": 20},  # meters
            "fov": {"horizontal": 110, "vertical": 70},  # degrees
            "imu": {
                "accelerometer": True,
                "gyroscope": True,
                "magnetometer": True
            },
            "ai_features": {
                "object_detection": True,
                "human_body_tracking": True,
                "spatial_mapping": True
            }
        }
        
        super().__post_init__()
