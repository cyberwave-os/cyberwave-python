"""
Intel RealSense D435 Specification

RGB-D camera with depth sensing capabilities.
"""

from dataclasses import dataclass
from ...base import DeviceSpec, Capability, Protocol, ConnectionInfo, SetupWizardField


@dataclass
class IntelRealSenseD435Spec(DeviceSpec):
    """Intel RealSense D435 RGB-D camera specification"""
    
    def __post_init__(self):
        # Core identification
        self.id = "intel/realsense-d435"
        self.name = "Intel RealSense D435"
        self.category = "depth_camera"
        self.manufacturer = "Intel"
        self.model = "D435"
        self.description = "RGB-D camera with stereo depth sensing"
        
        # Capabilities
        self.capabilities = [
            Capability(
                name="rgb_imaging",
                commands=["capture_rgb", "start_rgb_stream", "stop_rgb_stream"],
                description="RGB color imaging"
            ),
            Capability(
                name="depth_sensing",
                commands=["capture_depth", "start_depth_stream", "stop_depth_stream"],
                description="Stereo depth sensing"
            ),
            Capability(
                name="pointcloud",
                commands=["generate_pointcloud", "stream_pointcloud"],
                description="3D point cloud generation"
            )
        ]
        
        # Communication protocols
        self.protocols = [
            Protocol(
                type="usb",
                port="USB 3.0",
                commands=["capture_rgb", "capture_depth"],
                parameters={"interface": "librealsense2"}
            )
        ]
        
        # Connection information
        self.connection = ConnectionInfo(
            type="usb",
            setup_instructions=[
                "Connect camera to USB 3.0 port",
                "Install Intel RealSense SDK",
                "Verify camera detection with realsense-viewer"
            ]
        )
        
        # Setup wizard
        self.setup_wizard = [
            SetupWizardField(
                name="name",
                type="string",
                label="Camera Name",
                default="RealSense D435"
            ),
            SetupWizardField(
                name="rgb_resolution",
                type="select",
                label="RGB Resolution",
                options=["640x480", "1280x720", "1920x1080"],
                default="1280x720"
            ),
            SetupWizardField(
                name="depth_resolution", 
                type="select",
                label="Depth Resolution",
                options=["424x240", "640x480", "848x480"],
                default="640x480"
            ),
            SetupWizardField(
                name="framerate",
                type="select",
                label="Frame Rate",
                options=["15", "30", "60"],
                default="30"
            )
        ]
        
        # Technical specifications
        self.specs = {
            "rgb_sensor": {
                "resolution": "1920x1080",
                "framerate": 30,
                "fov": {"horizontal": 69, "vertical": 42}
            },
            "depth_sensor": {
                "technology": "Active IR Stereo",
                "resolution": "1280x720", 
                "range": {"min": 0.2, "max": 10},  # meters
                "accuracy": 0.002,  # meters at 2m distance
                "framerate": 90
            },
            "physical": {
                "dimensions": {"width": 90, "height": 25, "depth": 25},  # mm
                "weight": 72,  # grams
                "mounting": "1/4-20 UNC"
            },
            "connectivity": {
                "interface": "USB 3.0 Type-C",
                "power": "USB bus powered"
            }
        }
        
        super().__post_init__()
