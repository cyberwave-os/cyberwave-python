"""
Velodyne Puck (VLP-16) Specification

16-channel LIDAR sensor for 3D mapping and navigation.
"""

from dataclasses import dataclass
from ...base import DeviceSpec, Capability, Protocol, ConnectionInfo, SetupWizardField


@dataclass
class VelodynePuckSpec(DeviceSpec):
    """Velodyne Puck VLP-16 LIDAR specification"""
    
    def __post_init__(self):
        # Core identification
        self.id = "velodyne/puck"
        self.name = "Velodyne Puck (VLP-16)"
        self.category = "lidar"
        self.manufacturer = "Velodyne"
        self.model = "VLP-16"
        self.description = "16-channel LIDAR for 3D sensing and mapping"
        
        # Capabilities
        self.capabilities = [
            Capability(
                name="lidar_sensing",
                commands=["start_scan", "stop_scan", "get_pointcloud"],
                description="3D LIDAR scanning"
            ),
            Capability(
                name="mapping",
                commands=["generate_map", "localize", "detect_obstacles"],
                description="SLAM and navigation"
            )
        ]
        
        # Communication protocols
        self.protocols = [
            Protocol(
                type="ethernet",
                port=2368,
                commands=["start_scan", "get_pointcloud"],
                parameters={"data_port": 2368, "telemetry_port": 8308}
            )
        ]
        
        # Connection information
        self.connection = ConnectionInfo(
            type="ethernet",
            default_ip="192.168.1.201",
            setup_instructions=[
                "Connect LIDAR to network via ethernet",
                "Configure static IP (default: 192.168.1.201)",
                "Verify data stream on port 2368"
            ]
        )
        
        # Setup wizard
        self.setup_wizard = [
            SetupWizardField(
                name="name",
                type="string",
                label="LIDAR Name",
                default="Velodyne Puck"
            ),
            SetupWizardField(
                name="ip_address",
                type="ipv4",
                label="LIDAR IP Address",
                default="192.168.1.201"
            ),
            SetupWizardField(
                name="rotation_rate",
                type="select",
                label="Rotation Rate (RPM)",
                options=["300", "600", "1200"],
                default="600"
            )
        ]
        
        # Technical specifications
        self.specs = {
            "channels": 16,
            "range": 100,  # meters
            "accuracy": 0.03,  # meters
            "angular_resolution": 0.1,  # degrees
            "vertical_fov": 30,  # degrees
            "rotation_rate": {"min": 300, "max": 1200},  # RPM
            "points_per_second": 300000,
            "power": {"voltage": 12, "consumption": 8},  # V, W
            "operating_temp": {"min": -10, "max": 60}  # celsius
        }
        
        super().__post_init__()
