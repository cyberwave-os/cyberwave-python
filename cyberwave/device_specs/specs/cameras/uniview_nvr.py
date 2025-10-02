"""
Uniview NVR Specification

Network Video Recorder with multiple camera channel support.
"""

from dataclasses import dataclass
from ...base import DeviceSpec, Capability, Protocol, ConnectionInfo, SetupWizardField


@dataclass
class UniviewNVRSpec(DeviceSpec):
    """Uniview NVR system specification"""
    
    def __post_init__(self):
        # Core identification
        self.id = "uniview/nvr"
        self.name = "Uniview NVR"
        self.category = "nvr_system"
        self.manufacturer = "Uniview"
        self.model = "NVR Series"
        self.description = "Network Video Recorder with multi-channel camera support"
        
        # Capabilities with UI metadata for software-defined UI
        self.capabilities = [
            Capability(
                name="video_streaming",
                commands=["get_camera_stream", "list_cameras", "switch_camera"],
                description="Multi-channel video streaming",
                ui_metadata={
                    "component_type": "video_stream",
                    "icon": "video-camera",
                    "controls": [
                        {
                            "type": "camera_selector",
                            "label": "Camera Channel",
                            "options": ["D1", "D2", "D3", "D4"],
                            "command": "switch_camera"
                        },
                        {
                            "type": "button",
                            "label": "Refresh Stream",
                            "command": "refresh_stream",
                            "variant": "secondary"
                        }
                    ],
                    "video_config": {
                        "proxy_enabled": True,
                        "formats": ["mjpeg", "rtsp"],
                        "fallback": "snapshot"
                    }
                }
            ),
            Capability(
                name="video_recording",
                commands=["start_recording", "stop_recording", "get_recording_status"],
                description="Video recording and playback",
                ui_metadata={
                    "component_type": "recording_controls",
                    "icon": "record",
                    "controls": [
                        {
                            "type": "button",
                            "label": "Start Recording",
                            "command": "start_recording",
                            "variant": "primary"
                        },
                        {
                            "type": "button", 
                            "label": "Stop Recording",
                            "command": "stop_recording",
                            "variant": "danger"
                        }
                    ]
                }
            ),
            Capability(
                name="motion_detection",
                commands=["enable_motion_detection", "disable_motion_detection", "get_motion_events"],
                description="Motion detection and alerts",
                ui_metadata={
                    "component_type": "sensor_metrics",
                    "icon": "motion-detector",
                    "metrics": [
                        {
                            "name": "Motion Events",
                            "key": "motion_events",
                            "type": "counter",
                            "unit": "events"
                        },
                        {
                            "name": "Detection Status",
                            "key": "detection_status", 
                            "type": "status",
                            "values": ["enabled", "disabled"]
                        }
                    ]
                }
            )
        ]
        
        # Communication protocols
        self.protocols = [
            Protocol(
                type="http",
                port=80,
                commands=["list_cameras", "get_camera_stream"],
                parameters={"api_version": "1.0", "auth_required": True}
            ),
            Protocol(
                type="rtsp",
                port=554,
                commands=["get_camera_stream"],
                parameters={"streaming_protocol": "rtsp"}
            )
        ]
        
        # Connection information
        self.connection = ConnectionInfo(
            type="ethernet",
            default_ip="192.168.1.108",
            setup_instructions=[
                "Connect NVR to network via ethernet",
                "Access web interface at NVR IP address",
                "Configure admin credentials",
                "Add camera channels"
            ]
        )
        
        # Setup wizard
        self.setup_wizard = [
            SetupWizardField(
                name="name",
                type="string",
                label="NVR Name",
                default="Uniview NVR"
            ),
            SetupWizardField(
                name="ip_address",
                type="ipv4",
                label="NVR IP Address",
                default="192.168.1.108"
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
                default="",
                help_text="NVR admin password"
            ),
            SetupWizardField(
                name="channel_count",
                type="select",
                label="Number of Channels",
                options=["4", "8", "16", "32"],
                default="8"
            )
        ]
        
        # Technical specifications
        self.specs = {
            "channels": {"min": 4, "max": 32},
            "resolution": ["1080p", "4K"],
            "storage": {"max_capacity": "32TB", "raid_support": True},
            "network": {"ports": 2, "poe_support": True},
            "protocols": ["ONVIF", "RTSP", "HTTP"],
            "recording": {
                "continuous": True,
                "motion_triggered": True,
                "scheduled": True
            }
        }
        
        super().__post_init__()
