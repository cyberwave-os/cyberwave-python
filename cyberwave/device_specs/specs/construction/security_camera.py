"""
Security Camera Device Specifications

High-resolution security cameras for construction site monitoring and safety.
"""

from dataclasses import dataclass
from ...base import DeviceSpec, Capability, Protocol, ConnectionInfo, SetupWizardField, DependencySpec


@dataclass
class SecurityCameraSpec(DeviceSpec):
    """High-resolution security camera for construction site monitoring"""
    
    def __post_init__(self):
        # Core identification
        self.id = "generic/security_camera"
        self.name = "Security Camera"
        self.category = "security_camera"
        self.manufacturer = "Generic"
        self.model = "SecurityCam"
        self.description = "High-resolution security camera with AI-powered analytics for construction site monitoring"
        
        # Software-defined capabilities
        self.has_hardware_driver = True
        self.has_digital_asset = True
        self.has_simulation_model = True
        
        # Implementation details
        self.driver_class = "cyberwave_cli.drivers.security_camera.SecurityCameraDriver"
        self.asset_class = "cyberwave.assets.SecurityCamera"
        self.simulation_models = ["unity", "unreal"]
        self.fallback_asset_class = "cyberwave.assets.GenericCamera"
        
        # Extended capabilities for security monitoring
        self.extended_capabilities = {
            "has_ai_analytics": True,
            "has_motion_detection": True,
            "has_facial_recognition": True,
            "has_object_detection": True,
            "has_intrusion_detection": True,
            "has_night_vision": True
        }
        
        # Capabilities with UI metadata
        self.capabilities = [
            Capability(
                name="video_surveillance",
                commands=[
                    "start_recording", "stop_recording", "take_snapshot",
                    "zoom_in", "zoom_out", "focus_auto", "focus_manual"
                ],
                description="Video surveillance and recording capabilities",
                ui_metadata={
                    "component_type": "surveillance_controls",
                    "layout": "video_panel",
                    "controls": [
                        {
                            "id": "record_toggle",
                            "type": "toggle_button",
                            "labels": {"on": "Stop Recording", "off": "Start Recording"},
                            "icons": {"on": "square", "off": "circle"},
                            "commands": {"on": "stop_recording", "off": "start_recording"},
                            "variant": "destructive_when_on"
                        },
                        {
                            "id": "snapshot",
                            "type": "button",
                            "label": "Snapshot",
                            "icon": "camera",
                            "command": "take_snapshot"
                        },
                        {
                            "id": "zoom_controls",
                            "type": "button_group",
                            "controls": [
                                {"command": "zoom_in", "label": "+", "icon": "zoom-in"},
                                {"command": "zoom_out", "label": "-", "icon": "zoom-out"}
                            ]
                        }
                    ]
                }
            ),
            Capability(
                name="ai_analytics",
                commands=[
                    "detect_people", "detect_vehicles", "detect_equipment",
                    "analyze_safety_compliance", "count_personnel", "detect_ppe"
                ],
                description="AI-powered video analytics for safety and security",
                ui_metadata={
                    "component_type": "ai_analytics_panel",
                    "layout": "analytics_dashboard",
                    "analytics": [
                        {
                            "id": "people_detection",
                            "label": "Personnel Detection",
                            "icon": "users",
                            "type": "detection_overlay",
                            "command": "detect_people"
                        },
                        {
                            "id": "vehicle_detection",
                            "label": "Vehicle Detection", 
                            "icon": "truck",
                            "type": "detection_overlay",
                            "command": "detect_vehicles"
                        },
                        {
                            "id": "ppe_compliance",
                            "label": "PPE Compliance",
                            "icon": "hard-hat",
                            "type": "compliance_check",
                            "command": "detect_ppe"
                        },
                        {
                            "id": "safety_analysis",
                            "label": "Safety Analysis",
                            "icon": "shield-check",
                            "type": "safety_score",
                            "command": "analyze_safety_compliance"
                        }
                    ]
                }
            ),
            Capability(
                name="motion_detection",
                commands=["motion_zones", "intrusion_alert", "perimeter_breach"],
                description="Motion detection and intrusion monitoring",
                ui_metadata={
                    "component_type": "motion_detection",
                    "layout": "zone_overlay",
                    "zones": [
                        {
                            "id": "work_zone",
                            "type": "work_area",
                            "sensitivity": "medium",
                            "color": "green"
                        },
                        {
                            "id": "restricted_zone",
                            "type": "no_access",
                            "sensitivity": "high", 
                            "color": "red"
                        }
                    ]
                }
            ),
            Capability(
                name="telemetry",
                commands=["camera_status", "storage_usage", "network_quality", "temperature"],
                description="Camera system telemetry and health monitoring",
                ui_metadata={
                    "component_type": "telemetry_display",
                    "update_frequency": 30,
                    "layout": "compact_grid",
                    "metrics": [
                        {
                            "id": "camera_status",
                            "label": "Status",
                            "icon": "camera",
                            "type": "status",
                            "command": "camera_status"
                        },
                        {
                            "id": "storage_usage",
                            "label": "Storage",
                            "icon": "hard-drive",
                            "type": "progress",
                            "unit": "%",
                            "thresholds": {"critical": 90, "warning": 75, "good": 50},
                            "command": "storage_usage"
                        },
                        {
                            "id": "network_quality",
                            "label": "Network",
                            "icon": "wifi",
                            "type": "signal_strength",
                            "command": "network_quality"
                        }
                    ]
                }
            )
        ]
        
        # Communication protocols
        self.protocols = [
            Protocol(
                type="rtsp",
                port=554,
                commands=["video_stream"],
                parameters={"stream_path": "/live/main", "codec": "h264"}
            ),
            Protocol(
                type="http",
                port=80,
                commands=["take_snapshot", "camera_status"],
                parameters={"api_version": "v1", "auth": "basic"}
            ),
            Protocol(
                type="onvif",
                port=8080,
                commands=["ptz_control", "video_analytics"],
                parameters={"profile": "S", "version": "2.6"}
            )
        ]
        
        # Connection information
        self.connection = ConnectionInfo(
            type="ethernet",
            default_ip="192.168.1.64",
            setup_instructions=[
                "Connect camera to network via Ethernet",
                "Configure camera IP address (static recommended)",
                "Set up ONVIF profile for PTZ control",
                "Configure video stream settings",
                "Test RTSP stream connectivity"
            ]
        )
        
        # Setup wizard
        self.setup_wizard = [
            SetupWizardField(
                name="name",
                type="string",
                label="Camera Name",
                default="Security-Cam-01",
                help_text="Unique name for this security camera"
            ),
            SetupWizardField(
                name="ip_address",
                type="ipv4",
                label="IP Address",
                default="192.168.1.64",
                help_text="IP address of the security camera"
            ),
            SetupWizardField(
                name="location",
                type="string",
                label="Camera Location",
                default="Main Entrance",
                help_text="Physical location description"
            ),
            SetupWizardField(
                name="resolution",
                type="select",
                label="Video Resolution",
                default="1080p",
                options=["720p", "1080p", "4K"],
                help_text="Video stream resolution"
            ),
            SetupWizardField(
                name="enable_ai_analytics",
                type="boolean",
                label="Enable AI Analytics",
                default=True,
                help_text="Enable AI-powered video analytics"
            )
        ]
        
        # Technical specifications
        self.specs = {
            "sensor": {
                "type": "CMOS",
                "size": "1/2.8\"",
                "resolution": "1920x1080",
                "low_light": "0.01 lux"
            },
            "lens": {
                "focal_length": "2.8-12mm",
                "aperture": "F1.4",
                "field_of_view": "104°-30°"
            },
            "video": {
                "max_resolution": "1920x1080",
                "frame_rate": "30fps",
                "compression": "H.264/H.265",
                "bitrate": "up to 8Mbps"
            },
            "analytics": {
                "people_counting": True,
                "vehicle_detection": True,
                "facial_recognition": True,
                "license_plate_recognition": True,
                "intrusion_detection": True
            },
            "environmental": {
                "operating_temperature": {"min": -30, "max": 60},  # celsius
                "humidity": "≤95%",
                "ip_rating": "IP67",
                "vandal_resistance": "IK10"
            },
            "power": {
                "consumption": "12W",
                "input": "DC 12V ± 25% / PoE+",
                "backup": "Built-in supercapacitor"
            }
        }
        
        super().__post_init__()


@dataclass
class PTZSecurityCameraSpec(SecurityCameraSpec):
    """Pan-Tilt-Zoom security camera for wide area monitoring"""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Override identification
        self.id = "generic/ptz_security_camera"
        self.name = "PTZ Security Camera"
        self.model = "PTZ-SecurityCam"
        self.description = "Pan-Tilt-Zoom security camera with AI analytics for comprehensive site monitoring"
        
        # Add PTZ-specific capabilities
        ptz_capability = Capability(
            name="ptz_control",
            commands=[
                "pan_left", "pan_right", "tilt_up", "tilt_down",
                "zoom_in", "zoom_out", "preset_goto", "preset_set",
                "tour_start", "tour_stop", "auto_track"
            ],
            description="Pan, tilt, and zoom camera control",
            ui_metadata={
                "component_type": "ptz_controller",
                "layout": "joystick_panel",
                "controls": [
                    {
                        "id": "ptz_joystick",
                        "type": "joystick",
                        "commands": {
                            "pan": {"left": "pan_left", "right": "pan_right"},
                            "tilt": {"up": "tilt_up", "down": "tilt_down"}
                        },
                        "sensitivity": "medium"
                    },
                    {
                        "id": "zoom_slider",
                        "type": "slider",
                        "range": [1, 30],
                        "default": 1,
                        "label": "Zoom",
                        "commands": {"increase": "zoom_in", "decrease": "zoom_out"}
                    },
                    {
                        "id": "presets",
                        "type": "preset_buttons",
                        "presets": [
                            {"id": 1, "name": "Main Gate", "command": "preset_goto", "params": {"preset": 1}},
                            {"id": 2, "name": "Work Zone A", "command": "preset_goto", "params": {"preset": 2}},
                            {"id": 3, "name": "Equipment Yard", "command": "preset_goto", "params": {"preset": 3}}
                        ]
                    }
                ]
            }
        )
        
        self.capabilities.append(ptz_capability)
        
        # Enhanced specifications for PTZ
        self.specs.update({
            "ptz": {
                "pan_range": 360,  # degrees
                "tilt_range": 180,  # degrees
                "pan_speed": "0.1°-120°/s",
                "tilt_speed": "0.1°-120°/s",
                "zoom_ratio": "30x optical, 16x digital",
                "presets": 256,
                "tours": 8
            },
            "tracking": {
                "auto_tracking": True,
                "tracking_speed": "up to 60°/s",
                "tracking_accuracy": "±0.1°"
            }
        })
