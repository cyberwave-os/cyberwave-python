"""
Security Drone Device Specifications

Specialized drones for construction site security, monitoring, and inspection.
"""

from dataclasses import dataclass
from ...base import DeviceSpec, Capability, Protocol, ConnectionInfo, SetupWizardField, DependencySpec


@dataclass
class SecurityDroneSpec(DeviceSpec):
    """Professional security drone for construction site monitoring"""
    
    def __post_init__(self):
        # Core identification
        self.id = "generic/security_drone"
        self.name = "Security Drone"
        self.category = "security_drone"
        self.manufacturer = "Generic"
        self.model = "SecurityDrone-Pro"
        self.description = "Professional security drone with thermal imaging and AI analytics for construction site monitoring"
        
        # Software-defined capabilities
        self.has_hardware_driver = True
        self.has_digital_asset = True
        self.has_simulation_model = True
        
        # Implementation details
        self.driver_class = "cyberwave_cli.drivers.security_drone.SecurityDroneDriver"
        self.asset_class = "cyberwave.assets.SecurityDrone"
        self.simulation_models = ["airsim", "gazebo", "unity"]
        self.fallback_asset_class = "cyberwave.assets.GenericDrone"
        
        # Extended capabilities for security operations
        self.extended_capabilities = {
            "has_thermal_imaging": True,
            "has_night_vision": True,
            "has_ai_analytics": True,
            "has_autonomous_patrol": True,
            "has_emergency_response": True,
            "has_two_way_audio": True
        }
        
        # Capabilities with UI metadata
        self.capabilities = [
            Capability(
                name="security_flight",
                commands=[
                    "takeoff", "land", "emergency_land", "return_to_base",
                    "patrol_start", "patrol_stop", "follow_route", "hover",
                    "altitude_hold", "position_hold"
                ],
                description="Security-focused flight operations and patrol modes",
                ui_metadata={
                    "component_type": "security_flight_controller",
                    "layout": "mission_control",
                    "controls": [
                        {
                            "id": "takeoff",
                            "type": "button",
                            "label": "Launch",
                            "icon": "plane",
                            "variant": "primary",
                            "command": "takeoff",
                            "requires_confirmation": True
                        },
                        {
                            "id": "emergency_land",
                            "type": "button",
                            "label": "EMERGENCY LAND",
                            "icon": "alert-triangle",
                            "variant": "destructive",
                            "command": "emergency_land",
                            "requires_confirmation": True
                        },
                        {
                            "id": "patrol_mode",
                            "type": "toggle_button",
                            "labels": {"on": "Stop Patrol", "off": "Start Patrol"},
                            "icons": {"on": "pause", "off": "play"},
                            "commands": {"on": "patrol_stop", "off": "patrol_start"}
                        },
                        {
                            "id": "return_to_base",
                            "type": "button",
                            "label": "Return to Base",
                            "icon": "home",
                            "command": "return_to_base"
                        }
                    ]
                }
            ),
            Capability(
                name="surveillance_systems",
                commands=[
                    "thermal_imaging_on", "thermal_imaging_off", "night_vision_on", "night_vision_off",
                    "spotlight_on", "spotlight_off", "siren_activate", "announce_message"
                ],
                description="Advanced surveillance and deterrent systems",
                ui_metadata={
                    "component_type": "surveillance_panel",
                    "layout": "system_controls",
                    "systems": [
                        {
                            "id": "thermal_toggle",
                            "type": "toggle_button",
                            "label": "Thermal Imaging",
                            "icon": "thermometer",
                            "commands": {"on": "thermal_imaging_off", "off": "thermal_imaging_on"}
                        },
                        {
                            "id": "night_vision_toggle",
                            "type": "toggle_button",
                            "label": "Night Vision",
                            "icon": "eye",
                            "commands": {"on": "night_vision_off", "off": "night_vision_on"}
                        },
                        {
                            "id": "spotlight_toggle",
                            "type": "toggle_button",
                            "label": "Spotlight",
                            "icon": "flashlight",
                            "commands": {"on": "spotlight_off", "off": "spotlight_on"}
                        },
                        {
                            "id": "siren",
                            "type": "button",
                            "label": "Activate Siren",
                            "icon": "siren",
                            "variant": "destructive",
                            "command": "siren_activate",
                            "requires_confirmation": True
                        }
                    ]
                }
            ),
            Capability(
                name="ai_security_analytics",
                commands=[
                    "detect_intruders", "analyze_behavior", "identify_threats",
                    "count_personnel", "verify_credentials", "generate_report"
                ],
                description="AI-powered security analytics and threat detection",
                ui_metadata={
                    "component_type": "ai_security_panel",
                    "layout": "threat_dashboard",
                    "analytics": [
                        {
                            "id": "intruder_detection",
                            "label": "Intruder Detection",
                            "icon": "user-x",
                            "type": "threat_level",
                            "command": "detect_intruders"
                        },
                        {
                            "id": "behavior_analysis",
                            "label": "Behavior Analysis",
                            "icon": "activity",
                            "type": "behavior_score",
                            "command": "analyze_behavior"
                        },
                        {
                            "id": "personnel_count",
                            "label": "Personnel Count",
                            "icon": "users",
                            "type": "counter",
                            "command": "count_personnel"
                        }
                    ]
                }
            ),
            Capability(
                name="telemetry",
                commands=[
                    "battery_status", "gps_location", "altitude", "speed",
                    "signal_strength", "system_temperature", "flight_time"
                ],
                description="Real-time drone telemetry and status monitoring",
                ui_metadata={
                    "component_type": "drone_telemetry",
                    "update_frequency": 2,
                    "layout": "flight_dashboard",
                    "metrics": [
                        {
                            "id": "battery_status",
                            "label": "Battery",
                            "icon": "battery",
                            "type": "progress",
                            "unit": "%",
                            "thresholds": {"critical": 15, "warning": 30, "good": 50},
                            "command": "battery_status"
                        },
                        {
                            "id": "altitude",
                            "label": "Altitude",
                            "icon": "arrow-up",
                            "type": "value",
                            "unit": "m",
                            "command": "altitude"
                        },
                        {
                            "id": "speed",
                            "label": "Speed",
                            "icon": "gauge",
                            "type": "gauge",
                            "unit": "m/s",
                            "range": [0, 15],
                            "command": "speed"
                        },
                        {
                            "id": "signal_strength",
                            "label": "Signal",
                            "icon": "wifi",
                            "type": "signal_strength",
                            "command": "signal_strength"
                        }
                    ]
                }
            )
        ]
        
        # Communication protocols
        self.protocols = [
            Protocol(
                type="mavlink",
                port=14550,
                commands=["takeoff", "land", "goto", "mission_upload"],
                parameters={"system_id": 1, "component_id": 1, "heartbeat_rate": 1}
            ),
            Protocol(
                type="rtsp",
                port=554,
                commands=["video_stream", "thermal_stream"],
                parameters={"stream_path": "/live", "codec": "h264"}
            ),
            Protocol(
                type="tcp",
                port=5760,
                commands=["telemetry_stream"],
                parameters={"protocol": "mavlink", "stream_rate": 10}
            )
        ]
        
        # Connection information
        self.connection = ConnectionInfo(
            type="radio",
            default_ip="192.168.1.1",
            setup_instructions=[
                "Power on the security drone",
                "Establish radio link with ground control station",
                "Verify GPS lock and compass calibration",
                "Test all camera systems (visual and thermal)",
                "Confirm return-to-base functionality"
            ]
        )
        
        # Setup wizard
        self.setup_wizard = [
            SetupWizardField(
                name="name",
                type="string",
                label="Drone Name",
                default="Security-Drone-01",
                help_text="Unique identifier for this security drone"
            ),
            SetupWizardField(
                name="base_station_ip",
                type="ipv4",
                label="Base Station IP",
                default="192.168.1.1",
                help_text="IP address of the ground control station"
            ),
            SetupWizardField(
                name="patrol_zone",
                type="string",
                label="Patrol Zone",
                default="ZONE-ALPHA",
                help_text="Designated patrol area for this drone"
            ),
            SetupWizardField(
                name="max_altitude",
                type="int",
                label="Maximum Altitude (m)",
                default=50,
                help_text="Maximum allowed flight altitude"
            ),
            SetupWizardField(
                name="enable_thermal",
                type="boolean",
                label="Enable Thermal Imaging",
                default=True,
                help_text="Enable thermal imaging camera"
            ),
            SetupWizardField(
                name="auto_patrol",
                type="boolean",
                label="Auto Patrol Mode",
                default=False,
                help_text="Enable autonomous patrol when launched"
            )
        ]
        
        # Enhanced specifications for security drone
        self.specs.update({
            "flight_performance": {
                "max_flight_time": 45,  # minutes
                "max_speed": 15,  # m/s
                "max_altitude": 120,  # meters (legal limit)
                "wind_resistance": "12 m/s",
                "payload_capacity": 2.5  # kg
            },
            "cameras": {
                "visual": {
                    "resolution": "4K UHD",
                    "zoom": "30x optical, 6x digital",
                    "stabilization": "3-axis gimbal"
                },
                "thermal": {
                    "resolution": "640x512",
                    "temperature_range": "-40°C to 550°C",
                    "thermal_sensitivity": "±2°C or ±2%"
                }
            },
            "security_features": {
                "encrypted_communication": True,
                "secure_boot": True,
                "tamper_detection": True,
                "geofencing": True,
                "no_fly_zones": True
            },
            "environmental": {
                "operating_temperature": {"min": -20, "max": 50},
                "ip_rating": "IP43",
                "wind_resistance": "Level 5 (12 m/s)"
            }
        })
