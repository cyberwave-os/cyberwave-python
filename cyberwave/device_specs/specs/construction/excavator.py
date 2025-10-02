"""
Excavator Device Specifications

Heavy construction equipment for excavation, demolition, and material handling.
"""

from dataclasses import dataclass
from ...base import DeviceSpec, Capability, Protocol, ConnectionInfo, SetupWizardField, DependencySpec


@dataclass
class ExcavatorSpec(DeviceSpec):
    """Generic excavator specification for construction sites"""
    
    def __post_init__(self):
        # Core identification
        self.id = "generic/excavator"
        self.name = "Generic Excavator"
        self.category = "construction_equipment"
        self.manufacturer = "Generic"
        self.model = "Excavator"
        self.description = "Heavy construction excavator for digging, demolition, and material handling"
        
        # Software-defined capabilities
        self.has_hardware_driver = False  # Mock implementation
        self.has_digital_asset = True
        self.has_simulation_model = True
        
        # Implementation details
        self.driver_class = None  # No hardware driver for mock
        self.asset_class = "cyberwave.assets.GenericExcavator"
        self.simulation_models = ["mujoco", "gazebo"]
        self.fallback_asset_class = "cyberwave.assets.GenericHeavyMachinery"
        
        # Extended capabilities for WeBuild showcase
        self.extended_capabilities = {
            "has_telemetry_monitoring": True,
            "has_safety_systems": True,
            "has_remote_monitoring": True,
            "has_work_zone_detection": True
        }
        
        # Capabilities with UI metadata
        self.capabilities = [
            Capability(
                name="excavation",
                commands=[
                    "arm_extend", "arm_retract", "bucket_dig", "bucket_dump",
                    "boom_raise", "boom_lower", "swing_left", "swing_right",
                    "track_forward", "track_reverse", "emergency_stop"
                ],
                description="Excavation and material handling operations",
                ui_metadata={
                    "component_type": "excavator_controls",
                    "layout": "sectioned",
                    "sections": [
                        {
                            "title": "Arm Controls",
                            "controls": [
                                {
                                    "id": "arm_extend",
                                    "type": "button",
                                    "label": "Extend Arm",
                                    "icon": "arrow-right",
                                    "command": "arm_extend"
                                },
                                {
                                    "id": "arm_retract", 
                                    "type": "button",
                                    "label": "Retract Arm",
                                    "icon": "arrow-left",
                                    "command": "arm_retract"
                                }
                            ]
                        },
                        {
                            "title": "Bucket Controls",
                            "controls": [
                                {
                                    "id": "bucket_dig",
                                    "type": "button",
                                    "label": "Dig",
                                    "icon": "pickaxe",
                                    "command": "bucket_dig",
                                    "variant": "primary"
                                },
                                {
                                    "id": "bucket_dump",
                                    "type": "button", 
                                    "label": "Dump",
                                    "icon": "trash",
                                    "command": "bucket_dump"
                                }
                            ]
                        },
                        {
                            "title": "Movement",
                            "controls": [
                                {
                                    "id": "movement_pad",
                                    "type": "directional_pad",
                                    "commands": {
                                        "up": {"command": "track_forward", "params": {"speed": 0.5}},
                                        "down": {"command": "track_reverse", "params": {"speed": 0.5}},
                                        "left": {"command": "swing_left", "params": {"angle": 15}},
                                        "right": {"command": "swing_right", "params": {"angle": 15}}
                                    }
                                }
                            ]
                        }
                    ]
                }
            ),
            Capability(
                name="safety_monitoring",
                commands=["proximity_scan", "work_zone_check", "operator_status"],
                description="Safety systems and work zone monitoring",
                ui_metadata={
                    "component_type": "safety_monitor",
                    "layout": "dashboard",
                    "alerts": [
                        {
                            "id": "proximity_alert",
                            "type": "proximity",
                            "threshold": 5.0,  # meters
                            "severity": "warning"
                        },
                        {
                            "id": "work_zone_breach",
                            "type": "zone_violation", 
                            "severity": "critical"
                        }
                    ]
                }
            ),
            Capability(
                name="telemetry",
                commands=[
                    "engine_status", "hydraulic_pressure", "fuel_level", 
                    "operating_hours", "location", "load_weight"
                ],
                description="Real-time equipment telemetry and status",
                ui_metadata={
                    "component_type": "telemetry_display",
                    "update_frequency": 5,
                    "layout": "grid",
                    "metrics": [
                        {
                            "id": "engine_status",
                            "label": "Engine",
                            "icon": "zap",
                            "type": "status",
                            "command": "engine_status"
                        },
                        {
                            "id": "fuel_level",
                            "label": "Fuel",
                            "icon": "fuel",
                            "type": "progress",
                            "unit": "%",
                            "thresholds": {"critical": 10, "warning": 25, "good": 50},
                            "command": "fuel_level"
                        },
                        {
                            "id": "hydraulic_pressure",
                            "label": "Hydraulics",
                            "icon": "gauge",
                            "type": "gauge",
                            "unit": "PSI",
                            "range": [0, 3000],
                            "command": "hydraulic_pressure"
                        },
                        {
                            "id": "operating_hours",
                            "label": "Hours",
                            "icon": "clock",
                            "type": "value",
                            "unit": "hrs",
                            "command": "operating_hours"
                        }
                    ]
                }
            )
        ]
        
        # Communication protocols (mocked for showcase)
        self.protocols = [
            Protocol(
                type="can_bus",
                port=0,
                commands=["arm_extend", "arm_retract", "bucket_dig", "bucket_dump"],
                parameters={"baud_rate": 250000, "timeout": 1.0}
            ),
            Protocol(
                type="ethernet",
                port=502,  # Modbus TCP
                commands=["telemetry", "status_update"],
                parameters={"protocol": "modbus_tcp", "unit_id": 1}
            )
        ]
        
        # Connection information
        self.connection = ConnectionInfo(
            type="ethernet",
            default_ip="192.168.1.100",
            setup_instructions=[
                "Connect to excavator's onboard computer via Ethernet",
                "Configure network settings (static IP recommended)",
                "Verify CAN bus interface is operational",
                "Test communication with control system"
            ]
        )
        
        # Setup wizard
        self.setup_wizard = [
            SetupWizardField(
                name="name",
                type="string",
                label="Equipment Name",
                default="Excavator-01",
                help_text="Unique identifier for this excavator"
            ),
            SetupWizardField(
                name="ip_address",
                type="ipv4",
                label="IP Address",
                default="192.168.1.100",
                help_text="IP address of the excavator's control system"
            ),
            SetupWizardField(
                name="work_zone_id",
                type="string",
                label="Work Zone ID",
                default="ZONE-A",
                help_text="Designated work zone for this equipment"
            ),
            SetupWizardField(
                name="operator_id",
                type="string",
                label="Operator ID",
                required=True,
                help_text="ID of the certified operator"
            )
        ]
        
        # Technical specifications
        self.specs = {
            "weight": 20000,  # kg
            "dimensions": {"length": 9.5, "width": 2.8, "height": 3.2},  # meters
            "engine": {
                "power": 150,  # kW
                "type": "diesel",
                "displacement": 6.7  # liters
            },
            "hydraulics": {
                "max_pressure": 350,  # bar
                "flow_rate": 280  # l/min
            },
            "bucket": {
                "capacity": 1.2,  # m³
                "max_breakout_force": 130  # kN
            },
            "operating_range": {
                "max_dig_depth": 6.5,  # meters
                "max_reach": 9.8,  # meters
                "max_dump_height": 6.7  # meters
            }
        }
        
        super().__post_init__()


@dataclass
class CaterpillarExcavatorSpec(ExcavatorSpec):
    """Caterpillar 320 excavator specification"""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Override identification
        self.id = "caterpillar/320"
        self.name = "Caterpillar 320"
        self.manufacturer = "Caterpillar"
        self.model = "320"
        self.description = "Mid-size hydraulic excavator for general construction and excavation"
        
        # Enhanced specifications
        self.specs.update({
            "model_year": 2024,
            "fuel_capacity": 400,  # liters
            "travel_speed": {"low": 3.1, "high": 5.5},  # km/h
            "swing_speed": 11.2,  # rpm
            "ground_pressure": 47,  # kPa
            "grade_ability": 35,  # degrees
            "certifications": ["CE", "EPA Tier 4", "ISO 9001"]
        })
        
        # Add Caterpillar-specific capabilities
        self.extended_capabilities.update({
            "has_cat_connect": True,
            "has_grade_control": True,
            "has_payload_weighing": True
        })
