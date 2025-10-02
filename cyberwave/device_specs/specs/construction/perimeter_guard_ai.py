"""AI perimeter security service specification."""

from dataclasses import dataclass

from ...base import Capability, DeviceSpec


@dataclass
class PerimeterGuardAISpec(DeviceSpec):
    """Virtual security agent that coordinates patrols and threat analysis."""

    def __post_init__(self):
        self.id = "ai/perimeter_guard"
        self.name = "Perimeter Guard AI"
        self.category = "ai_service"
        self.manufacturer = "Cyberwave"
        self.model = "PerimeterGuard"
        self.description = "AI orchestration layer for perimeter monitoring and autonomous patrol control"

        self.has_digital_asset = True
        self.has_simulation_model = True

        self.capabilities = [
            Capability(
                name="patrol_management",
                commands=["patrol_start", "patrol_stop", "follow_route"],
                description="Manage patrol tasks for connected security twins",
                metadata={
                    "command_schemas": {
                        "patrol_start": {
                            "type": "object",
                            "properties": {
                                "zone": {"type": "string", "description": "Zone or route identifier"}
                            },
                        },
                        "follow_route": {
                            "type": "object",
                            "properties": {
                                "route": {"type": "string"},
                                "speed": {"type": "number"},
                            },
                        },
                    }
                },
            ),
            Capability(
                name="threat_analytics",
                commands=["detect_intruders", "identify_threats", "analyze_behavior"],
                description="Real-time threat detection and verification",
                metadata={
                    "command_schemas": {
                        "detect_intruders": {
                            "type": "object",
                            "properties": {
                                "mode": {
                                    "type": "string",
                                    "enum": ["stream", "snapshot"],
                                }
                            },
                        },
                        "identify_threats": {
                            "type": "object",
                            "properties": {
                                "target": {"type": "string", "description": "Event or detection identifier"}
                            },
                        },
                    }
                },
            ),
            Capability(
                name="alerting",
                commands=["create_alert", "notify_team"],
                description="Emit alerts or notifications to downstream systems",
                metadata={
                    "command_schemas": {
                        "create_alert": {
                            "type": "object",
                            "properties": {
                                "message": {"type": "string"},
                                "severity": {"type": "string"},
                            },
                        }
                    }
                },
            ),
        ]

        self.extended_capabilities = {
            "real_time_monitoring": True,
            "multimodal_fusion": True,
            "autonomous_dispatch": True,
        }

        super().__post_init__()
