"""Company building digital twin specification."""

from dataclasses import dataclass

from ...base import Capability, DeviceSpec


@dataclass
class CompanyBuildingSpec(DeviceSpec):
    """High-level site twin used for orchestration examples."""

    def __post_init__(self):
        self.id = "your_company/your_building"
        self.name = "Company Headquarters"
        self.category = "facility"
        self.manufacturer = "Cyberwave"
        self.model = "HQ-Digital-Twin"
        self.description = "Aggregated facility digital twin describing floors, zones, and security endpoints"

        self.has_digital_asset = True
        self.has_simulation_model = True

        self.capabilities = [
            Capability(
                name="facility_overview",
                commands=["list_zones", "get_zone_status", "create_alert"],
                description="Facility-wide monitoring and alerting",
            ),
        ]

        self.extended_capabilities = {
            "multi_level": True,
            "supports_agents": True,
        }

        super().__post_init__()
