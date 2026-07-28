"""Driver self-telemetry (twin/telemetry) default is capped at 1 Hz."""
from cyberwave.driver.interface.registry_mixin import InterfaceRegistryMixin


def test_telemetry_default_rate_is_1hz():
    assert InterfaceRegistryMixin.TELEMETRY_PUBLISH_RATE_HZ == 1.0
