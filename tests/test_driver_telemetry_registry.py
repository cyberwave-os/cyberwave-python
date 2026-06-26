"""BaseDriver registers twin/telemetry publisher at init."""

from __future__ import annotations

from types import SimpleNamespace

from cyberwave.driver import CallbackGroup, CommandArgs, BaseDriver, TopicSpec
from cyberwave.driver.interface.registry import DriverInterfaceRegistry

_TEST_TWIN = SimpleNamespace(
    uuid="00000000-0000-0000-0000-000000000002",
    environment_id="env-00000000-0000-0000-0000-000000000099",
    name="telemetry-test",
    client=None,
)


class _TelemetryDriver(BaseDriver):
    REGISTRY_ID = "acme/telemetry-test"
    driver_family = "python"

    def define_interface(self, iface: DriverInterfaceRegistry) -> None:
        cmd = TopicSpec(
            namespace="twin",
            leaf="command",
            payload_schema_ref="TwinCommandPayload",
        )
        iface.add_listener(
            cmd,
            CallbackGroup(callback=lambda _e: None),
            command=CommandArgs(name="ping"),
        )

    async def on_configure(self) -> None:
        pass

    async def on_connect_to_device(self) -> None:
        pass

    async def on_register_callbacks(self) -> None:
        pass

    async def on_activate(self) -> None:
        pass

    async def on_shutdown(self) -> None:
        pass

    @classmethod
    def create(cls) -> _TelemetryDriver:
        return cls(_TEST_TWIN)


def test_define_interface_defaults_registers_telemetry_publisher() -> None:
    driver = _TelemetryDriver(_TEST_TWIN)
    publishers = driver._interface.publishers_for_mode(driver.operation_mode)
    slugs = [(e.topic.namespace, e.topic.leaf) for e in publishers]
    assert ("twin", "telemetry") in slugs
