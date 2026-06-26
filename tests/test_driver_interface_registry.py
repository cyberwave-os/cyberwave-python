"""Driver interface registry and manifest export."""

from __future__ import annotations

import asyncio

import pytest

from cyberwave.driver import (
    CallbackGroup,
    CommandArgs,
    DriverInterfaceRegistry,
    DriverOperationMode,
    TopicSpec,
    default_management_commands,
)
from cyberwave.driver.interface.args import PublisherArgs, effective_publish_mode
from cyberwave.driver.interface.registry_mixin import InterfaceRegistryMixin
from cyberwave.manifest.driver_config import TWIN_IMU_TOPIC_SLUG


def test_async_mqtt_handler_runs_on_driver_loop() -> None:
    received: list[dict[str, object]] = []

    async def _run() -> None:
        async def async_handler(envelope: dict[str, object]) -> None:
            received.append(envelope)

        loop = asyncio.get_running_loop()
        wrapper = InterfaceRegistryMixin._adapt_async_mqtt_handler(
            async_handler, loop=loop, path="cyberwave/twin/uuid/command"
        )
        wrapper({"command": "rotate"})
        await asyncio.sleep(0.05)

    asyncio.run(_run())
    assert received == [{"command": "rotate"}]


def test_add_listener_command_exports_cw_driver_root() -> None:
    registry = DriverInterfaceRegistry()
    cmd = TopicSpec(
        namespace="twin",
        leaf="command",
        payload_schema_ref="TwinCommandPayload",
        description="Commands",
    )
    registry.add_listener(
        cmd,
        CallbackGroup(lambda _e: None),
        command=CommandArgs(name="stop"),
    )
    raw = registry.to_cw_driver_dict(registry_id="acme/test")
    assert raw["mqtt"]["twin"]["command"]["payload_schema_ref"] == "TwinCommandPayload"
    assert "stop" in raw["mqtt"]["commands"]["supported"]


def test_default_management_commands() -> None:
    registry = DriverInterfaceRegistry()
    default_management_commands(
        registry,
        on_controller_changed=CallbackGroup(lambda _e: None),
        on_teleoperate=CallbackGroup(lambda _e: None),
    )
    table = registry.command_dispatch_table(DriverOperationMode.NO_OP)
    assert "controller-changed" in table
    assert "teleoperate" in table


def test_catalog_hidden_command_dispatches_but_is_not_advertised() -> None:
    registry = DriverInterfaceRegistry()
    cmd = TopicSpec(
        namespace="twin",
        leaf="command",
        payload_schema_ref="TwinCommandPayload",
    )
    registry.add_listener(
        cmd,
        CallbackGroup(lambda _e: None),
        command=CommandArgs(name="stop", catalog_hidden=True),
    )
    registry.add_listener(
        cmd,
        CallbackGroup(lambda _e: None),
        command=CommandArgs(name="grab"),
    )
    raw = registry.to_cw_driver_dict(registry_id="acme/test")
    supported = raw["mqtt"]["commands"]["supported"]
    assert "grab" in supported
    assert "stop" not in supported  # hidden from catalog
    # …but still dispatchable on the wire.
    table = registry.command_dispatch_table(DriverOperationMode.NO_OP)
    assert "stop" in table


def test_default_management_commands_catalog_hidden() -> None:
    registry = DriverInterfaceRegistry()
    default_management_commands(
        registry,
        on_controller_changed=CallbackGroup(lambda _e: None),
        on_teleoperate=CallbackGroup(lambda _e: None),
        on_remoteoperate=CallbackGroup(lambda _e: None),
        on_stop=CallbackGroup(lambda _e: None),
        catalog_hidden=True,
    )
    raw = registry.to_cw_driver_dict(registry_id="acme/test")
    commands = raw["mqtt"].get("commands", {})
    supported = commands.get("supported", [])
    names = {e["name"] if isinstance(e, dict) else e for e in supported}
    assert names.isdisjoint(
        {"controller-changed", "teleoperate", "remoteoperate", "stop"}
    )
    # All four still dispatch.
    table = registry.command_dispatch_table(DriverOperationMode.NO_OP)
    for cmd_name in ("controller-changed", "teleoperate", "remoteoperate", "stop"):
        assert cmd_name in table


def test_duplicate_command_raises() -> None:
    registry = DriverInterfaceRegistry()
    cmd = TopicSpec(
        namespace="twin",
        leaf="command",
        payload_schema_ref="TwinCommandPayload",
    )
    registry.add_listener(
        cmd,
        CallbackGroup(lambda _e: None),
        command=CommandArgs(name="stop"),
    )
    registry.add_listener(
        cmd,
        CallbackGroup(lambda _e: None),
        command=CommandArgs(name="stop"),
    )
    try:
        registry.command_dispatch_table(DriverOperationMode.NO_OP)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()


def test_enable_zenoh_exports_zenoh_section() -> None:
    registry = DriverInterfaceRegistry()
    imu = TopicSpec(
        topic_slug=TWIN_IMU_TOPIC_SLUG,
        payload_schema_ref="ImuPayload",
        description="IMU",
        enable_zenoh=True,
        zenoh_channel="imu",
    )
    registry.add_publisher(
        imu,
        CallbackGroup(lambda: {"sensor_id": "x"}),
        publisher=PublisherArgs(rate_hz=50.0),
    )
    raw = registry.to_cw_driver_dict(registry_id="acme/test")
    assert "zenoh" in raw
    assert raw["zenoh"]["channels"]["imu"]["payload_schema_ref"] == "ImuPayload"


def test_inferred_publish_mode_for_dual_spec() -> None:
    dual = TopicSpec(
        topic_slug=TWIN_IMU_TOPIC_SLUG,
        payload_schema_ref="ImuPayload",
        enable_zenoh=True,
        zenoh_channel="imu",
    )
    assert effective_publish_mode(dual, PublisherArgs()) == "dual"


def test_registry_zenoh_enabled_without_data_backend_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cyberwave.driver.interface.registry_mixin import InterfaceRegistryMixin

    monkeypatch.delenv("CYBERWAVE_DATA_BACKEND", raising=False)

    class _Probe(InterfaceRegistryMixin):
        def __init__(self) -> None:
            self._interface = DriverInterfaceRegistry()
            self._registry_zenoh_bus = None
            self._registry_zenoh_subscriptions = []

    probe = _Probe()
    imu = TopicSpec(
        topic_slug=TWIN_IMU_TOPIC_SLUG,
        payload_schema_ref="ImuPayload",
        enable_zenoh=True,
        zenoh_channel="imu",
    )
    probe._interface.add_publisher(
        imu,
        CallbackGroup(lambda: {}),
        publisher=PublisherArgs(rate_hz=1.0),
    )
    assert probe._registry_zenoh_requested() is True
    assert probe._registry_zenoh_enabled() is True


def test_enable_zenoh_forbidden_on_command() -> None:
    with pytest.raises(ValueError, match="MQTT-only"):
        TopicSpec(
            namespace="twin",
            leaf="command",
            payload_schema_ref="TwinCommandPayload",
            enable_zenoh=True,
            zenoh_channel="commands/bad",
        )


def test_enable_zenoh_defaults_channel_from_slug() -> None:
    imu = TopicSpec(
        topic_slug=TWIN_IMU_TOPIC_SLUG,
        payload_schema_ref="ImuPayload",
        enable_zenoh=True,
    )
    assert imu.resolved_zenoh_channel() == "imu"


def test_unwire_interface_without_client_does_not_mask_startup_error() -> None:
    """Teardown after a failed startup (no cloud client) must be a safe no-op.

    Regression: when startup fails before MQTT connects, run_async's finally
    block calls _unwire_interface_from_registry. If that requires a client it
    raises RuntimeError and masks the real startup exception.
    """

    class _Bare(InterfaceRegistryMixin):
        def __init__(self) -> None:
            self._cw = None
            self._wired_mqtt_handlers = [("cyberwave/twin/x/command", lambda _e: None)]

        async def _teardown_zenoh_registry(self) -> None:
            return None

    bare = _Bare()
    asyncio.run(bare._unwire_interface_from_registry())  # must not raise
    assert bare._wired_mqtt_handlers == []
