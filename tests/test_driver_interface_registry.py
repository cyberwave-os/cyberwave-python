"""Driver interface registry and manifest export."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cyberwave.driver import (
    CallbackGroup,
    CommandArg,
    CommandArgs,
    DriverInterfaceRegistry,
    DriverOperationMode,
    TopicSpec,
    default_management_commands,
)
from cyberwave.driver.interface.args import PublisherArgs, effective_publish_mode
from cyberwave.driver.interface.registry import resolve_topic_path
from cyberwave.driver.interface.registry_mixin import InterfaceRegistryMixin
from cyberwave.manifest.driver_config import TWIN_IMU_TOPIC_SLUG


def test_typed_command_export_matches_shared_contract_and_yaml_round_trip(
    tmp_path,
) -> None:
    from cyberwave.driver.interface.cw_driver import (
        dump_cw_driver_yml,
        load_cw_driver_yml,
        resolve_driver_config_dict,
    )

    args = (
        CommandArg(
            "action",
            type="string",
            required=True,
            enum=("grip", "release", "reset"),
            description="Gripper action",
        ),
        CommandArg("speed", 0, "m/s", type="number", minimum=0, maximum=1),
        CommandArg("joints", type="array", required=True, min_items=1),
        CommandArg("enabled", False, type="boolean"),
        CommandArg("repetitions", 1, type="integer", minimum=1),
        CommandArg("options", {}, type="object"),
    )
    registry = DriverInterfaceRegistry()
    registry.add_listener(
        TopicSpec(
            namespace="twin",
            leaf="command",
            description="Commands",
            payload_schema_ref="TwinCommandPayload",
        ),
        CallbackGroup(lambda _e: None),
        command=CommandArgs(name="configured", args=args),
    )
    root = registry.to_cw_driver_dict(registry_id="test/typed-command")
    expected = json.loads(
        (Path(__file__).parent / "fixtures/driver-command-arguments.json").read_text()
    )
    assert root["mqtt"]["commands"]["supported"][0]["args"] == expected
    exported = dump_cw_driver_yml(root, tmp_path / "cw-driver.yml")
    assert resolve_driver_config_dict(load_cw_driver_yml(exported)) == root


def test_legacy_command_args_keep_positional_constructor_and_exact_wire_shape() -> None:
    assert CommandArg("forward", 0, "m").to_catalog_dict() == {
        "name": "forward",
        "default": 0,
        "unit": "m",
    }
    assert CommandArg("action").to_catalog_dict() == {
        "name": "action",
        "default": None,
        "unit": None,
    }
    with pytest.raises(TypeError):
        CommandArg("action", None, None, "string")
    with pytest.raises(ValueError, match="require type"):
        CommandArg("action", required=True).to_catalog_dict()


class _Probe(InterfaceRegistryMixin):
    def get_driver_manifest(self, *, compiled: bool = False) -> dict:
        return {"mqtt": {"commands": {"supported": ["home"]}}}

    def _emit_driver_info_after_schema(self, schema: dict) -> None:
        pass


_COMPILED = {"commands": {"supported": ["home"]}}


def _probe_on_twin(record=None, mqtt=None):
    """A probe whose twin holds *record* and an ``mqtt`` catalog block."""
    probe = _Probe()
    metadata = {} if mqtt is None else {"mqtt": mqtt}
    if record is not None:
        metadata[_Probe.CATALOG_HASH_METADATA_KEY] = record
    twin = SimpleNamespace(
        uuid="t-1",
        driver=MagicMock(),
        client=SimpleNamespace(twins=MagicMock()),
        _data={"metadata": metadata},
    )

    def set_schema(driver_config, *, merge):
        # Like the real handle: the twin is reloaded with the compiled blocks.
        twin._data = {"metadata": {**twin._data["metadata"], "mqtt": _COMPILED}}
        return {"mqtt": _COMPILED}

    twin.driver.set_schema.side_effect = set_schema
    twin.driver.get_schemas.return_value = {"mqtt": _COMPILED}
    probe._twin = twin
    return probe, twin


def _recorded(twin) -> dict:
    ((_, kwargs),) = twin.client.twins.update.call_args_list
    return kwargs["metadata"][_Probe.CATALOG_HASH_METADATA_KEY]


def test_a_changed_catalog_replaces_the_twins_and_records_its_hash() -> None:
    probe, twin = _probe_on_twin(record={"input": "an older catalog"})
    probe.register_interface_on_twin()
    twin.driver.set_schema.assert_called_once_with(probe.cw_driver, merge=False)
    record = _recorded(twin)
    assert record["input"] == probe.catalog_hash()
    # The digest of what the twin holds after the write, not before it.
    assert record["written"] == probe._written_catalog_hash({"mqtt": _COMPILED})


def test_an_unchanged_catalog_is_not_rewritten() -> None:
    probe, twin = _probe_on_twin()
    probe.register_interface_on_twin()
    probe, twin = _probe_on_twin(record=_recorded(twin), mqtt=_COMPILED)
    schemas = probe.register_interface_on_twin()
    twin.driver.set_schema.assert_not_called()
    twin.client.twins.update.assert_not_called()
    assert schemas == twin.driver.get_schemas.return_value
    # Forced, or asked to merge into a changed one, it is written.
    probe.register_interface_on_twin(force=True, merge=True)
    twin.driver.set_schema.assert_called_once_with(probe.cw_driver, merge=True)


def test_a_catalog_another_writer_replaced_is_registered_again() -> None:
    # v2 registers; a rollback to a driver that predates the record (or an
    # edit) rewrites the catalog and leaves the record; v2 starts again.
    probe, twin = _probe_on_twin()
    probe.register_interface_on_twin()
    older = {"commands": {"supported": ["home", "legacy"]}}
    probe, twin = _probe_on_twin(record=_recorded(twin), mqtt=older)
    probe.register_interface_on_twin()
    twin.driver.set_schema.assert_called_once_with(probe.cw_driver, merge=False)


def test_the_catalog_hash_ignores_key_order() -> None:
    first, second = _Probe(), _Probe()
    second.get_driver_manifest = lambda compiled=False: {  # type: ignore[method-assign]
        "mqtt": {"commands": {"supported": ["home"]}}
    }
    assert first.catalog_hash() == second.catalog_hash()
    second.get_driver_manifest = lambda compiled=False: {  # type: ignore[method-assign]
        "mqtt": {"commands": {"supported": ["home", "stop"]}}
    }
    assert first.catalog_hash() != second.catalog_hash()


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


def test_fallback_state_topic_substitutes_the_twin_uuid() -> None:
    """New state leaves are not necessarily enumerated in the static slug map."""
    motion = TopicSpec(
        namespace="state",
        leaf="motion",
        payload_schema_ref="TwinMotionStatusPayload",
    )

    assert resolve_topic_path(motion, "twin-123", prefix="local") == (
        "localcyberwave/twin/twin-123/state/motion"
    )


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


def test_default_management_commands_always_hides_teleop_controller_from_catalog() -> (
    None
):
    """controller-changed/teleoperate/remoteoperate are internal plumbing, never
    advertised as robot capabilities — even when catalog_hidden defaults to False.
    ``stop`` is unaffected and stays visible (existing behavior)."""
    registry = DriverInterfaceRegistry()
    default_management_commands(
        registry,
        on_controller_changed=CallbackGroup(lambda _e: None),
        on_teleoperate=CallbackGroup(lambda _e: None),
        on_remoteoperate=CallbackGroup(lambda _e: None),
        on_stop=CallbackGroup(lambda _e: None),
    )
    raw = registry.to_cw_driver_dict(registry_id="acme/test")
    commands = raw["mqtt"].get("commands", {})
    supported = commands.get("supported", [])
    names = {e["name"] if isinstance(e, dict) else e for e in supported}
    assert names.isdisjoint({"controller-changed", "teleoperate", "remoteoperate"})
    assert "stop" in names
    # All four still dispatch.
    table = registry.command_dispatch_table(DriverOperationMode.NO_OP)
    for cmd_name in ("controller-changed", "teleoperate", "remoteoperate", "stop"):
        assert cmd_name in table


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


def test_command_topic_subscribes_at_qos1() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    class Probe(InterfaceRegistryMixin):
        def __init__(self) -> None:
            self._cw = SimpleNamespace(
                mqtt=MagicMock(), config=SimpleNamespace(source_type="edge")
            )
            self._interface = DriverInterfaceRegistry()
            self._operation_mode = DriverOperationMode.NO_OP
            self._wired_mqtt_handlers = []

        def _twin_uuid_for_wire(self) -> str:
            return "arm-1"

        def _mqtt_prefix_for_wire(self) -> str:
            return ""

        async def _wire_zenoh_from_registry(self) -> None:
            pass

    probe = Probe()
    probe._interface.add_listener(
        TopicSpec(namespace="twin", leaf="command", payload_schema_ref="TwinCommandPayload"),
        CallbackGroup(lambda _e: None),
        command=CommandArgs(name="grip", args=()),
    )
    asyncio.run(probe._wire_interface_from_registry())
    call = probe._cw.mqtt.subscribe.call_args
    assert call.args[0] == "cyberwave/twin/arm-1/command"
    assert call.kwargs["qos"] == 1
