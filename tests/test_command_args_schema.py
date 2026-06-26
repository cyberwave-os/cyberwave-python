"""Command argument declarations on CommandArgs and their cw-driver emission."""

from __future__ import annotations

from cyberwave.driver import CommandArg, CommandArgs


def test_command_args_carries_arg_declarations() -> None:
    cmd = CommandArgs(
        name="ee_move",
        args=(
            CommandArg("forward", default=0.0, unit="m"),
            CommandArg("frame", default="tool"),
        ),
    )
    assert cmd.name == "ee_move"
    assert cmd.args[0].name == "forward"
    assert cmd.args[0].default == 0.0
    assert cmd.args[0].unit == "m"
    assert cmd.args[1].default == "tool"
    assert cmd.args[1].unit is None


def test_command_args_defaults_to_empty_tuple() -> None:
    assert CommandArgs(name="grab").args == ()


from cyberwave.driver import CallbackGroup, TopicSpec
from cyberwave.driver.interface.registry import DriverInterfaceRegistry


def _command_topic() -> TopicSpec:
    return TopicSpec(
        namespace="twin",
        leaf="command",
        payload_schema_ref="TwinCommandPayload",
    )


def test_cw_driver_emits_command_args() -> None:
    reg = DriverInterfaceRegistry()
    reg.add_listener(
        _command_topic(),
        CallbackGroup(callback=lambda payload: None),
        command=CommandArgs(
            name="ee_move",
            args=(CommandArg("forward", default=0.0, unit="m"),),
        ),
    )
    root = reg.to_cw_driver_dict(registry_id="agilex/piper", driver_family="ros_python")
    supported = root["mqtt"]["commands"]["supported"]
    entry = next(e for e in supported if isinstance(e, dict) and e["name"] == "ee_move")
    assert entry["args"] == [{"name": "forward", "default": 0.0, "unit": "m"}]


from cyberwave.manifest.driver_config import command_args


def test_command_args_reads_specs_args() -> None:
    bundle = {
        "topics": {"cyberwave/twin/{twin_uuid}/command": {}},
        "commands": {
            "supported": ["ee_move"],
            "specs": {
                "ee_move": {
                    "args": [
                        {"name": "forward", "default": 0.0, "unit": "m"},
                        {"name": "frame", "default": "tool"},
                    ]
                }
            },
        },
    }
    args = command_args(bundle, "ee_move")
    assert [a["name"] for a in args] == ["forward", "frame"]
    assert command_args(bundle, "unknown") == []
