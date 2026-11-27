"""Joint ``acceleration`` is its own wire channel, never an alias of ``efforts``.

The joint bus carries four measured channels (``positions`` / ``velocities`` /
``accelerations`` / ``efforts``) and their ``target_*`` setpoints. The SDK used to
map ``joints.set(..., what_data="acceleration")`` onto ``efforts`` (a 2 rad/s²
request reached the driver as a 2 N·m torque command) and to copy inbound efforts
into the cached ``acceleration`` field (``get(what_data=["acceleration"])``
reported N·m). Acceleration is rad/s² (m/s² for prismatic joints) and travels on
``accelerations``; effort stays on ``efforts``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cyberwave.data.state_representation import (
    MEASURED_JOINT_FIELDS,
    TARGET_JOINT_FIELDS,
    parse_joint_mqtt_payload,
)
from cyberwave.mqtt import CyberwaveMQTTClient
from cyberwave.twin.capabilities import joints as _joints
from cyberwave.twin.classes import JointTwin


def _joint_twin() -> JointTwin:
    client = SimpleNamespace(
        mqtt=MagicMock(),
        assets=MagicMock(),
        config=SimpleNamespace(runtime_mode="live", topic_prefix="", source_type="tele"),
        twins=SimpleNamespace(api=None),
    )
    client.assets.get.return_value = SimpleNamespace(
        metadata={
            "mqtt": {
                "topics": {"cyberwave/joint/{twin_uuid}/update": {}},
                "commands": {"supported": []},
            }
        }
    )
    return JointTwin(client, SimpleNamespace(uuid="arm-1", name="Arm", asset_uuid="asset-1"))


def _cached(handle: _joints.JointsHandle, joint: str) -> list[dict[str, float]]:
    return [state[joint] for state in handle._curr_joints_by_mode.values() if joint in state]


# --- write side: joints.set -------------------------------------------------


def test_set_acceleration_publishes_accelerations_not_efforts() -> None:
    twin = _joint_twin()
    with patch.object(_joints, "controllable_joint_names", return_value=["j1"]):
        with patch.object(twin, "_prepare_outbound_command"):
            twin.joints.set({"j1": 2.0}, what_data="acceleration", mode="relative")
    payload = twin._outbound_log[-1].payload
    assert payload["accelerations"] == {"j1": 2.0}
    assert "efforts" not in payload
    assert "velocities" not in payload


def test_set_acceleration_absolute_rides_alongside_positions() -> None:
    # Absolute mode needs a positions map; the acceleration channel must not
    # sneak in as anything else when it is missing.
    twin = _joint_twin()
    with patch.object(_joints, "controllable_joint_names", return_value=["j1"]):
        with patch.object(twin, "_prepare_outbound_command"):
            with pytest.raises(ValueError, match="positions"):
                twin.joints.set({"j1": 2.0}, what_data="acceleration", mode="absolute")
    assert twin._outbound_log == []


def test_set_effort_still_publishes_efforts_only() -> None:
    twin = _joint_twin()
    with patch.object(_joints, "controllable_joint_names", return_value=["j1"]):
        with patch.object(twin, "_prepare_outbound_command"):
            twin.joints.set({"j1": 2.0}, what_data="effort", mode="relative")
    payload = twin._outbound_log[-1].payload
    assert payload["efforts"] == {"j1": 2.0}
    assert "accelerations" not in payload


def test_set_acceleration_updates_only_the_cached_acceleration() -> None:
    twin = _joint_twin()
    with patch.object(_joints, "controllable_joint_names", return_value=["j1"]):
        with patch.object(twin, "_prepare_outbound_command"):
            twin.joints.set({"j1": 2.0}, what_data="acceleration", mode="relative")
        cached = _cached(twin.joints, "j1")
    assert cached
    for state in cached:
        assert state["acceleration"] == 2.0
        assert state["effort"] == 0.0


# --- read side: inbound /update ---------------------------------------------


def test_inbound_accelerations_fill_acceleration_and_not_effort() -> None:
    twin = _joint_twin()
    with patch.object(_joints, "controllable_joint_names", return_value=["j1"]):
        handle = twin.joints
        handle._on_joint_payload(
            {
                "positions": {"j1": 0.1},
                "accelerations": {"j1": 3.5},
                "efforts": {"j1": 2.0},
                "source_type": "edge",
            }
        )
        cached = _cached(handle, "j1")
    assert cached, "the joint update was not cached"
    for state in cached:
        assert state["acceleration"] == 3.5
        assert state["effort"] == 2.0


def test_inbound_effort_is_not_reported_as_acceleration() -> None:
    twin = _joint_twin()
    with patch.object(_joints, "controllable_joint_names", return_value=["j1"]):
        handle = twin.joints
        handle._on_joint_payload(
            {"positions": {"j1": 0.1}, "efforts": {"j1": 2.0}, "source_type": "edge"}
        )
        cached = _cached(handle, "j1")
    assert cached, "the joint update was not cached"
    for state in cached:
        assert state["effort"] == 2.0
        assert state["acceleration"] == 0.0


def test_inbound_acceleration_only_update_is_applied() -> None:
    twin = _joint_twin()
    with patch.object(_joints, "controllable_joint_names", return_value=["j1"]):
        handle = twin.joints
        handle._on_joint_payload({"positions": {"j1": 0.1}, "source_type": "edge"})
        handle._on_joint_payload(
            {"positions": {}, "accelerations": {"j1": -1.25}, "source_type": "edge"}
        )
        cached = _cached(handle, "j1")
    for state in cached:
        assert state["position"] == 0.1  # untouched by an acceleration-only update
        assert state["acceleration"] == -1.25


# --- parser -----------------------------------------------------------------


def test_parser_channel_sets_include_accelerations() -> None:
    assert MEASURED_JOINT_FIELDS == {"positions", "velocities", "accelerations", "efforts"}
    assert TARGET_JOINT_FIELDS == {
        "target_positions",
        "target_velocities",
        "target_accelerations",
        "target_efforts",
    }


def test_parser_aggregated_keeps_accelerations_and_targets_apart() -> None:
    parsed = parse_joint_mqtt_payload(
        {
            "positions": {"j1": 0.1},
            "accelerations": {"j1": 0.0},  # explicit zero is a real reading
            "efforts": {"j1": 4.0},
            "target_accelerations": {"j1": 9.0},
        }
    )
    assert parsed.accelerations == {"j1": 0.0}
    assert parsed.efforts == {"j1": 4.0}
    assert parsed.target_accelerations == {"j1": 9.0}
    assert parsed.has_targets


def test_parser_absent_accelerations_stay_absent() -> None:
    parsed = parse_joint_mqtt_payload({"positions": {"j1": 0.1}, "efforts": {"j1": 4.0}})
    assert parsed.accelerations == {}
    assert parsed.target_accelerations == {}


def test_parser_single_joint_acceleration() -> None:
    parsed = parse_joint_mqtt_payload(
        {
            "joint_name": "j1",
            "joint_state": {"position": 0.1, "acceleration": 1.5, "effort": 4.0},
            "target_joint_state": {"position": 0.2, "acceleration": 2.5},
        }
    )
    assert parsed.accelerations == {"j1": 1.5}
    assert parsed.efforts == {"j1": 4.0}
    assert parsed.target_accelerations == {"j1": 2.5}


def test_parser_flat_payload_does_not_treat_accelerations_as_a_joint() -> None:
    parsed = parse_joint_mqtt_payload(
        {"j1": 0.1, "accelerations": 3.0, "target_accelerations": 2.0, "source_type": "edge"}
    )
    assert parsed.positions == {"j1": 0.1}


def test_parser_positional_construction_still_works() -> None:
    from cyberwave.data.state_representation.space.joint import ParsedJointMqttUpdate

    parsed = ParsedJointMqttUpdate({"j1": 0.1}, {}, {})
    assert parsed.accelerations == {}
    assert parsed.has_measured


# --- low-level publisher ----------------------------------------------------


def _bare_mqtt_client() -> CyberwaveMQTTClient:
    client = CyberwaveMQTTClient.__new__(CyberwaveMQTTClient)
    client.topic_prefix = ""
    client.publish = MagicMock()
    client._get_effective_source_type = lambda st: st or "edge"
    client._handle_twin_update_with_telemetry = MagicMock()
    client._is_rate_limited = lambda key: False
    return client


def test_update_joints_state_publishes_accelerations() -> None:
    client = _bare_mqtt_client()
    client.update_joints_state("tw", {"j1": 0.1}, accelerations={"j1": 2.0})
    message = client.publish.call_args.args[1]
    assert message["positions"] == {"j1": 0.1}
    assert message["accelerations"] == {"j1": 2.0}
    assert "efforts" not in message


def test_update_joints_state_target_accelerations_only() -> None:
    client = _bare_mqtt_client()
    client.update_joints_state("tw", {}, accelerations={"j1": 2.0}, as_targets=True)
    message = client.publish.call_args.args[1]
    assert message["target_accelerations"] == {"j1": 2.0}
    assert "accelerations" not in message
    assert "target_positions" not in message
    assert "target_efforts" not in message


def test_update_joint_state_single_acceleration() -> None:
    client = _bare_mqtt_client()
    client.update_joint_state("tw", "j1", position=0.1, acceleration=2.0)
    message = client.publish.call_args.args[1]
    assert message["joint_state"] == {"position": 0.1, "acceleration": 2.0}
