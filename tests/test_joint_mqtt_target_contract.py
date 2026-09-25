"""MQTT joint contract: measured state vs commanded targets.

Measured fields (``positions`` / ``velocities`` / ``efforts``) and target
fields (``target_positions`` / ``target_velocities`` / ``target_efforts``) must
parse into separate buckets so a consumer never renders a command as state.
"""

from __future__ import annotations

from cyberwave.data.state_representation import (
    joint_dict_from_payload,
    joint_target_dict_from_payload,
    parse_joint_mqtt_payload,
)


def test_measured_positions_only_populate_measured_bucket() -> None:
    parsed = parse_joint_mqtt_payload(
        {
            "source_type": "sim",
            "positions": {"_1": 0.5, "_2": -0.25},
            "velocities": {"_1": 0.1, "_2": 0.0},
            "efforts": {"_1": 1.0, "_2": 2.0},
        }
    )
    assert parsed.positions == {"_1": 0.5, "_2": -0.25}
    assert parsed.velocities == {"_1": 0.1, "_2": 0.0}
    assert parsed.efforts == {"_1": 1.0, "_2": 2.0}
    assert parsed.target_positions == {}
    assert parsed.target_velocities == {}
    assert parsed.target_efforts == {}
    assert parsed.has_measured is True
    assert parsed.has_targets is False


def test_target_positions_only_populate_target_bucket() -> None:
    parsed = parse_joint_mqtt_payload(
        {
            "source_type": "sim_tele",
            "target_positions": {"_1": 0.7, "_2": 0.2},
            "target_velocities": {"_1": 0.0, "_2": 0.0},
            "target_efforts": {"_1": 0.0, "_2": 0.0},
        }
    )
    assert parsed.target_positions == {"_1": 0.7, "_2": 0.2}
    assert parsed.target_velocities == {"_1": 0.0, "_2": 0.0}
    assert parsed.target_efforts == {"_1": 0.0, "_2": 0.0}
    # A command payload must NOT leak into the measured bucket.
    assert parsed.positions == {}
    assert parsed.velocities == {}
    assert parsed.efforts == {}
    assert parsed.has_measured is False
    assert parsed.has_targets is True


def test_payload_carrying_both_keeps_buckets_separate() -> None:
    parsed = parse_joint_mqtt_payload(
        {
            "source_type": "sim",
            "positions": {"_1": 0.5},
            "target_positions": {"_1": 0.9},
        }
    )
    assert parsed.positions == {"_1": 0.5}
    assert parsed.target_positions == {"_1": 0.9}
    assert parsed.has_measured is True
    assert parsed.has_targets is True


def test_single_joint_target_state_routes_to_target_bucket() -> None:
    parsed = parse_joint_mqtt_payload(
        {
            "source_type": "tele",
            "joint_name": "_3",
            "target_joint_state": {"position": 1.1, "velocity": 0.2, "effort": 0.3},
        }
    )
    assert parsed.target_positions == {"_3": 1.1}
    assert parsed.target_velocities == {"_3": 0.2}
    assert parsed.target_efforts == {"_3": 0.3}
    assert parsed.positions == {}


def test_convenience_helpers_split_measured_and_target() -> None:
    payload = {
        "source_type": "sim",
        "positions": {"_1": 0.5},
        "target_positions": {"_1": 0.9},
    }
    assert joint_dict_from_payload(payload) == {"_1": 0.5}
    assert joint_target_dict_from_payload(payload) == {"_1": 0.9}


def test_controllable_filter_applies_to_both_buckets() -> None:
    parsed = parse_joint_mqtt_payload(
        {
            "positions": {"_1": 0.1, "_x": 9.9},
            "target_positions": {"_1": 0.2, "_x": 8.8},
        },
        controllable_names=frozenset({"_1"}),
    )
    assert parsed.positions == {"_1": 0.1}
    assert parsed.target_positions == {"_1": 0.2}
