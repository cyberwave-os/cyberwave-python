"""Tests for FlyingTwin's canonical drone-command surface.

This file complements ``test_flying_twin_hovering.py`` (which focuses on
the local hovering-metadata bookkeeping) by asserting the wire contract
shared with every Cyberwave drone driver — see
``cyberwave-edge-nodes/cyberwave-edge-dji-mini-android`` for the
authoritative consumer:

  topic   : ``{topic_prefix}cyberwave/twin/{twin_uuid}/command``
  payload : ``{"source_type", "command", "data", "timestamp"}``

Covered:
- takeoff / land / hover go to the canonical topic with the proper
  ``source_type`` (``tele`` in live mode, ``sim_tele`` in simulation)
- return_to_home / cancel_return_to_home / cancel_takeoff /
  cancel_landing / set_home_here / start_compass_calibration /
  stop_compass_calibration / reboot / emergency_stop
- gimbal_rotate (default + per-axis + relative + duration)
- gimbal_recenter sends pitch=0 / mode=absolute (matches the
  ``N`` keyboard binding in ``controller:dji-keyboard:v1``)
- gimbal_rotate_speed forwards 0.1°/s units
- Inherited LocomoteTwin behaviour: move_forward also lands on the
  canonical topic (since Mini-class drones can be teleop'd off-RC for
  simulator-only flows even if the real driver currently drops it)
- topic_prefix is honoured (multi-environment broker routing)
- explicit ``source_type="sim"`` is normalised to ``sim_tele``
- invalid source_type raises ValueError before MQTT is touched
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from cyberwave.twin import FlyingTwin


CANONICAL_TOPIC = "cyberwave/twin/drone-uuid/command"


def _make_client(
    *,
    runtime_mode: str = "live",
    topic_prefix: str = "",
) -> SimpleNamespace:
    mqtt = MagicMock()
    mqtt.connected = True
    return SimpleNamespace(
        mqtt=mqtt,
        config=SimpleNamespace(
            runtime_mode=runtime_mode,
            source_type="edge" if runtime_mode == "live" else "sim",
            topic_prefix=topic_prefix,
        ),
        twins=MagicMock(),
    )


def _make_drone(
    *,
    runtime_mode: str = "live",
    topic_prefix: str = "",
) -> tuple[FlyingTwin, SimpleNamespace]:
    client = _make_client(runtime_mode=runtime_mode, topic_prefix=topic_prefix)
    data = SimpleNamespace(
        uuid="drone-uuid",
        name="DJI Mini 4 Pro",
        metadata={},
    )
    return FlyingTwin(client, data), client


def _last_command_publish(client: SimpleNamespace) -> tuple[str, dict[str, Any]]:
    """
    Return ``(topic, payload)`` for the most recent canonical-command
    publish. Skips any metadata side-publishes (the hover-status
    twins.update path doesn't go through MQTT, but other tests in the
    suite might add other publishes later — this keeps us robust).
    """
    matching = [
        (call.args[0], call.args[1])
        for call in client.mqtt.publish.call_args_list
        if isinstance(call.args[1], dict)
        and call.args[1].get("command") is not None
    ]
    assert matching, (
        f"No canonical-command publish recorded; "
        f"actual calls: {client.mqtt.publish.call_args_list}"
    )
    return matching[-1]


# ---------------------------------------------------------------------------
# Topic-prefix routing
# ---------------------------------------------------------------------------


class TestTopicPrefix:
    def test_empty_prefix_publishes_to_canonical_topic(self):
        twin, client = _make_drone(topic_prefix="")
        twin.takeoff(altitude=1.5)

        topic, _ = _last_command_publish(client)
        assert topic == CANONICAL_TOPIC

    def test_environment_prefix_is_honoured(self):
        twin, client = _make_drone(topic_prefix="dev/")
        twin.takeoff(altitude=1.5)

        topic, _ = _last_command_publish(client)
        assert topic == "dev/cyberwave/twin/drone-uuid/command"


# ---------------------------------------------------------------------------
# Source-type resolution
# ---------------------------------------------------------------------------


class TestSourceType:
    def test_live_mode_defaults_to_tele(self):
        twin, client = _make_drone(runtime_mode="live")
        twin.land()

        _, payload = _last_command_publish(client)
        assert payload["source_type"] == "tele"

    def test_simulation_mode_defaults_to_sim_tele(self):
        twin, client = _make_drone(runtime_mode="simulation")
        twin.land()

        _, payload = _last_command_publish(client)
        assert payload["source_type"] == "sim_tele"

    def test_explicit_sim_is_normalised_to_sim_tele(self):
        """Legacy callers that pass ``source_type='sim'`` keep working."""
        twin, client = _make_drone(runtime_mode="live")
        twin.land(source_type="sim")

        _, payload = _last_command_publish(client)
        assert payload["source_type"] == "sim_tele"

    def test_invalid_source_type_raises_before_publish(self):
        twin, client = _make_drone()
        with pytest.raises(ValueError, match="Invalid source type"):
            twin.takeoff(source_type="edit")

        client.mqtt.publish.assert_not_called()


# ---------------------------------------------------------------------------
# Aircraft-state commands (data-less / single-field)
# ---------------------------------------------------------------------------


class TestAircraftStateCommands:
    @pytest.mark.parametrize(
        "method_name,command,expected_data",
        [
            ("cancel_takeoff", "cancel_takeoff", {}),
            ("cancel_landing", "cancel_landing", {}),
            ("return_to_home", "return_to_home", {}),
            ("cancel_return_to_home", "cancel_return_to_home", {}),
            ("set_home_here", "set_home_here", {}),
            ("start_compass_calibration", "start_compass_calibration", {}),
            ("stop_compass_calibration", "stop_compass_calibration", {}),
            ("reboot", "reboot", {}),
            ("emergency_stop", "emergency_stop", {}),
        ],
    )
    def test_each_command_publishes_canonical_envelope(
        self, method_name: str, command: str, expected_data: dict
    ):
        twin, client = _make_drone()
        getattr(twin, method_name)()

        topic, payload = _last_command_publish(client)
        assert topic == CANONICAL_TOPIC
        assert payload["command"] == command
        assert payload["data"] == expected_data
        assert payload["source_type"] == "tele"
        assert isinstance(payload["timestamp"], float)


# ---------------------------------------------------------------------------
# Gimbal — angle command
# ---------------------------------------------------------------------------


class TestGimbalRotate:
    def test_recenter_default_pitch_absolute(self):
        twin, client = _make_drone()
        twin.gimbal_recenter()

        topic, payload = _last_command_publish(client)
        assert topic == CANONICAL_TOPIC
        assert payload["command"] == "gimbal_rotate"
        assert payload["data"] == {"pitch": 0.0, "mode": "absolute"}

    def test_pitch_only(self):
        twin, client = _make_drone()
        twin.gimbal_rotate(pitch=-45.0)

        _, payload = _last_command_publish(client)
        assert payload["data"] == {"pitch": -45.0, "mode": "absolute"}

    def test_relative_mode_with_duration(self):
        twin, client = _make_drone()
        twin.gimbal_rotate(pitch=10.0, mode="relative", duration=2.5)

        _, payload = _last_command_publish(client)
        # Field order doesn't matter for dict equality.
        assert payload["data"] == {
            "pitch": 10.0,
            "duration": 2.5,
            "mode": "relative",
        }

    def test_pitch_roll_yaw_all_set(self):
        twin, client = _make_drone()
        twin.gimbal_rotate(pitch=-30.0, roll=5.0, yaw=15.0)

        _, payload = _last_command_publish(client)
        assert payload["data"] == {
            "pitch": -30.0,
            "roll": 5.0,
            "yaw": 15.0,
            "mode": "absolute",
        }

    def test_unspecified_axes_are_omitted_from_payload(self):
        """
        The driver distinguishes "leave this axis alone" (key absent)
        from "command axis to 0" (key=0). The SDK must preserve that
        distinction by only emitting axes the caller explicitly set.
        """
        twin, client = _make_drone()
        twin.gimbal_rotate(pitch=-45.0)

        _, payload = _last_command_publish(client)
        assert "roll" not in payload["data"]
        assert "yaw" not in payload["data"]
        assert "duration" not in payload["data"]

    def test_no_axes_sends_only_mode(self):
        """
        Calling ``gimbal_rotate()`` with no kwargs is a no-op-with-mode
        — the driver treats this the same as ``gimbal_recenter()`` and
        recenters to pitch 0 / absolute.
        """
        twin, client = _make_drone()
        twin.gimbal_rotate()

        _, payload = _last_command_publish(client)
        assert payload["data"] == {"mode": "absolute"}


# ---------------------------------------------------------------------------
# Gimbal — speed command
# ---------------------------------------------------------------------------


class TestGimbalRotateSpeed:
    def test_pitch_only_in_deci_deg_per_sec(self):
        twin, client = _make_drone()
        twin.gimbal_rotate_speed(pitch=100.0)  # 10°/s

        topic, payload = _last_command_publish(client)
        assert topic == CANONICAL_TOPIC
        assert payload["command"] == "gimbal_rotate_speed"
        assert payload["data"] == {"pitch": 100.0}

    def test_all_axes(self):
        twin, client = _make_drone()
        twin.gimbal_rotate_speed(pitch=50.0, roll=-25.0, yaw=10.0)

        _, payload = _last_command_publish(client)
        assert payload["data"] == {
            "pitch": 50.0,
            "roll": -25.0,
            "yaw": 10.0,
        }

    def test_no_axes_sends_empty_data(self):
        twin, client = _make_drone()
        twin.gimbal_rotate_speed()

        _, payload = _last_command_publish(client)
        assert payload["data"] == {}


# ---------------------------------------------------------------------------
# Inherited locomotion — move_forward goes through the canonical topic too
# ---------------------------------------------------------------------------


class TestLocomoteInheritance:
    def test_flying_twin_exposes_locomote_methods(self):
        twin, _ = _make_drone()
        # No AttributeError — FlyingTwin now inherits from LocomoteTwin.
        assert callable(twin.move_forward)
        assert callable(twin.move_backward)
        assert callable(twin.turn_left)
        assert callable(twin.turn_right)

    def test_move_forward_publishes_to_canonical_topic(self):
        twin, client = _make_drone(runtime_mode="simulation")
        twin.move_forward(1.5)

        topic, payload = _last_command_publish(client)
        assert topic == CANONICAL_TOPIC
        assert payload["command"] == "move_forward"
        assert payload["source_type"] == "sim_tele"
        assert payload["data"] == {"linear_x": 1.5, "angular_z": 0.0}
