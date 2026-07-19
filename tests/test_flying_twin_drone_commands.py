"""Tests for FlyingTwin's canonical drone-command surface.

This file complements ``test_flying_twin_hovering.py`` (which focuses on
the local hovering-metadata bookkeeping) by asserting the wire contract
shared with every Cyberwave drone driver — see
``cyberwave-edge-nodes/cyberwave-edge-dji-mini-android`` for the
authoritative consumer:

  topic   : ``{topic_prefix}cyberwave/twin/{twin_uuid}/command``
  payload : ``{"source_type", "command", "data", "timestamp"}``

Covered:
- takeoff / land / hover go to the canonical topic with ``source_type=tele``
  in live mode; they also publish (with ``source_type=sim_tele``) in a
  simulation runtime — flight is PLAYGROUND-compatible, since the browser
  playground renders these commands directly
- return_to_home / cancel_return_to_home / cancel_takeoff /
  cancel_landing / set_home_here / start_compass_calibration /
  stop_compass_calibration / reboot / emergency_stop
- gimbal_rotate (default + per-axis + relative + duration)
- gimbal_recenter sends pitch=0 / mode=absolute (matches the
  ``;`` keyboard binding in ``controller:dji-keyboard:v1``;
  held ``J`` / ``N`` drive the gimbal up / down via
  ``gimbal_rotate_speed``)
- gimbal_rotate_speed forwards 0.1°/s units
- pan_camera yaws the airframe via repeated turn_left / turn_right
  publishes inside the 500 ms watchdog, with an explicit zero at
  the end (Mini-class single-axis-gimbal workaround)
- Inherited LocomoteTwin behaviour: move_forward also lands on the
  canonical topic (since Mini-class drones can be teleop'd off-RC for
  simulator-only flows even if the real driver currently drops it)
- topic_prefix is honoured (multi-environment broker routing)
- explicit ``source_type="sim"`` is normalised to ``sim_tele``
- invalid source_type raises ValueError before MQTT is touched
"""

import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

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
    assets = MagicMock()
    assets.get.return_value = SimpleNamespace(metadata={"mqtt": {"topics": {}, "commands": {"supported": []}}})
    return SimpleNamespace(
        mqtt=mqtt,
        assets=assets,
        config=SimpleNamespace(
            runtime_mode=runtime_mode,
            source_type="edge" if runtime_mode == "live" else "sim",
            topic_prefix=topic_prefix,
        ),
        twins=SimpleNamespace(
            api=None,
            update=MagicMock(return_value=SimpleNamespace(uuid="drone-uuid", metadata={})),
        ),
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
        asset_uuid="asset-uuid",
        metadata={},
    )
    twin = FlyingTwin(client, data)
    twin._prepare_outbound_command = lambda: None  # type: ignore[method-assign]
    return twin, client


def _last_command_publish(twin: FlyingTwin) -> tuple[str, dict[str, Any]]:
    """Return topic/payload from PR1 mock outbound log."""
    assert twin._outbound_log, "Expected mock outbound command on twin"
    resolved = twin._outbound_log[-1]
    return resolved.topic, resolved.payload


def _run_drone_cmd(twin: FlyingTwin, fn, *args, **kwargs) -> tuple[str, dict[str, Any]]:
    with patch.object(twin, "_prepare_outbound_command"):
        fn(*args, **kwargs)
    return _last_command_publish(twin)


# ---------------------------------------------------------------------------
# Topic-prefix routing
# ---------------------------------------------------------------------------


class TestTopicPrefix:
    def test_empty_prefix_publishes_to_canonical_topic(self):
        twin, client = _make_drone(topic_prefix="")
        twin.takeoff(altitude=1.5)

        topic, _ = _last_command_publish(twin)
        assert topic == CANONICAL_TOPIC

    def test_environment_prefix_is_honoured(self):
        twin, client = _make_drone(topic_prefix="dev/")
        twin.takeoff(altitude=1.5)

        topic, _ = _last_command_publish(twin)
        assert topic == "dev/cyberwave/twin/drone-uuid/command"


# ---------------------------------------------------------------------------
# Source-type resolution
# ---------------------------------------------------------------------------


class TestSourceType:
    def test_live_mode_defaults_to_tele(self):
        twin, client = _make_drone(runtime_mode="live")
        twin.land()

        _, payload = _last_command_publish(twin)
        assert payload["source_type"] == "tele"

    def test_drone_commands_publish_sim_tele_in_simulation_mode(self):
        twin, client = _make_drone(runtime_mode="simulation")
        twin.land()

        _, payload = _last_command_publish(twin)
        assert payload["source_type"] == "sim_tele"

    def test_explicit_sim_is_normalised_to_sim_tele(self):
        """Legacy callers that pass ``source_type='sim'`` keep working."""
        twin, client = _make_drone(runtime_mode="live")
        twin.land(source_type="sim")

        _, payload = _last_command_publish(twin)
        assert payload["source_type"] == "sim_tele"

    def test_invalid_source_type_raises_before_publish(self):
        twin, client = _make_drone()
        with pytest.raises(ValueError, match="Invalid source type"):
            twin.takeoff(source_type="edit")

        assert twin._outbound_log == []


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

        topic, payload = _last_command_publish(twin)
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

        topic, payload = _last_command_publish(twin)
        assert topic == CANONICAL_TOPIC
        assert payload["command"] == "gimbal_rotate"
        assert payload["data"] == {"pitch": 0.0, "mode": "absolute"}

    def test_pitch_only(self):
        twin, client = _make_drone()
        twin.gimbal_rotate(pitch=-45.0)

        _, payload = _last_command_publish(twin)
        assert payload["data"] == {"pitch": -45.0, "mode": "absolute"}

    def test_relative_mode_with_duration(self):
        twin, client = _make_drone()
        twin.gimbal_rotate(pitch=10.0, mode="relative", duration=2.5)

        _, payload = _last_command_publish(twin)
        # Field order doesn't matter for dict equality.
        assert payload["data"] == {
            "pitch": 10.0,
            "duration": 2.5,
            "mode": "relative",
        }

    def test_pitch_roll_yaw_all_set(self):
        twin, client = _make_drone()
        twin.gimbal_rotate(pitch=-30.0, roll=5.0, yaw=15.0)

        _, payload = _last_command_publish(twin)
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

        _, payload = _last_command_publish(twin)
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

        _, payload = _last_command_publish(twin)
        assert payload["data"] == {"mode": "absolute"}


# ---------------------------------------------------------------------------
# Gimbal — speed command
# ---------------------------------------------------------------------------


class TestGimbalRotateSpeed:
    def test_pitch_only_in_deci_deg_per_sec(self):
        twin, client = _make_drone()
        twin.gimbal_rotate_speed(pitch=100.0)  # 10°/s

        topic, payload = _last_command_publish(twin)
        assert topic == CANONICAL_TOPIC
        assert payload["command"] == "gimbal_rotate_speed"
        assert payload["data"] == {"pitch": 100.0}

    def test_all_axes(self):
        twin, client = _make_drone()
        twin.gimbal_rotate_speed(pitch=50.0, roll=-25.0, yaw=10.0)

        _, payload = _last_command_publish(twin)
        assert payload["data"] == {
            "pitch": 50.0,
            "roll": -25.0,
            "yaw": 10.0,
        }

    def test_no_axes_sends_empty_data(self):
        twin, client = _make_drone()
        twin.gimbal_rotate_speed()

        _, payload = _last_command_publish(twin)
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
        # Live mode: locomotion inherited by flying twins publishes normally.
        twin, client = _make_drone(runtime_mode="live")
        twin.move_forward(1.5, duration=0)

        topic, payload = _last_command_publish(twin)
        assert topic == CANONICAL_TOPIC
        assert payload["command"] == "move_forward"
        assert payload["source_type"] == "tele"
        assert payload["data"] == {"linear_x": 1.5, "angular_z": 0.0}

    def test_move_forward_publishes_sim_tele_in_simulation(self):
        twin, client = _make_drone(runtime_mode="simulation")
        twin.move_forward(1.5, duration=0)

        _, payload = _last_command_publish(twin)
        assert payload["source_type"] == "sim_tele"


# ---------------------------------------------------------------------------
# pan_camera — Mini-class workaround for the missing gimbal yaw axis
# ---------------------------------------------------------------------------


def _yaw_publishes(twin: FlyingTwin) -> list[dict[str, Any]]:
    """All ``turn_left`` / ``turn_right`` payloads from mock outbound log."""
    return [
        resolved.payload
        for resolved in twin._outbound_log
        if resolved.command in {"turn_left", "turn_right"}
    ]


def _install_fake_clock(monkeypatch, ticks: list[float]) -> dict[str, list[float]]:
    """
    Replace ``cyberwave.twin.time.monotonic`` / ``time.sleep`` with deterministic
    stand-ins so the watchdog-refresh loop converges in zero wall time.

    ``ticks`` is the sequence ``monotonic()`` returns on successive calls; the
    helper records every ``sleep`` duration so tests can assert on the cadence.

    Note: ``cyberwave.__init__`` rebinds the name ``twin`` to the ``twin()``
    function from ``cyberwave.compact``, which shadows the ``cyberwave.twin``
    submodule attribute. We therefore reach for ``sys.modules`` directly to
    get the real module object that owns the ``time`` reference we want to
    patch.
    """
    import sys

    twin_module = sys.modules["cyberwave.twin"]

    iterator = iter(ticks)
    last = ticks[-1] if ticks else 0.0

    def fake_monotonic() -> float:
        nonlocal last
        try:
            last = next(iterator)
        except StopIteration:
            pass
        return last

    sleeps: list[float] = []
    monkeypatch.setattr(twin_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(twin_module.time, "sleep", lambda s: sleeps.append(s))
    return {"sleeps": sleeps}


class TestPanCamera:
    def test_positive_angle_yaws_left_with_refresh_then_zero(self, monkeypatch):
        # 90° at 30°/s → 3.0 s total; with refresh_hz=5 we expect 15
        # refresh publishes followed by one explicit zero. The fake
        # clock advances by 0.2 s on each `monotonic()` call so the
        # loop terminates after exactly 15 iterations.
        ticks = [0.0] + [0.2 * i for i in range(1, 17)]  # 0.0 … 3.2
        recorder = _install_fake_clock(monkeypatch, ticks)

        twin, client = _make_drone()
        twin.pan_camera(angle_deg=90.0)

        publishes = _yaw_publishes(twin)
        # 15 sustained-yaw publishes (3.0 s × 5 Hz) + 1 zero.
        assert len(publishes) == 16
        assert all(p["command"] == "turn_left" for p in publishes)
        # All sustained publishes carry the same rad/s magnitude:
        # 30°/s in rad/s.
        expected_rate = math.radians(30.0)
        for p in publishes[:-1]:
            assert p["data"]["angular_z"] == pytest.approx(expected_rate)
            assert p["data"]["linear_x"] == 0
        # Final publish is the explicit zero.
        assert publishes[-1]["data"]["angular_z"] == 0.0
        # Refresh cadence: every sleep is the configured 1/refresh_hz = 0.2 s
        # (the loop may shrink the last interval — we assert "at most" period).
        assert recorder["sleeps"], "pan_camera never slept — watchdog refresh missing"
        assert all(s <= 0.2 + 1e-9 for s in recorder["sleeps"])

    def test_negative_angle_yaws_right(self, monkeypatch):
        # 60° at 60°/s → 1.0 s total. Use refresh_hz=5 → 5 publishes.
        ticks = [0.0] + [0.2 * i for i in range(1, 7)]
        _install_fake_clock(monkeypatch, ticks)

        twin, client = _make_drone()
        twin.pan_camera(angle_deg=-60.0, yaw_rate_deg_s=60.0)

        publishes = _yaw_publishes(twin)
        assert len(publishes) == 6  # 5 sustained + 1 zero
        assert all(p["command"] == "turn_right" for p in publishes)
        expected_rate = math.radians(60.0)
        for p in publishes[:-1]:
            assert p["data"]["angular_z"] == pytest.approx(expected_rate)
        assert publishes[-1]["data"]["angular_z"] == 0.0

    def test_zero_angle_is_noop(self):
        twin, client = _make_drone()
        twin.pan_camera(angle_deg=0.0)

        # Not even a final zero — pan_camera shorts out before
        # touching the watchdog surface, so the MQTT history stays
        # empty for this twin.
        assert _yaw_publishes(twin) == []

    def test_subdegree_angle_is_noop(self):
        # Floating-point dust shouldn't trip the loop; the
        # implementation treats anything within 1 µ-degree of zero
        # as "do nothing".
        twin, client = _make_drone()
        twin.pan_camera(angle_deg=1e-9)
        assert _yaw_publishes(twin) == []

    def test_non_positive_rate_raises(self):
        twin, _ = _make_drone()
        with pytest.raises(ValueError, match="yaw_rate_deg_s must be positive"):
            twin.pan_camera(angle_deg=90.0, yaw_rate_deg_s=0.0)
        with pytest.raises(ValueError, match="yaw_rate_deg_s must be positive"):
            twin.pan_camera(angle_deg=90.0, yaw_rate_deg_s=-1.0)

    def test_too_slow_refresh_raises(self):
        # The 500 ms command-stale watchdog on the DJI Android driver
        # snaps the target to zero if it goes 0.5 s without a fresh
        # command, so anything ≤ 2 Hz is unsafe.
        twin, _ = _make_drone()
        with pytest.raises(ValueError, match="500 ms command-stale watchdog"):
            twin.pan_camera(angle_deg=90.0, refresh_hz=2.0)
        with pytest.raises(ValueError, match="500 ms command-stale watchdog"):
            twin.pan_camera(angle_deg=90.0, refresh_hz=0.5)

    def test_source_type_threads_through(self, monkeypatch):
        # Verifies the SDK still honours an explicit `source_type`
        # override (drone simulators want `sim_tele` even when the
        # client is configured for live mode).
        ticks = [0.0, 0.2, 0.4]  # 1 sustained publish + zero
        _install_fake_clock(monkeypatch, ticks)

        twin, client = _make_drone()
        twin.pan_camera(angle_deg=6.0, source_type="sim_tele")

        publishes = _yaw_publishes(twin)
        assert all(p["source_type"] == "sim_tele" for p in publishes)
