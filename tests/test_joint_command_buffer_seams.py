"""JointCommandBufferMixin opt-in seams: MQTT-outage motion stop + joint-target refusal."""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cyberwave.driver.control.joint_teleop import JointCommandBufferMixin


class _Registry:
    """Stands in for InterfaceRegistryMixin (its _invoke_listener is a staticmethod)."""

    @staticmethod
    async def _invoke_listener(callback, envelope) -> None:
        result = callback(envelope)
        if inspect.isawaitable(result):
            await result


class _Driver(JointCommandBufferMixin, _Registry):
    def __init__(self) -> None:
        self._init_joint_command_buffer()
        self.published: list[dict[str, float]] = []
        self.homed = 0
        self.logger = MagicMock()

    def get_logger(self):
        return self.logger

    def _publish_joint_command(self, positions: dict[str, float]) -> None:
        self.published.append(dict(positions))

    def _return_to_home(self) -> None:
        self.homed += 1


class _OutageDriver(_Driver):
    STOP_MOTION_ON_MQTT_OUTAGE = True

    def __init__(self, *, connected: bool = True, count: int = 0) -> None:
        super().__init__()
        self.mqtt = SimpleNamespace(connected=connected, disconnect_count=count)
        self._cw = SimpleNamespace(mqtt=self.mqtt)
        self.stops = 0

    def _stop_motion(self) -> None:
        self.stops += 1
        super()._stop_motion()


def _listen(driver, callback, envelope=None) -> None:
    asyncio.run(driver._invoke_listener(callback, envelope or {}))


# ── unchanged unless opted in ────────────────────────────────────────────────


def test_without_opt_in_a_disconnected_client_changes_nothing() -> None:
    driver = _Driver()
    driver._cw = SimpleNamespace(
        mqtt=SimpleNamespace(connected=False, disconnect_count=7)
    )
    assert driver._mqtt_motion_ready() is True
    assert driver.submit_joint_targets({"joint1": 0.5})
    assert driver.pump_joint_commands() == {"joint1": 0.5}
    assert driver.published == [{"joint1": 0.5}]
    seen = []
    _listen(driver, seen.append, {"command": "home"})
    assert seen == [{"command": "home"}]


def test_without_a_refusal_override_no_merge_or_check_happens() -> None:
    driver = _Driver()
    driver._joint_target_baseline = MagicMock()  # would be consulted by a check
    assert driver.publish_joint_targets({"joint1": 0.1})
    driver._joint_target_baseline.assert_not_called()


def test_listener_without_a_registry_parent_still_runs_the_callback() -> None:
    class Bare(JointCommandBufferMixin):
        def __init__(self) -> None:
            self._init_joint_command_buffer()

    async def handler(envelope):
        seen.append(envelope)

    seen: list = []
    asyncio.run(Bare()._invoke_listener(handler, {"x": 1}))
    assert seen == [{"x": 1}]


# ── MQTT outage ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("short_outage", [False, True])
def test_an_outage_stops_motion_once_and_drops_queued_input(short_outage) -> None:
    driver = _OutageDriver()
    driver.submit_joint_targets({"joint1": 1.0})
    driver.request_home()

    driver.mqtt.disconnect_count += 1
    driver.mqtt.connected = short_outage  # the whole outage can fall between ticks
    assert driver.pump_joint_commands() is None
    assert driver.stops == 1
    assert driver._tele_inbox.pending_count == 0
    assert not driver._home_pending
    assert driver.published == [] and driver.homed == 0
    driver.pump_joint_commands()
    assert driver.stops == 1  # once per outage, not once per tick
    driver.logger.warning.assert_called_once()

    if not short_outage:
        ignored = MagicMock()
        _listen(driver, ignored, {"command": "home"})
        ignored.assert_not_called()
        assert not driver.submit_joint_targets({"joint1": 0.9})
        detached = MagicMock()
        _listen(driver, detached, {"command": "controller_changed"})
        detached.assert_called_once()

    driver.mqtt.connected = True
    # The first fresh command works before the next tick, without reattaching.
    _listen(driver, lambda _e: driver.submit_joint_targets({"joint1": 0.4}))
    driver.pump_joint_commands()
    assert driver.published == [{"joint1": 0.4}]
    assert driver.stops == 1


@pytest.mark.parametrize("during_check", [False, True])
def test_reconnect_between_drain_and_publish_does_not_replay_an_old_target(
    during_check,
) -> None:
    driver = _OutageDriver(count=0 if during_check else 1)
    if during_check:

        class Checking(_OutageDriver):
            def _joint_target_refusal(self, merged):
                self.mqtt.disconnect_count += 1

        driver = Checking()
    assert driver.publish_joint_targets({"joint1": 1.0}) is False
    assert driver.stops == 1
    assert driver.published == []
    # The first new input observes recovery before being handed to its handler.
    driver.mqtt.disconnect_count += 1
    _listen(driver, lambda _e: driver.submit_joint_targets({"joint1": 0.4}))
    assert driver.stops == 2
    assert driver._tele_inbox.drain() == {"joint1": 0.4}


def test_a_reconnect_resumes_without_another_stop() -> None:
    driver = _OutageDriver()
    driver._mqtt_motion_paused = True  # an outage already handled
    assert driver._mqtt_motion_ready()
    assert driver.stops == 0
    driver.logger.info.assert_called_once()


def test_no_bound_client_means_ready() -> None:
    driver = _OutageDriver()
    driver._cw = None
    assert driver._mqtt_motion_ready()


def test_sync_adopts_disconnects_from_before_wiring_without_stopping() -> None:
    driver = _OutageDriver(count=3)
    driver.sync_mqtt_disconnect_count()
    assert driver.mqtt_motion_epoch() == 3
    assert driver._mqtt_motion_ready()
    assert driver.stops == 0


def test_multi_step_publishers_can_detect_an_outage_between_steps() -> None:
    driver = _OutageDriver()
    epoch = driver.mqtt_motion_epoch()
    assert driver.mqtt_motion_still_ready(epoch)
    driver.mqtt.disconnect_count += 1
    assert not driver.mqtt_motion_still_ready(epoch)


# ── joint-target refusal ─────────────────────────────────────────────────────


class _Gated(_Driver):
    def __init__(self, reason=None) -> None:
        super().__init__()
        self._joint_states = {f"joint{i}": 0.0 for i in range(1, 7)}
        self.reason = reason
        self.checked: list[dict[str, float]] = []

    def _joint_target_refusal(self, merged):
        self.checked.append(dict(merged))
        return self.reason(merged) if callable(self.reason) else self.reason


def test_the_refusal_sees_the_merged_posture_not_the_delta() -> None:
    driver = _Gated()
    driver._joint_states.update({"joint1": 0.7, "joint5": -0.3})
    assert driver.publish_joint_targets({"joint2": 0.4})
    assert driver.checked[0]["joint1"] == 0.7
    assert driver.checked[0]["joint5"] == -0.3
    assert driver.checked[0]["joint2"] == 0.4
    # Only the requested delta is handed to the write seam.
    assert driver.published == [{"joint2": 0.4}]


def test_a_refused_target_is_not_published_counted_and_log_throttled() -> None:
    driver = _Gated("self-collision")
    assert not driver.publish_joint_targets({"joint2": 1.2})
    assert not driver.publish_joint_targets({"joint2": 1.3})
    assert driver.published == []
    assert driver.joint_target_refusals == 2
    driver.logger.warning.assert_called_once()
    message = driver.logger.warning.call_args.args[0]
    assert "self-collision" in message and "joint2=1.200" in message


def test_pump_routes_through_the_refusal_and_consumes_the_target() -> None:
    driver = _Gated("blocked")
    driver.submit_joint_targets({"joint2": 1.2})
    assert driver.pump_joint_commands() == {"joint2": 1.2}
    assert driver.published == []
    assert driver.pump_joint_commands() is None  # never replayed


def test_a_checker_fault_fails_open_and_is_logged() -> None:
    def explode(_merged):
        raise RuntimeError("geometry went away")

    driver = _Gated(explode)
    assert driver.publish_joint_targets({"joint2": 0.4})
    assert driver.published == [{"joint2": 0.4}]
    driver.logger.error.assert_called_once()
    assert driver.joint_target_refusals == 0


def test_home_bypasses_the_refusal_on_purpose() -> None:
    """Refusing to park the arm is not a safe outcome."""

    class Homing(_Gated):
        _return_to_home = JointCommandBufferMixin._return_to_home

    driver = Homing("blocked")
    driver._home_position = {"joint1": 0.0}
    driver.request_home()
    driver.pump_joint_commands()
    assert driver.published == [{"joint1": 0.0}]
