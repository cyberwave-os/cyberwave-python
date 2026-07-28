"""JointCommandBufferMixin: thread-safe submit/drain/home hand-off (no ROS)."""

from __future__ import annotations

from cyberwave.driver.control.joint_teleop import (
    HomePositionNotDefinedError,
    JointCommandBufferMixin,
)


class _FakeDriver(JointCommandBufferMixin):
    def __init__(self) -> None:
        self._init_joint_command_buffer()
        self.published: list[dict[str, float]] = []
        self.homed = 0

    def _publish_joint_command(self, positions: dict[str, float]) -> None:
        self.published.append(positions)

    def _return_to_home(self) -> None:
        self.homed += 1


def test_submit_then_pump_publishes() -> None:
    d = _FakeDriver()
    assert d.submit_joint_targets({"joint1": 0.5}) is True
    d.pump_joint_commands()
    assert d.published == [{"joint1": 0.5}]


def test_pump_with_nothing_pending_does_not_publish() -> None:
    d = _FakeDriver()
    d.pump_joint_commands()
    assert d.published == []


def test_request_home_runs_home_and_discards_queued() -> None:
    d = _FakeDriver()
    d.submit_joint_targets({"joint1": 0.5})
    d.request_home()
    d.pump_joint_commands()  # home branch
    assert d.homed == 1
    assert d.published == []  # queued command discarded on home
    d.pump_joint_commands()  # nothing left
    assert d.published == []


def test_stale_timestamp_dropped() -> None:
    d = _FakeDriver()
    assert d.submit_joint_targets({"j": 1.0}, timestamp=10.0) is True
    assert d.submit_joint_targets({"j": 2.0}, timestamp=5.0) is False  # older → stale
    d.pump_joint_commands()
    assert d.published == [{"j": 1.0}]


def test_invalid_timestamp_coerced_to_none_and_accepted() -> None:
    d = _FakeDriver()
    assert d.submit_joint_targets({"j": 1.0}, timestamp="not-a-number") is True
    d.pump_joint_commands()
    assert d.published == [{"j": 1.0}]


def test_cancel_pending_drops_queue_and_clears_home() -> None:
    d = _FakeDriver()
    d.submit_joint_targets({"joint1": 0.5})
    d.request_home()
    d.cancel_pending()
    d.pump_joint_commands()
    assert d.published == []   # queue dropped
    assert d.homed == 0        # home request cleared


def test_default_return_to_home_publishes_home_position() -> None:
    class _D(JointCommandBufferMixin):
        def __init__(self) -> None:
            self._init_joint_command_buffer()
            self._home_position = {"joint1": 0.0, "joint2": 0.0}
            self.published: list[dict[str, float]] = []

        def _publish_joint_command(self, positions: dict[str, float]) -> None:
            self.published.append(positions)
        # NOTE: no _return_to_home override → exercises the mixin default

    d = _D()
    d.request_home()
    d.pump_joint_commands()
    assert d.published == [{"joint1": 0.0, "joint2": 0.0}]


def test_home_without_position_raises() -> None:
    import pytest

    class _D(JointCommandBufferMixin):
        def __init__(self) -> None:
            self._init_joint_command_buffer()

        def _publish_joint_command(self, positions: dict[str, float]) -> None:
            pass

    d = _D()
    d.request_home()
    with pytest.raises(HomePositionNotDefinedError):
        d.pump_joint_commands()


def test_counters_initialized_to_zero():
    d = _FakeDriver()
    assert d.tele_mqtt_rx == 0
    assert d.tele_ros_publish == 0


def test_submit_increments_rx_only_on_accept():
    d = _FakeDriver()
    assert d.submit_joint_targets({"joint1": 1.0}, timestamp=10.0) is True
    assert d.tele_mqtt_rx == 1
    # stale (older timestamp) → dropped, counter unchanged
    assert d.submit_joint_targets({"joint1": 2.0}, timestamp=5.0) is False
    assert d.tele_mqtt_rx == 1


def test_pump_increments_publish_count():
    d = _FakeDriver()
    d.submit_joint_targets({"joint1": 1.0})
    d.pump_joint_commands()
    assert d.tele_ros_publish == 1


def test_tele_log_due_first_then_throttled():
    d = _FakeDriver()
    assert d.tele_log_due(1, now=100.0) is True     # first
    assert d.tele_log_due(2, now=102.0) is False    # <5s later
    assert d.tele_log_due(3, now=106.0) is True      # >=5s later
