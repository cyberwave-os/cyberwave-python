"""Standard teleop status fields + throttled log helpers on JointCommandBufferMixin."""

from __future__ import annotations

import logging

from cyberwave.driver.interface import DriverOperationMode
from cyberwave.driver.control.joint_teleop import JointCommandBufferMixin


class _Driver(JointCommandBufferMixin):
    def __init__(self, mode=DriverOperationMode.TELEOP_REMOTE):
        self._init_joint_command_buffer()
        self.operation_mode = mode

    def controller_policy_snapshot(self):
        return {"controller_type": "keyboard", "controller_policy_uuid": "abc"}

    def _publish_joint_command(self, positions):
        pass


def test_status_fields_teleop_mode():
    d = _Driver()
    d.submit_joint_targets({"j1": 0.5})
    fields = d.teleop_status_fields()
    assert fields["teleop_robot_status"] == "teleop"
    assert fields["teleop_operation_mode"] == "teleop_remote"
    assert fields["teleop_mqtt_rx"] == 1
    assert fields["teleop_ros_publish"] == 0
    assert fields["teleop_pending_joints"] == 1
    assert fields["controller_type"] == "keyboard"


def test_status_fields_idle_in_no_op():
    d = _Driver(mode=DriverOperationMode.NO_OP)
    assert d.teleop_status_fields()["teleop_robot_status"] == "idle"


def test_status_fields_without_composed_surfaces():
    class _Bare(JointCommandBufferMixin):
        def __init__(self):
            self._init_joint_command_buffer()

    fields = _Bare().teleop_status_fields()
    assert fields["teleop_operation_mode"] is None
    assert "controller_type" not in fields


def test_log_tele_rx_throttled(caplog):
    d = _Driver()
    with caplog.at_level(logging.INFO):
        d.tele_mqtt_rx = 1
        d.log_tele_rx({"source_type": "tele"}, {"j1": 0.5})   # first: logs
        d.tele_mqtt_rx = 2
        d.log_tele_rx({"source_type": "tele"}, {"j1": 0.6})   # within 5s: silent
    lines = [r.message for r in caplog.records if "tele RX" in r.message]
    assert len(lines) == 1
    assert "j1" in lines[0]


def test_log_tele_publish_includes_extra(caplog):
    class _WithExtra(_Driver):
        def teleop_log_extra(self):
            return "; arm not ready"

    d = _WithExtra()
    with caplog.at_level(logging.INFO):
        d.tele_ros_publish = 1
        d.log_tele_publish({"j1": 0.5})
    lines = [r.message for r in caplog.records if "tele publish" in r.message]
    assert len(lines) == 1
    assert lines[0].endswith("; arm not ready")


def test_log_tele_ignored_only_first_events(caplog):
    d = _Driver()
    with caplog.at_level(logging.WARNING):
        for _ in range(6):
            d.log_tele_ignored({"source_type": "tele", "bogus": 1})
            d.tele_mqtt_rx += 1
    lines = [r.message for r in caplog.records if "ignored" in r.message]
    assert len(lines) == 4  # logged while tele_mqtt_rx <= 3, silent after
