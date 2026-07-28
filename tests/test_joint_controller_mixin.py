"""JointControllerMixin: dispatch, staleness, pacer, overlap superseding, lifecycle."""

from __future__ import annotations

import asyncio

import pytest

from cyberwave.driver.control import (
    JointControllerConfig,
    MotionRequest,
    TrajectoryWaypoint,
)
from cyberwave.driver.interface.registry import DriverInterfaceRegistry
from cyberwave.driver.control.joint_controller import JointControllerMixin


def _fast_motion(req: MotionRequest):
    """Three waypoints, 10ms apart — fast pacer tests."""
    if not req.target:
        return ()
    steps = []
    for i, frac in enumerate((0.33, 0.66, 1.0)):
        pos = {
            j: req.current.get(j, v) + frac * (v - req.current.get(j, v))
            for j, v in req.target.items()
        }
        steps.append(TrajectoryWaypoint(positions=pos, velocities=None, time_from_start=i * 0.01))
    return tuple(steps)


class _TerminalBase:
    def define_interface(self, iface):
        pass

    async def on_exit_operation(self):
        self.exited = True


class _Driver(JointControllerMixin, _TerminalBase):
    use_joint_controller = True

    def __init__(self, cfg: JointControllerConfig | None, motion=_fast_motion):
        self._cfg = cfg
        self._motion = motion
        self._joints = {"j1": 0.0, "j2": 0.0, "left_j1": 0.0, "right_j1": 0.0}
        self.sent: list[dict[str, float]] = []
        self.exited = False

    def joint_controller_config(self):
        return self._cfg

    def joint_controller_motion(self):
        return self._motion

    def joint_controller_after_process(self):
        return lambda cmd: self.sent.append(dict(cmd.positions))

    def _current_joint_positions(self):
        return dict(self._joints)


async def _drain(driver, timeout=1.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while driver._active_streams and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.005)


def test_no_config_disables_listener_registration():
    d = _Driver(cfg=None)
    reg = DriverInterfaceRegistry()
    d.define_interface(reg)
    assert not reg._listeners
    assert d._joint_controller is None


def test_listener_registered_with_config():
    d = _Driver(cfg=JointControllerConfig())
    reg = DriverInterfaceRegistry()
    d.define_interface(reg)
    assert len(reg._listeners) == 1
    assert reg._listeners[0].topic.namespace == "joint"
    assert reg._listeners[0].topic.leaf == "update"


async def test_forward_motion_none_applies_immediately():
    d = _Driver(cfg=JointControllerConfig(), motion=None)
    await d._on_joint_target({"target_positions": {"j1": 1.0}})
    assert d.sent == [{"j1": 1.0}]


async def test_flat_positions_payload_accepted_as_target():
    d = _Driver(cfg=JointControllerConfig(), motion=None)
    await d._on_joint_target({"j1": 0.7, "source_type": "tele"})
    assert d.sent == [{"j1": 0.7}]


async def test_stale_timestamp_dropped_before_planning():
    d = _Driver(cfg=JointControllerConfig(), motion=None)
    await d._on_joint_target({"target_positions": {"j1": 1.0}, "timestamp": 100.0})
    await d._on_joint_target({"target_positions": {"j1": 2.0}, "timestamp": 99.0})
    assert d.sent == [{"j1": 1.0}]  # stale command never planned


async def test_pacer_streams_all_waypoints():
    d = _Driver(cfg=JointControllerConfig())
    await d._on_joint_target({"target_positions": {"j1": 1.0}})
    await _drain(d)
    assert len(d.sent) == 3
    assert d.sent[-1]["j1"] == pytest.approx(1.0)


async def test_overlapping_plan_supersedes_stream():
    d = _Driver(cfg=JointControllerConfig())
    await d._on_joint_target({"target_positions": {"j1": 1.0}})
    await d._on_joint_target({"target_positions": {"j1": -1.0}})  # same joint: cancels
    await _drain(d)
    finals = [s["j1"] for s in d.sent]
    assert finals[-1] == pytest.approx(-1.0)
    assert len(d.sent) < 6  # first stream did not run to completion


async def test_disjoint_plans_run_concurrently():
    d = _Driver(cfg=JointControllerConfig())
    await d._on_joint_target({"target_positions": {"left_j1": 1.0}})
    await d._on_joint_target({"target_positions": {"right_j1": 1.0}})
    await _drain(d)
    lefts = [s for s in d.sent if "left_j1" in s]
    rights = [s for s in d.sent if "right_j1" in s]
    assert len(lefts) == 3 and len(rights) == 3  # neither cancelled the other


async def test_apply_joint_targets_routes_through_controller():
    d = _Driver(cfg=JointControllerConfig())
    d._apply_joint_targets({"j1": 0.5})
    await _drain(d)
    assert d.sent[-1]["j1"] == pytest.approx(0.5)


async def test_on_exit_operation_cancels_streams():
    d = _Driver(cfg=JointControllerConfig())
    await d._on_joint_target({"target_positions": {"j1": 1.0}})
    await d.on_exit_operation()
    await asyncio.sleep(0.05)
    n = len(d.sent)
    await asyncio.sleep(0.05)
    assert len(d.sent) == n  # stream stopped emitting
    assert d.exited  # cooperative super() reached the base
