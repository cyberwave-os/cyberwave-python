"""JointController.plan: filtering, clamping, passthrough, velocity resolution."""

from __future__ import annotations

import pytest

from cyberwave.driver.control import (
    JointController,
    JointControllerConfig,
    MotionRequest,
    TrajectoryWaypoint,
    waypoint_command,
)


def _fake_motion(req: MotionRequest):
    """Two-waypoint linear stand-in; records the request on the function."""
    _fake_motion.last_request = req  # type: ignore[attr-defined]
    if not req.target:
        return ()
    mid = {j: (req.current.get(j, v) + v) / 2 for j, v in req.target.items()}
    return (
        TrajectoryWaypoint(positions=mid, velocities=None, time_from_start=0.05),
        TrajectoryWaypoint(positions=dict(req.target), velocities=None, time_from_start=0.1),
    )


def test_forward_passthrough_when_motion_none():
    ctrl = JointController(JointControllerConfig(), motion=None)
    plan = ctrl.plan({"j1": 0.0}, {"j1": 1.0})
    assert len(plan.waypoints) == 1
    assert plan.waypoints[0].positions == {"j1": 1.0}
    assert plan.waypoints[0].time_from_start == 0.0
    assert plan.duration == 0.0
    assert plan.joints == frozenset({"j1"})


def test_empty_target_empty_plan():
    ctrl = JointController(JointControllerConfig(), motion=_fake_motion)
    plan = ctrl.plan({"j1": 0.0}, {})
    assert plan.waypoints == ()


def test_joints_filter_drops_foreign_joints():
    cfg = JointControllerConfig(joints=("j1",))
    ctrl = JointController(cfg, motion=None)
    plan = ctrl.plan({"j1": 0.0}, {"j1": 0.5, "mimic8": -0.5})
    assert plan.waypoints[0].positions == {"j1": 0.5}


def test_position_clamp_config_over_provider():
    cfg = JointControllerConfig(position_limits={"j1": (-0.25, 0.25)})
    ctrl = JointController(
        cfg, motion=None, position_limits=lambda: {"j1": (-9.0, 9.0), "j2": (0.0, 0.1)}
    )
    plan = ctrl.plan({"j1": 0.0, "j2": 0.0}, {"j1": 1.0, "j2": 1.0})
    assert plan.waypoints[0].positions["j1"] == 0.25  # config clamp wins
    assert plan.waypoints[0].positions["j2"] == 0.1   # provider fallback clamps


def test_velocity_resolution_precedence():
    cfg = JointControllerConfig(
        default_velocities={"j2": 2.0}, default_velocity=0.5
    )
    ctrl = JointController(
        cfg, motion=_fake_motion, velocity_limits=lambda: {"j3": 3.0}
    )
    ctrl.plan(
        {"j1": 0.0, "j2": 0.0, "j3": 0.0, "j4": 0.0},
        {"j1": 1.0, "j2": 1.0, "j3": 1.0, "j4": 1.0},
        velocities={"j1": 9.0},  # wire payload wins for j1
    )
    resolved = _fake_motion.last_request.velocities  # type: ignore[attr-defined]
    assert resolved == {"j1": 9.0, "j2": 2.0, "j3": 3.0, "j4": 0.5}


def test_provider_failure_falls_back_to_default():
    def boom():
        raise RuntimeError("no urdf")

    ctrl = JointController(
        JointControllerConfig(default_velocity=0.7), motion=_fake_motion,
        velocity_limits=boom,
    )
    ctrl.plan({"j1": 0.0}, {"j1": 1.0})
    assert _fake_motion.last_request.velocities == {"j1": 0.7}  # type: ignore[attr-defined]


def test_passthrough_joint_full_target_on_every_waypoint():
    cfg = JointControllerConfig(passthrough_joints=("grip",))
    ctrl = JointController(cfg, motion=_fake_motion)
    plan = ctrl.plan({"j1": 0.0, "grip": 0.0}, {"j1": 1.0, "grip": 0.04})
    assert len(plan.waypoints) == 2
    for wp in plan.waypoints:
        assert wp.positions["grip"] == 0.04
    assert plan.waypoints[0].positions["j1"] == 0.5  # shaped midpoint
    assert plan.joints == frozenset({"j1", "grip"})


def test_unknown_current_joint_becomes_passthrough():
    ctrl = JointController(JointControllerConfig(), motion=_fake_motion)
    plan = ctrl.plan({"j1": 0.0}, {"j1": 1.0, "jX": 0.3})
    for wp in plan.waypoints:
        assert wp.positions["jX"] == 0.3  # applied directly, never shaped from 0.0


def test_only_passthrough_yields_single_immediate_waypoint():
    cfg = JointControllerConfig(passthrough_joints=("grip",))
    ctrl = JointController(cfg, motion=_fake_motion)
    plan = ctrl.plan({"grip": 0.0}, {"grip": 0.04})
    assert len(plan.waypoints) == 1
    assert plan.waypoints[0].time_from_start == 0.0


def test_waypoint_command_channel_gating():
    wp = TrajectoryWaypoint(
        positions={"j1": 0.5}, velocities={"j1": 0.2}, time_from_start=0.1
    )
    pos_only = waypoint_command(wp, JointControllerConfig())
    assert pos_only.positions == {"j1": 0.5} and pos_only.velocities is None
    with_vel = waypoint_command(
        wp, JointControllerConfig(command_interfaces=("position", "velocity"))
    )
    assert with_vel.velocities == {"j1": 0.2}
