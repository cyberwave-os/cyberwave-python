"""driver/control/types.py — frozen dataclasses + config validation."""

from __future__ import annotations

import pytest

from cyberwave.driver.control.types import (
    JointCommand,
    JointControllerConfig,
    MotionRequest,
    TrajectoryPlan,
    TrajectoryWaypoint,
)


def test_joint_command_defaults():
    cmd = JointCommand(positions={"j1": 0.5})
    assert cmd.positions == {"j1": 0.5}
    assert cmd.velocities is None
    assert cmd.efforts is None


def test_trajectory_plan_empty_is_noop():
    plan = TrajectoryPlan(waypoints=(), duration=0.0, joints=frozenset())
    assert not plan.waypoints
    assert plan.duration == 0.0


def test_config_defaults():
    cfg = JointControllerConfig()
    assert cfg.joints is None
    assert cfg.passthrough_joints == ()
    assert cfg.default_velocity == 1.0
    assert cfg.rate_hz == 50.0
    assert cfg.command_interfaces == ("position",)


def test_config_mappings_immutable():
    cfg = JointControllerConfig(default_velocities={"j1": 2.0}, position_limits={"j1": (-1.0, 1.0)})
    with pytest.raises(TypeError):
        cfg.default_velocities["j2"] = 3.0  # type: ignore[index]
    with pytest.raises(TypeError):
        cfg.position_limits["j2"] = (0.0, 1.0)  # type: ignore[index]


def test_config_rejects_nonpositive_rate_and_velocity():
    with pytest.raises(ValueError):
        JointControllerConfig(rate_hz=0.0)
    with pytest.raises(ValueError):
        JointControllerConfig(default_velocity=0.0)


def test_motion_request_and_waypoint_shape():
    req = MotionRequest(current={"j1": 0.0}, target={"j1": 1.0}, velocities={"j1": 1.0}, rate_hz=50.0)
    wp = TrajectoryWaypoint(positions={"j1": 0.5}, velocities={"j1": 1.0}, time_from_start=0.5)
    assert req.rate_hz == 50.0
    assert wp.time_from_start == 0.5
