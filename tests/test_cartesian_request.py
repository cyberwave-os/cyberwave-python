"""The one funnel every Cartesian intent goes through, and how it refuses.

Pure: no ROS, no solver, no planner. The preconditions and the reachability
check are injected, so the whole refusal matrix is exercised directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from cyberwave.driver.control.cartesian_request import (
    SUPPORTED_FRAMES,
    CartesianPlanner,
    CartesianRejection,
    Preconditions,
    UnavailablePlanner,
    validate_cartesian_request,
)
from cyberwave.driver.motion.guards import MotionRefusal, check_motion_preconditions

_MODULE = (
    Path(__file__).resolve().parents[1]
    / "cyberwave"
    / "driver"
    / "control"
    / "cartesian_request.py"
)


@dataclass
class _Pose:
    x: float = 0.3
    y: float = 0.0
    z: float = 0.25
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


_OK = Preconditions()
_SOLVED = {f"joint{i}": 0.0 for i in range(1, 7)}


def _validate(**kwargs):
    kwargs.setdefault("source", "ee_pose")
    kwargs.setdefault("preconditions", _OK)
    kwargs.setdefault("solve", lambda _pose: _SOLVED)
    return validate_cartesian_request(kwargs.pop("target", _Pose()), **kwargs)


def test_a_valid_target_is_accepted_and_carries_its_origin() -> None:
    decision = _validate(source="ee_move")
    assert decision.accepted
    assert decision.status() == "accepted"
    assert decision.request.source == "ee_move"
    assert decision.request.target == (0.3, 0.0, 0.25, 0.0, 0.0, 0.0)


def test_the_ik_solution_is_carried_as_a_seed_not_as_a_path() -> None:
    """It proves reachability and seeds the planner. It is not a trajectory."""
    request = _validate().request
    assert request.seed_joints == _SOLVED


# ── Every precondition fails closed ──────────────────────────────────────────


@pytest.mark.parametrize(
    "preconditions,expected",
    [
        (Preconditions(arm_faulted=True), CartesianRejection.ARM_FAULTED),
        (
            Preconditions(control_mode_ready=False),
            CartesianRejection.CONTROL_MODE_UNAVAILABLE,
        ),
        (
            Preconditions(hardware_verified=False),
            CartesianRejection.HARDWARE_UNVERIFIED,
        ),
        (
            Preconditions(geometry_validated=False),
            CartesianRejection.GEOMETRY_UNVALIDATED,
        ),
        (
            Preconditions(collision_model_valid=False),
            CartesianRejection.COLLISION_MODEL_INVALID,
        ),
    ],
)
def test_each_precondition_refuses_with_its_own_code(preconditions, expected) -> None:
    decision = _validate(preconditions=preconditions)
    assert not decision.accepted
    assert decision.rejection is expected


def test_the_control_mode_refusal_is_robot_agnostic_by_default() -> None:
    detail = _validate(preconditions=Preconditions(control_mode_ready=False)).detail
    assert detail == "the arm is not in its command mode"
    custom = Preconditions(control_mode_ready=False, control_mode_detail="vendor mode")
    assert _validate(preconditions=custom).detail == "vendor mode"


def test_the_fault_check_comes_first() -> None:
    """A faulted arm is refused for being faulted, whatever else is also wrong."""
    decision = _validate(
        preconditions=Preconditions(arm_faulted=True, hardware_verified=False)
    )
    assert decision.rejection is CartesianRejection.ARM_FAULTED


@pytest.mark.parametrize(
    "overrides",
    [
        {"arm_faulted": True},
        {"control_mode_ready": False},
        {"hardware_verified": False},
        {"geometry_validated": False},
        {"collision_model_valid": False},
        {"measured_state_seen": False},
    ],
)
def test_the_motion_guards_refuse_the_same_preconditions_with_the_same_code(
    overrides,
) -> None:
    """``ee_pose`` and the planned primitives are the same act, same vocabulary."""
    preconditions = Preconditions(**overrides)
    cartesian = _validate(preconditions=preconditions).rejection
    motion = check_motion_preconditions(preconditions, plan_only=False)
    assert motion is not None
    assert MotionRefusal(cartesian.value) is motion.code


# ── Target sanity ────────────────────────────────────────────────────────────


def test_an_unknown_frame_is_refused_not_assumed() -> None:
    """A request in an unrecognised frame is a request nobody can check."""
    decision = _validate(frame="wrist_ish")
    assert decision.rejection is CartesianRejection.UNKNOWN_FRAME


@pytest.mark.parametrize("frame", sorted(SUPPORTED_FRAMES))
def test_the_supported_frames_are_accepted(frame) -> None:
    assert _validate(frame=frame).accepted


def test_a_driver_may_narrow_the_supported_frames() -> None:
    decision = _validate(frame="tool", supported_frames=frozenset({"base"}))
    assert decision.rejection is CartesianRejection.UNKNOWN_FRAME


@pytest.mark.parametrize("bad", [float("nan"), math.inf, -math.inf])
def test_a_non_finite_target_is_refused(bad) -> None:
    decision = _validate(target=_Pose(z=bad))
    assert decision.rejection is CartesianRejection.NON_FINITE_TARGET


def test_an_unreachable_target_is_refused_before_planning() -> None:
    decision = _validate(solve=lambda _pose: None)
    assert decision.rejection is CartesianRejection.UNREACHABLE


def test_a_solver_exception_is_a_refusal_not_a_crash() -> None:
    def _boom(_pose):
        raise RuntimeError("singular configuration")

    decision = _validate(solve=_boom)
    assert decision.rejection is CartesianRejection.UNREACHABLE
    assert "singular configuration" in decision.detail


def test_validation_without_a_solver_still_works() -> None:
    """IK-off drivers skip pre-flight reachability; the planner owns it."""
    decision = _validate(solve=None)
    assert decision.accepted
    assert decision.request.seed_joints is None


# ── No planner is a first-class state, not an absent one ─────────────────────


def test_the_unavailable_planner_refuses_with_a_readable_status() -> None:
    planner = UnavailablePlanner()
    assert isinstance(planner, CartesianPlanner)
    assert planner.available is False
    decision = planner.plan_and_execute(_validate().request)
    assert decision.rejection is CartesianRejection.PLANNER_UNAVAILABLE
    assert "planner" in decision.detail
    assert decision.status().startswith("planner_unavailable: ")


def test_the_unavailable_planner_carries_the_drivers_own_reason() -> None:
    decision = UnavailablePlanner("move_group is down").plan_and_execute(
        _validate().request
    )
    assert decision.detail == "move_group is down"


def test_the_status_line_names_the_code_and_the_reason() -> None:
    status = _validate(preconditions=Preconditions(arm_faulted=True)).status()
    assert status.startswith("arm_faulted:")


def test_nothing_in_this_module_can_execute_motion() -> None:
    """A validator that can move the arm is not a validator."""
    source = _MODULE.read_text()
    for forbidden in ("CartesianPathQueue", "ros_publish", "rclpy", "import numpy"):
        assert forbidden not in source, forbidden


# ── measured state: refuse to plan from a configuration nobody reported ──────


def test_a_request_is_refused_before_any_joint_feedback_has_arrived() -> None:
    """Cartesian motion is relative to where the arm IS.

    ``ArmController._start`` completes any joint the feedback omitted with 0.0,
    so with no feedback at all the solver silently plans from a fabricated
    all-zeros posture -- and the symptom reads as a reach problem.
    """
    decision = _validate(
        source="ee_move",
        preconditions=Preconditions(measured_state_seen=False),
    )
    assert not decision.accepted
    assert decision.rejection is CartesianRejection.NO_MEASURED_STATE
    assert "feedback" in (decision.detail or "")


def test_the_measured_state_gate_precedes_the_solver() -> None:
    """No IK attempt at all: a solve seeded from an assumed posture is a wrong
    answer, not a slow one."""
    calls: list = []

    def solve(target):
        calls.append(target)
        return dict(_SOLVED)

    _validate(
        source="ee_move",
        preconditions=Preconditions(measured_state_seen=False),
        solve=solve,
    )
    assert calls == [], "the solver ran despite there being no measured state"


def test_measured_state_defaults_permissive_for_pure_unit_paths() -> None:
    """Manifest export has nothing to move; the driver supplies the real answer."""
    assert Preconditions().measured_state_seen is True


def test_an_ik_miss_reports_the_callers_reason_when_one_is_supplied() -> None:
    """``solve_refusal`` is how the fused UNREACHABLE answer names its cause."""
    decision = _validate(
        solve=lambda _pose: None,
        solve_refusal=lambda: "refused by the collision model: floor",
    )
    assert decision.rejection is CartesianRejection.UNREACHABLE
    assert "floor" in (decision.detail or "")


def test_the_generic_detail_survives_without_a_reason_supplier() -> None:
    decision = _validate(solve=lambda _pose: None)
    assert decision.rejection is CartesianRejection.UNREACHABLE
    assert "no in-limit, collision-free solution" in (decision.detail or "")


def test_a_raising_reason_supplier_does_not_break_the_refusal() -> None:
    """Diagnosis must never turn a refusal into a crash."""

    def boom():
        raise RuntimeError("nope")

    decision = _validate(solve=lambda _pose: None, solve_refusal=boom)
    assert decision.rejection is CartesianRejection.UNREACHABLE
    assert decision.detail
