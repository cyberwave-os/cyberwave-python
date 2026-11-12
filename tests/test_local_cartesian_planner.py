"""In-process Cartesian planning, still executed through a trajectory controller.

The planner never streams raw IK waypoints at hardware: it hands one timed,
audited trajectory to an injected ``send_trajectory``. Fakes cover the control
flow; the ``simple_arm`` URDF fixture covers it end to end through pinocchio.
"""

from __future__ import annotations

import io
import re
import tokenize
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cyberwave.driver.control.cartesian import EEPose
from cyberwave.driver.control.cartesian_request import (
    CartesianPlanner,
    CartesianRejection,
    CartesianRequest,
)
from cyberwave.driver.control.local_planner import LocalCartesianPlanner
from cyberwave.driver.kinematics.arm.controller import pose_to_matrix
from cyberwave.driver.kinematics.arm.trajectory import (
    PosePathPreflightError,
    waypoint_count,
)

pytest.importorskip("scipy")

JOINTS = tuple(f"joint{i}" for i in range(1, 7))
_START = dict.fromkeys(JOINTS, 0.0)
_TARGET = (0.36, 0.0, 0.25, 0.0, 0.0, 0.0)
_ACCELERATION = 2.0
_VELOCITY = dict.fromkeys(JOINTS, 0.3)
_MODULE = (
    Path(__file__).resolve().parents[1]
    / "cyberwave"
    / "driver"
    / "control"
    / "local_planner.py"
)
_URDF = str(Path(__file__).resolve().parent / "fixtures" / "simple_arm.urdf")


class _Controller:
    """Straight-line solver stand-in with the ArmController surface used here."""

    def __init__(self, path=None, raises=None) -> None:
        self._path = path
        self._raises = raises
        self.calls = []
        self.ik_kwargs = []

    def ee_pose(self, joints):
        return [[1, 0, 0, 0.30], [0, 1, 0, 0.0], [0, 0, 1, 0.25], [0, 0, 0, 1]]

    def pose_path_to_joints(
        self, start, pose, *, steps, max_joint_step, ik_kwargs=None
    ):
        if self._raises:
            raise self._raises
        self.ik_kwargs.append(ik_kwargs)
        self.calls.append(steps)
        if self._path is not None:
            return self._path
        return [{j: 0.05 * (n + 1) for j in JOINTS} for n in range(steps)]

    def pose_matrix_path_to_joints(
        self, start, poses, *, max_joint_step, allow_ik_gaps=False, ik_kwargs=None
    ):
        if self._raises:
            raise self._raises
        self.ik_kwargs.append(ik_kwargs)
        self.calls.append((len(poses), max_joint_step, allow_ik_gaps))
        if self._path is not None:
            return self._path
        return [{j: 0.05 * (n + 1) for j in JOINTS} for n in range(len(poses))]


def _linear_waypoints(start, target, steps):
    return [
        np.asarray(start) + (np.asarray(target) - np.asarray(start)) * index / steps
        for index in range(1, steps + 1)
    ]


def _kinematics(**overrides):
    fields = {
        "joint_limits": lambda: dict.fromkeys(JOINTS, (-5.0, 5.0)),
        "collision_reason": lambda _point: None,
        "ik": lambda _start, _target: {**_START, "joint1": 0.25},
        "pose_waypoints": _linear_waypoints,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _planner(controller, sent=None, *, accept=True, **kwargs):
    def send(trajectory):
        if sent is not None:
            sent.append(trajectory)
        return accept

    kwargs.setdefault("velocity_limits", _VELOCITY)
    kwargs.setdefault("max_joint_acceleration", _ACCELERATION)
    kwargs.setdefault("arm_joints", JOINTS)
    return LocalCartesianPlanner(
        controller=controller,
        current_joints=lambda: dict(_START),
        send_trajectory=send,
        **kwargs,
    )


def _request(**kwargs):
    return CartesianRequest(target=_TARGET, source="ee_move", **kwargs)


def _target_matrix():
    return pose_to_matrix(EEPose(*_TARGET))


# ── It executes through the injected sender, not at a vendor topic ──────────


def test_it_satisfies_the_cartesian_planner_protocol() -> None:
    assert isinstance(_planner(_Controller()), CartesianPlanner)


def test_a_reachable_target_becomes_a_trajectory_goal() -> None:
    sent = []
    decision = _planner(_Controller(), sent).plan_and_execute(_request())
    assert decision.accepted
    trajectory = sent[0]
    assert list(trajectory.positions[0]) == list(JOINTS)
    assert len(trajectory.positions) == len(trajectory.time_from_start_s) > 1
    assert trajectory.positions[0] == _START


def test_joint_names_come_from_the_controller_config_by_default() -> None:
    controller = _Controller()
    controller.config = SimpleNamespace(arm_joints=("joint2", "joint1"))
    planner = LocalCartesianPlanner(
        controller=controller,
        current_joints=lambda: dict(_START),
        send_trajectory=lambda _t: True,
        velocity_limits=_VELOCITY,
        max_joint_acceleration=_ACCELERATION,
    )
    assert planner.arm_joints == ("joint2", "joint1")


def test_rotation_only_path_gets_more_than_translation_minimum() -> None:
    controller = _Controller()
    request = CartesianRequest(
        target=(0.30, 0.0, 0.25, 0.12, 0.0, 0.0), source="ee_rotate_left"
    )
    assert _planner(controller).plan_and_execute(request).accepted
    assert controller.calls == [6]
    assert waypoint_count(0.0, 0.12) == 6


def test_a_plan_superseded_during_ik_is_never_submitted() -> None:
    sent = []
    decision = _planner(_Controller(), sent).plan_and_execute(
        _request(), is_current=lambda: False
    )
    assert decision.rejection is CartesianRejection.SUPERSEDED
    assert not sent


def test_it_is_available_without_waiting_on_another_process() -> None:
    assert _planner(_Controller()).available is True
    assert _planner(None).available is False
    decision = _planner(None).plan_and_execute(_request())
    assert decision.rejection is CartesianRejection.PLANNER_UNAVAILABLE


def _code_only(path: Path) -> str:
    """Source with comments and every string literal removed: the contract is
    about what the CODE references, not what the prose explains."""
    return " ".join(
        token.string
        for token in tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
        if token.type not in (tokenize.COMMENT, tokenize.STRING)
    )


def test_the_module_streams_nothing_and_depends_on_no_planner_process() -> None:
    code = _code_only(_MODULE)
    for forbidden in (
        "CartesianPathQueue",
        "rclpy",
        "moveit",
        "move_group",
        "MoveGroup",
        "piper",
    ):
        assert forbidden.lower() not in code.lower(), forbidden


# ── All-or-nothing ──────────────────────────────────────────────────────────


def test_an_unsolvable_path_is_refused_not_approximated() -> None:
    sent = []
    decision = _planner(_Controller(path=[]), sent).plan_and_execute(_request())
    assert decision.rejection is CartesianRejection.PLANNING_FAILED
    assert sent == []


def test_a_solver_exception_is_a_refusal_not_a_crash() -> None:
    decision = _planner(_Controller(raises=RuntimeError("singular"))).plan_and_execute(
        _request()
    )
    assert decision.rejection is CartesianRejection.PLANNING_FAILED
    assert "singular" in decision.detail


def test_a_rejected_goal_is_reported() -> None:
    decision = _planner(_Controller(), accept=False).plan_and_execute(_request())
    assert decision.rejection is CartesianRejection.PLANNING_FAILED
    assert "did not accept" in decision.detail


def test_diagnosis_never_raises_when_the_controller_exposes_no_solver() -> None:
    decision = _planner(_Controller(path=[])).plan_and_execute(_request())
    assert decision.rejection is CartesianRejection.PLANNING_FAILED
    assert "solver unavailable" in (decision.detail or "")


# ── Timing uses the one configured contract ─────────────────────────────────


def test_ee_move_uses_the_configured_acceleration_bound() -> None:
    sent = []
    acceleration = 0.05
    planner = _planner(
        _Controller(path=[{**_START, "joint1": 0.04}]),
        sent,
        max_joint_acceleration=acceleration,
    )
    assert planner.plan_and_execute(_request()).accepted
    assert sent[0].time_from_start_s[-1] >= 2.0 * (0.04 / acceleration) ** 0.5


def test_every_local_request_uses_the_one_resolved_setting() -> None:
    """There is no per-request override: the request has no speed."""
    assert not hasattr(_request(), "speed_scale")
    sent = []
    _planner(_Controller(), sent).plan_and_execute(_request())
    _planner(_Controller(), sent).plan_and_execute(_request())
    assert sent[1].time_from_start_s[-1] == pytest.approx(sent[0].time_from_start_s[-1])


def test_smoothing_checks_collisions_between_ik_samples() -> None:
    controller = _Controller(path=[{**_START, "joint1": 0.1}])
    controller.kinematics = SimpleNamespace(
        collision_reason=lambda point: (
            "obstacle" if 0.04 < point["joint1"] < 0.06 else None
        ),
    )
    sent = []
    decision = _planner(controller, sent).plan_and_execute(_request())
    assert not decision.accepted
    assert "smoothed trajectory: obstacle" in decision.detail
    assert not sent


# ── Path solving must not change IK branch ──────────────────────────────────


def test_every_path_solve_forwards_the_configured_continuity_kwargs() -> None:
    """A multi-seed solver swaps IK branch mid-path unless told not to; the
    caller's kwargs reach every path solve (jog and LIN alike)."""
    controller = _Controller()
    planner = _planner(controller, [], path_ik_kwargs={"restart": False})

    assert planner.plan_and_execute(_request()).accepted
    controller.kinematics = SimpleNamespace(
        pose_waypoints=lambda start, target, steps: [np.asarray(target)] * steps
    )
    planner.preflight_move(dict(_START), _target_matrix(), motion_type="lin")

    assert len(controller.ik_kwargs) == 2
    for kwargs in controller.ik_kwargs:
        assert kwargs == {"restart": False}, kwargs


def test_a_single_seed_solver_gets_no_extra_kwargs_by_default() -> None:
    controller = _Controller()
    assert _planner(controller).plan_and_execute(_request()).accepted
    assert controller.ik_kwargs == [None]


# ── Preflight: PTP, LIN and ranked grasps ───────────────────────────────────


def test_preflight_refuses_a_bad_target_or_motion_type() -> None:
    controller = _Controller()
    controller.kinematics = _kinematics()
    planner = _planner(controller)
    with pytest.raises(PosePathPreflightError, match="finite 4x4"):
        planner.preflight_move(_START, np.eye(3), motion_type="ptp")
    with pytest.raises(PosePathPreflightError, match="motion_type"):
        planner.preflight_move(_START, _target_matrix(), motion_type="circ")
    with pytest.raises(PosePathPreflightError, match="no arm controller"):
        _planner(None).preflight_move(_START, _target_matrix(), motion_type="ptp")


@pytest.mark.parametrize(
    "start, match",
    [
        ({k: v for k, v in _START.items() if k != "joint3"}, "missing joint3"),
        ({**_START, "joint2": float("nan")}, "non-finite joint2"),
        ({**_START, "joint1": 6.0}, r"joint1=6\.000000 rad is outside"),
    ],
)
def test_preflight_validates_the_exact_starting_configuration(start, match) -> None:
    controller = _Controller()
    controller.kinematics = _kinematics()
    with pytest.raises(PosePathPreflightError, match=match):
        _planner(controller).preflight_move(start, _target_matrix(), motion_type="ptp")


def test_a_start_in_collision_is_refused() -> None:
    controller = _Controller()
    controller.kinematics = _kinematics(collision_reason=lambda _point: "floor")
    with pytest.raises(PosePathPreflightError, match="start configuration: floor"):
        _planner(controller).preflight_move(_START, _target_matrix(), motion_type="ptp")


def test_local_ptp_preflight_checks_each_interpolated_collision_sample() -> None:
    controller = _Controller()
    controller.kinematics = _kinematics(
        collision_reason=lambda point: (
            "floor contact" if point["joint1"] > 0.2 else None
        ),
        ik=lambda _start, _target: {**_START, "joint1": 0.3},
        diagnose_ik=lambda *_args: None,
    )
    with pytest.raises(PosePathPreflightError, match="local PTP transition sample"):
        _planner(controller).preflight_move(_START, _target_matrix(), motion_type="ptp")


def test_ranked_grasp_preflight_couples_ptp_transit_and_lin_approach() -> None:
    sent = []
    controller = _Controller()
    controller.kinematics = _kinematics()

    trajectory = _planner(controller, sent).preflight_grasp(
        _START,
        _target_matrix(),
        approach=(0.0, 0.0, -1.0),
        approach_distance_m=0.06,
        lift_distance_m=0.06,
    )

    assert sent == []
    assert trajectory.requested_pose_count == 7
    assert len(trajectory.positions) == len(trajectory.time_from_start_s) > 7
    assert trajectory.duration_s > 0.0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"approach": (0.0, 0.0, 0.0)}, "approach must be"),
        ({"approach_distance_m": 0.0}, "approach_distance_m"),
        ({"lift_distance_m": -1.0}, "lift_distance_m"),
    ],
)
def test_grasp_preflight_validates_its_geometry(kwargs, match) -> None:
    controller = _Controller()
    controller.kinematics = _kinematics()
    arguments = {
        "approach": (0.0, 0.0, -1.0),
        "approach_distance_m": 0.06,
        "lift_distance_m": 0.06,
        **kwargs,
    }
    with pytest.raises(PosePathPreflightError, match=match):
        _planner(controller).preflight_grasp(_START, _target_matrix(), **arguments)


class _LiftRejectingController(_Controller):
    """Refuses any LIN whose final pose is above the given height."""

    ceiling = 0.35

    def pose_matrix_path_to_joints(
        self, start, poses, *, max_joint_step, allow_ik_gaps=False, ik_kwargs=None
    ):
        if float(poses[-1][2, 3]) > self.ceiling:
            return []
        return super().pose_matrix_path_to_joints(
            start,
            poses,
            max_joint_step=max_joint_step,
            allow_ik_gaps=allow_ik_gaps,
            ik_kwargs=ik_kwargs,
        )


def test_ranked_grasp_preflight_refuses_a_candidate_whose_required_lift_has_no_ik() -> (
    None
):
    """Selection must not close on a grasp that cannot make its LIN lift."""
    controller = _LiftRejectingController()
    controller.kinematics = _kinematics(
        ik=lambda _start, target: (
            None if float(target[2, 3]) > 0.35 else {**_START, "joint1": 0.25}
        ),
    )
    with pytest.raises(PosePathPreflightError, match="local LIN via point"):
        _planner(controller).preflight_grasp(
            _START,
            _target_matrix(),
            approach=(0.0, 0.0, -1.0),
            approach_distance_m=0.06,
            lift_distance_m=0.12,
        )


def test_the_lift_direction_is_configurable() -> None:
    """A base frame whose +Z is not up lifts along the axis it is told to."""
    controller = _LiftRejectingController()
    controller.kinematics = _kinematics()
    lifted_sideways = _planner(controller, lift_direction=(1.0, 0.0, 0.0))
    # +X lift keeps z at the grasp height (0.25), under the 0.35 ceiling.
    trajectory = lifted_sideways.preflight_grasp(
        _START,
        _target_matrix(),
        approach=(0.0, 0.0, -1.0),
        approach_distance_m=0.06,
        lift_distance_m=0.12,
    )
    assert trajectory.duration_s > 0.0


def test_a_grasp_preflight_comes_to_rest_at_pregrasp() -> None:
    """Transit and approach are two profiles, so the approach starts from rest."""
    controller = _Controller()
    controller.kinematics = _kinematics()

    trajectory = _planner(controller).preflight_grasp(
        _START,
        _target_matrix(),
        approach=(0.0, 0.0, -1.0),
        approach_distance_m=0.06,
        lift_distance_m=0.06,
    )

    positions, times = trajectory.positions, trajectory.time_from_start_s
    # The transit moves joint1 alone; the LIN stub's points move every joint, so
    # the pregrasp is the last joint2 == 0 sample.
    pregrasp_index = max(
        i for i, point in enumerate(positions) if point["joint2"] == 0.0
    )
    assert positions[pregrasp_index]["joint1"] == pytest.approx(0.25)
    assert trajectory.velocities[pregrasp_index] == dict.fromkeys(JOINTS, 0.0)
    assert trajectory.accelerations[pregrasp_index] == dict.fromkeys(JOINTS, 0.0)
    assert all(b > a for a, b in pairwise(times))
    first_travel = max(
        abs(positions[pregrasp_index + 1][j] - positions[pregrasp_index][j])
        for j in JOINTS
    )
    first_duration = times[pregrasp_index + 1] - times[pregrasp_index]
    # From rest, travel d under acceleration a takes at least sqrt(2 d / a).
    accel = _ACCELERATION * 0.5
    assert first_duration >= (2.0 * first_travel / accel) ** 0.5 * (1 - 1e-6)


def test_a_ranked_grasp_search_can_skip_the_diagnostic_re_solve() -> None:
    """``diagnose_ik`` re-runs the solve to name the refusing guard; a ranked
    search that will not report this candidate's reason need not pay for it."""
    diagnosed = []
    controller = _Controller()
    controller.kinematics = _kinematics(
        ik=lambda _start, _target: None,
        diagnose_ik=lambda *args: (
            diagnosed.append(args) or "every converged IK solution was refused: floor"
        ),
    )
    planner = _planner(controller)
    arguments = {
        "approach": (0.0, 0.0, -1.0),
        "approach_distance_m": 0.06,
        "lift_distance_m": 0.06,
    }
    with pytest.raises(
        PosePathPreflightError, match="no IK solution from the current configuration"
    ):
        planner.preflight_grasp(_START, _target_matrix(), diagnose=False, **arguments)
    assert diagnosed == []
    with pytest.raises(PosePathPreflightError, match="floor"):
        planner.preflight_grasp(_START, _target_matrix(), **arguments)
    assert len(diagnosed) == 1


def test_an_undiagnosed_ranked_search_still_names_a_collision_refusal() -> None:
    """``last_ik_refusal`` costs nothing: the solver set it during the failed ik."""
    controller = _Controller()
    controller.kinematics = _kinematics(
        ik=lambda _start, _target: None,
        diagnose_ik=lambda *_args: pytest.fail("diagnose_ik must not re-solve"),
        last_ik_refusal="contact with the floor plane at z=-0.0150 m",
    )
    with pytest.raises(PosePathPreflightError, match="floor plane at z=-0.0150"):
        _planner(controller).preflight_move(
            _START, _target_matrix(), motion_type="ptp", diagnose=False
        )


def test_a_ptp_without_a_solver_is_refused() -> None:
    controller = _Controller()
    controller.kinematics = SimpleNamespace(joint_limits=dict)
    with pytest.raises(PosePathPreflightError, match="solver is unavailable"):
        _planner(controller).preflight_move(_START, _target_matrix(), motion_type="ptp")


# ── End to end through pinocchio on the simple_arm fixture ──────────────────
#
# simple_arm has two revolute joints: joint1 yaws about base Z, joint2 pitches.
# At joint2 = 0 the tool sits on the yaw axis, so a pure yaw of the tool is a
# straight (rotation-only) LIN the 2-DOF arm can follow exactly.

_ARM = ("joint1", "joint2")


def _real_controller():
    try:
        import pinocchio  # noqa: F401
    except ImportError:
        pytest.skip("pinocchio unavailable")
    from cyberwave.driver.kinematics import ArmKinematicsConfig
    from cyberwave.driver.kinematics.arm.controller import ArmController

    return ArmController(
        ArmKinematicsConfig(
            urdf_path=_URDF,
            ee_frame="tool_link",
            arm_joints=_ARM,
            locked_joints=("gripper_joint",),
        )
    )


def _real_planner(controller, start, sent=None, **kwargs):
    kwargs.setdefault("velocity_limits", controller.kinematics.velocity_limits())
    kwargs.setdefault("max_joint_acceleration", 1.0)
    return LocalCartesianPlanner(
        controller=controller,
        current_joints=lambda: dict(start),
        send_trajectory=lambda t: (
            (sent.append(t) if sent is not None else None) or True
        ),
        **kwargs,
    )


def _rpy(matrix):
    r = np.asarray(matrix)[:3, :3]
    return (
        float(np.arctan2(r[2, 1], r[2, 2])),
        float(np.arcsin(max(-1.0, min(1.0, -r[2, 0])))),
        float(np.arctan2(r[1, 0], r[0, 0])),
    )


def _pose_of(controller, joints):
    here = controller.ee_pose(joints)
    return (float(here[0][3]), float(here[1][3]), float(here[2][3]), *_rpy(here))


def test_a_real_rotation_only_jog_plans_and_respects_the_urdf_limits() -> None:
    controller = _real_controller()
    start = {"joint1": 0.0, "joint2": 0.0}
    target = _pose_of(controller, {"joint1": 0.3, "joint2": 0.0})
    sent = []
    decision = _real_planner(controller, start, sent).plan_and_execute(
        CartesianRequest(target=target, source="ee_rotate_left")
    )
    assert decision.accepted, decision.detail
    trajectory = sent[0]
    assert list(trajectory.positions[0]) == list(_ARM)
    assert trajectory.positions[0] == start
    assert trajectory.positions[-1]["joint1"] == pytest.approx(0.3, abs=1e-3)
    # 0.3 rad of rotation at 50 samples/rad.
    assert len(trajectory.positions) == 1 + 15
    peak = max(abs(v["joint1"]) for v in trajectory.velocities)
    assert peak <= 1.0


def test_a_real_ptp_preflight_reaches_an_fk_target() -> None:
    controller = _real_controller()
    start = {"joint1": 0.0, "joint2": 0.0}
    target = controller.ee_pose({"joint1": 0.3, "joint2": 0.5})
    trajectory = _real_planner(controller, start).preflight_move(
        start, target, motion_type="ptp"
    )
    reached = controller.ee_pose(trajectory.positions[-1])
    assert np.allclose(reached[:3, 3], target[:3, 3], atol=1e-3)
    assert trajectory.requested_pose_count == 1


def test_the_real_refusal_names_a_waypoint_and_a_specific_guard() -> None:
    """Not "one of three things went wrong somewhere along the line"."""
    controller = _real_controller()
    start = {"joint1": 0.2, "joint2": 0.4}
    decision = _real_planner(controller, start).plan_and_execute(
        CartesianRequest(target=(5.0, 5.0, 5.0, 0.0, 0.0, 0.0), source="ee_move")
    )
    detail = decision.detail or ""
    assert decision.rejection is CartesianRejection.PLANNING_FAILED
    assert re.match(r"waypoint \d+/\d+: ", detail), detail
    assert "no IK solution" in detail, detail
    assert "no in-limit, collision-free straight-line path" not in detail, detail


def test_the_real_refusal_names_the_joint_step_cap_when_that_is_what_fired() -> None:
    """A cap of zero refuses a path whose every waypoint solves cleanly."""
    controller = _real_controller()
    start = {"joint1": 0.0, "joint2": 0.0}
    target = _pose_of(controller, {"joint1": 0.3, "joint2": 0.0})
    decision = _real_planner(
        controller, start, max_joint_step_rad=0.0
    ).plan_and_execute(CartesianRequest(target=target, source="ee_move"))
    detail = decision.detail or ""
    assert "cap 0.000" in detail, detail
    assert "rad in one step" in detail, detail
    assert "branch flip" in detail, detail


def test_pose_path_diagnosis_mirrors_the_path_solve() -> None:
    # Same solver kwargs and joint-step cap as the path solve: an off-branch
    # solution is a refused jump, not a "start configuration moved" report.
    seen = []

    def ik(seed, target, **kwargs):
        seen.append(kwargs)
        return {**seed, "joint4": seed["joint4"] + 2.0}

    controller = SimpleNamespace(kinematics=_kinematics(ik=ik))
    planner = _planner(controller, path_ik_kwargs={"restart": False})
    reason = planner._explain_pose_path(dict(_START), [_target_matrix()])
    assert seen == [{"restart": False}]
    assert "joint4 would jump 2.000 rad" in reason
    assert "start configuration moved" not in reason
