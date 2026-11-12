"""In-process IK planning for any arm, executing through its trajectory controller.

This is the planner of the built-in ``"pinocchio"`` kinematics backend
(:class:`~cyberwave.driver.motion.pinocchio_backend.PinocchioMotionBackend`),
and of the Cartesian jogs. It is never a silent fallback from another
backend: a request names the backend it wants, or gets the driver's default.

What it is NOT is streaming raw IK waypoints at a vendor command topic. That
path bypasses trajectory tolerances, preemption, cancellation and arbitration,
and it stays ruled out. Only the *planner* is in-process:

    request -> this module (straight-line IK) -> injected send -> trajectory controller

Everything downstream is the driver's: the trajectory controller still owns
interpolation, path and goal tolerances, preemption and cancellation. Execution
is injected as a callback, so this module needs no ROS.

What it does not have is a world scene beyond what the solver exposes. Each
sampled configuration is checked against the solver's joint limits and, when
the kinematics object provides ``collision_reason(joints) -> str | None``,
against its collision model; nothing else (a desk, segmented objects, an
attached payload) is seen unless that model contains it. A request whose
declared PTP or LIN motion does not work is refused; the motion type is never
changed as a fallback.

LIN here is straight to its SAMPLING resolution, not exactly: the tool path is
solved and checked at :func:`waypoint_count` samples and the trajectory
controller interpolates in joint space between them, so the tool point bows
slightly off the line within one sample interval. An exact-path guarantee is a
property of a dedicated Cartesian planner, not of this one.

Optional solver hooks, all looked up with ``getattr`` so a plain
:class:`~cyberwave.driver.kinematics.arm.base.BaseKinematicsManipulator` works
and richer solvers get better diagnostics:

* ``joint_limits() -> {joint: (lo, hi)}`` -- start-state and spline audit.
* ``collision_reason(joints) -> str | None`` -- per-sample collision audit.
* ``diagnose_ik(seed, target) -> str | None`` -- names the guard that refused.
* ``last_ik_refusal: str | None`` and ``IK_REFUSED_PREFIX`` -- a free reason
  from the most recent failed ``ik`` call.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ..kinematics.arm.controller import pose_to_matrix
from ..kinematics.arm.trajectory import (
    DEFAULT_WAYPOINT_SAMPLING,
    JOINT_SAMPLE_RAD,
    LocalTrajectory,
    PosePathPreflightError,
    WaypointSampling,
    _import_spline,
    interpolate_joint_transitions,
    smooth_trajectory,
)
from .cartesian import EEPose
from .cartesian_request import (
    CartesianDecision,
    CartesianRejection,
    CartesianRequest,
)

#: Default per-waypoint joint jump (rad) treated as an IK branch flip mid-path.
#: A backstop, not the continuity guarantee (see ``path_ik_kwargs``); at the
#: default 1 cm sampling a legitimate step is far smaller than this.
DEFAULT_MAX_JOINT_STEP_RAD = 0.2

#: Default base-frame direction of the post-grasp lift preflighted by
#: :meth:`LocalCartesianPlanner.preflight_grasp`: straight up, for a base frame
#: whose +Z opposes gravity.
DEFAULT_LIFT_DIRECTION: tuple[float, float, float] = (0.0, 0.0, 1.0)


class LocalCartesianPlanner:
    """Straight-line Cartesian planning, executed through the arm controller.

    Satisfies :class:`~cyberwave.driver.control.cartesian_request.CartesianPlanner`,
    so a driver hands over a validated absolute TCP target and gets a decision
    back regardless of which planner it composed.

    ``velocity_limits`` (rad/s per joint) and ``max_joint_acceleration``
    (rad/s²) are required: they are the arm's physical contract, and a
    trajectory timed against a made-up default is one the hardware lags and
    the trajectory controller aborts.
    """

    def __init__(
        self,
        *,
        controller: Any,
        current_joints: Callable[[], dict[str, float]],
        send_trajectory: Callable[[LocalTrajectory], bool],
        velocity_limits: Mapping[str, float],
        max_joint_acceleration: float,
        arm_joints: Sequence[str] | None = None,
        max_joint_step_rad: float = DEFAULT_MAX_JOINT_STEP_RAD,
        sampling: WaypointSampling = DEFAULT_WAYPOINT_SAMPLING,
        joint_sample_rad: float = JOINT_SAMPLE_RAD,
        path_ik_kwargs: Mapping[str, Any] | None = None,
        lift_direction: tuple[float, float, float] = DEFAULT_LIFT_DIRECTION,
    ) -> None:
        """Compose the planner over an ``ArmController``-shaped *controller*.

        ``arm_joints`` defaults to ``controller.config.arm_joints`` and then to
        the keys of ``velocity_limits``; it fixes which joints are timed and the
        key order of every trajectory row.

        ``path_ik_kwargs`` is forwarded to the solver for every PATH solve (LIN
        and straight-line jogs), never for an absolute PTP target. A solver
        that restarts from other seeds when the previous-solution seed fails
        will silently swap IK branch mid-path -- the wrist wanders while the TCP
        stays on the line -- so such a solver should be given the kwarg that
        disables its restart (e.g. ``{"restart": False}``). ``None`` passes
        nothing, which is right for a single-seed solver.
        """
        self._controller = controller
        self._current_joints = current_joints
        self._send_trajectory = send_trajectory
        if arm_joints is None:
            config = getattr(controller, "config", None)
            arm_joints = getattr(config, "arm_joints", None) or tuple(velocity_limits)
        self._joints: tuple[str, ...] = tuple(arm_joints)
        # Immutable, per joint, resolved once. There is no per-request override:
        # one configured ceiling times every local trajectory, exactly as it
        # caps the hardware command.
        self._velocity_limits = {
            joint: float(velocity_limits[joint]) for joint in self._joints
        }
        self._max_joint_acceleration = float(max_joint_acceleration)
        self._max_joint_step_rad = float(max_joint_step_rad)
        self._sampling = sampling
        self._joint_sample_rad = float(joint_sample_rad)
        self._path_ik_kwargs = dict(path_ik_kwargs) if path_ik_kwargs else None
        self._lift_direction = np.asarray(lift_direction, dtype=float)
        # Pay the scipy.interpolate import here, at driver start, rather than
        # on the driver loop inside the first plan. Absent scipy is reported by
        # that plan, as before.
        try:
            _import_spline()
        except ImportError:
            pass

    @property
    def arm_joints(self) -> tuple[str, ...]:
        return self._joints

    @property
    def available(self) -> bool:
        """True whenever a solver is composed. No external process to wait on."""
        return self._controller is not None

    def plan_and_execute(
        self,
        request: CartesianRequest,
        *,
        is_current: Callable[[], bool] = lambda: True,
    ) -> CartesianDecision:
        """Plan a straight TCP line to ``request.target`` and send it.

        ``is_current`` is asked once more after planning: a request superseded
        while IK ran is never submitted.
        """
        if self._controller is None:
            return CartesianDecision(
                rejection=CartesianRejection.PLANNER_UNAVAILABLE,
                detail="no arm controller composed",
            )
        try:
            start = dict(self._current_joints())
            here = self._controller.ee_pose(start)
            target = request.target
            distance = math.dist(
                (float(here[0][3]), float(here[1][3]), float(here[2][3])),
                (target[0], target[1], target[2]),
            )
            target_matrix = pose_to_matrix(EEPose(*target))
            rotation = np.asarray(here)[:3, :3].T @ target_matrix[:3, :3]
            angle = math.acos(max(-1.0, min(1.0, (float(np.trace(rotation)) - 1) / 2)))
            steps = self._sampling.count(distance, angle)
            path = self._controller.pose_path_to_joints(
                start,
                EEPose(*target),
                steps=steps,
                max_joint_step=self._max_joint_step_rad,
                ik_kwargs=self._path_ik_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - a solver crash is a refusal
            return CartesianDecision(
                rejection=CartesianRejection.PLANNING_FAILED,
                detail=f"{type(exc).__name__}: {exc}",
            )

        if not path:
            # All-or-nothing: if any waypoint misses IK, leaves the joint limits,
            # or collides, the whole request is refused. A partially executed
            # straight line leaves the arm somewhere nobody asked for.
            #
            # Which of those fired is the whole diagnosis: "move the wrist away
            # from the shoulder" and "that pose is outside the arm's travel" have
            # nothing in common except a generic message.
            return CartesianDecision(
                rejection=CartesianRejection.PLANNING_FAILED,
                detail=self._explain(start, target, steps),
            )

        try:
            trajectory = self._time_path(start, path, requested_pose_count=len(path))
        except (ValueError, PosePathPreflightError) as exc:
            return CartesianDecision(
                rejection=CartesianRejection.PLANNING_FAILED, detail=str(exc)
            )
        if not is_current():
            return CartesianDecision(
                rejection=CartesianRejection.SUPERSEDED,
                detail="a newer command replaced this Cartesian target",
            )
        if not self._send_trajectory(trajectory):
            return CartesianDecision(
                rejection=CartesianRejection.PLANNING_FAILED,
                detail="arm controller did not accept the trajectory goal",
            )
        return CartesianDecision(request=request)

    def preflight_move(
        self,
        start: dict[str, float],
        target: Any,
        *,
        motion_type: str,
        diagnose: bool = True,
    ) -> LocalTrajectory:
        """Build one timed local-IK PTP or base-frame LIN trajectory, without moving.

        ``target`` is a base-frame 4x4 TCP transform. LIN is straight at
        :func:`waypoint_count` resolution, not exactly; see the module
        docstring. ``diagnose=False`` skips the second full IK pass that names
        WHICH guard refused a PTP target: a ranked search that will not report
        this candidate's reason has no use for it, and it doubles the cost of
        every failing pose.
        """
        if self._controller is None:
            raise PosePathPreflightError("no arm controller composed")
        target_matrix = np.asarray(target, dtype=float)
        if target_matrix.shape != (4, 4) or not np.isfinite(target_matrix).all():
            raise PosePathPreflightError("target must be a finite 4x4 transform")
        start = {name: float(value) for name, value in start.items()}
        start_refusal = self._starting_configuration_refusal(start)
        if start_refusal:
            raise PosePathPreflightError(start_refusal)

        if motion_type == "ptp":
            raw_path = self._preflight_ptp_path(start, target_matrix, diagnose=diagnose)
            requested_pose_count = 1
        elif motion_type == "lin":
            raw_path, requested_pose_count = self._preflight_lin_path(
                start, target_matrix
            )
        else:
            raise PosePathPreflightError("motion_type must be 'ptp' or 'lin'")

        return self._time_path(
            start, raw_path, requested_pose_count=requested_pose_count
        )

    def _time_path(
        self,
        start: dict[str, float],
        path: list[dict[str, float]],
        *,
        requested_pose_count: int,
    ) -> LocalTrajectory:
        kin = getattr(self._controller, "kinematics", None)
        get_limits = getattr(kin, "joint_limits", None)
        return smooth_trajectory(
            path,
            start,
            velocity_limits=self._velocity_limits,
            max_joint_acceleration=self._max_joint_acceleration,
            requested_pose_count=requested_pose_count,
            joint_limits=get_limits() if callable(get_limits) else None,
            collision_reason=getattr(kin, "collision_reason", None),
            joints=self._joints,
        )

    def preflight_grasp(
        self,
        start: dict[str, float],
        grasp_target: Any,
        *,
        approach: tuple[float, float, float],
        approach_distance_m: float,
        lift_distance_m: float,
        lift_direction: tuple[float, float, float] | None = None,
        diagnose: bool = True,
    ) -> LocalTrajectory:
        """Build a local-IK PTP-to-pregrasp plus base-frame LIN approach trajectory.

        The two phases are timed as separate profiles, so the arm comes to rest
        at the pregrasp pose before the straight approach begins. One profile
        over the concatenation would carry cruise speed into the last
        centimetres before contact.

        The returned trajectory stops at the grasp pose. Before selecting this
        candidate, the required separate post-close LIN lift along
        *lift_direction* (default: the configured one) is preflighted. This does not cache or
        submit the lift: after gripper closure the executor must still plan
        from measured joints.
        """
        direction = np.asarray(approach, dtype=float)
        norm = float(np.linalg.norm(direction))
        if direction.shape != (3,) or not math.isfinite(norm) or norm < 1e-9:
            raise PosePathPreflightError("approach must be a finite non-zero vector")
        if not math.isfinite(approach_distance_m) or approach_distance_m <= 0.0:
            raise PosePathPreflightError("approach_distance_m must be positive")
        if not math.isfinite(lift_distance_m) or lift_distance_m <= 0.0:
            raise PosePathPreflightError("lift_distance_m must be positive")
        direction /= norm
        grasp = np.asarray(grasp_target, dtype=float)
        if grasp.shape != (4, 4) or not np.isfinite(grasp).all():
            raise PosePathPreflightError("grasp target must be a finite 4x4 transform")
        pregrasp = grasp.copy()
        pregrasp[:3, 3] -= direction * float(approach_distance_m)

        start = {name: float(value) for name, value in start.items()}
        transit = self.preflight_move(
            start, pregrasp, motion_type="ptp", diagnose=diagnose
        )
        pregrasp_joints = dict(transit.positions[-1])
        approach_trajectory = self.preflight_move(
            pregrasp_joints, grasp, motion_type="lin"
        )
        grasp_joints = dict(approach_trajectory.positions[-1])
        lift = grasp.copy()
        up = (
            self._lift_direction
            if lift_direction is None
            else np.asarray(lift_direction, dtype=float)
        )
        lift[:3, 3] += up * float(lift_distance_m)
        self.preflight_move(grasp_joints, lift, motion_type="lin")
        positions = [dict(point) for point in transit.positions]
        positions.extend(dict(point) for point in approach_trajectory.positions[1:])
        transit_end = (
            transit.time_from_start_s[-1] if transit.time_from_start_s else 0.0
        )
        times = list(transit.time_from_start_s)
        times.extend(
            transit_end + when for when in approach_trajectory.time_from_start_s[1:]
        )
        return LocalTrajectory(
            positions=tuple(positions),
            time_from_start_s=tuple(times),
            velocities=transit.velocities + approach_trajectory.velocities[1:],
            accelerations=transit.accelerations + approach_trajectory.accelerations[1:],
            requested_pose_count=1 + approach_trajectory.requested_pose_count,
            solved_pose_count=1 + approach_trajectory.solved_pose_count,
        )

    def _preflight_ptp_path(
        self, start: dict[str, float], target: np.ndarray, *, diagnose: bool = True
    ) -> list[dict[str, float]]:
        kin = getattr(self._controller, "kinematics", None)
        solve = getattr(kin, "ik", None)
        if not callable(solve):
            raise PosePathPreflightError("local IK solver is unavailable")
        solution = solve(start, target)
        if solution is None:
            explain = getattr(kin, "diagnose_ik", None) if diagnose else None
            reason = explain(start, target) if callable(explain) else None
            if reason is None:
                # Free: the solver already knows whether a seed converged and
                # was refused by a collision guard. Without it a ranked search
                # reports "no IK" for a pose that solves but touches the floor.
                refusal = getattr(kin, "last_ik_refusal", None)
                if refusal:
                    prefix = getattr(
                        type(kin),
                        "IK_REFUSED_PREFIX",
                        "every converged IK solution was refused: ",
                    )
                    reason = f"{prefix}{refusal}"
            raise PosePathPreflightError(
                reason or "no IK solution from the current configuration"
            )
        path, refusal = self._audit_joint_transitions(
            start, [solution], context="local PTP"
        )
        if refusal:
            raise PosePathPreflightError(refusal)
        return path

    def _preflight_lin_path(
        self, start: dict[str, float], target: np.ndarray
    ) -> tuple[list[dict[str, float]], int]:
        here = np.asarray(self._controller.ee_pose(start), dtype=float)
        distance = math.dist(tuple(here[:3, 3]), tuple(target[:3, 3]))
        count = self._sampling.count(distance)
        kin = getattr(self._controller, "kinematics", None)
        interpolate = getattr(kin, "pose_waypoints", None)
        if not callable(interpolate):
            raise PosePathPreflightError("local Cartesian interpolator is unavailable")
        poses = interpolate(here, target, count)
        path = self._controller.pose_matrix_path_to_joints(
            start,
            poses,
            max_joint_step=self._max_joint_step_rad,
            ik_kwargs=self._path_ik_kwargs,
        )
        if not path:
            raise PosePathPreflightError(self._explain_pose_path(start, poses))
        return [dict(point) for point in path], count

    def _starting_configuration_refusal(self, start: dict[str, float]) -> str | None:
        """Validate the exact joint state from which the trajectory will start."""
        for joint in self._joints:
            if joint not in start:
                return f"start configuration is missing {joint}"
            if not math.isfinite(start[joint]):
                return f"start configuration has non-finite {joint}"

        kin = getattr(self._controller, "kinematics", None)
        joint_limits = getattr(kin, "joint_limits", None)
        if callable(joint_limits):
            for joint, (lower, upper) in joint_limits().items():
                if joint in start and not lower <= start[joint] <= upper:
                    return (
                        f"start configuration {joint}={start[joint]:.6f} rad "
                        f"is outside [{lower:.6f}, {upper:.6f}] rad"
                    )

        collision_reason = getattr(kin, "collision_reason", None)
        if callable(collision_reason):
            reason = collision_reason(start)
            if reason:
                return f"start configuration: {reason}"
        return None

    def _audit_joint_transitions(
        self,
        start: dict[str, float],
        path: list[dict[str, float]],
        *,
        context: str,
    ) -> tuple[list[dict[str, float]], str | None]:
        """Sample joint-space transitions and apply the local collision model."""
        expanded = interpolate_joint_transitions(
            path, start, sample_rad=self._joint_sample_rad
        )
        kin = getattr(self._controller, "kinematics", None)
        collision_reason = getattr(kin, "collision_reason", None)
        if not callable(collision_reason):
            return expanded, None
        total = len(expanded)
        for index, positions in enumerate(expanded, start=1):
            reason = collision_reason(positions)
            if reason:
                return [], (f"{context} transition sample {index}/{total}: {reason}")
        return expanded, None

    def _explain_pose_path(self, start: dict[str, float], poses: list[Any]) -> str:
        """Name the local LIN via point and guard that prevented a trajectory."""
        kin = getattr(self._controller, "kinematics", None)
        if kin is None:
            return (
                "local LIN path refused; solver unavailable for a per-via-point "
                "diagnosis"
            )
        # Same solver kwargs and joint-step check as the path solve: a
        # restarted, off-branch solution would otherwise let this walk finish and
        # report a moved start configuration that never happened.
        seed = dict(start)
        total = len(poses)
        for index, target in enumerate(poses, start=1):
            solution = kin.ik(seed, target, **(self._path_ik_kwargs or {}))
            if solution is None:
                reason = getattr(kin, "diagnose_ik", lambda *_: None)(seed, target)
                return (
                    f"local LIN via point {index}/{total}: "
                    f"{reason or 'no IK solution from the current configuration'}"
                )
            worst = max(
                ((abs(solution[n] - seed.get(n, 0.0)), n) for n in solution),
                default=(0.0, ""),
            )
            if worst[0] > self._max_joint_step_rad:
                return (
                    f"local LIN via point {index}/{total}: {worst[1]} would jump "
                    f"{worst[0]:.3f} rad in one step (cap {self._max_joint_step_rad:.3f}) "
                    "-- an IK branch flip mid-path, refused before anything moves"
                )
            seed = solution
        return (
            "local LIN path was refused but every required pose re-solved during "
            "diagnosis; start configuration moved"
        )

    def _explain(
        self, start: dict[str, float], target: tuple[float, ...], steps: int
    ) -> str:
        """Name the guard that refused the path, and the waypoint it refused.

        Runs only after ``pose_path_to_joints`` has already returned nothing, so
        the cost is irrelevant and nothing on the success path changes.

        This mirrors ``ArmController.pose_path_to_joints`` deliberately: same
        waypoints, same seeding, same joint-step comparison. If that walk
        changes, this one has to change with it, or the message starts
        describing a path the planner never tried.
        """
        # A refusal must never become a crash: diagnosis is best-effort, and a
        # controller that exposes no solver (a test double) still has to
        # produce a refusal the caller can read.
        kin = getattr(self._controller, "kinematics", None)
        if kin is None:
            return (
                f"no in-limit, collision-free straight-line path over {steps} "
                "waypoint(s); solver unavailable for a per-waypoint diagnosis"
            )
        seed = dict(start)
        try:
            waypoints = kin.pose_waypoints(
                kin.fk(seed), pose_to_matrix(EEPose(*target)), steps
            )
        except Exception as exc:  # noqa: BLE001 - diagnosis must never raise
            return f"could not sample the straight line: {type(exc).__name__}: {exc}"

        for index, waypoint in enumerate(waypoints, start=1):
            solution = kin.ik(seed, waypoint, **(self._path_ik_kwargs or {}))
            if solution is None:
                reason = getattr(kin, "diagnose_ik", lambda *_: None)(seed, waypoint)
                return (
                    f"waypoint {index}/{steps}: "
                    f"{reason or 'no IK solution from the current configuration'}"
                )
            worst = max(
                ((abs(solution[n] - seed.get(n, 0.0)), n) for n in solution),
                default=(0.0, ""),
            )
            if worst[0] > self._max_joint_step_rad:
                return (
                    f"waypoint {index}/{steps}: {worst[1]} would jump "
                    f"{worst[0]:.3f} rad in one step (cap {self._max_joint_step_rad:.3f}) "
                    "-- an IK branch flip mid-path, refused before anything moves"
                )
            seed = solution
        # The walk succeeded on the retry. Report that rather than inventing a
        # cause: it means the refusal depends on state that moved underneath us,
        # which is a real answer and a different bug from an unreachable pose.
        return (
            f"straight line over {steps} waypoint(s) was refused but re-solved on "
            "diagnosis -- the start configuration moved between plan and retry"
        )
