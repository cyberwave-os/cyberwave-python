"""ArmController — pure joint-target math for robot arms.

Extracted from ``ArmCapabilityMixin`` so single-arm and dual-arm paths share one
engine. This class **computes joint targets and never applies them**; it knows
nothing about ``self``, MQTT, seams, or ROS. Kinematics (pinocchio via
``BaseKinematicsManipulator``) is built lazily on the first EE/pose call, so
drivers that never issue EE/pose commands never import pinocchio.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from .config import ArmKinematicsConfig, tool_offset_from_xyz_rpy


def _jog_ik_kwargs(kin: Any) -> dict[str, Any]:
    """``restart=False`` for a solver that has it.

    A jog must stay on the arm's current IK branch: a multi-seed restart answers
    a 1 cm jog with a different posture reaching the same point, and the arm
    swings through a large, unplanned motion to get there. Refusing is right.
    """
    try:
        parameters = inspect.signature(kin.ik).parameters
    except (TypeError, ValueError):
        return {}
    return {"restart": False} if "restart" in parameters else {}


class EEPoseLike(Protocol):
    """Structural type for an absolute end-effector pose (no hard import).

    Members are read-only properties so a frozen dataclass (``EEPose``) satisfies
    the protocol.
    """

    @property
    def x(self) -> float: ...
    @property
    def y(self) -> float: ...
    @property
    def z(self) -> float: ...
    @property
    def roll(self) -> float: ...
    @property
    def pitch(self) -> float: ...
    @property
    def yaw(self) -> float: ...


def pose_to_matrix(pose: EEPoseLike) -> np.ndarray:
    """4x4 homogeneous transform from x,y,z + roll,pitch,yaw.

    Standard fixed-axis (extrinsic XYZ) RPY convention: R = Rz(yaw) @ Ry(pitch)
    @ Rx(roll), matching ROS/URDF ``rpy`` semantics. The formula is
    :func:`~cyberwave.driver.kinematics.arm.config.tool_offset_from_xyz_rpy`'s;
    there is one.
    """
    return tool_offset_from_xyz_rpy(
        pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw
    )


def validate_kinematics_candidate(
    installed: Any | None, candidate: Any, obstacles: tuple[Any, ...]
) -> None:
    """Raise ``ValueError`` unless *candidate* may replace *installed*.

    The rules every rebuild obeys: a reviewed self-collision model that loaded
    must survive the rebuild, and requested world geometry must actually be
    paired with the arm -- otherwise a scene would be silently ignored while
    reported as active.
    """
    if getattr(installed, "collision_model_valid", False) and not getattr(
        candidate, "collision_model_valid", False
    ):
        raise ValueError("the reviewed self-collision model did not load")
    if obstacles and not getattr(candidate, "obstacle_collision_available", False):
        raise ValueError("the requested world geometry was not paired with the arm")


class ArmController:
    """Config-driven pure math: computes joint targets, never applies them."""

    def __init__(
        self,
        config: ArmKinematicsConfig,
        *,
        build_kinematics: Callable[[ArmKinematicsConfig], Any] | None = None,
    ) -> None:
        self._config = config
        self._build_kinematics = build_kinematics
        self._kin: Any = None

    @property
    def config(self) -> ArmKinematicsConfig:
        return self._config

    def _kinematics(self) -> Any:
        if self._kin is None:
            build = self._build_kinematics
            if build is None:
                from .base import BaseKinematicsManipulator  # lazy: imports pinocchio

                build = BaseKinematicsManipulator
            self._kin = build(self._config)
        return self._kin

    @property
    def base_frame(self) -> str:
        """The frame :meth:`ee_pose` is expressed in: ``config.base_frame``, or
        the URDF root link the kinematics resolved when that is unset."""
        if self._config.base_frame is not None:
            return self._config.base_frame
        resolved = getattr(self._kinematics(), "base_frame", None)
        return str(resolved) if resolved else "base_link"

    @property
    def loaded_kinematics(self) -> Any | None:
        """The kinematics instance if it has been built, else ``None``.

        For status reads that must not trigger the (pinocchio) build, e.g. a
        precondition asking whether the loaded collision model is valid.
        """
        return self._kin

    def replace_kinematics(self, candidate: Any) -> None:
        """Install a rebuilt solver (new world geometry, a carried payload).

        Validate it first with :func:`validate_kinematics_candidate`.
        """
        self._kin = candidate

    @property
    def kinematics(self) -> Any:
        """The single kinematics instance this controller owns (built lazily).

        Public so read-only consumers — e.g. the ``state/kinematics`` publisher —
        borrow this model instead of constructing a second one from the same
        URDF. Prefer :meth:`ee_pose` over touching this directly.
        """
        return self._kinematics()

    def ee_pose(self, current: dict[str, float]) -> np.ndarray:
        """Measured TCP pose (4x4) in :attr:`base_frame` for *current*.

        The one place forward kinematics is run for reporting, so transport
        layers never call ``fk`` themselves. Honours ``config.tool_offset``:
        the result is the tool centre point when one is configured.
        """
        return self._kinematics().fk(self._start(current))

    def _start(self, current: dict[str, float]) -> dict[str, float]:
        return {n: current.get(n, 0.0) for n in self._config.arm_joints}

    # ── EE motion (FK -> transform -> IK) ────────────────────────────────────

    def translate(
        self,
        current: dict[str, float],
        components: dict[str, float],
        frame: str = "tool",
    ) -> dict[str, float] | None:
        kin = self._kinematics()
        start = self._start(current)
        target = kin.apply_translation(kin.fk(start), components, frame=frame)
        return kin.ik(start, target, **_jog_ik_kwargs(kin))

    def rotate(
        self, current: dict[str, float], axis: str, angle: float, frame: str = "tool"
    ) -> dict[str, float] | None:
        kin = self._kinematics()
        start = self._start(current)
        target = kin.apply_rotation(kin.fk(start), axis, angle, frame=frame)
        return kin.ik(start, target, **_jog_ik_kwargs(kin))

    def pose_to_joints(
        self, current: dict[str, float], pose: EEPoseLike
    ) -> dict[str, float] | None:
        kin = self._kinematics()
        return kin.ik(self._start(current), pose_to_matrix(pose))

    def pose_path_to_joints(
        self,
        current: dict[str, float],
        pose: EEPoseLike,
        *,
        steps: int = 1,
        max_joint_step: float | None = None,
        ik_kwargs: dict[str, Any] | None = None,
    ) -> list[dict[str, float]] | None:
        """Solve a straight-line TCP path to *pose*, one waypoint at a time.

        Each waypoint is seeded from the previous solution. Returns ``None`` —
        never a partial path — if any waypoint misses IK or any consecutive joint
        delta exceeds *max_joint_step* (rad).

        ``max_joint_step`` is a backstop, not the continuity guarantee. A solver
        that retries from other seeds can return a solution on a DIFFERENT IK
        branch when the previous-solution seed fails to converge, and a branch
        change smaller than *max_joint_step* passes this check. ``ik_kwargs`` is
        forwarded verbatim to the solver so a caller that needs strict continuity
        can disable that retry for path solving; ``None`` changes nothing.
        """
        kin = self._kinematics()
        seed = self._start(current)
        waypoints = kin.pose_waypoints(kin.fk(seed), pose_to_matrix(pose), steps)
        path: list[dict[str, float]] = []
        for target in waypoints:
            solution = kin.ik(seed, target, **(ik_kwargs or {}))
            if solution is None:
                return None
            if max_joint_step is not None and any(
                abs(solution[name] - seed.get(name, 0.0)) > max_joint_step
                for name in solution
            ):
                return None
            path.append(solution)
            seed = solution
        return path

    def pose_matrix_path_to_joints(
        self,
        current: dict[str, float],
        poses: list[np.ndarray],
        *,
        max_joint_step: float | None = None,
        allow_ik_gaps: bool = False,
        ik_kwargs: dict[str, Any] | None = None,
    ) -> list[dict[str, float]] | None:
        """Solve an ordered base-frame TCP path of 4x4 pose via points.

        The path is geometry only; the caller owns time parameterisation and
        trajectory execution. Every solve is seeded from the preceding solution.
        ``allow_ik_gaps`` explicitly permits unresolved interior CIRC samples to
        become a joint-space PTP transition; collision refusals and an unresolved
        final target still refuse the whole path.
        """
        kin = self._kinematics()
        seed = self._start(current)
        path: list[dict[str, float]] = []
        final_solved = False
        for index, target in enumerate(poses):
            solution = kin.ik(
                seed, np.asarray(target, dtype=float), **(ik_kwargs or {})
            )
            if solution is None:
                if allow_ik_gaps and not getattr(kin, "last_ik_refusal", None):
                    final_solved = False
                    continue
                return None
            if max_joint_step is not None and any(
                abs(solution[name] - seed.get(name, 0.0)) > max_joint_step
                for name in solution
            ):
                return None
            path.append(solution)
            seed = solution
            final_solved = index == len(poses) - 1
        return path if path and final_solved else None

    def position_path_to_joints(
        self,
        current: dict[str, float],
        positions_m: list[np.ndarray],
        *,
        max_joint_step: float | None = None,
        ik_kwargs: dict[str, Any] | None = None,
    ) -> list[dict[str, float]] | None:
        """Solve base-frame TCP positions while leaving tool orientation free."""
        kin = self._kinematics()
        seed = self._start(current)
        posture_reference = dict(seed)
        path: list[dict[str, float]] = []
        for position in positions_m:
            solution = kin.ik_position(
                seed, position, posture_reference=posture_reference,
                **(ik_kwargs or {}),
            )
            if solution is None:
                return None
            if max_joint_step is not None and any(
                abs(solution[name] - seed.get(name, 0.0)) > max_joint_step
                for name in solution
            ):
                return None
            path.append(solution)
            seed = solution
        return path

    # ── base joint nudge (no IK) ─────────────────────────────────────────────

    def base_rotate(self, current: dict[str, float], angle: float) -> dict[str, float]:
        base = self._config.base_joint
        assert base is not None, "base_rotate requires config.base_joint"
        cur = current.get(base, 0.0)
        if self._config.base_joint_limits is not None:
            lo, hi = self._config.base_joint_limits
        else:
            lo, hi = (
                self._kinematics()
                .joint_limits()
                .get(base, (float("-inf"), float("inf")))
            )
        return {base: max(lo, min(hi, cur + angle))}

    # ── gripper command + feedback ───────────────────────────────────────────

    def gripper_target(self, force: float) -> dict[str, float]:
        g = self._config.gripper
        if g is None:
            return {}
        force = max(0.0, min(1.0, force))
        target = g.open + force * (g.closed - g.open)
        positions: dict[str, float] = {name: target for name in g.names}
        for mimic, factor in g.mimic.items():
            positions[mimic] = factor * target
        return positions

    def classify_gripper(self, value: float) -> str:
        g = self._config.gripper
        assert g is not None, "classify_gripper requires a gripper config"
        return "open" if abs(value - g.open) <= abs(value - g.closed) else "closed"

    def binarize(self, payload: dict[str, float]) -> dict[str, float]:
        g = self._config.gripper
        if g is None or not g.binarize or not g.binarize_measured:
            return payload
        result = payload
        for name in g.names:
            if name in payload:
                snapped = (
                    g.open
                    if self.classify_gripper(payload[name]) == "open"
                    else g.closed
                )
                result = {**result, name: snapped}
                for mimic, factor in g.mimic.items():
                    if mimic in result:
                        result = {**result, mimic: factor * snapped}
        return result

    def hold(self, latched_state: str | None) -> dict[str, float]:
        g = self._config.gripper
        if g is None:
            return {}
        value = g.open if (latched_state or "open") == "open" else g.closed
        return {name: value for name in g.names}
