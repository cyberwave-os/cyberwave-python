"""PlannedMotionMixin -- planning, motion and trajectories behind ONE backend name.

A driver declares which planner answers its planned-motion commands::

    class MyArmDriver(
        PlannedMotionMixin,        # first: it overrides seams of the ones below
        WorldObstaclesMixin,
        ArmCapabilityMixin,
        EECartesianPoseMixin,
        JointCommandBufferMixin,
        BaseROS2Driver,
    ):
        kinematics_backend = "pinocchio"   # or a registered name -- the only switch
        use_base_commands = True
        use_cartesian_pose = True
        PLANNED_MOTION_MAX_JOINT_ACCELERATION_RAD_S2 = 2.0

        def arm_config(self): ...
        def _motion_preconditions(self): return Preconditions(...)

and gets, identically for every backend:

* ``set_pose`` (a target pose -- absolute or relative, PTP or LIN, in any
  frame of the robot description), ``grasp`` (ranked candidates),
  ``scene_set``, ``attach_payload`` and ``detach_payload``, with one argument
  table, one catalog description and one refusal vocabulary; a request may
  name another registered backend in ``backend``;
* ``ee_pose`` (absolute TCP pose) as ``set_pose`` with a PTP motion;
* ``state/motion`` published from the driver tick (edge-triggered + heartbeat);
* ``stop`` (and ``_stop_motion``) cancelling the active planned motion;
* :meth:`replace_planned_motion`, for driver-owned motion (a jog) that must
  take the arm only once the planned motion has ended.

``kinematics_backend`` may also be assigned on the instance (e.g. from config
or an environment variable in ``configure()``); it names the default, read on
every request.

## Backends

``"pinocchio"`` -- in-process IK and collision checks, executed through the
arm's trajectory controller -- is built in. A driver that has another planner
(an external framework running beside it, say) implements
:class:`~cyberwave.driver.motion.backend.MotionBackend` in its own package and
registers it under a name::

    def _motion_backend_factories(self):
        return {**super()._motion_backend_factories(), "sampler": self._build_sampler}

    def configure(self, ...):
        self.kinematics_backend = "sampler" if config.sampler else "pinocchio"

Selecting a name nobody registered fails loudly, naming the registered ones.
Every backend that has been used holds the same task scene and payload.

## MRO

``PlannedMotionMixin`` must come BEFORE ``EECartesianPoseMixin`` (it overrides
``_dispatch_ee_pose`` so the existing ``ee_pose`` registration routes to the
planner -- with ``use_cartesian_pose = False``, or without that mixin, it
registers ``ee_pose`` itself), before ``WorldObstaclesMixin`` (so an
``add_obstacles`` rebuild keeps the task scene and the carried payload), and
before ``ArmCapabilityMixin`` / ``JointCommandBufferMixin`` (its default
``_build_arm_kinematics`` builds collision-aware kinematics, and its
``_stop_motion`` cancels the planned motion first). A class that orders them
otherwise is refused when it is defined.

## Seams (all optional; defaults in brackets)

* :meth:`_motion_preconditions` -- the gate matrix for moves [all pass];
* :meth:`_motion_arm_controller` -- the ``ArmController`` [``self._arm``];
* :meth:`_motion_trajectory_executor` -- ``send(traj, on_active, on_done)`` /
  ``cancel()`` [SDK ``TrajectoryExecutor`` on
  ``PLANNED_MOTION_TRAJECTORY_ACTION`` under ``namespaced()``];
* :meth:`_motion_velocity_limits` / :meth:`_motion_max_joint_acceleration` --
  trajectory timing for the in-process planner [URDF velocity limits /
  ``PLANNED_MOTION_MAX_JOINT_ACCELERATION_RAD_S2``];
* :meth:`_planned_motion_kinematics_kwargs` -- extra
  ``CollisionAwareKinematics`` kwargs (``srdf_path``, floor, limit overrides);
* :meth:`_motion_backend_factories` -- backend name -> zero-argument factory
  [``{"pinocchio": _build_pinocchio_motion_backend}``];
* :meth:`_planned_motion_config` -- task tolerances [``PlannedMotionConfig()``];
* :meth:`_motion_frame_source` -- the arm's frames for relative targets and
  reference frames [forward kinematics of the arm controller];
* :meth:`_motion_planning_frame`, :meth:`_motion_wrist_payload_present`,
  :meth:`_refresh_planned_motion_status`, :meth:`_on_grasp_candidate_refused`.
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

from ..command_response import CommandOutcome
from ..interface import CallbackGroup, CommandArg, CommandArgs, TopicSpec
from ..interface.args import TELEOP_MODES, DriverOperationMode, PublisherArgs
from ..kinematics.arm.controller import validate_kinematics_candidate
from ..kinematics.arm.trajectory import LocalTrajectory
from ..motion.backend import PINOCCHIO_BACKEND, MotionBackend
from ..motion.guards import Refusal
from ..motion.pinocchio_backend import PinocchioMotionBackend
from ..motion.planner import (
    MOTION_COMMANDS,
    PlannedMotionConfig,
    PlannedMotionController,
)
from ..motion.requests import DEFAULT_APPROACH_DISTANCE_M, DEFAULT_LIFT_DISTANCE_M
from ..motion.scene import DEFAULT_PAYLOAD_ID, PLANNING_FRAME, Primitive
from ..motion.status import MotionStatus, MotionStatusStream
from ..motion.targets import FrameSource
from ..support.utils import driver_logger
from .cartesian import EE_POSE_ARGS, EECartesianPoseMixin, EEPose, is_cartesian_jog
from .cartesian_request import (
    CartesianDecision,
    CartesianRejection,
    CartesianRequest,
    Preconditions,
)
from .local_planner import LocalCartesianPlanner

logger = logging.getLogger(__name__)

_ALL_OPERATION_MODES = frozenset(DriverOperationMode)


# ── one catalog for every backend ────────────────────────────────────────────

#: How a target may be written (see :mod:`cyberwave.driver.motion.targets`):
#: always a position and a rotation, metres and radians.
POSE_FORMS = (
    "[x,y,z,roll,pitch,yaw] | [x,y,z,qx,qy,qz,qw] | [[x,y,z],[rotation]] | "
    "{position:{x,y,z}, orientation:{x,y,z,w}, reference_frame}"
)


def planned_motion_command_args(
    frame: str = PLANNING_FRAME,
) -> dict[str, tuple[CommandArg, ...]]:
    """Wire arguments of every planned-motion command, in catalog order.

    One table whichever backend answers: an executor written against a
    pinocchio driver must work unchanged against any other backend. Every
    argument that is an option has its default here; ``None`` with a
    ``required`` hint marks the few that are data (the target, the grasp
    candidates, the payload size).
    """
    backend = CommandArg("backend", None, "the driver's kinematics_backend")
    expected = CommandArg(
        "expected_joints", None, "rad per joint; refuse if the arm has moved"
    )
    revision = CommandArg("scene_revision", None, "refuse unless this scene is active")
    plan_only = CommandArg("plan_only", False, "plan, report, move nothing")
    return {
        "set_pose": (
            CommandArg("pose", None, f"required: {POSE_FORMS}"),
            CommandArg("mode", "absolute", "absolute|relative"),
            CommandArg("motion", "ptp", "ptp|lin"),
            CommandArg(
                "reference_frame",
                None,
                f"{frame} when absolute, the tool when relative; any frame "
                "of the robot description, base or tool",
            ),
            backend,
            expected,
            revision,
            plan_only,
        ),
        "grasp": (
            CommandArg(
                "candidates",
                None,
                "required: ranked [{pose, approach:{x,y,z}}]; approach "
                "defaults to the tool's own axis",
            ),
            CommandArg("reference_frame", None, frame),
            CommandArg("approach_distance_m", DEFAULT_APPROACH_DISTANCE_M, "m"),
            CommandArg("lift_distance_m", DEFAULT_LIFT_DISTANCE_M, "m"),
            backend,
            expected,
            revision,
            plan_only,
        ),
        "scene_set": (
            CommandArg("objects", [], f"box/cylinder/sphere primitives in {frame}"),
            CommandArg("allow_touch", [], "object ids the gripper may contact"),
            CommandArg("scene_revision", None, "numbered by the driver"),
            CommandArg("frame", frame),
        ),
        "attach_payload": (
            CommandArg("dimensions", None, "required: m [x, y, z]"),
            CommandArg("object_id", DEFAULT_PAYLOAD_ID),
            CommandArg("pose", None, "relative to the TCP; the TCP itself"),
        ),
        "detach_payload": (CommandArg("object_id", None, "the attached payload"),),
    }


PLANNED_MOTION_DESCRIPTIONS: dict[str, str] = {
    "set_pose": (
        "Move the TCP to a pose -- a position and a rotation, or a waypoint, in "
        "any frame of the robot description -- through the driver's kinematics backend, "
        "or the one named in backend. motion=ptp (default) finds any "
        "collision-free path; lin moves the tool along a straight line, "
        "collision-checked along the whole line and refused rather than bent. "
        "mode=relative moves the tool by the pose instead of to it (along the "
        "tool's own axes unless reference_frame says otherwise), starting from "
        "the measured tool pose when it arrives. Results on state/motion."
    ),
    "grasp": (
        "Grasp the first reachable candidate of a ranked list: PTP to the "
        "pregrasp pose (approach_distance_m back from the grasp), a straight "
        "approach onto the grasp pose, and a straight lift_distance_m lift "
        "checked from there. The first candidate whose three phases all plan "
        "is executed up to the grasp pose; the lift is checked, not executed, "
        "and the gripper is left to grip. state/motion reports the "
        "selected_candidate."
    ),
    "scene_set": (
        "Replace this driver's task collision objects (box/cylinder/sphere "
        "primitives in the planning frame). allow_touch lists object ids the "
        "gripper may contact, which is what makes a grasp pose plannable at "
        "all. Without data it clears the task scene. Every backend holds the "
        "same scene."
    ),
    "attach_payload": (
        "Carry an inflated box on the TCP in every later plan, collision-checked "
        "against the world and the arm but not the gripper holding it. An object "
        "of the same id leaves the world scene."
    ),
    "detach_payload": (
        "Remove the attached payload from the robot and from the world. It is "
        "not re-added at its carried pose: send a fresh scene_set describing "
        "where it actually landed."
    ),
}

#: ``ee_pose`` when this mixin owns it.
PLANNED_EE_POSE_DESCRIPTION = (
    "Absolute TCP pose [x, y, z, roll, pitch, yaw] (extrinsic XYZ): set_pose "
    "with motion=ptp, planned by the driver's kinematics backend and reported "
    "on state/motion."
)

#: The SDK classes whose seams this mixin overrides; it must precede them.
_MUST_PRECEDE = (
    "EECartesianPoseMixin",
    "WorldObstaclesMixin",
    "ArmCapabilityMixin",
    "JointCommandBufferMixin",
)


class PlannedMotionMixin:
    """Planned arm motion through one selectable kinematics backend."""

    kinematics_backend: str = PINOCCHIO_BACKEND
    """The name of the backend answering planned motion: the ONLY switch.

    ``"pinocchio"`` (built in) or any name :meth:`_motion_backend_factories`
    registers. Resolved on the instance, so a driver may assign it from config
    in ``configure()``.
    """

    use_planned_motion_commands: bool = True
    """Register set_pose / grasp / scene_set / attach_payload / detach_payload."""

    use_planned_ee_pose: bool = True
    """Route ``ee_pose`` through the planner (registering it if nothing else does)."""

    use_motion_state_publisher: bool = True
    """Publish ``state/motion`` from the driver tick."""

    MOTION_STATE_RATE_HZ: float = 10.0
    """Tick rate of the ``state/motion`` publisher (it only emits when due)."""

    PLANNED_MOTION_TRAJECTORY_ACTION: str = "arm_controller/follow_joint_trajectory"
    """Relative ``FollowJointTrajectory`` action the default executor sends to."""

    PLANNED_MOTION_MAX_JOINT_ACCELERATION_RAD_S2: float | None = None
    """Physical joint acceleration bound timing in-process trajectories (rad/s²).

    No default on purpose: a trajectory timed against a made-up bound is one the
    hardware lags and the controller aborts. Unset means the pinocchio backend
    refuses with ``backend_unavailable``.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Only the type is checked here: which names exist is decided by the
        # instance's _motion_backend_factories, and resolved on the instance.
        if "kinematics_backend" in cls.__dict__:
            value = cls.__dict__["kinematics_backend"]
            if not isinstance(value, str) or not value.strip():
                raise TypeError(
                    f"{cls.__name__}: kinematics_backend must be a non-empty "
                    f"backend name, got {value!r}"
                )
        mro = cls.__mro__
        own = mro.index(PlannedMotionMixin)
        for index, base in enumerate(mro):
            if (
                base.__name__ in _MUST_PRECEDE
                and base.__module__.startswith("cyberwave.driver.")
                and index < own
            ):
                raise TypeError(
                    f"{cls.__name__}: PlannedMotionMixin must come before "
                    f"{base.__name__} in the base-class list (it overrides that "
                    "mixin's seams)"
                )

    # ── backend selection ────────────────────────────────────────────────────

    @property
    def selected_kinematics_backend(self) -> str:
        """The selected backend name (class attribute or instance override).

        Raises ``ValueError`` naming the registered backends when the name is
        not one of them.
        """
        name = str(self.kinematics_backend or "").strip().lower()
        registered = self._motion_backend_factories()
        if name not in registered:
            raise ValueError(
                f"{type(self).__name__}: kinematics_backend {name!r} is not a "
                f"registered motion backend; registered: {sorted(registered)}"
            )
        return name

    @property
    def planned_motion(self) -> PlannedMotionController:
        """The orchestrator over every registered backend, built on first use.

        The default backend is read from ``kinematics_backend`` on every
        request, so assigning it in ``configure()`` needs no rebuild.
        """
        controller: PlannedMotionController | None = getattr(
            self, "_planned_motion_ctrl", None
        )
        if controller is None:
            controller = self._build_planned_motion()
            self._planned_motion_ctrl = controller
        return controller

    def _planned_motion_or_none(self) -> PlannedMotionController | None:
        """The built controller, without building one (stop paths, status)."""
        return getattr(self, "_planned_motion_ctrl", None)

    def _motion_status_stream(self) -> MotionStatusStream:
        stream = getattr(self, "_planned_motion_stream", None)
        if stream is None:
            stream = MotionStatusStream()
            self._planned_motion_stream = stream
        return stream

    def _build_planned_motion(self) -> PlannedMotionController:
        name = self.selected_kinematics_backend
        controller = PlannedMotionController(
            backends=self._motion_backend_factories,
            default_backend=lambda: self.selected_kinematics_backend,
            current_joints=self._current_joint_positions,  # type: ignore[attr-defined]
            preconditions=self._motion_preconditions,
            config=self._planned_motion_config(),
            frames=self._motion_frame_source,
            stream=self._motion_status_stream(),
            wrist_payload_present=self._motion_wrist_payload_present,
            refresh_status=self._refresh_planned_motion_status,
            on_failed=self._report_planned_motion_failure,
            log=self._planned_motion_info,
        )
        self._planned_motion_info(
            f"planned motion: kinematics backend {name} "
            f"({'available' if controller.available else 'not available yet'}); "
            f"registered: {', '.join(controller.registered_backends)}"
        )
        return controller

    def _motion_backend_factories(self) -> dict[str, Callable[[], MotionBackend]]:
        """Backend name -> zero-argument factory building that backend.

        The registry ``kinematics_backend`` is resolved against. Default: the
        built-in in-process backend only. A driver with another planner
        extends it rather than replacing it, so ``"pinocchio"`` stays
        selectable::

            def _motion_backend_factories(self):
                return {
                    **super()._motion_backend_factories(),
                    "sampler": self._build_sampler_backend,
                }

        A factory is called once per (re)build, on first use, so it may read
        configuration ``configure()`` loaded. It should not raise for a
        missing runtime dependency: build a backend that reports
        ``available() == False`` and says why, so the driver still starts and
        refuses planned motion with ``backend_unavailable``.
        """
        return {PINOCCHIO_BACKEND: self._build_pinocchio_motion_backend}

    def _build_pinocchio_motion_backend(self) -> MotionBackend:
        """The built-in in-process backend (IK + collision checks on the URDF)."""
        return PinocchioMotionBackend(
            planner=self._local_cartesian_planner,
            current_joints=self._current_joint_positions,  # type: ignore[attr-defined]
            execute=self._motion_execute,
            cancel_execution=self._motion_cancel_execution,
            rebuild_scene=self._rebuild_planned_motion_scene,
            planning_frame=self._motion_planning_frame(),
            executor_name=type(self).PLANNED_MOTION_TRAJECTORY_ACTION,
            unavailable_detail=self._local_planner_unavailable_detail,
            on_candidate_refused=lambda detail: self._on_grasp_candidate_refused(
                "grasp", detail
            ),
        )

    # ── seams: task level ────────────────────────────────────────────────────

    def _motion_preconditions(self) -> Any:
        """The gate matrix every move must clear (``Preconditions``-shaped).

        Default: every gate open. A hardware driver overrides this with its
        fault latch, control mode, firmware check and -- most importantly --
        ``measured_state_seen``, so nothing is planned from an assumed posture.
        """
        return Preconditions()

    def _planned_motion_config(self) -> PlannedMotionConfig:
        """Task tolerances. Default: the grasp approach axis is the arm's
        ``tool_axes["forward"]`` when it declares one."""
        return PlannedMotionConfig(tool_approach_axis=self._motion_tool_approach_axis())

    def _motion_tool_approach_axis(self) -> tuple[float, float, float] | None:
        try:
            config = getattr(self, "arm_config", lambda: None)()
        except Exception:  # noqa: BLE001 - no arm config means no default axis
            return None
        axis = (getattr(config, "tool_axes", None) or {}).get("forward")
        if axis is None:
            return None
        values = tuple(float(v) for v in axis)
        return values if len(values) == 3 else None  # type: ignore[return-value]

    def _motion_frame_source(self) -> FrameSource | None:
        """The arm's frames at its current posture, for resolving targets.

        Default: forward kinematics of the arm controller -- the TCP, and any
        frame of the robot description through ``kinematics.frame_pose``.
        ``None`` (no arm controller) accepts only absolute planning-frame
        targets.
        """
        arm = self._motion_arm_controller()
        if arm is None:
            return None
        joints = dict(self._current_joint_positions())  # type: ignore[attr-defined]
        kinematics = arm.kinematics
        start = {name: joints.get(name, 0.0) for name in arm.config.arm_joints}
        return FrameSource(
            planning_frame=self._motion_planning_frame(),
            tcp=arm.ee_pose(joints),
            lookup=lambda name: kinematics.frame_pose(start, name),
        )

    def _motion_planning_frame(self) -> str:
        """The one frame scenes and targets are expressed in, as the catalog says.

        Default: the arm's ``base_frame`` (``base_link`` unless configured).
        The in-process backend plans in it; a driver whose registered backend
        plans in another frame overrides this to return that backend's
        ``planning_frame``, so the catalog describes the frame it plans in.
        """
        try:
            config = getattr(self, "arm_config", lambda: None)()
        except Exception:  # noqa: BLE001 - manifest export must not fail here
            return PLANNING_FRAME
        return getattr(config, "base_frame", None) or PLANNING_FRAME

    def _motion_wrist_payload_present(self) -> bool:
        """Is a wrist camera/bracket in the collision model? Reported, not detected."""
        return False

    def _refresh_planned_motion_status(self, status: MotionStatus) -> None:
        """Fill driver-level status fields right before ``state/motion`` leaves."""
        del status

    # ── seams: in-process planning (pinocchio) ───────────────────────────────

    def _motion_arm_controller(self) -> Any:
        """The ``ArmController`` the in-process planner solves with, or ``None``."""
        try:
            return getattr(self, "_arm", None)
        except Exception:  # noqa: BLE001 - arm_config() None before configure
            return None

    def _planned_motion_kinematics_kwargs(self) -> dict[str, Any]:
        """Extra ``CollisionAwareKinematics`` kwargs: ``srdf_path``, floor, overrides."""
        return {}

    def _build_arm_kinematics(
        self,
        config: Any,
        *,
        obstacles: tuple[Primitive, ...] | None = None,
        attached_payload: Primitive | None = None,
    ) -> Any:
        """Collision-aware kinematics carrying the world and the payload.

        Overrides the ``ArmCapabilityMixin`` default (bare kinematics): planned
        motion needs a collision model. A driver overriding this keeps the two
        keyword arguments -- the scene rebuild passes them.
        """
        from ..kinematics.arm.collision import CollisionAwareKinematics

        kwargs = dict(self._planned_motion_kinematics_kwargs())
        if obstacles is not None:
            kwargs["obstacles"] = tuple(obstacles)
        if attached_payload is not None:
            kwargs["attached_payload"] = attached_payload
        return CollisionAwareKinematics(config, **kwargs)

    def _motion_velocity_limits(self) -> Mapping[str, float] | None:
        """Per-joint velocity limits (rad/s) timing in-process trajectories.

        Default: the URDF's, for every arm joint; ``None`` when any is missing
        (an unlimited joint cannot be timed honestly).
        """
        arm = self._motion_arm_controller()
        if arm is None:
            return None
        limits = arm.kinematics.velocity_limits()
        joints = tuple(arm.config.arm_joints)
        if not all(joint in limits for joint in joints):
            return None
        return {joint: float(limits[joint]) for joint in joints}

    def _motion_max_joint_acceleration(self) -> float | None:
        return type(self).PLANNED_MOTION_MAX_JOINT_ACCELERATION_RAD_S2

    def _motion_local_planner_kwargs(self, arm: Any) -> dict[str, Any]:
        """Extra ``LocalCartesianPlanner`` kwargs.

        Default: ``path_ik_kwargs={"restart": False}`` when the solver supports
        it, so a path solve never silently swaps IK branch mid-line.
        """
        try:
            parameters = inspect.signature(arm.kinematics.ik).parameters
        except (TypeError, ValueError):
            return {}
        return {"path_ik_kwargs": {"restart": False}} if "restart" in parameters else {}

    def _local_cartesian_planner(self) -> LocalCartesianPlanner | None:
        """The in-process planner the pinocchio backend plans with."""
        planner = getattr(self, "_motion_local_planner_obj", None)
        if planner is not None:
            return planner
        arm = self._motion_arm_controller()
        if arm is None:
            return None
        limits = self._motion_velocity_limits()
        acceleration = self._motion_max_joint_acceleration()
        if not limits or acceleration is None:
            return None
        planner = LocalCartesianPlanner(
            controller=arm,
            current_joints=self._current_joint_positions,  # type: ignore[attr-defined]
            send_trajectory=self._motion_send_trajectory,
            velocity_limits=limits,
            max_joint_acceleration=float(acceleration),
            arm_joints=tuple(arm.config.arm_joints),
            **self._motion_local_planner_kwargs(arm),
        )
        self._motion_local_planner_obj = planner
        return planner

    def _local_planner_unavailable_detail(self) -> str:
        if self._motion_arm_controller() is None:
            return "no arm controller is composed (arm_config() returned None)"
        if self._motion_max_joint_acceleration() is None:
            return (
                "PLANNED_MOTION_MAX_JOINT_ACCELERATION_RAD_S2 is not set; the "
                "in-process planner will not time a trajectory against a made-up "
                "acceleration bound"
            )
        try:
            limits = self._motion_velocity_limits()
        except Exception as exc:  # noqa: BLE001 - the reason IS the detail
            return (
                f"the arm kinematics could not be built ({type(exc).__name__}: {exc})"
            )
        if not limits:
            return (
                "some arm joint has no velocity limit; override "
                "_motion_velocity_limits() with the arm's physical limits"
            )
        return "the in-process IK planner is unavailable"

    def _local_planner_or_none(self) -> LocalCartesianPlanner | None:
        try:
            return self._local_cartesian_planner()
        except Exception as exc:  # noqa: BLE001 - surfaced as unavailable
            self._planned_motion_warning(f"in-process planner unavailable: {exc}")
            return None

    def _on_grasp_candidate_refused(self, command: str, detail: str) -> None:
        """One ranked grasp candidate was refused (not the terminal result)."""
        self._planned_motion_warning(f"{command}: planning_failed: {detail}")

    # ── seams: execution ─────────────────────────────────────────────────────

    def _motion_namespaced(self, name: str) -> str:
        namespaced = getattr(self, "namespaced", None)
        if callable(namespaced):
            return str(namespaced(name))
        return "/" + name.lstrip("/")

    def _motion_trajectory_executor(self) -> Any:
        """``send(trajectory, on_active, on_done) -> bool`` and ``cancel()``.

        Default: one SDK ``TrajectoryExecutor`` on
        ``PLANNED_MOTION_TRAJECTORY_ACTION`` under the driver's namespace, with
        the start-state check at ``PlannedMotionConfig.start_state_tolerance_rad``.
        """
        executor = getattr(self, "_motion_trajectory_executor_obj", None)
        if executor is None:
            from ..ros2.trajectory_executor import TrajectoryExecutor

            arm = self._motion_arm_controller()
            if arm is None:
                raise RuntimeError("no arm controller to name the trajectory joints")
            executor = TrajectoryExecutor(
                self,
                self._motion_namespaced(type(self).PLANNED_MOTION_TRAJECTORY_ACTION),
                tuple(arm.config.arm_joints),
                current_positions=self._current_joint_positions,  # type: ignore[attr-defined]
                start_state_tolerance_rad=(
                    self._planned_motion_config().start_state_tolerance_rad
                ),
            )
            self._motion_trajectory_executor_obj = executor
        return executor

    def _motion_execute(
        self, trajectory: LocalTrajectory, on_active: Any, on_done: Any
    ) -> bool:
        try:
            return bool(
                self._motion_trajectory_executor().send(trajectory, on_active, on_done)
            )
        except Exception as exc:  # noqa: BLE001 - "not accepted" is the honest answer
            self._planned_motion_warning(f"trajectory submission failed: {exc}")
            return False

    def _motion_send_trajectory(self, trajectory: LocalTrajectory) -> bool:
        def on_done(outcome: str, detail: str) -> None:
            if outcome == "failed":
                self._report_planned_motion_failure(
                    "trajectory", f"local Cartesian execution: {detail}"
                )

        return self._motion_execute(trajectory, lambda: None, on_done)

    def _motion_cancel_execution(self) -> None:
        executor = getattr(self, "_motion_trajectory_executor_obj", None)
        if executor is None:
            try:
                executor = self._motion_trajectory_executor()
            except Exception:  # noqa: BLE001 - nothing was ever sent
                return
        executor.cancel()

    # ── scene rebuild (pinocchio) ────────────────────────────────────────────

    def _planned_motion_task_scene(
        self,
    ) -> tuple[tuple[Primitive, ...], Primitive | None]:
        controller = self._planned_motion_or_none()
        backend = (
            None if controller is None else controller.built_backend(PINOCCHIO_BACKEND)
        )
        if isinstance(backend, PinocchioMotionBackend):
            return backend.scene_objects, backend.payload
        return (), None

    def _rebuild_planned_motion_scene(
        self, objects: tuple[Primitive, ...], payload: Primitive | None
    ) -> None:
        """Install a collision model = standing world + task scene + payload."""
        if getattr(self, "world_obstacles_updating", False):
            raise ValueError("a world obstacle update is in progress")
        current_world = getattr(self, "current_world_obstacles", None)
        world = tuple(current_world()) if callable(current_world) else ()
        candidate = self._build_planned_motion_kinematics(
            world + tuple(objects), payload
        )
        install = getattr(self, "_install_world_obstacle_kinematics", None)
        if callable(install):
            install(candidate)
        else:
            self._motion_arm_controller().replace_kinematics(candidate)

    def _build_planned_motion_kinematics(
        self, obstacles: tuple[Primitive, ...], payload: Primitive | None
    ) -> Any:
        arm = self._motion_arm_controller()
        if arm is None:
            raise ValueError("the kinematics solver is not configured")
        kwargs: dict[str, Any] = {"obstacles": tuple(obstacles)}
        if payload is not None:
            kwargs["attached_payload"] = payload
        candidate = self._build_arm_kinematics(arm.config, **kwargs)
        validate_kinematics_candidate(arm.loaded_kinematics, candidate, obstacles)
        if (
            payload is not None
            and getattr(candidate, "attached_payload_id", None) != payload.id
        ):
            raise ValueError(
                "the kinematics builder did not carry the payload; accept "
                "attached_payload= in _build_arm_kinematics"
            )
        return candidate

    def _build_world_obstacle_kinematics(self, obstacles: tuple[Primitive, ...]) -> Any:
        """``add_obstacles`` keeps the task scene and the carried payload."""
        objects, payload = self._planned_motion_task_scene()
        if not objects and payload is None:
            return super()._build_world_obstacle_kinematics(obstacles)  # type: ignore[misc]
        return self._build_planned_motion_kinematics(
            tuple(obstacles) + objects, payload
        )

    def _world_obstacles_busy_detail(self) -> str | None:
        controller = self._planned_motion_or_none()
        detail = None if controller is None else controller.busy_detail()
        if detail is not None:
            return detail
        parent = getattr(super(), "_world_obstacles_busy_detail", None)
        return parent() if callable(parent) else None

    # ── registration ─────────────────────────────────────────────────────────

    def _ee_pose_registered_elsewhere(self) -> bool:
        return (
            isinstance(self, EECartesianPoseMixin)
            and bool(getattr(self, "use_cartesian_pose", False))
            and not bool(getattr(self, "use_dual_arm", False))
        )

    def define_interface(self, iface: Any) -> None:
        super().define_interface(iface)  # type: ignore[misc]
        cmd_topic = TopicSpec(
            namespace="twin",
            leaf="command",
            payload_schema_ref="TwinCommandPayload",
            description="Planned arm motion commands",
        )
        if self.use_planned_motion_commands:
            table = planned_motion_command_args(self._motion_planning_frame())
            for name in MOTION_COMMANDS:
                iface.add_listener(
                    cmd_topic,
                    CallbackGroup(callback=self._on_planned_motion_command),
                    command=CommandArgs(
                        name=name,
                        args=table[name],
                        description=PLANNED_MOTION_DESCRIPTIONS[name],
                    ),
                    operation_modes=TELEOP_MODES,
                )
        if self.use_planned_ee_pose and not self._ee_pose_registered_elsewhere():
            iface.add_listener(
                cmd_topic,
                CallbackGroup(callback=self._on_planned_ee_pose),
                command=CommandArgs(
                    name="ee_pose",
                    args=EE_POSE_ARGS,
                    description=PLANNED_EE_POSE_DESCRIPTION,
                ),
                operation_modes=TELEOP_MODES,
            )
        if self.use_motion_state_publisher:
            iface.add_publisher(
                TopicSpec(
                    namespace="state",
                    leaf="motion",
                    payload_schema_ref="TwinMotionStatusPayload",
                    description=(
                        "Planned-motion status: the default kinematics backend and "
                        "its availability, the active scene revision, any attached "
                        "payload, motion_state (idle|planning|executing|failed|"
                        "cancelled), the command and backend of the current motion, "
                        "the planner and timing of the last plan, and last_result / "
                        "last_refusal naming their command. Published on every "
                        "transition, at 2 Hz while a motion is active and every 5 s "
                        "while idle."
                    ),
                ),
                # Returns None when nothing is due; the tick publisher skips it.
                CallbackGroup(callback=self._planned_motion_state_payload),  # type: ignore[arg-type]
                publisher=PublisherArgs(rate_hz=type(self).MOTION_STATE_RATE_HZ),
                operation_modes=_ALL_OPERATION_MODES,
            )

    # ── dispatch ─────────────────────────────────────────────────────────────

    def _on_planned_motion_command(self, payload: dict[str, Any]) -> CommandOutcome:
        """Run one command; a requester that asked for a reply gets the verdict.

        ``accepted`` once the command is underway (its result follows on
        ``state/motion``), ``rejected`` with the refusal line otherwise.
        """
        command = str(payload.get("command") or "")
        data = payload.get("data")
        try:
            # Refusals are reported inside handle_command (status + failure hook).
            refusal = self.planned_motion.handle_command(
                command, {} if data is None else data
            )
        except Exception as exc:  # noqa: BLE001 - dispatch must never crash
            detail = f"{type(exc).__name__}: {exc}"
            self._report_planned_motion_failure(command, detail)
            return CommandOutcome.rejected(detail)
        if refusal is not None:
            return CommandOutcome.rejected(refusal.status_line())
        return CommandOutcome.accept()

    def _on_planned_ee_pose(self, payload: dict[str, Any]) -> None:
        try:
            pose = EEPose.from_payload(payload.get("data") or {})
        except (TypeError, ValueError) as exc:
            self._report_planned_motion_failure("ee_pose", str(exc))
            return
        self.submit_planned_ee_pose(pose)

    def _on_arm_command(self, payload: dict[str, Any]) -> None:
        """A Cartesian jog takes the arm from a running planned motion.

        Never both at once: the planned goal is cancelled, and the jog waits for
        it to end, then solves from the measured state. Every arm composing this
        mixin gets the handoff; ``grip`` / ``release`` / ``home`` pass straight
        through.
        """
        parent = super()._on_arm_command  # type: ignore[misc]
        if is_cartesian_jog(str(payload.get("command"))) and (
            self.replace_planned_motion(lambda: parent(payload))
        ):
            return
        parent(payload)

    def _dispatch_ee_pose(
        self, pose: EEPose, *, controller: Any, arm: str | None
    ) -> None:
        """The ``EECartesianPoseMixin`` registration, routed to the planner."""
        if self.use_planned_ee_pose and arm is None:
            self.submit_planned_ee_pose(pose)
            return
        super()._dispatch_ee_pose(pose, controller=controller, arm=arm)  # type: ignore[misc]

    def submit_planned_ee_pose(
        self, pose: Any, *, source: str = "ee_pose"
    ) -> CartesianDecision:
        """Plan an absolute TCP pose as ``set_pose`` with a PTP motion.

        The controller runs the same gates as ``set_pose`` and reports any
        refusal itself, on ``state/motion`` and through the failure hook, under
        *source*. Planning/execution results arrive on ``state/motion``.
        """
        try:
            target = tuple(
                float(getattr(pose, axis))
                for axis in ("x", "y", "z", "roll", "pitch", "yaw")
            )
            return self.planned_motion.plan_and_execute(
                CartesianRequest(target=target, source=source)  # type: ignore[arg-type]
            )
        except Exception as exc:  # noqa: BLE001 - a crash is a refusal
            decision = CartesianDecision(
                rejection=CartesianRejection.PLANNING_FAILED,
                detail=f"{type(exc).__name__}: {exc}",
            )
            self._report_planned_motion_failure(source, decision.status())
            return decision

    def _planned_motion_state_payload(self) -> dict[str, Any] | None:
        """``state/motion`` tick: edge-triggered, with heartbeats.

        Publishes even when the backend is unavailable: going silent then would
        make "the planner is down" indistinguishable from "this driver does not
        publish the stream".
        """
        try:
            return self.planned_motion.status_payload_if_due(time.monotonic())
        except Exception as exc:  # noqa: BLE001 - the tick must never raise
            self._planned_motion_warning(f"state/motion unavailable: {exc}")
            return None

    # ── stop ─────────────────────────────────────────────────────────────────

    def cancel_planned_motion(self) -> None:
        """Cancel the active planned motion, if any."""
        controller = self._planned_motion_or_none()
        if controller is not None:
            controller.cancel()

    def replace_planned_motion(self, replacement: Any) -> bool:
        """Hand the arm from the planned motion to driver-owned motion (a jog).

        ``True``: a planned motion was in flight; it is cancelled and
        *replacement* (no arguments) runs once it has ended. ``False``: nothing
        to wait for; run the work now.
        See :meth:`PlannedMotionController.replace_after_cancel`.
        """
        controller = self._planned_motion_or_none()
        return controller is not None and controller.replace_after_cancel(replacement)

    def _stop_motion(self) -> None:
        # Planner first: cancelling its goal is what cancels the controller goal
        # beneath it; dropping queued joint targets comes after.
        self.cancel_planned_motion()
        parent = getattr(super(), "_stop_motion", None)
        if callable(parent):
            parent()

    async def _on_stop_cmd(self, envelope: dict[str, Any]) -> None:
        if envelope.get("command") != "status":
            self.cancel_planned_motion()
        parent = getattr(super(), "_on_stop_cmd", None)
        if callable(parent):
            await parent(envelope)

    # ── reporting ────────────────────────────────────────────────────────────

    def refuse_planned_motion(self, command: str, refusal: Refusal) -> None:
        """Report a driver-decided refusal on ``state/motion`` and the failure hook."""
        self.planned_motion.refuse(command, refusal)

    def _report_planned_motion_failure(self, command: str | None, detail: str) -> None:
        on_failed = getattr(self, "_on_arm_command_failed", None)
        if callable(on_failed):
            on_failed(command, detail)
        else:
            self._planned_motion_warning(f"{command}: {detail}")

    def _planned_motion_logger(self) -> Any:
        return driver_logger(self, logger)

    def _planned_motion_info(self, message: str) -> None:
        self._planned_motion_logger().info(message)

    def _planned_motion_warning(self, message: str) -> None:
        self._planned_motion_logger().warning(message)


__all__ = [
    "PLANNED_EE_POSE_DESCRIPTION",
    "PLANNED_MOTION_DESCRIPTIONS",
    "PlannedMotionMixin",
    "planned_motion_command_args",
]
