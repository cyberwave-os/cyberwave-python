"""One planned-motion layer for every arm driver, whichever planner answers.

:class:`PlannedMotionController` owns everything a caller can observe about a
planned motion -- so it is identical whichever
:class:`~cyberwave.driver.motion.backend.MotionBackend` answers (the built-in
in-process pinocchio one, or one a driver registers):

* the commands -- ``set_pose``, ``grasp``, ``scene_set``, ``attach_payload``,
  ``detach_payload`` -- and the absolute ``ee_pose`` they subsume;
* which backend answers: the driver's default, or the one a request names in
  ``backend``;
* the ONE shared ``state/motion`` status (:class:`MotionStatusStream`);
* the guards and their ORDER (below);
* the active scene revision and the attached payload. Every backend that has
  been used holds the same scene: a scene or payload change reaches all of
  them (or none), and a backend used for the first time is given the current
  scene and payload before its first goal;
* where a target is: relative targets and reference frames are resolved into
  the planning frame here (:mod:`~cyberwave.driver.motion.targets`), so a
  backend only ever sees planning-frame poses;
* one motion at a time: a new move supersedes the active one, which is
  cancelled first; the newest replacement starts when the old one reports
  its terminal result, and a manual stop drops any pending replacement.

Guard order (each command stops at its first refusal; the order is part of the
contract -- a stale robot state must not be reported when the arm was faulted,
or an operator chases the wrong problem):

1. the command is known and its ``data`` is an object;
2. the backend is registered and available (``backend_unavailable``);
3. moves only -- ``plan_only`` is a boolean, then
   :func:`~cyberwave.driver.motion.guards.check_motion_preconditions`
   (faults, control mode, hardware, geometry, collision model, measured state;
   a probe skips the ones that only prevent motion);
4. scene and payload edits only -- busy: refused while a motion is planning or
   executing (moves never refuse busy: they supersede);
5. moves only -- the scene revision, then the start state
   (``expected_joints`` vs measured joints), each only when the request sends
   one;
6. the request itself (target, candidates, scene/attach payload), then the
   backend's own goal check.

A refusal is reported three ways at once: returned, written to
``last_refusal`` with the command name (``motion_state`` is unchanged -- the
refused command never started), and passed to ``on_failed`` for the driver's
alerting. Asynchronous failures of an accepted motion use the same two sinks.

Pure Python: no ROS, no pinocchio.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..control.cartesian_request import (
    CartesianDecision,
    CartesianRejection,
    CartesianRequest,
)
from .backend import (
    GraspGoal,
    MotionBackend,
    MotionGoal,
    MotionOutcome,
    MotionResult,
    MotionType,
    MotionUpdate,
)
from .guards import (
    MotionRefusal,
    Refusal,
    check_motion_preconditions,
    check_not_busy,
    check_revision,
    check_start_state,
)
from .requests import parse_grasp
from .scene import (
    DEFAULT_PAYLOAD_ID,
    AttachRequest,
    SceneError,
    SceneRequest,
    SceneTracker,
    parse_attach,
    parse_scene_set,
)
from .status import MotionState, MotionStatus, MotionStatusStream, planner_name
from .targets import (
    FrameSource,
    TargetMode,
    needs_frames,
    parse_pose_spec,
    reference_frame,
    resolve_target,
)

SET_POSE = "set_pose"
GRASP = "grasp"
SCENE_SET = "scene_set"
ATTACH_PAYLOAD = "attach_payload"
DETACH_PAYLOAD = "detach_payload"

#: The planned-motion commands, in catalog order.
MOTION_COMMANDS: tuple[str, ...] = (
    SET_POSE,
    GRASP,
    SCENE_SET,
    ATTACH_PAYLOAD,
    DETACH_PAYLOAD,
)
_MOVES = frozenset({SET_POSE, GRASP})

BackendSource = MotionBackend | Callable[[], MotionBackend]


@dataclass(frozen=True)
class PlannedMotionConfig:
    """Task-level settings shared by every backend."""

    #: How far a measured joint may be from the requester's ``expected_joints``
    #: before the request is refused as ``stale_robot_state`` (rad).
    start_state_tolerance_rad: float = 0.02
    #: Planning-frame direction of the lift a ``grasp`` checks after the grasp.
    grasp_lift_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    #: Tool-frame axis a ``grasp`` candidate without ``approach`` travels
    #: along; ``None`` makes ``approach`` required.
    tool_approach_axis: tuple[float, float, float] | None = None


@dataclass
class _Slot:
    """A backend that has been used, and what it was told about the scene."""

    backend: MotionBackend
    tracker: SceneTracker


@dataclass
class _Motion:
    """The single in-flight motion."""

    command: str
    backend: str
    goal: MotionGoal | GraspGoal
    #: Numbered by the driver when the motion starts (see ``MotionStatus``).
    motion_id: int = 0


def _parse_motion_type(value: object) -> MotionType:
    if value is None:
        return MotionType.PTP
    for member in MotionType:
        if member.value == value:
            return member
    raise SceneError(
        MotionRefusal.INVALID_REQUEST,
        f"motion must be one of {sorted(m.value for m in MotionType)}, got {value!r}",
    )


def _as_backend(source: BackendSource) -> MotionBackend:
    if isinstance(source, MotionBackend):
        return source
    return source()


class PlannedMotionController:
    """Guards, status, the shared scene and one-motion-at-a-time over the backends.

    ``backends`` maps every registered name to a backend or a zero-argument
    factory (or is a zero-argument callable returning that mapping, read on
    each lookup); a backend is built the first time a request needs it.
    ``default_backend`` names the one that answers requests naming none (or
    is a zero-argument callable returning that name). ``frames`` returns the
    arm's :class:`~cyberwave.driver.motion.targets.FrameSource` at its
    current posture, or ``None`` when the driver has no forward kinematics
    (then only absolute planning-frame poses are accepted).

    Satisfies :class:`~cyberwave.driver.control.cartesian_request.CartesianPlanner`
    (``available`` / ``plan_and_execute``), so an absolute ``ee_pose`` is the
    same physical act as ``set_pose`` and reports on the same stream.

    ``preconditions`` returns a
    :class:`~cyberwave.driver.control.cartesian_request.Preconditions` (or
    anything with the same attributes), read fresh per move; ``None`` means "no
    opinion". ``refresh_status`` fills driver-level status fields right before a
    payload leaves.
    """

    COMMANDS: tuple[str, ...] = MOTION_COMMANDS

    def __init__(
        self,
        *,
        backends: Mapping[str, BackendSource]
        | Callable[[], Mapping[str, BackendSource]],
        default_backend: str | Callable[[], str],
        current_joints: Callable[[], Mapping[str, float]],
        preconditions: Callable[[], Any] | None = None,
        config: PlannedMotionConfig | None = None,
        frames: Callable[[], FrameSource | None] | None = None,
        stream: MotionStatusStream | None = None,
        wrist_payload_present: bool | Callable[[], bool] = False,
        refresh_status: Callable[[MotionStatus], None] | None = None,
        on_failed: Callable[[str, str], None] = lambda _command, _detail: None,
        log: Callable[[str], None] = lambda _message: None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._backends = backends
        self._clock = clock
        self._last_stamp_ms = 0
        self._motions_started = 0
        self._default_backend = default_backend
        self._current_joints = current_joints
        self._preconditions = preconditions
        self._config = config or PlannedMotionConfig()
        self._frames = frames
        self._stream = MotionStatusStream() if stream is None else stream
        self._status = self._stream.status
        self._wrist_payload_present = wrist_payload_present
        self._refresh_status = refresh_status
        self._on_failed = on_failed
        self._log = log
        self._slots: dict[str, _Slot] = {}
        #: The planning frame, fixed by the first backend built.
        self._frame: str | None = None
        #: The scene every used backend holds (the attached object removed),
        #: and the carried payload: what a newly used backend is given.
        self._scene: SceneRequest | None = None
        self._attached: AttachRequest | None = None
        self._scenes_set = 0
        self._motion: _Motion | None = None
        self._replacement: Callable[[], object] | None = None
        self._cancel_requested = False
        #: Commands arrive on the driver's loop thread, backend results on the
        #: ROS executor's. Without one lock, a pose sent as the running motion
        #: finishes could be stored as the replacement of a motion that just
        #: ended -- accepted, and never run. Reentrant: a backend may answer
        #: synchronously from inside plan_and_execute, and a replacement runs
        #: from inside _on_done.
        self._lock = threading.RLock()

    # ── backends ─────────────────────────────────────────────────────────────

    def _registry(self) -> Mapping[str, BackendSource]:
        backends = self._backends
        return backends() if callable(backends) else backends

    @property
    def backend_name(self) -> str:
        """The default backend's name, as ``kinematics_backend`` reports it."""
        name = self._default_backend
        return str(name() if callable(name) else name)

    @property
    def registered_backends(self) -> tuple[str, ...]:
        return tuple(sorted(self._registry()))

    def built_backend(self, name: str) -> MotionBackend | None:
        """The backend registered as *name* if it has been used, else ``None``."""
        slot = self._slots.get(name)
        return None if slot is None else slot.backend

    @property
    def backend(self) -> MotionBackend:
        """The default backend (built on first use)."""
        slot, refusal = self._slot(self.backend_name)
        if slot is None:
            raise ValueError(refusal.detail if refusal else "no default backend")
        return slot.backend

    def _slot(self, name: str) -> tuple[_Slot | None, Refusal | None]:
        """The backend *name*, built and given the current scene on first use."""
        slot = self._slots.get(name)
        if slot is not None:
            return slot, None
        registry = self._registry()
        source = registry.get(name)
        if source is None:
            return None, Refusal(
                MotionRefusal.INVALID_REQUEST,
                f"backend {name!r} is not registered on this driver; "
                f"registered: {sorted(registry)}",
            )
        backend = _as_backend(source)
        frame = backend.planning_frame
        if self._frame is None:
            self._frame = frame
        elif frame != self._frame:
            return None, Refusal(
                MotionRefusal.INVALID_REQUEST,
                f"backend {name!r} plans in {frame!r}, but this driver's planning "
                f"frame is {self._frame!r}",
            )
        slot = _Slot(backend, SceneTracker(frame))
        # A backend used for the first time must hold the world the others
        # hold before it plans anything: otherwise the same scene_revision
        # would mean two different worlds depending on which backend answers.
        if self._scene is not None:
            refusal = backend.apply_scene(
                self._scene, slot.tracker.plan_update(self._scene)
            )
            if refusal is not None:
                return None, refusal
            slot.tracker.commit_update(self._scene)
        if self._attached is not None:
            refusal = backend.attach(self._attached)
            if refusal is not None:
                return None, refusal
            slot.tracker.commit_attach(self._attached)
        self._slots[name] = slot
        return slot, None

    def _available_slot(self, name: str) -> tuple[_Slot | None, Refusal | None]:
        slot, refusal = self._slot(name)
        if slot is None:
            return None, refusal
        if not slot.backend.available():
            return None, slot.backend.unavailable_refusal()
        return slot, None

    def _requested_backend(self, data: dict[str, Any]) -> str:
        name = data.get("backend")
        if name is None:
            return self.backend_name
        if not isinstance(name, str) or not name.strip():
            raise SceneError(
                MotionRefusal.INVALID_REQUEST, "backend must be a registered name"
            )
        return name.strip().lower()

    # ── identity ─────────────────────────────────────────────────────────────

    @property
    def stream(self) -> MotionStatusStream:
        return self._stream

    @property
    def config(self) -> PlannedMotionConfig:
        return self._config

    @property
    def planning_frame(self) -> str:
        if self._frame is None:
            return self.backend.planning_frame
        return self._frame

    @property
    def active_revision(self) -> str | None:
        return None if self._scene is None else self._scene.revision

    @property
    def attached_payload(self) -> str | None:
        return None if self._attached is None else self._attached.object_id

    @property
    def available(self) -> bool:
        """The default backend can accept a goal now."""
        try:
            return bool(self.backend.available())
        except Exception:  # noqa: BLE001 - status must never be the failure
            return False

    @property
    def active(self) -> bool:
        """A motion is in flight."""
        return self._motion is not None

    def busy_detail(self) -> str | None:
        """Why a collision-model change must wait, or ``None``."""
        if not self.active:
            return None
        state = self._status.motion_state
        if state in (MotionState.PLANNING, MotionState.EXECUTING):
            return f"a planned motion is {state.value}"
        return "a planned motion is in flight"

    # ── status ───────────────────────────────────────────────────────────────

    def _refresh(self, status: MotionStatus) -> None:
        try:
            name = self.backend_name
        except Exception:  # noqa: BLE001 - status must never be the failure
            name = None
        status.kinematics_backend = name
        status.backend_available = self.available
        present = self._wrist_payload_present
        status.wrist_payload_present = bool(present() if callable(present) else present)
        status.active_scene_revision = self.active_revision
        status.attached_payload = self.attached_payload
        if self._refresh_status is not None:
            self._refresh_status(status)

    def status(self) -> MotionStatus:
        with self._lock:
            self._refresh(self._status)
            return self._status

    def status_payload_if_due(self, now: float) -> dict[str, Any] | None:
        """Edge-triggered, plus a heartbeat. ``None`` = nothing new to say."""
        with self._lock:
            self._refresh(self._status)
            return self._stream.payload_if_due(now)

    # ── CartesianPlanner: absolute ee_pose ───────────────────────────────────

    def plan_and_execute(self, request: CartesianRequest) -> CartesianDecision:
        """Absolute ``ee_pose``: literally ``set_pose`` with a PTP motion.

        The same path, gates and refusal stream as ``set_pose``; only the label
        on ``state/motion`` keeps the name the caller sent. The decision is
        "accepted" once submitted; planning and execution results arrive on
        ``state/motion`` and through ``on_failed``.
        """
        refusal = self.handle_command(
            SET_POSE,
            {"pose": list(request.target), "motion": "ptp"},
            label=request.source,
        )
        if refusal is None:
            return CartesianDecision(request=request)
        return CartesianDecision(
            rejection=_cartesian_rejection(refusal.code), detail=refusal.detail
        )

    # ── commands ─────────────────────────────────────────────────────────────

    def handle_command(
        self, command: str, data: Any, *, label: str | None = None
    ) -> Refusal | None:
        """Run one command. Returns the refusal (also reported), or ``None``.

        ``label`` is the name reported on ``state/motion`` when it differs from
        the command run (an ``ee_pose`` alias runs ``set_pose``).
        """
        label = label or command
        with self._lock:
            try:
                refusal = self._dispatch(command, {} if data is None else data, label)
            except SceneError as exc:
                refusal = exc.refusal
            except (TypeError, ValueError) as exc:
                refusal = Refusal(MotionRefusal.INVALID_REQUEST, str(exc))
            if refusal is not None:
                self._report_refusal(label, refusal)
            return refusal

    def refuse(self, command: str, refusal: Refusal) -> None:
        """Publish a refusal decided outside the controller on the shared stream."""
        with self._lock:
            self._report_refusal(command, refusal)

    def _dispatch(self, command: str, data: Any, label: str) -> Refusal | None:
        if command not in self.COMMANDS:
            return Refusal(
                MotionRefusal.INVALID_REQUEST, f"unknown motion command {command!r}"
            )
        if not isinstance(data, dict):
            return Refusal(
                MotionRefusal.INVALID_REQUEST, f"{command} data must be an object"
            )
        if command in _MOVES:
            return self._move(command, data, label)
        _slot, refusal = self._available_slot(self.backend_name)
        if refusal is not None:
            return refusal
        # A scene edit or a payload change underneath a running trajectory would
        # invalidate the collision decision that authorised it.
        busy = check_not_busy(self._status.motion_state)
        if busy is None and self.active:
            busy = Refusal(MotionRefusal.BUSY, "a planned motion is in flight")
        if busy is not None:
            return busy
        if command == SCENE_SET:
            return self._scene_set(data)
        if command == ATTACH_PAYLOAD:
            return self._attach(data)
        return self._detach(data)

    def _frame_source(self) -> FrameSource | None:
        return None if self._frames is None else self._frames()

    def _move(self, command: str, data: dict[str, Any], label: str) -> Refusal | None:
        name = self._requested_backend(data)
        slot, refusal = self._available_slot(name)
        if slot is None:
            return refusal
        plan_only = data.get("plan_only", False)
        if not isinstance(plan_only, bool):
            return Refusal(
                MotionRefusal.INVALID_REQUEST, "plan_only must be true or false"
            )
        # Before the request is parsed: a refusal on policy must not depend on
        # whether the target happens to be well formed.
        if self._preconditions is not None:
            refusal = check_motion_preconditions(
                self._preconditions(), plan_only=plan_only
            )
            if refusal is not None:
                return refusal
        refusal = check_revision(data.get("scene_revision"), self.active_revision)
        if refusal is not None:
            return refusal
        expected = data.get("expected_joints")
        if expected is not None and not isinstance(expected, dict):
            return Refusal(
                MotionRefusal.INVALID_REQUEST,
                "expected_joints must be an object of joint name -> radians",
            )
        refusal = check_start_state(
            expected,
            dict(self._current_joints()),
            self._config.start_state_tolerance_rad,
        )
        if refusal is not None:
            return refusal
        goal: MotionGoal | GraspGoal
        if command == GRASP:
            goal = parse_grasp(
                data,
                frames=self._frame_source(),
                lift_direction=self._config.grasp_lift_direction,
                tool_approach_axis=self._config.tool_approach_axis,
                plan_only=plan_only,
            )
        else:
            if data.get("pose") is None:
                return Refusal(MotionRefusal.INVALID_REQUEST, "pose is required")
            spec = parse_pose_spec(data["pose"])
            mode = TargetMode.parse(data.get("mode"))
            motion_type = _parse_motion_type(data.get("motion"))
            frame = reference_frame(spec.reference_frame, data.get("reference_frame"))
            frames = self._frame_source() if needs_frames(spec, mode, frame) else None
            goal = MotionGoal(
                target=resolve_target(spec, mode=mode, frame=frame, frames=frames),
                motion_type=motion_type,
                plan_only=plan_only,
                command=label,
            )
        refusal = slot.backend.validate_goal(goal)
        if refusal is not None:
            return refusal
        self._start(_Motion(label, name, goal))
        return None

    # ── scene and payload: every used backend, or none ───────────────────────

    def _scene_set(self, data: dict[str, Any]) -> Refusal | None:
        request = parse_scene_set(
            data,
            planning_frame=self.planning_frame,
            default_revision=f"scene-{self._scenes_set + 1}",
        )
        if self._attached is not None:
            # The carried object is part of the robot now, not of the world.
            request = _without(request, self._attached.object_id)
        previous = self._scene or SceneRequest("", (), frozenset())
        applied: list[_Slot] = []
        for slot in list(self._slots.values()):
            refusal = slot.backend.apply_scene(
                request, slot.tracker.plan_update(request)
            )
            if refusal is not None:
                # The revision is NOT advanced, and the backends that took it
                # are put back: a revision active on one backend only would let
                # the next motion pass its revision guard on another backend
                # against a world that does not exist there.
                for done in applied:
                    self._restore_scene(done, previous)
                return refusal
            slot.tracker.commit_update(request)
            applied.append(slot)
        self._scene = request
        self._scenes_set += 1
        self._status.active_scene_revision = request.revision
        self._set_result(
            SCENE_SET,
            "accepted",
            scene_revision=request.revision,
            object_count=len(request.objects),
        )
        return None

    def _restore_scene(self, slot: _Slot, previous: SceneRequest) -> None:
        refusal = slot.backend.apply_scene(previous, slot.tracker.plan_update(previous))
        if refusal is None:
            slot.tracker.commit_update(previous)
            return
        self._drop_slot(slot, f"could not restore its previous scene: {refusal.detail}")

    def _drop_slot(self, slot: _Slot, why: str) -> None:
        """Forget a backend whose state is unknown; it is rebuilt on next use."""
        for name, entry in list(self._slots.items()):
            if entry is slot:
                del self._slots[name]
                self._log(f"planned motion: backend {name} {why}; rebuilt on next use")

    def _attach(self, data: dict[str, Any]) -> Refusal | None:
        request = parse_attach(data)
        if self._attached is not None:
            return Refusal(
                MotionRefusal.PAYLOAD_ALREADY_ATTACHED,
                f"{self._attached.object_id!r} is already attached; detach it first",
            )
        applied: list[_Slot] = []
        for slot in list(self._slots.values()):
            refusal = slot.backend.attach(request)
            if refusal is not None:
                for done in applied:
                    undo = done.backend.detach(request.object_id)
                    if undo is None:
                        done.tracker.commit_detach()
                    else:
                        self._drop_slot(
                            done, f"kept a payload it was told to drop: {undo.detail}"
                        )
                return refusal
            slot.tracker.commit_attach(request)
            applied.append(slot)
        self._attached = request
        if self._scene is not None:
            self._scene = _without(self._scene, request.object_id)
        self._status.attached_payload = request.object_id
        self._set_result(ATTACH_PAYLOAD, "accepted", object_id=request.object_id)
        return None

    def _detach(self, data: dict[str, Any]) -> Refusal | None:
        object_id = data.get("object_id")
        if object_id is None:
            object_id = (
                DEFAULT_PAYLOAD_ID
                if self._attached is None
                else self._attached.object_id
            )
        if self._attached is None or object_id != self._attached.object_id:
            return Refusal(
                MotionRefusal.INVALID_REQUEST,
                f"no payload {object_id!r} is attached "
                f"(attached: {self.attached_payload!r})",
            )
        carried = self._attached
        applied: list[_Slot] = []
        for slot in list(self._slots.values()):
            refusal = slot.backend.detach(object_id)
            if refusal is not None:
                for done in applied:
                    undo = done.backend.attach(carried)
                    if undo is None:
                        done.tracker.commit_attach(carried)
                    else:
                        self._drop_slot(
                            done, f"lost a payload it still carries: {undo.detail}"
                        )
                return refusal
            slot.tracker.commit_detach()
            applied.append(slot)
        self._attached = None
        self._status.attached_payload = None
        self._set_result(DETACH_PAYLOAD, "accepted", object_id=object_id)
        return None

    # ── one motion at a time ─────────────────────────────────────────────────

    def replace_after_cancel(self, replacement: Callable[[], object]) -> bool:
        """Cancel the active motion; run *replacement* after it has ended.

        For motion this controller does not plan -- a driver's Cartesian jog,
        say -- that must not reach the trajectory controller while a planned
        goal still owns it. Sending it at once would race the cancelled goal:
        the planner would report the preemption as a controller failure, and
        the replaced goal's terminal result could land on the new motion.

        Returns ``True`` when a motion was in flight: *replacement* runs once
        that motion reports its terminal result. It is the one replaceable
        intent -- a newer move, a later call or :meth:`cancel` drops it.
        Returns ``False`` when nothing is in flight: the caller runs its work
        now.
        """
        with self._lock:
            if self._motion is None:
                return False
            self._replacement = replacement
            self._request_cancel()
            return True

    def _start(self, motion: _Motion) -> None:
        if self._motion is not None:
            # One replaceable intent, never a backlog: the newest wins.
            self._replacement = lambda: self._start(motion)
            self._request_cancel()
            return
        slot = self._slots[motion.backend]
        self._motions_started += 1
        motion.motion_id = self._motions_started
        self._motion = motion
        self._cancel_requested = False
        status = self._status
        status.motion_state = MotionState.PLANNING
        status.command, status.backend = motion.command, motion.backend
        status.motion_id = motion.motion_id
        status.planner = None
        status.plan_time_ms = None
        status.trajectory_points = None
        status.trajectory_duration_s = None
        status.progress = None
        self._stream.mark_dirty()
        slot.backend.plan_and_execute(
            motion.goal,
            lambda update: self._locked(self._on_update, motion, update),
            lambda result: self._locked(self._on_done, motion, result),
        )

    def _locked(self, callback: Callable[..., None], *args: Any) -> None:
        """Run a backend callback (any thread) under the controller's lock."""
        with self._lock:
            callback(*args)

    def _on_update(self, motion: _Motion, update: MotionUpdate) -> None:
        if self._motion is not motion:
            return
        status = self._status
        self._apply_stats(update)
        # A plan-only probe never reaches a controller. Reporting EXECUTING for
        # a motion that cannot move is the one lie this stream must not tell.
        if (
            update.executing
            and not motion.goal.plan_only
            and not self._cancel_requested
            and status.motion_state is MotionState.PLANNING
        ):
            status.motion_state = MotionState.EXECUTING
        self._stream.mark_dirty()

    def _apply_stats(self, stats: MotionUpdate | MotionResult) -> None:
        status = self._status
        if stats.planner is not None:
            status.planner = planner_name(stats.planner)
        if stats.plan_time_ms is not None:
            status.plan_time_ms = stats.plan_time_ms
        if stats.trajectory_points is not None:
            status.trajectory_points = stats.trajectory_points
        if stats.trajectory_duration_s is not None:
            status.trajectory_duration_s = stats.trajectory_duration_s

    def _on_done(self, motion: _Motion, result: MotionResult) -> None:
        if self._motion is not motion:
            # A result for a motion already terminated: acting on it would
            # resurrect state for a motion nobody is waiting on.
            return
        self._motion = None
        status = self._status
        self._apply_stats(result)
        if self._cancel_requested:
            self._cancel_requested = False
            replacement, self._replacement = self._replacement, None
            status.motion_state = MotionState.CANCELLED
            status.last_refusal = {
                **self._refusal_payload(
                    motion.command,
                    Refusal(MotionRefusal.CANCELLED, "superseded or stopped"),
                ),
                "motion_id": motion.motion_id,
            }
            self._set_result(
                motion.command,
                "cancelled",
                backend=motion.backend,
                motion_id=motion.motion_id,
            )
            if replacement is not None:
                replacement()
            return
        if result.ok:
            status.motion_state = MotionState.IDLE
            planned = result.outcome is MotionOutcome.PLANNED
            # A probe is NOT progress=1.0 and NOT "succeeded": nothing moved, and
            # a consumer that cannot tell the two apart believes the arm is
            # somewhere it never went.
            status.progress = None if planned else 1.0
            extra: dict[str, Any] = {
                "backend": motion.backend,
                "motion_id": motion.motion_id,
            }
            if isinstance(motion.goal, MotionGoal):
                extra["motion"] = motion.goal.motion_type.value
            if status.planner is not None:
                extra["planner"] = planner_name(status.planner)
            if result.selected_candidate is not None:
                extra["selected_candidate"] = result.selected_candidate
            self._set_result(motion.command, result.outcome.value, **extra)
            self._log(
                f"{motion.command}: {extra.get('planner', '?')} "
                f"{'found a plan (probe; nothing executed)' if planned else 'succeeded'}"
                f" in {status.plan_time_ms} ms, {status.trajectory_points} point(s)"
            )
            return
        refusal = result.refusal or Refusal(
            MotionRefusal.PLANNING_FAILED, f"{motion.command} {result.outcome.value}"
        )
        status.motion_state = (
            MotionState.CANCELLED
            if refusal.code is MotionRefusal.CANCELLED
            else MotionState.FAILED
        )
        status.last_refusal = {
            **self._refusal_payload(motion.command, refusal),
            "motion_id": motion.motion_id,
        }
        self._stream.mark_dirty()
        self._on_failed(motion.command, refusal.status_line())

    def _request_cancel(self) -> None:
        motion = self._motion
        if motion is not None and not self._cancel_requested:
            self._cancel_requested = True
            self._slots[motion.backend].backend.cancel()

    def cancel(self) -> None:
        """Stop the active motion and drop any replacement.

        Fire and forget: the terminal status arrives through the backend's
        result. The scene revision and any attached payload survive -- the
        executor decides what to do next.
        """
        with self._lock:
            self._replacement = None
            self._request_cancel()

    # ── reporting ────────────────────────────────────────────────────────────

    def _stamp_ms(self) -> int:
        """When a result was recorded: wall-clock ms, strictly increasing.

        Results carry no request id, so two identical answers in a row (two
        ``set_pose`` successes, say) are told apart by this alone: a requester
        reads the first result for its command whose stamp it has not seen.
        """
        stamp = max(int(self._clock() * 1000), self._last_stamp_ms + 1)
        self._last_stamp_ms = stamp
        return stamp

    def _refusal_payload(self, command: str, refusal: Refusal) -> dict[str, Any]:
        return {**refusal.to_payload(command), "timestamp_ms": self._stamp_ms()}

    def _set_result(self, command: str, outcome: str, **extra: Any) -> None:
        self._status.last_result = {
            "command": command,
            "outcome": outcome,
            **extra,
            "timestamp_ms": self._stamp_ms(),
        }
        self._stream.mark_dirty()

    def _report_refusal(self, command: str, refusal: Refusal) -> None:
        self._status.last_refusal = self._refusal_payload(command, refusal)
        self._stream.mark_dirty()
        self._on_failed(command, refusal.status_line())


def _cartesian_rejection(code: MotionRefusal) -> CartesianRejection:
    """The ``CartesianRejection`` naming the same cause as *code*.

    The gate codes share their strings by design; an unavailable backend is
    ``planner_unavailable``; anything else refused the plan.
    """
    if code is MotionRefusal.BACKEND_UNAVAILABLE:
        return CartesianRejection.PLANNER_UNAVAILABLE
    try:
        return CartesianRejection(code.value)
    except ValueError:
        return CartesianRejection.PLANNING_FAILED


def _without(request: SceneRequest, object_id: str) -> SceneRequest:
    """*request* with one object (and its touch allowance) taken out."""
    return SceneRequest(
        request.revision,
        tuple(entry for entry in request.objects if entry.id != object_id),
        request.allow_touch - {object_id},
    )


__all__ = [
    "ATTACH_PAYLOAD",
    "DETACH_PAYLOAD",
    "GRASP",
    "MOTION_COMMANDS",
    "SCENE_SET",
    "SET_POSE",
    "PlannedMotionConfig",
    "PlannedMotionController",
]
