"""The ``state/motion`` payload every planned-motion arm driver publishes.

One dataclass, because two producers of the same wire shape drift. The driver's
publisher, the telemetry fields and the tests all read this.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

#: Status is edge-triggered, plus this heartbeat while a motion is running, so a
#: consumer that missed a transition learns the motion is still alive without the
#: driver publishing an unchanged payload ten times a second while idle.
ACTIVE_HEARTBEAT_S = 0.5

#: And a slow heartbeat while idle, so a subscriber that connects AFTER the last
#: transition still learns the state. Pure edge triggering left a late consumer
#: with no payload at all until something happened -- indistinguishable from a
#: driver that does not publish this stream, which is exactly the ambiguity an
#: executor gating on a field in it cannot afford.
IDLE_HEARTBEAT_S = 5.0


class MotionState(str, Enum):
    """Where the single active motion is, or ``IDLE`` when there is none.

    ``FAILED`` and ``CANCELLED`` are terminal but not blocking: a new motion may
    start from either. They are distinct because "the planner could not" and
    "somebody stopped it" call for different operator responses.
    """

    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PlannerKind(str, Enum):
    """The SDK's own planner, as reported in ``planner``.

    ``planner`` on ``state/motion`` is a plain string: the name of the planner
    that produced the trajectory, as the answering backend reports it. This is
    the value the SDK itself reports; a registered backend reports its own
    (e.g. one per planner of an external framework). Reported rather than
    inferred: a backend may fall back from one planner to another, and "the
    arm took a joint-space detour" is only explicable if the status says which
    planner answered.
    """

    #: The in-process planner of the pinocchio backend: IK and collision checks
    #: on the robot description, sampled PTP/LIN paths, the grasp search.
    PINOCCHIO = "pinocchio"


def planner_name(planner: str | Enum | None) -> str | None:
    """The wire string for a reported planner (an enum member or a plain str)."""
    if planner is None:
        return None
    if isinstance(planner, Enum):
        return str(planner.value)
    return str(planner)


@dataclass
class MotionStatus:
    """Mutable snapshot the planner owns and the driver publishes.

    Deliberately not frozen: this is one live object with a publisher reading it,
    not a stream of values. ``to_payload`` is the only thing that leaves.
    """

    #: The driver's default backend -- the one answering every request that
    #: names none (``"pinocchio"``, or a backend the driver registered) -- and
    #: whether it can accept goals now. ``None`` for a driver that composes no
    #: planned-motion backend.
    kinematics_backend: str | None = None
    backend_available: bool = False
    #: False means the planner is reasoning about a BARE wrist. If a camera or
    #: bracket is physically mounted, its geometry is invisible to every
    #: collision decision reported here -- and on a descent the camera is the
    #: part most likely to make contact. The driver cannot detect the hardware;
    #: this reports what it was told, so a consumer can refuse to run a task
    #: against a planner that cannot see the end effector it is flying.
    wrist_payload_present: bool = False
    active_scene_revision: str | None = None
    attached_payload: str | None = None
    motion_state: MotionState = MotionState.IDLE
    #: The command of the current/last motion (``set_pose``, ``grasp``, ...),
    #: the backend answering it, and its number: the driver numbers every
    #: motion it starts, and a motion's terminal ``last_result`` /
    #: ``last_refusal`` carries the same ``motion_id``. A requester that sent
    #: a command watches for the first new ``motion_id`` with that command,
    #: then for the result naming it; the cancellation of a motion it
    #: replaced names the older id.
    command: str | None = None
    backend: str | None = None
    motion_id: int | None = None
    #: The planner that produced the current/last trajectory, as the backend
    #: names it (see :class:`PlannerKind` for the SDK's own values).
    planner: str | None = None
    plan_time_ms: int | None = None
    trajectory_points: int | None = None
    trajectory_duration_s: float | None = None
    progress: float | None = None
    #: ``{"command", "outcome", ..., "timestamp_ms"}`` for the last command that
    #: was accepted.
    last_result: dict[str, Any] | None = None
    #: ``{"command", "code", "detail", "timestamp_ms"}`` for the last refusal,
    #: kept until the next one. ``timestamp_ms`` is unique and increasing, so
    #: two identical answers in a row stay distinct; a result that ends a
    #: motion also names its ``motion_id``, and a refusal of a command that
    #: never started names none. One that needs its own verdict sends the
    #: command as an MQTT v5 request: the driver replies ``accepted`` or
    #: ``rejected`` with the refusal line, and this stream carries what
    #: happened next.
    last_refusal: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        """The wire dict. Enums become their string values; nothing is dropped.

        Every key is always present, including the ``None`` ones: a consumer
        checking ``payload["attached_payload"]`` should not have to distinguish
        "no payload" from "this driver version does not report payloads".
        """
        payload = asdict(self)
        payload["motion_state"] = self.motion_state.value
        payload["planner"] = planner_name(self.planner)
        return payload


class MotionStatusStream:
    """The one edge-triggered publisher for the one shared ``MotionStatus``.

    ``state/motion`` is a latest-value topic with two motion backends behind it.
    While two motion backends each owned a status object, whichever
    published last replaced the other's ``last_result``, ``last_refusal``,
    ``active_scene_revision`` and ``attached_payload``: a consumer waiting for a
    terminal result could miss it and time out on a motion that had actually
    succeeded, and the scene fields dropped to null for every payload the
    other backend published. One status object and one heartbeat is what
    makes the stream readable.

    ``refresh`` is the driver's hook for the fields no backend owns alone
    (e.g. whether the wrist payload is in the collision model); it runs
    immediately before the payload leaves.
    """

    def __init__(
        self,
        status: MotionStatus | None = None,
        *,
        refresh: Callable[[MotionStatus], None] | None = None,
    ) -> None:
        self.status = MotionStatus() if status is None else status
        self._refresh = refresh
        #: Publish once on the first tick: a consumer that subscribed before the
        #: driver came up must not have to wait for a transition to learn state.
        self._dirty = True
        self._last_published: float | None = None

    def mark_dirty(self) -> None:
        """Something in the shared status changed; publish on the next tick."""
        self._dirty = True

    def payload_if_due(self, now: float) -> dict[str, Any] | None:
        """Edge-triggered, plus a heartbeat. ``None`` = nothing new to say."""
        active = self.status.motion_state in (
            MotionState.PLANNING,
            MotionState.EXECUTING,
        )
        interval = ACTIVE_HEARTBEAT_S if active else IDLE_HEARTBEAT_S
        heartbeat = (
            self._last_published is not None and now - self._last_published >= interval
        )
        if not (self._dirty or heartbeat):
            return None
        self._dirty = False
        self._last_published = now
        if self._refresh is not None:
            self._refresh(self.status)
        return self.status.to_payload()
