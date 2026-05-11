"""Deterministic motion executor for the SO-101 arm.

The executor consumes a `MotionPlan` (4 action types: `set_joint`, `set_pose`,
`wait`, `home`) and runs it on a Cyberwave robot twin with:

  * **Joint clamping** — every commanded angle is clamped to a configurable
    safe range (`DEFAULT_JOINT_LIMITS`) before being sent. The arm is
    physically incapable of an out-of-range request from a hallucinating LLM.
  * **Duration caps** — no single action may exceed `MAX_DURATION_S`, no plan
    may have more than `MAX_ACTIONS_PER_PLAN` actions.
  * **Smooth ramping** — joint moves linearly interpolate from the executor's
    current pose to the target pose at `RAMP_HZ`, instead of snapping. This
    is what makes the demo look intentional rather than jerky.

Phase 3: hand-crafted `MotionPlan` instances drive the executor.
Phase 4: an LLM produces the same shape via `MotionPlan.from_dict(claude_json)`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Protocol


JointName = str  # canonical SO-101 names: "1".."6"

ActionType = Literal["set_joint", "set_pose", "wait", "home"]
ALLOWED_ACTION_TYPES: set[str] = {"set_joint", "set_pose", "wait", "home"}

JOINTS: tuple[str, ...] = ("1", "2", "3", "4", "5", "6")

# Per-joint angular limits, degrees. Conservative envelope for a public-demo
# arm — wide enough to look expressive, narrow enough that a worst-case
# hallucinated value can't bash the robot into itself or a table.
DEFAULT_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "1": (-90, 90),
    "2": (-60, 60),
    "3": (-60, 60),
    "4": (-60, 60),
    "5": (-60, 60),
    "6": (-60, 60),
}

MAX_DURATION_S: float = 5.0
MAX_ACTIONS_PER_PLAN: int = 8
DEFAULT_DURATION_S: float = 1.0
RAMP_HZ: int = 20


class _RobotJoints(Protocol):
    def set(  # noqa: D401 — match SDK signature
        self,
        joint_name: str,
        position: float,
        degrees: bool = True,
    ) -> Any: ...


class _Robot(Protocol):
    @property
    def joints(self) -> _RobotJoints: ...


# ---------------------------------------------------------------------------
# Plan dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Action:
    """One step of a motion plan.

    `type` determines which fields matter:
      * `set_joint`  → joint, angle, [duration]
      * `set_pose`   → pose,  [duration]
      * `wait`       → duration
      * `home`       → [duration]
    """

    type: ActionType
    joint: str | None = None
    angle: float | None = None
    pose: dict[str, float] | None = None
    duration: float = DEFAULT_DURATION_S

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Action":
        return cls(
            type=data["type"],
            joint=data.get("joint"),
            angle=data.get("angle"),
            pose=data.get("pose"),
            duration=float(data.get("duration", DEFAULT_DURATION_S)),
        )


@dataclass
class MotionPlan:
    """A short, validated sequence of motion actions."""

    say: str = ""
    actions: list[Action] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MotionPlan":
        return cls(
            say=str(data.get("say", "")),
            actions=[Action.from_dict(a) for a in data.get("actions", [])],
        )


# ---------------------------------------------------------------------------
# Validation + clamping
# ---------------------------------------------------------------------------


def clamp(
    joint: str,
    angle: float,
    limits: dict[str, tuple[float, float]] | None = None,
) -> float:
    """Clamp `angle` to the safe range for `joint`."""
    bounds = (limits or DEFAULT_JOINT_LIMITS).get(joint, (-60.0, 60.0))
    return max(bounds[0], min(bounds[1], float(angle)))


def validate_plan(plan: MotionPlan) -> list[str]:
    """Return a list of human-readable error strings; empty list = valid."""
    errors: list[str] = []

    if len(plan.actions) > MAX_ACTIONS_PER_PLAN:
        errors.append(
            f"too many actions ({len(plan.actions)} > {MAX_ACTIONS_PER_PLAN})"
        )

    for i, a in enumerate(plan.actions):
        prefix = f"action[{i}] ({a.type})"

        if a.type not in ALLOWED_ACTION_TYPES:
            errors.append(f"{prefix}: unknown type, must be one of {ALLOWED_ACTION_TYPES}")
            continue

        if a.duration < 0:
            errors.append(f"{prefix}: duration must be ≥ 0, got {a.duration}")
        if a.duration > MAX_DURATION_S:
            errors.append(
                f"{prefix}: duration {a.duration}s exceeds MAX_DURATION_S ({MAX_DURATION_S}s)"
            )

        if a.type == "set_joint":
            if a.joint not in DEFAULT_JOINT_LIMITS:
                errors.append(f"{prefix}: unknown joint {a.joint!r}, expected one of {JOINTS}")
            if a.angle is None:
                errors.append(f"{prefix}: missing 'angle'")

        elif a.type == "set_pose":
            if not a.pose:
                errors.append(f"{prefix}: missing or empty 'pose'")
            else:
                for j in a.pose:
                    if j not in DEFAULT_JOINT_LIMITS:
                        errors.append(f"{prefix}: pose contains unknown joint {j!r}")

    return errors


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class MotionExecutor:
    """Runs validated MotionPlans on a Cyberwave robot twin.

    Tracks an in-memory `_current_pose` so it can interpolate from the last
    commanded pose, regardless of the robot's actual physical state. For Phase
    3 this is good enough; in Phase 5+ we can subscribe to live joint states
    to bootstrap from the real position.
    """

    def __init__(
        self,
        robot: _Robot,
        *,
        joint_limits: dict[str, tuple[float, float]] | None = None,
        ramp_hz: int = RAMP_HZ,
        dry_run: bool = False,
    ) -> None:
        self.robot = robot
        self.joint_limits = joint_limits or DEFAULT_JOINT_LIMITS
        self.ramp_hz = ramp_hz
        self.dry_run = dry_run
        self._current_pose: dict[str, float] = {j: 0.0 for j in JOINTS}

    # ---- public API ------------------------------------------------------

    def home(self, duration: float = 1.5) -> None:
        """Convenience: ramp every joint back to 0°."""
        self._ramp_to({j: 0.0 for j in JOINTS}, duration)

    def execute(self, plan: MotionPlan) -> None:
        """Validate, log, then execute every action in `plan`."""
        errors = validate_plan(plan)
        if errors:
            raise ValueError("invalid plan:\n  - " + "\n  - ".join(errors))

        if plan.say:
            print(f"  💬  {plan.say}")

        for i, action in enumerate(plan.actions, 1):
            print(f"  ▶  [{i}/{len(plan.actions)}] {self._describe(action)}", flush=True)
            self._run(action)

        print(f"  ✅  plan complete  (final pose: {self._format_pose()})")

    # ---- internals -------------------------------------------------------

    def _describe(self, a: Action) -> str:
        if a.type == "wait":
            return f"wait {a.duration:.2f}s"
        if a.type == "home":
            return f"home over {a.duration:.2f}s"
        if a.type == "set_joint":
            clamped = clamp(a.joint or "", a.angle or 0.0, self.joint_limits)
            return f"joint {a.joint} → {clamped:+.1f}° over {a.duration:.2f}s"
        if a.type == "set_pose":
            parts = ", ".join(
                f"{j}={clamp(j, v, self.joint_limits):+.1f}°" for j, v in (a.pose or {}).items()
            )
            return f"pose {{{parts}}} over {a.duration:.2f}s"
        return f"<unknown:{a.type}>"

    def _run(self, action: Action) -> None:
        if action.type == "wait":
            time.sleep(action.duration)
            return

        if action.type == "home":
            self._ramp_to({j: 0.0 for j in JOINTS}, action.duration)
            return

        if action.type == "set_joint":
            assert action.joint is not None and action.angle is not None
            target = clamp(action.joint, action.angle, self.joint_limits)
            new_pose = {**self._current_pose, action.joint: target}
            self._ramp_to(new_pose, action.duration)
            return

        if action.type == "set_pose":
            assert action.pose is not None
            new_pose = dict(self._current_pose)
            for j, v in action.pose.items():
                new_pose[j] = clamp(j, v, self.joint_limits)
            self._ramp_to(new_pose, action.duration)
            return

        raise ValueError(f"unknown action type: {action.type}")

    def _ramp_to(self, target_pose: dict[str, float], duration: float) -> None:
        """Linearly interpolate every joint from current → target over `duration`."""
        if duration <= 0:
            self._snap_to(target_pose)
            return

        steps = max(2, int(duration * self.ramp_hz))
        start = dict(self._current_pose)
        dt = duration / steps

        for s in range(1, steps + 1):
            t = s / steps
            interp = {
                j: start.get(j, 0.0) + (target_pose[j] - start.get(j, 0.0)) * t
                for j in target_pose
            }
            self._snap_to(interp)
            time.sleep(dt)

    def _snap_to(self, pose: dict[str, float]) -> None:
        for joint, angle in pose.items():
            if not self.dry_run:
                self.robot.joints.set(joint, angle, degrees=True)
            self._current_pose[joint] = angle

    def _format_pose(self) -> str:
        return ", ".join(f"{j}={self._current_pose[j]:+.1f}°" for j in JOINTS)
