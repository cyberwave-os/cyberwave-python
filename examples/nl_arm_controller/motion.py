"""Deterministic motion executor for the AgileX PiPER arm.

The executor consumes a `MotionPlan` (5 action types: `set_joint`, `set_pose`,
`set_gripper`, `wait`, `home`) and runs it on a Cyberwave robot twin with:

  * **Joint clamping** — every commanded angle is clamped to a safe range
    before being sent. The envelope is the intersection of the conservative
    demo limits in `ARM_JOINTS` and the twin's own URDF limits (fetched via
    `robot.get_schema()`), so a hallucinating LLM cannot drive the arm past
    what the model itself allows.
  * **Duration caps** — no single action may exceed `MAX_DURATION_S`, no plan
    may have more than `MAX_ACTIONS_PER_PLAN` actions.
  * **Smooth ramping** — joint moves linearly interpolate from the executor's
    current pose to the target pose at `RAMP_HZ`, instead of snapping. This
    is what makes the demo look intentional rather than jerky.
  * **Change-only publishing** — a ramp tick only publishes the joints that
    actually moved, so a single-joint wave sends ~20 msg/s instead of 140.

This module is the single source of truth for the PiPER's joint model: names,
directional semantics, demo limits, and the keyboard bindings that mirror the
dashboard's "Keyboard (PiPER)" controller. `planner.py` builds its LLM prompts
from `joint_table_for_prompt()` and `teleop.py` builds its key map from
`KEY_TO_ACTION`, so the joint model is described in exactly one place.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


# ---------------------------------------------------------------------------
# Joint model — single source of truth
# ---------------------------------------------------------------------------

JointName = str  # PiPER platform joint names: "joint1".."joint7"

ActionType = Literal["set_joint", "set_pose", "set_gripper", "wait", "home"]
ALLOWED_ACTION_TYPES: set[str] = {
    "set_joint",
    "set_pose",
    "set_gripper",
    "wait",
    "home",
}


@dataclass(frozen=True)
class JointSpec:
    """One controllable revolute joint of the PiPER arm.

    `lower`/`upper` are the *demo* envelope in degrees — deliberately narrower
    than the hardware range so a worst-case plan still looks like a gesture
    rather than a collision. `key_increase`/`key_decrease` mirror the
    dashboard's keyboard controller so muscle memory transfers between the
    browser teleop panel and `--keys` mode here.
    """

    name: str
    label: str
    positive: str
    lower: float
    upper: float
    key_increase: str
    key_decrease: str


# Demo envelope, degrees. The PiPER's real URDF range is wider (and asymmetric
# for joint2/joint3 — the shoulder only sweeps forward from zero and the elbow
# only folds back), so these are narrowed on purpose. At runtime
# `effective_limits()` intersects them with the twin's actual limits, so these
# numbers can only ever be *more* conservative than the model allows.
ARM_JOINTS: tuple[JointSpec, ...] = (
    JointSpec(
        name="joint1",
        label="base rotation",
        positive="swings the arm to the operator's LEFT (counter-clockwise seen from above)",
        lower=-90.0,
        upper=90.0,
        key_increase="1",
        key_decrease="2",
    ),
    JointSpec(
        name="joint2",
        label="shoulder pitch",
        positive="pitches the upper arm FORWARD and DOWN from vertical (this joint only moves one way from zero)",
        lower=0.0,
        upper=90.0,
        key_increase="3",
        key_decrease="4",
    ),
    JointSpec(
        name="joint3",
        label="elbow",
        positive="unfolds the forearm; negative FOLDS the elbow back (this joint only moves one way from zero)",
        lower=-90.0,
        upper=0.0,
        key_increase="5",
        key_decrease="6",
    ),
    JointSpec(
        name="joint4",
        label="forearm roll",
        positive="rolls the forearm clockwise as seen from the base",
        lower=-60.0,
        upper=60.0,
        key_increase="7",
        key_decrease="8",
    ),
    JointSpec(
        name="joint5",
        label="wrist pitch",
        positive="pitches the tool UP",
        lower=-60.0,
        upper=60.0,
        key_increase="9",
        key_decrease="0",
    ),
    JointSpec(
        name="joint6",
        label="wrist roll",
        positive="rolls the tool clockwise",
        lower=-60.0,
        upper=60.0,
        key_increase="Q",
        key_decrease="W",
    ),
)


@dataclass(frozen=True)
class GripperSpec:
    """The PiPER gripper — platform joint `joint7`, with `joint8` mimicking it.

    Plans talk about the gripper in **percent open** (0 = closed, 100 = fully
    open) so the LLM never has to know the native unit. The executor converts
    to native units at publish time: metres for the prismatic finger joint on
    the real model, radians if a given twin models it as revolute. `joint8`
    is a mimic joint (factor -1.0) and is driven by the platform — never
    command it directly.
    """

    name: str = "joint7"
    label: str = "gripper"
    key_increase: str = "E"  # open
    key_decrease: str = "R"  # close
    closed_native: float = 0.0
    open_native: float = 0.035  # metres of finger travel
    mimic: str = "joint8"
    mimic_factor: float = -1.0


GRIPPER = GripperSpec()

JOINTS: tuple[str, ...] = tuple(s.name for s in ARM_JOINTS)
JOINT_SPECS: dict[str, JointSpec] = {s.name: s for s in ARM_JOINTS}

DEFAULT_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    s.name: (s.lower, s.upper) for s in ARM_JOINTS
}

# key → (joint name, +1 / -1). Mirrors the dashboard's "Keyboard (PiPER)"
# controller exactly: odd digits / Q / E increase, even digits / W / R decrease.
KEY_TO_ACTION: dict[str, tuple[str, float]] = {
    **{s.key_increase: (s.name, +1.0) for s in ARM_JOINTS},
    **{s.key_decrease: (s.name, -1.0) for s in ARM_JOINTS},
    GRIPPER.key_increase: (GRIPPER.name, +1.0),
    GRIPPER.key_decrease: (GRIPPER.name, -1.0),
}

MAX_DURATION_S: float = 5.0
MAX_ACTIONS_PER_PLAN: int = 8
DEFAULT_DURATION_S: float = 1.0
RAMP_HZ: int = 20
# Below this, a ramp tick is not worth an MQTT publish for that joint.
PUBLISH_EPSILON_DEG: float = 0.05


def joint_table_for_prompt() -> str:
    """Render the joint table the LLM prompts embed.

    Keeping this generated means the planner can never drift out of sync with
    the limits the executor actually enforces.
    """
    lines = []
    for s in ARM_JOINTS:
        rng = f"[{s.lower:+.0f}°, {s.upper:+.0f}°]"
        lines.append(f'  "{s.name}" — {s.label:<14} {rng:<18} positive {s.positive}')
    lines.append(
        f'  "{GRIPPER.name}" — {GRIPPER.label:<14} [0, 100] percent open '
        f"(0 = closed, 100 = open) — command with set_gripper, never set_joint"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runtime limit discovery
# ---------------------------------------------------------------------------


def read_twin_joint_schema(robot: Any) -> dict[str, dict[str, Any]]:
    """Pull `{joint_name: {type, lower, upper}}` out of the twin's schema.

    Limits in the schema are radians (revolute) or metres (prismatic). Returns
    an empty dict if the schema is unavailable — the caller then falls back to
    the hardcoded demo envelope.
    """
    try:
        schema = robot.get_schema() or {}
    except Exception:
        return {}
    if not isinstance(schema, dict):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for j in schema.get("joints") or []:
        if not isinstance(j, dict):
            continue
        name = j.get("name")
        if not name:
            continue
        limits = j.get("limits") or {}
        out[str(name)] = {
            "type": j.get("type"),
            "lower": limits.get("lower"),
            "upper": limits.get("upper"),
        }
    return out


def effective_limits(
    twin_joints: dict[str, dict[str, Any]] | None = None,
) -> dict[str, tuple[float, float]]:
    """Intersect the demo envelope with the twin's own revolute limits."""
    limits = dict(DEFAULT_JOINT_LIMITS)
    if not twin_joints:
        return limits

    for name, (demo_lo, demo_hi) in list(limits.items()):
        info = twin_joints.get(name)
        if not info or info.get("type") == "prismatic":
            continue
        lo, hi = info.get("lower"), info.get("upper")
        if lo is None or hi is None:
            continue
        schema_lo, schema_hi = math.degrees(float(lo)), math.degrees(float(hi))
        tight_lo, tight_hi = max(demo_lo, schema_lo), min(demo_hi, schema_hi)
        # A degenerate intersection means the twin models this joint on a
        # different convention than we assume — keep the demo envelope rather
        # than pinning the joint to a single angle.
        if tight_lo < tight_hi:
            limits[name] = (tight_lo, tight_hi)
    return limits


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
      * `set_joint`   → joint, angle, [duration]
      * `set_pose`    → pose,  [duration]
      * `set_gripper` → opening (percent), [duration]
      * `wait`        → duration
      * `home`        → [duration]
    """

    type: ActionType
    joint: str | None = None
    angle: float | None = None
    pose: dict[str, float] | None = None
    opening: float | None = None
    duration: float = DEFAULT_DURATION_S

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Action":
        return cls(
            type=data["type"],
            joint=data.get("joint"),
            angle=data.get("angle"),
            pose=data.get("pose"),
            opening=data.get("opening"),
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
    table = limits or DEFAULT_JOINT_LIMITS
    bounds = table.get(joint)
    if bounds is None:
        # Unknown joint: fall back to the tightest envelope we ship.
        bounds = (-60.0, 60.0)
    return max(bounds[0], min(bounds[1], float(angle)))


def clamp_opening(opening: float) -> float:
    """Clamp a gripper command to 0–100 percent open."""
    return max(0.0, min(100.0, float(opening)))


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
            if a.joint == GRIPPER.name:
                errors.append(
                    f"{prefix}: {GRIPPER.name} is the gripper — use set_gripper with 'opening'"
                )
            elif a.joint not in DEFAULT_JOINT_LIMITS:
                errors.append(f"{prefix}: unknown joint {a.joint!r}, expected one of {JOINTS}")
            if a.angle is None:
                errors.append(f"{prefix}: missing 'angle'")

        elif a.type == "set_pose":
            if not a.pose:
                errors.append(f"{prefix}: missing or empty 'pose'")
            else:
                for j in a.pose:
                    if j == GRIPPER.name:
                        errors.append(
                            f"{prefix}: {GRIPPER.name} is the gripper — use set_gripper"
                        )
                    elif j not in DEFAULT_JOINT_LIMITS:
                        errors.append(f"{prefix}: pose contains unknown joint {j!r}")

        elif a.type == "set_gripper":
            if a.opening is None:
                errors.append(f"{prefix}: missing 'opening' (0–100 percent)")

    return errors


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class MotionExecutor:
    """Runs validated MotionPlans on a Cyberwave PiPER twin.

    Tracks an in-memory `_current_pose` so it can interpolate from the last
    commanded pose, regardless of the robot's actual physical state. Pass a
    robot and the executor will narrow its clamps to the twin's own URDF
    limits on construction; pass `joint_limits` to override entirely.
    """

    def __init__(
        self,
        robot: _Robot,
        *,
        joint_limits: dict[str, tuple[float, float]] | None = None,
        ramp_hz: int = RAMP_HZ,
        dry_run: bool = False,
        discover_limits: bool = True,
    ) -> None:
        self.robot = robot
        self.ramp_hz = ramp_hz
        self.dry_run = dry_run

        twin_joints = (
            read_twin_joint_schema(robot) if (discover_limits and not dry_run) else {}
        )
        self.twin_joints = twin_joints
        self.joint_limits = joint_limits or effective_limits(twin_joints)

        self._current_pose: dict[str, float] = {j: 0.0 for j in JOINTS}
        self._gripper_pct: float = 0.0
        # Last value actually published per joint, so ramp ticks can skip
        # joints that did not move.
        self._published: dict[str, float] = {}

    # ---- public API ------------------------------------------------------

    @property
    def gripper_percent(self) -> float:
        return self._gripper_pct

    def home(self, duration: float = 1.5) -> None:
        """Ramp every arm joint back to 0°. Leaves the gripper as-is."""
        self._ramp_to({j: 0.0 for j in JOINTS}, duration)

    def set_gripper(self, opening_pct: float, duration: float = 0.6) -> None:
        """Ramp the gripper to `opening_pct` percent open (0 = closed)."""
        target = clamp_opening(opening_pct)
        if duration <= 0:
            self._publish_gripper(target)
            return
        steps = max(2, int(duration * self.ramp_hz))
        start = self._gripper_pct
        dt = duration / steps
        for s in range(1, steps + 1):
            self._publish_gripper(start + (target - start) * (s / steps))
            time.sleep(dt)

    def nudge(self, joint: str, delta: float) -> float:
        """Step one joint by `delta` (degrees, or percent for the gripper).

        Used by keyboard teleop — snaps rather than ramps, since the key
        repeat rate is the ramp.
        """
        if joint == GRIPPER.name:
            self._publish_gripper(self._gripper_pct + delta)
            return self._gripper_pct
        target = clamp(joint, self._current_pose.get(joint, 0.0) + delta, self.joint_limits)
        self._snap_to({**self._current_pose, joint: target})
        return target

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
            return f"{a.joint} → {clamped:+.1f}° over {a.duration:.2f}s"
        if a.type == "set_gripper":
            return (
                f"gripper → {clamp_opening(a.opening or 0.0):.0f}% open "
                f"over {a.duration:.2f}s"
            )
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

        if action.type == "set_gripper":
            assert action.opening is not None
            self.set_gripper(action.opening, action.duration)
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
            self._current_pose[joint] = angle
            last = self._published.get(joint)
            # Only publish joints that actually moved this tick — a
            # single-joint gesture then costs one message per tick, not six.
            if last is not None and abs(last - angle) < PUBLISH_EPSILON_DEG:
                continue
            if not self.dry_run:
                self.robot.joints.set(joint, angle, degrees=True)
            self._published[joint] = angle

    def _publish_gripper(self, opening_pct: float) -> None:
        """Send the gripper in the twin's native unit (metres or radians).

        `joint7` is prismatic on the stock PiPER model, so `degrees=True`
        would silently scale the command by π/180. We resolve the joint's
        real range from the twin schema and always publish native units.
        """
        pct = clamp_opening(opening_pct)
        self._gripper_pct = pct

        low, high = self._gripper_native_range()
        native = low + (high - low) * (pct / 100.0)

        last = self._published.get(GRIPPER.name)
        span = abs(high - low) or 1.0
        if last is not None and abs(last - native) < span * 1e-3:
            return
        if not self.dry_run:
            # joint8 mimics joint7 (factor -1.0) and is driven by the
            # platform — commanding it here would fight the mimic.
            self.robot.joints.set(GRIPPER.name, native, degrees=False)
        self._published[GRIPPER.name] = native

    def _gripper_native_range(self) -> tuple[float, float]:
        info = self.twin_joints.get(GRIPPER.name) or {}
        lo, hi = info.get("lower"), info.get("upper")
        if lo is not None and hi is not None and float(hi) != float(lo):
            return float(lo), float(hi)
        return GRIPPER.closed_native, GRIPPER.open_native

    def _format_pose(self) -> str:
        arm = ", ".join(f"{j}={self._current_pose[j]:+.1f}°" for j in JOINTS)
        return f"{arm}, gripper={self._gripper_pct:.0f}%"
