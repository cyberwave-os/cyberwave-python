"""The SDK's generic sections: one entry per thing an arm driver can expose.

Each entry says what it needs from the twin and calls the
:class:`~cyberwave.test.driver_testing.harness.StressHarness` method that drives it. The
order here is the run order: joints first (no planner involved), then Cartesian
jogs, planned motion, scene and grasp, gripper, base, obstacles, navigation,
and the refusal probes last.
"""

from __future__ import annotations

from typing import Any

from .discovery import TwinProfile
from .harness import NO_MEASURED_TCP
from .registry import section

_ROTATIONS = {"ee_rotate_left", "ee_rotate_right", "ee_rotate_up", "ee_rotate_down"}


def _default_backend_ready(profile: TwinProfile) -> str | None:
    if profile.motion_status is not None and not profile.backend_available:
        return f"default backend {profile.default_backend or 'unknown'} reports unavailable"
    return None


def _needs_tcp(profile: TwinProfile) -> str | None:
    if "kinematics" not in profile.topics:
        return NO_MEASURED_TCP
    return _default_backend_ready(profile)


_ARM: dict[str, Any] = {"needs_arm_joints": True, "topics": {"joints"}}


# ── joints ─────────────────────────────────────────────────────────────────


@section(
    "joints.single",
    area="joints",
    summary="each arm joint alone to a target inside its limits, measured, and back",
    **_ARM,
)
def _single(harness: Any) -> Any:
    return harness.single_joints()


@section(
    "joints.multi",
    area="joints",
    summary="every arm joint at once to a target inside its limits, measured, and back",
    **_ARM,
)
def _multi(harness: Any) -> Any:
    return harness.multi_joint()


@section(
    "joints.latest_target",
    area="joints",
    summary="a newer joint target replaces an older one immediately (latest target wins)",
    **_ARM,
)
def _latest(harness: Any) -> Any:
    return harness.partial_targets()


@section(
    "joints.sine",
    area="joints",
    summary="sinusoidal joint teleop load, then measured return to the baseline",
    **_ARM,
)
def _sine(harness: Any) -> Any:
    return harness.sine()


@section(
    "joints.home",
    area="joints",
    summary="home: the arm settles at its measured (and declared) home",
    commands={"home"},
    **_ARM,
)
def _home(harness: Any) -> Any:
    return harness.home()


# ── Cartesian jogs ─────────────────────────────────────────────────────────


@section(
    "cartesian.ee_move",
    area="cartesian",
    summary="relative ee_move up and its inverse, measured displacement and restore",
    commands={"ee_move"},
    topics={"pose"},
)
def _jog(harness: Any) -> Any:
    return harness.jog()


@section(
    "cartesian.ee_rotate",
    area="cartesian",
    summary="every ee_rotate_* the catalog has, each undone by its inverse: measured rotation and restore",
    any_commands=_ROTATIONS,
    topics={"pose"},
)
def _rotate(harness: Any) -> Any:
    return harness.tool_rotation()


@section(
    "cartesian.ee_pose",
    area="cartesian",
    summary="absolute ee_pose to the TCP raised by a step, measured, and back",
    commands={"ee_pose"},
    topics={"pose"},
    check=_needs_tcp,
)
def _ee_pose(harness: Any) -> Any:
    return harness.absolute_pose()


# ── planned motion ─────────────────────────────────────────────────────────

_PLANNED: dict[str, Any] = {
    "commands": {"set_pose"},
    "topics": {"pose", "motion", "joints"},
    "needs_arm_joints": True,
}


@section(
    "planned.ptp",
    area="planned",
    summary="set_pose motion=ptp: plan only, execute, measured arrival, return",
    per_backend=True,
    **_PLANNED,
)
def _ptp(harness: Any, backend: str) -> Any:
    return harness.cartesian(backend, "ptp")


@section(
    "planned.lin",
    area="planned",
    summary="set_pose motion=lin: plan only, execute, measured arrival, return",
    per_backend=True,
    **_PLANNED,
)
def _lin(harness: Any, backend: str) -> Any:
    return harness.cartesian(backend, "lin")


@section(
    "planned.replace",
    area="planned",
    summary="a newer set_pose replaces an executing one; its endpoint holds",
    check=_needs_tcp,
    **_PLANNED,
)
def _replace(harness: Any) -> Any:
    return harness.command_replacement()


@section(
    "planned.stop",
    area="planned",
    summary="stop ends an executing set_pose with a measured hold; a fresh command works",
    **_PLANNED,
    check=_default_backend_ready,
)
def _stop(harness: Any) -> Any:
    return harness.command_replacement(stop=True)


@section(
    "planned.teleop_preempts",
    area="planned",
    summary="a joint target preempts an executing set_pose and holds",
    **_PLANNED,
    check=_default_backend_ready,
)
def _preempt(harness: Any) -> Any:
    return harness.teleop_takeover()


# ── grasp and scene ────────────────────────────────────────────────────────


@section(
    "scene.collision_guard",
    area="scene",
    summary="scene_set with a test fixture; a plan-only set_pose into it is refused for collision",
    commands={"scene_set", "set_pose"},
    topics={"pose", "motion", "joints"},
    needs_arm_joints=True,
    check=_default_backend_ready,
)
def _scene(harness: Any) -> Any:
    return harness.scene_collision_guard()


@section(
    "scene.payload",
    area="scene",
    summary="attach_payload then detach_payload, as state/motion reports them",
    commands={"attach_payload", "detach_payload"},
    topics={"motion"},
)
def _payload(harness: Any) -> Any:
    return harness.payload()


@section(
    "grasp.candidates",
    area="grasp",
    summary="plan-only grasp: an unreachable candidate falls through to a reachable one",
    commands={"grasp"},
    topics={"motion"},
)
def _grasp(harness: Any) -> Any:
    return harness.grasp_search()


@section(
    "grasp.execute",
    area="grasp",
    summary="executed grasp on the tool's pose in free space: measured approach and arrival, gripper untouched",
    commands={"grasp"},
    topics={"motion", "pose", "joints"},
    needs_arm_joints=True,
    per_backend=True,
)
def _grasp_execute(harness: Any, backend: str) -> Any:
    return harness.grasp_execute(backend)


# ── gripper ────────────────────────────────────────────────────────────────


@section(
    "gripper.endpoints",
    area="gripper",
    summary="release, grip, release: measured jaw endpoints",
    commands={"grip", "release"},
    topics={"joints"},
    needs_gripper_joint=True,
)
def _gripper(harness: Any) -> Any:
    return harness.gripper()


@section(
    "gripper.independent",
    area="gripper",
    summary="grip and release while an arm set_pose completes uncancelled",
    commands={"grip", "release", "set_pose"},
    topics={"joints", "pose", "motion"},
    needs_gripper_joint=True,
    needs_arm_joints=True,
    check=_default_backend_ready,
)
def _gripper_overlap(harness: Any) -> Any:
    return harness.gripper(overlap=True)


# ── base rotation ──────────────────────────────────────────────────────────


@section(
    "base.rotate",
    area="base",
    summary="base_rotate_left/right: one joint turns by the angle and back, measured",
    commands={"base_rotate_left", "base_rotate_right"},
    **_ARM,
)
def _base(harness: Any) -> Any:
    return harness.base_rotation()


# ── world obstacles ────────────────────────────────────────────────────────


@section(
    "obstacles.roundtrip",
    area="obstacles",
    summary="add_obstacles with a fixture, refused inside it (checked with the pinocchio backend only), "
    "remove_obstacle, original list restored",
    commands={"add_obstacles", "remove_obstacle"},
    topics={"pose", "joints"},
    needs_arm_joints=True,
    check=_default_backend_ready,
)
def _obstacles(harness: Any) -> Any:
    return harness.obstacles()


# ── navigation ─────────────────────────────────────────────────────────────


@section(
    "navigation.mission",
    area="navigation",
    summary="a navigate waypoint mission out and back, measured",
    topics={"navigate", "pose", "joints"},
    needs_arm_joints=True,
)
def _mission(harness: Any) -> Any:
    return harness.waypoint_mission()


@section(
    "navigation.cancel",
    area="navigation",
    summary="a navigate mission stopped while moving: cancelled, measured hold",
    topics={"navigate", "pose", "joints"},
    needs_arm_joints=True,
)
def _cancel(harness: Any) -> Any:
    return harness.navigation_cancel()


# ── faults and gates ───────────────────────────────────────────────────────


@section(
    "faults.refusals",
    area="faults",
    summary="guarded requests are refused with their code on state/motion, not timed out",
    commands={"set_pose"},
    topics={"motion", "joints"},
    needs_arm_joints=True,
)
def _refusals(harness: Any) -> Any:
    return harness.refusals()
