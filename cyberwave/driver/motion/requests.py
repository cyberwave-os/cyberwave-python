"""The ``grasp`` request: ranked candidates, read into a backend-neutral goal.

Whichever backend answers -- the in-process one or a registered planner --
the request is the same: ranked TCP poses, each with the direction the tool
travels onto it, plus the distances of the staged approach and lift. Pixels,
depth maps and point clouds never reach a planner; perception owns turning
those into candidates.

Pure: no ROS and no planner, so which payload is refused, and with which code,
runs in the unit suite.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .backend import GraspCandidate, GraspGoal
from .guards import MotionRefusal
from .poses import pose_to_transform
from .scene import SceneError, _finite
from .targets import (
    FrameSource,
    TargetMode,
    parse_pose_spec,
    reference_frame,
    resolve_direction,
    resolve_target,
)

#: Default bound on ranked candidates per request. Each candidate costs a full
#: plan, so an unbounded list is an unbounded busy period for the driver.
DEFAULT_MAX_GRASP_CANDIDATES = 64
#: Straight approach onto the grasp pose (m).
DEFAULT_APPROACH_DISTANCE_M = 0.06
#: Lift checked after the grasp pose, before the candidate is chosen (m).
DEFAULT_LIFT_DISTANCE_M = 0.05

_CODE = MotionRefusal.INVALID_REQUEST


def _positive(data: dict[str, Any], key: str, default: float) -> float:
    value = data.get(key)
    if value is None:
        return default
    number = _finite(value, key, _CODE)
    if number <= 0.0:
        raise SceneError(_CODE, f"{key} must be positive")
    return number


def parse_grasp(
    data: Any,
    *,
    frames: FrameSource | None,
    lift_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    tool_approach_axis: tuple[float, float, float] | None = None,
    plan_only: bool = False,
    max_candidates: int = DEFAULT_MAX_GRASP_CANDIDATES,
) -> GraspGoal:
    """Validate one ``grasp`` request completely, or refuse all of it.

    ``candidates`` is the only required argument: a ranked list of
    ``{pose, approach}``. ``pose`` takes any form ``set_pose`` takes, in
    ``reference_frame`` (default: the planning frame). ``approach`` is the
    direction the tool travels onto the grasp pose, in the same frame;
    without one, the tool approaches along its own axis ``tool_approach_axis``
    at the grasp pose (the driver's ``tool_axes["forward"]``). Every
    candidate leaves in the planning frame.

    The per-candidate budget is an argument, never read from ``data``: a
    caller able to widen it could hold the driver busy past the timeout it
    then reports against.
    """
    if not isinstance(data, dict):
        raise SceneError(_CODE, "grasp data must be an object")
    raw_candidates = data.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise SceneError(_CODE, "candidates must be a non-empty list")
    if len(raw_candidates) > max_candidates:
        raise SceneError(
            _CODE,
            f"candidates is bounded at {max_candidates}, got {len(raw_candidates)}",
        )
    candidates: list[GraspCandidate] = []
    for index, raw in enumerate(raw_candidates):
        what = f"candidates[{index}]"
        if not isinstance(raw, dict) or "pose" not in raw:
            raise SceneError(_CODE, f"{what} must be an object with a pose")
        spec = parse_pose_spec(raw.get("pose"), what=f"{what}.pose")
        frame = reference_frame(spec.reference_frame, data.get("reference_frame"))
        pose = resolve_target(
            spec, mode=TargetMode.ABSOLUTE, frame=frame, frames=frames
        )
        if raw.get("approach") is not None:
            approach = resolve_direction(
                raw["approach"], frame=frame, frames=frames, what=f"{what}.approach"
            )
        elif tool_approach_axis is not None:
            rotation = pose_to_transform(pose)[:3, :3]
            axis = rotation @ np.asarray(tool_approach_axis, dtype=float)
            axis /= float(np.linalg.norm(axis))
            approach = (float(axis[0]), float(axis[1]), float(axis[2]))
        else:
            raise SceneError(
                _CODE,
                f"{what}.approach is required: this driver declares no tool "
                "forward axis to approach along",
            )
        candidates.append(GraspCandidate(pose, approach))
    return GraspGoal(
        candidates=tuple(candidates),
        approach_distance_m=_positive(
            data, "approach_distance_m", DEFAULT_APPROACH_DISTANCE_M
        ),
        lift_distance_m=_positive(data, "lift_distance_m", DEFAULT_LIFT_DISTANCE_M),
        lift_direction=lift_direction,
        plan_only=plan_only,
    )


__all__ = [
    "DEFAULT_APPROACH_DISTANCE_M",
    "DEFAULT_LIFT_DISTANCE_M",
    "DEFAULT_MAX_GRASP_CANDIDATES",
    "parse_grasp",
]
