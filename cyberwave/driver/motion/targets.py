"""Where a ``set_pose`` / ``grasp`` target is, resolved into the planning frame.

A target is always a full Cartesian pose, a position AND a rotation, in one
of the forms a requester already has it in:

* ``[x, y, z, roll, pitch, yaw]`` -- extrinsic XYZ in radians, the ``ee_pose``
  convention;
* ``[x, y, z, qx, qy, qz, qw]`` -- the ``pose_xyzw`` convention of ``twin/pose``;
* ``[[x, y, z], [roll, pitch, yaw]]`` or ``[[x, y, z], [qx, qy, qz, qw]]`` --
  the same, as a position and a rotation;
* a waypoint ``{position, orientation | rotation, reference_frame | frame_id}``
  -- ``position`` as ``{x, y, z}`` or ``[x, y, z]``, the rotation as a named
  ``{x, y, z, w}`` quaternion: the shape
  :class:`~cyberwave.navigation.NavigationPlan` builds.

There is no position-only form and no degrees: a pose without a rotation is
not a pose, and angles are radians.

and in the frame it has it in. ``reference_frame`` names any frame of the
robot description (``base_link``, ``link6``, a camera link), read at the arm's
current posture, or one of two aliases: ``"base"`` (the planning frame) and
``"tool"`` / ``"tcp"`` (the tool centre point).

``mode`` says what the numbers mean:

* ``"absolute"`` (default) -- the target IS the pose, expressed in the
  reference frame (default: the planning frame);
* ``"relative"`` -- the target is a displacement of the current tool pose,
  along the reference frame's axes (default: the tool's own axes), which is
  what ``ee_move`` / ``ee_rotate_*`` do. The translation moves the tool centre
  point; the rotation turns the tool about it. The displacement is taken from
  the MEASURED tool pose when the request arrives, and a newer motion replaces
  a running one, so two relative targets sent back to back do not add up: the
  second starts from wherever the first had got to. Send absolute targets for
  a sequence that must land exactly.

Everything leaves as a :class:`~cyberwave.driver.motion.scene.Pose` in the
planning frame, so a backend never sees a relative or foreign-frame target.
Pure numpy over the shared geometry core: no ROS, no pinocchio.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .guards import MotionRefusal
from .poses import pose_from_xyz_rpy, pose_to_transform, transform_to_pose
from .scene import Pose, SceneError, _finite

#: Aliases of the planning frame.
BASE_FRAMES = frozenset({"base"})
#: Aliases of the tool centre point.
TOOL_FRAMES = frozenset({"tool", "tcp"})

_UNIT_QUATERNION_TOLERANCE = 1e-3
_CODE = MotionRefusal.INVALID_REQUEST


class TargetMode(str, Enum):
    """Whether a target is a pose (``absolute``) or a displacement (``relative``)."""

    ABSOLUTE = "absolute"
    RELATIVE = "relative"

    @classmethod
    def parse(cls, value: object) -> TargetMode:
        """``None`` is ``absolute``; anything outside the closed set is refused."""
        if value is None:
            return cls.ABSOLUTE
        for member in cls:
            if member.value == value:
                return member
        raise SceneError(
            _CODE, f"mode must be one of {sorted(m.value for m in cls)}, got {value!r}"
        )


@dataclass(frozen=True)
class PoseSpec:
    """A target as the requester wrote it: numbers, and the frame they are in.

    ``reference_frame`` is ``None`` when the target names no frame of its own.
    """

    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    reference_frame: str | None = None


def _unit_quaternion(
    values: Any, what: str, code: MotionRefusal = _CODE
) -> tuple[float, float, float, float]:
    quaternion = tuple(_finite(v, what, code) for v in values)
    norm = math.sqrt(sum(component * component for component in quaternion))
    if abs(norm - 1.0) > _UNIT_QUATERNION_TOLERANCE:
        raise SceneError(
            code,
            f"{what} is not a unit quaternion (norm {norm:.4f}); "
            "send a normalised {x, y, z, w}",
        )
    return quaternion  # type: ignore[return-value]


def _frame_name(value: Any, what: str, code: MotionRefusal = _CODE) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SceneError(code, f"{what} must be a frame name")
    return value.strip()


def parse_pose_spec(
    value: Any, *, what: str = "pose", code: MotionRefusal = _CODE
) -> PoseSpec:
    """Read one target in any of the module's forms, refusing anything else.

    The one pose parser: motion targets, scene objects and payloads all read
    poses here. ``code`` is the refusal a malformed pose is reported as.
    """
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(part, (list, tuple)) for part in value)
    ):
        position_part, rotation_part = value
        if len(position_part) != 3 or len(rotation_part) not in (3, 4):
            raise SceneError(
                code,
                f"{what} as a pair must be [[x, y, z], [roll, pitch, yaw]] or "
                "[[x, y, z], [qx, qy, qz, qw]]",
            )
        value = [*position_part, *rotation_part]
    if isinstance(value, (list, tuple)):
        numbers = tuple(_finite(v, f"{what}[{i}]", code) for i, v in enumerate(value))
        if len(numbers) == 6:
            pose = pose_from_xyz_rpy(*numbers)
            return PoseSpec(pose.position, pose.orientation)
        if len(numbers) == 7:
            return PoseSpec(
                numbers[:3],  # type: ignore[arg-type]
                _unit_quaternion(numbers[3:], f"{what} quaternion", code),
            )
    elif isinstance(value, dict):
        position_data = value.get("position")
        if isinstance(position_data, dict):
            position = tuple(
                _finite(position_data.get(axis), f"{what}.position.{axis}", code)
                for axis in "xyz"
            )
        elif isinstance(position_data, (list, tuple)) and len(position_data) == 3:
            position = tuple(
                _finite(v, f"{what}.position", code) for v in position_data
            )
        else:
            raise SceneError(code, f"{what}.position must be {{x, y, z}} or [x, y, z]")
        if "orientation" in value and "rotation" in value:
            raise SceneError(code, f"{what}: send orientation or rotation, not both")
        orientation_data = value.get("orientation", value.get("rotation"))
        if not isinstance(orientation_data, dict):
            raise SceneError(
                code,
                f"{what} needs a rotation: orientation (or rotation) as a named "
                "{x, y, z, w} quaternion",
            )
        orientation = _unit_quaternion(
            (orientation_data.get(axis) for axis in "xyzw"),
            f"{what}.orientation",
            code,
        )
        if (
            "reference_frame" in value
            and "frame_id" in value
            and value["reference_frame"] != value["frame_id"]
        ):
            raise SceneError(code, f"{what}: reference_frame and frame_id disagree")
        frame = _frame_name(
            value.get("reference_frame", value.get("frame_id")),
            f"{what}.reference_frame",
            code,
        )
        return PoseSpec(position, orientation, frame)  # type: ignore[arg-type]
    raise SceneError(
        code,
        f"{what} must be a pose: [x, y, z, roll, pitch, yaw], "
        "[x, y, z, qx, qy, qz, qw], [[x, y, z], rotation] or "
        "{position, orientation}",
    )


def reference_frame(spec_frame: str | None, argument: Any) -> str | None:
    """The one frame a target is in: its own, or the request's ``reference_frame``.

    Both set and different is a contradiction, refused rather than resolved.
    """
    requested = _frame_name(argument, "reference_frame")
    if spec_frame is not None and requested is not None and spec_frame != requested:
        raise SceneError(
            _CODE,
            f"the target is in {spec_frame!r} but reference_frame is {requested!r}",
        )
    return spec_frame or requested


@dataclass(frozen=True)
class FrameSource:
    """What resolving a target reads from the arm, at its current posture.

    ``tcp`` is the tool centre point and ``lookup(name)`` any other frame of the
    robot description, both as 4x4 transforms in the planning frame.
    ``lookup`` raises ``KeyError`` / ``ValueError`` for a frame the description
    does not have.
    """

    planning_frame: str
    tcp: np.ndarray
    lookup: Callable[[str], np.ndarray]

    def transform(self, name: str | None) -> np.ndarray:
        """The planning-frame transform of *name* (aliases resolved)."""
        if name is None or name in BASE_FRAMES or name == self.planning_frame:
            return np.eye(4)
        if name in TOOL_FRAMES:
            return np.asarray(self.tcp, dtype=float)
        try:
            return np.asarray(self.lookup(name), dtype=float)
        except (KeyError, ValueError) as exc:
            raise SceneError(
                _CODE,
                f"reference_frame {name!r} is not a frame of the robot "
                f"description ({exc})",
            ) from None


def needs_frames(spec: PoseSpec, mode: TargetMode, frame: str | None) -> bool:
    """Does resolving this target read the arm at all?

    An absolute, fully specified pose in the planning frame does not, so a
    driver without forward kinematics can still be sent one.
    """
    return mode is TargetMode.RELATIVE or not (frame is None or frame in BASE_FRAMES)


def resolve_target(
    spec: PoseSpec,
    *,
    mode: TargetMode,
    frame: str | None,
    frames: FrameSource | None,
) -> Pose:
    """The planning-frame pose *spec* asks for, given the arm's current posture.

    ``frames`` may be ``None`` only when :func:`needs_frames` is ``False``;
    otherwise the target is refused with the reason.
    """
    if frames is None:
        if needs_frames(spec, mode, frame):
            raise SceneError(
                _CODE,
                "this target needs the arm's current pose (a relative target or "
                "a reference frame other than the planning frame), and this "
                "driver has no forward kinematics",
            )
        return Pose(spec.position, spec.orientation)
    if frame is not None and frame == frames.planning_frame:
        frame = None
    tcp = np.asarray(frames.tcp, dtype=float)
    if mode is TargetMode.RELATIVE:
        axes = frames.transform("tool" if frame is None else frame)[:3, :3]
        delta = pose_to_transform(Pose((0.0, 0.0, 0.0), spec.orientation))[:3, :3]
        target = np.eye(4)
        target[:3, 3] = tcp[:3, 3] + axes @ np.asarray(spec.position, dtype=float)
        target[:3, :3] = axes @ delta @ axes.T @ tcp[:3, :3]
        return transform_to_pose(target)
    target = frames.transform(frame) @ pose_to_transform(
        Pose(spec.position, spec.orientation)
    )
    return transform_to_pose(target)


def resolve_direction(
    vector: Any, *, frame: str | None, frames: FrameSource | None, what: str
) -> tuple[float, float, float]:
    """A direction given in *frame*, as a unit vector in the planning frame."""
    if isinstance(vector, dict):
        values = tuple(
            _finite(vector.get(axis), f"{what}.{axis}", _CODE) for axis in "xyz"
        )
    elif isinstance(vector, (list, tuple)) and len(vector) == 3:
        values = tuple(_finite(v, what, _CODE) for v in vector)
    else:
        raise SceneError(_CODE, f"{what} must be {{x, y, z}} or [x, y, z]")
    direction = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise SceneError(_CODE, f"{what} must be non-zero")
    if not (frame is None or frame in BASE_FRAMES):
        if frames is None:
            raise SceneError(
                _CODE,
                f"{what} is in {frame!r}, and this driver has no forward "
                "kinematics to read that frame",
            )
        if frame != frames.planning_frame:
            direction = frames.transform(frame)[:3, :3] @ direction
    direction /= norm
    return (float(direction[0]), float(direction[1]), float(direction[2]))


__all__ = [
    "BASE_FRAMES",
    "TOOL_FRAMES",
    "FrameSource",
    "PoseSpec",
    "TargetMode",
    "needs_frames",
    "parse_pose_spec",
    "reference_frame",
    "resolve_direction",
    "resolve_target",
]
