"""Planning-scene intermediate representation shared by every arm planner.

A robot description contains no desk, box, or target object, so a planner given
nothing routes confidently through a table it cannot see. This module is how the
world gets in: a requester describes it as conservative primitives in the arm's
planning frame, and every one of them is validated here before any planner or
collision model sees it.

Pure by construction. No ROS import, so the whole validation surface -- which
payload is refused and with which code -- runs in the unit suite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from .guards import MotionRefusal, Refusal

#: The default frame scene geometry and motion targets are expressed in.
#:
#: A driver whose planning frame differs passes ``planning_frame=`` to
#: :func:`parse_scene_set`. Whatever the frame, it is ONE frame: a scene half in
#: the base and half in a tool frame is a collision decision nobody can check,
#: and a driver publishes no TF for an arbitrary requester frame. Anything else
#: is refused.
PLANNING_FRAME = "base_link"

PrimitiveKind = Literal["box", "cylinder", "sphere"]

#: ``shape_msgs/SolidPrimitive`` dimension counts. Meshes are deliberately absent:
#: a mesh on the wire is unbounded input reaching a collision checker, and every
#: obstacle this task needs (desk, box walls, object bounds) is over-boundable by
#: a primitive. Over-bounding is the safe direction -- an oversized obstacle
#: refuses plans, an absent one permits bad ones.
_DIMENSION_COUNT: dict[str, int] = {"box": 3, "cylinder": 2, "sphere": 1}


class SceneError(ValueError):
    """A wire payload that cannot become a scene, carrying its closed-set refusal.

    An exception rather than a return value because parsing is recursive (scene,
    object, pose) and threading ``Refusal | None`` through every level would
    bury the first real failure. The caller converts ``.refusal`` back into the
    same reported shape every other guard produces.
    """

    def __init__(self, code: MotionRefusal, detail: str) -> None:
        super().__init__(detail)
        self.refusal = Refusal(code, detail)


@dataclass(frozen=True)
class Pose:
    """A pose in the planning frame: metres, and an explicitly ordered quaternion.

    Named ``(x, y, z, w)`` on the wire and stored in that order. The convention
    is stated because the alternative ordering is the classic silent bug, and
    because the driver's own ``EEPose`` path is extrinsic-XYZ RPY -- the two meet
    in ``requests.pose_from_xyz_rpy``.
    """

    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]


@dataclass(frozen=True)
class Primitive:
    """One world collision object: an id, a shape, and where it is."""

    id: str
    kind: PrimitiveKind
    dimensions: tuple[float, ...]
    pose: Pose


@dataclass(frozen=True)
class SceneRequest:
    """One immutable observation of the world, addressed by its revision.

    ``allow_touch`` is the mechanism that makes a grasp plannable at all: with
    the target object in the scene, NO path to a pose at the target is collision
    free, because the fingers must reach it. Listing the target here allows the
    gripper links to contact that one object, and nothing else.
    """

    revision: str
    objects: tuple[Primitive, ...]
    allow_touch: frozenset[str]


@dataclass(frozen=True)
class AttachRequest:
    """An inflated payload proxy to carry on the tool frame after a verified grasp."""

    object_id: str
    dimensions: tuple[float, float, float]
    #: Relative to the tool frame, not to the planning frame: the payload travels with
    #: the gripper, so its pose is fixed in the tool frame.
    pose: Pose


def _finite(value: Any, what: str, code: MotionRefusal) -> float:
    """Coerce to float and reject NaN/inf, naming the field that was wrong."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise SceneError(code, f"{what} is not a number: {value!r}") from None
    if not math.isfinite(number):
        raise SceneError(code, f"{what} is not finite: {value!r}")
    return number


def parse_pose(
    data: Any, *, code: MotionRefusal = MotionRefusal.INVALID_REQUEST
) -> Pose:
    """A scene-object or payload pose, in every form ``set_pose`` accepts.

    Built on :func:`~cyberwave.driver.motion.targets.parse_pose_spec`, so there
    is one parser and one set of accepted forms. These poses are already in
    their fixed frame (the planning frame, or the tool frame for a payload),
    so a pose that names a ``reference_frame`` of its own is refused.
    """
    from .targets import parse_pose_spec  # targets imports this module

    spec = parse_pose_spec(data, code=code)
    if spec.reference_frame is not None:
        raise SceneError(
            code,
            f"pose: reference_frame {spec.reference_frame!r} is not accepted here; "
            "scene and payload poses are in their fixed frame",
        )
    return Pose(spec.position, spec.orientation)


def _parse_primitive(data: Any) -> Primitive:
    code = MotionRefusal.INVALID_SCENE
    if not isinstance(data, dict) or not data.get("id"):
        raise SceneError(code, "every scene object needs a non-empty id")
    identifier = str(data["id"])
    kind = data.get("primitive")
    if kind not in _DIMENSION_COUNT:
        raise SceneError(
            code,
            f"object {identifier!r}: unknown primitive {kind!r}; "
            f"use one of {sorted(_DIMENSION_COUNT)}",
        )
    raw = data.get("dimensions")
    if not isinstance(raw, (list, tuple)) or len(raw) != _DIMENSION_COUNT[kind]:
        raise SceneError(
            code,
            f"object {identifier!r}: {kind} takes {_DIMENSION_COUNT[kind]} "
            f"dimension(s), got {raw!r}",
        )
    dimensions = tuple(
        _finite(d, f"object {identifier!r} dimension", code) for d in raw
    )
    if any(d <= 0.0 for d in dimensions):
        raise SceneError(code, f"object {identifier!r}: dimensions must be positive")
    return Primitive(
        identifier, kind, dimensions, parse_pose(data.get("pose"), code=code)
    )


def parse_scene_set(
    data: Any,
    *,
    planning_frame: str = PLANNING_FRAME,
    default_revision: str = "",
) -> SceneRequest:
    """Validate one ``scene_set`` payload completely, or refuse all of it.

    Every argument has a default: ``scene_revision`` is *default_revision*
    (the driver numbers scenes itself), ``objects`` and ``allow_touch`` are
    empty, ``frame`` is the planning frame. ``scene_set`` with no data
    therefore clears the task scene.

    Partial application is never attempted: a scene half-applied is a collision
    model nobody described, and the revision that names it would be a lie.
    """
    code = MotionRefusal.INVALID_SCENE
    if not isinstance(data, dict):
        raise SceneError(code, "scene_set data must be an object")
    revision = data.get("scene_revision")
    if revision is None:
        revision = default_revision
    if not isinstance(revision, str) or not revision:
        raise SceneError(code, "scene_revision must be a non-empty string")
    # Absent means the planning frame; an explicit null names no frame at all.
    frame = data.get("frame", planning_frame)
    if frame != planning_frame:
        raise SceneError(code, f"frame must be {planning_frame!r}, got {frame!r}")
    raw_objects = data.get("objects")
    if raw_objects is None:
        raw_objects = []
    if not isinstance(raw_objects, list):
        raise SceneError(code, "objects must be a list (an empty list is allowed)")
    objects = tuple(_parse_primitive(entry) for entry in raw_objects)
    identifiers = [entry.id for entry in objects]
    if len(set(identifiers)) != len(identifiers):
        raise SceneError(code, "duplicate object ids in one scene")
    allow_touch = data.get("allow_touch") or []
    if not isinstance(allow_touch, list) or not all(
        isinstance(a, str) for a in allow_touch
    ):
        raise SceneError(code, "allow_touch must be a list of object ids")
    unknown = sorted(set(allow_touch) - set(identifiers))
    if unknown:
        raise SceneError(
            code, f"allow_touch names objects that are not in this scene: {unknown}"
        )
    return SceneRequest(revision, objects, frozenset(allow_touch))


#: ``attach_payload`` / ``detach_payload`` without an ``object_id``.
DEFAULT_PAYLOAD_ID = "payload"
#: ``attach_payload`` without a ``pose``: centred on the tool centre point.
TOOL_ORIGIN = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def parse_attach(data: Any) -> AttachRequest:
    """Validate one ``attach_payload`` payload.

    ``dimensions`` is the one required argument. ``object_id`` defaults to
    ``"payload"`` and ``pose`` (relative to the TCP) to the TCP itself.

    Dimensions are taken as given and NOT inflated here: the executor measured
    the object and chose its own margin, and a second inflation applied silently
    downstream would make the carried proxy disagree with the one it reasoned
    about.
    """
    code = MotionRefusal.INVALID_REQUEST
    if not isinstance(data, dict):
        raise SceneError(code, "attach_payload data must be an object")
    object_id = data.get("object_id")
    if object_id is None:
        object_id = DEFAULT_PAYLOAD_ID
    if not isinstance(object_id, str) or not object_id:
        raise SceneError(code, "object_id must be a non-empty string")
    raw = data.get("dimensions")
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise SceneError(code, "attach_payload dimensions must be [x, y, z] in metres")
    dimensions = tuple(_finite(d, "payload dimension", code) for d in raw)
    if any(d <= 0.0 for d in dimensions):
        raise SceneError(code, "payload dimensions must be positive")
    raw_pose = data.get("pose")
    return AttachRequest(
        object_id,
        dimensions,  # type: ignore[arg-type]
        TOOL_ORIGIN if raw_pose is None else parse_pose(raw_pose, code=code),
    )


# ── what the active planner holds, whichever planner it is ──────────────────


@dataclass(frozen=True)
class SceneDiff:
    """What must change in a planner's world to reach a requested scene.

    A diff rather than a replacement because an external planner's scene (e.g.
    MoveIt's, applied as a diff) must be updated without discarding the robot
    state, the attached payload, or anything another component put there. ``remove_ids`` covers only ids
    this driver previously added, so a scene update cannot delete geometry it
    does not own. A backend that rebuilds its whole collision model (the
    in-process one) reads the :class:`SceneRequest` instead and ignores this.
    """

    frame: str
    remove_ids: tuple[str, ...]
    add: tuple[Primitive, ...]
    allow_touch: frozenset[str]
    revoke_touch: tuple[str, ...]


class SceneTracker:
    """What this driver believes its planner holds, and which ids it owns.

    Split into ``plan_*`` and ``commit_*`` on purpose: the tracker advances only
    after the planner acknowledges the change. A tracker that updated on send
    would report an active revision for a scene that was never applied, and the
    next motion would pass its revision guard against a world that does not
    exist. One tracker per driver, whichever backend answers.
    """

    def __init__(self, frame: str = PLANNING_FRAME) -> None:
        self.frame = frame
        self.active_revision: str | None = None
        self.attached_payload: str | None = None
        #: Ids this driver added. Never anything else -- a scene update must not
        #: delete geometry another component put in the scene.
        self._owned_ids: set[str] = set()
        self._touch_ids: set[str] = set()

    def plan_update(self, request: SceneRequest) -> SceneDiff:
        """The diff for *request*, without committing to it."""
        return SceneDiff(
            frame=self.frame,
            remove_ids=tuple(sorted(self._owned_ids)),
            add=request.objects,
            allow_touch=request.allow_touch,
            revoke_touch=tuple(sorted(self._touch_ids - request.allow_touch)),
        )

    def commit_update(self, request: SceneRequest) -> None:
        """Record *request* as applied. Called only after the planner acknowledges."""
        self._owned_ids = {entry.id for entry in request.objects}
        self._touch_ids = set(request.allow_touch)
        self.active_revision = request.revision

    @property
    def touch_ids(self) -> tuple[str, ...]:
        """Target objects whose contact allowance belongs to the active revision."""
        return tuple(sorted(self._touch_ids))

    def owns(self, object_id: str) -> bool:
        """Is *object_id* currently a world object this driver put there?"""
        return object_id in self._owned_ids

    def check_attach(self, request: AttachRequest) -> Refusal | None:
        """At most one payload. Two would mean the robot model carries an object
        the gripper is not holding, and every later plan is wrong by its volume."""
        if self.attached_payload is not None:
            return Refusal(
                MotionRefusal.PAYLOAD_ALREADY_ATTACHED,
                f"{self.attached_payload!r} is already attached; detach it first",
            )
        return None

    def commit_attach(self, request: AttachRequest) -> None:
        """The object leaves the world and becomes part of the robot."""
        self.attached_payload = request.object_id
        self._owned_ids.discard(request.object_id)
        self._touch_ids.discard(request.object_id)

    def check_detach(self, object_id: str) -> Refusal | None:
        if not object_id or self.attached_payload != object_id:
            return Refusal(
                MotionRefusal.INVALID_REQUEST,
                f"no payload {object_id!r} is attached "
                f"(attached: {self.attached_payload!r})",
            )
        return None

    def commit_detach(self) -> None:
        """The payload is gone from the robot AND from the world.

        It is not re-added as a world object: after a release the executor knows
        where it actually landed, and its next ``scene_set`` says so. Putting it
        back at the carried pose would assert a position nobody observed.
        """
        self.attached_payload = None
