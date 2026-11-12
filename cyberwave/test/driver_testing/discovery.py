"""What a twin can do, read from the twin itself before anything moves.

:func:`profile_from` is pure: it turns the four things a twin reports (its
capabilities, its MQTT command catalog, its universal schema and one
``state/motion`` payload) into a :class:`TwinProfile`. :func:`discover` reads
those from a live :class:`~cyberwave.twin.Twin`. The section registry plans on
the profile alone, so a test can hand it a fake twin's data.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Joint types a driver can command.
_CONTROLLABLE = frozenset({"revolute", "prismatic", "continuous"})
#: A joint whose name says it is part of a gripper.
_GRIPPER_NAME = re.compile(r"grip|finger|jaw|hand|claw|knuckle", re.IGNORECASE)


@dataclass(frozen=True)
class JointInfo:
    """One controllable joint of the universal schema."""

    name: str
    type: str
    lower: float | None = None
    upper: float | None = None
    home: float | None = None

    @property
    def limited(self) -> bool:
        return (
            self.lower is not None
            and self.upper is not None
            and self.lower < self.upper
        )


@dataclass
class TwinProfile:
    """Everything the registry needs to choose sections, and nothing live."""

    uuid: str
    capabilities: dict[str, Any] = field(default_factory=dict)
    #: Command names in the catalog's ``commands.supported``.
    commands: frozenset[str] = frozenset()
    #: Arguments per catalog command: ``{name: {arg: default}}``.
    command_args: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Topic families the catalog declares: ``joints`` (``joint/update``),
    #: ``pose`` (``twin/pose``), ``kinematics`` (``state/kinematics``),
    #: ``motion`` (``state/motion``) and ``navigate`` (``navigate/*``).
    topics: frozenset[str] = frozenset()
    arm_joints: tuple[JointInfo, ...] = ()
    gripper_joints: tuple[JointInfo, ...] = ()
    #: The driver's default planned-motion backend (``state/motion``).
    default_backend: str | None = None
    backend_available: bool = False
    #: Every backend a planned section runs on: the default, then any asked for.
    backends: tuple[str, ...] = ()
    motion_status: dict[str, Any] | None = None

    @property
    def has_planned_motion(self) -> bool:
        return "set_pose" in self.commands and "motion" in self.topics

    @property
    def gripper_joint(self) -> str | None:
        return self.gripper_joints[0].name if self.gripper_joints else None

    def joint(self, name: str) -> JointInfo | None:
        for joint in (*self.arm_joints, *self.gripper_joints):
            if joint.name == name:
                return joint
        return None

    def home(self) -> dict[str, float] | None:
        """The schema's home posture, when every arm joint declares one."""
        if not self.arm_joints or any(j.home is None for j in self.arm_joints):
            return None
        return {j.name: float(j.home) for j in self.arm_joints}  # type: ignore[arg-type]

    def to_json(self) -> dict[str, Any]:
        return {
            "uuid": self.uuid,
            "commands": sorted(self.commands),
            "topics": sorted(self.topics),
            "arm_joints": [j.__dict__ for j in self.arm_joints],
            "gripper_joints": [j.__dict__ for j in self.gripper_joints],
            "default_backend": self.default_backend,
            "backend_available": self.backend_available,
            "backends": list(self.backends),
        }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def schema_joints(universal_schema: Mapping[str, Any] | None) -> list[JointInfo]:
    """Controllable joints with their limits and home, in schema order."""
    joints = []
    for raw in (universal_schema or {}).get("joints", []) or []:
        if not isinstance(raw, Mapping) or not raw.get("name"):
            continue
        kind = str(raw.get("type", "")).lower()
        if kind not in _CONTROLLABLE:
            continue
        limits = raw.get("limits") or raw.get("limit") or {}
        if not isinstance(limits, Mapping):
            limits = {}
        joints.append(
            JointInfo(
                name=str(raw["name"]),
                type=kind,
                lower=_number(limits.get("lower")) if kind != "continuous" else None,
                upper=_number(limits.get("upper")) if kind != "continuous" else None,
                home=_number(raw.get("home_position")),
            )
        )
    return joints


def split_joints(
    joints: Sequence[JointInfo],
    *,
    arm: Iterable[str] | None = None,
    gripper: Iterable[str] | None = None,
) -> tuple[tuple[JointInfo, ...], tuple[JointInfo, ...]]:
    """Arm and gripper joints: named explicitly, or told apart by name and type.

    Without names, a joint named like a gripper part, or any prismatic joint
    after a revolute one, is a gripper joint; everything else is the arm.
    """
    by_name = {j.name: j for j in joints}
    if arm is not None or gripper is not None:
        arm_names = list(arm) if arm is not None else []
        grip_names = list(gripper) if gripper is not None else []
        if arm is None:
            arm_names = [j.name for j in joints if j.name not in grip_names]
        if gripper is None:
            grip_names = [
                j.name
                for j in joints
                if j.name not in arm_names and _is_gripper(j, joints)
            ]
        missing = [n for n in (*arm_names, *grip_names) if n not in by_name]
        if missing:
            raise ValueError(f"Joints not in the twin's universal schema: {missing}")
        return tuple(by_name[n] for n in arm_names), tuple(
            by_name[n] for n in grip_names
        )
    gripper_joints = tuple(j for j in joints if _is_gripper(j, joints))
    arm_joints = tuple(j for j in joints if j not in gripper_joints)
    return arm_joints, gripper_joints


def _is_gripper(joint: JointInfo, joints: Sequence[JointInfo]) -> bool:
    if _GRIPPER_NAME.search(joint.name):
        return True
    revolute_before = any(
        j.type in ("revolute", "continuous")
        for j in joints[: list(joints).index(joint)]
    )
    return joint.type == "prismatic" and revolute_before


def catalog_commands(catalog: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """``{command: {arg: default}}`` from the catalog's ``commands.supported``."""
    supported = (catalog.get("commands") or {}).get("supported") or []
    commands: dict[str, dict[str, Any]] = {}
    for entry in supported:
        if isinstance(entry, str):
            commands[entry] = {}
        elif isinstance(entry, Mapping) and entry.get("name"):
            args = {}
            for arg in entry.get("args") or []:
                if isinstance(arg, Mapping) and arg.get("name"):
                    args[str(arg["name"])] = arg.get("default")
            commands[str(entry["name"])] = args
    return commands


def catalog_topics(catalog: Mapping[str, Any]) -> frozenset[str]:
    """The feedback and mission topic families a catalog declares."""
    found = set()

    def has(group: str, leaf: str) -> bool:
        block = catalog.get(group)
        return isinstance(block, Mapping) and isinstance(block.get(leaf), Mapping)

    # The backend serves a flat topics map; local manifests use nested blocks.
    paths = (catalog.get("topics") or {}).keys()
    if has("joint", "update") or any(
        re.fullmatch(r"cyberwave/joint/[^/]+/update", p) for p in paths
    ):
        found.add("joints")
    if has("twin", "pose") or any(
        re.fullmatch(r"cyberwave/twin/[^/]+/pose", p) for p in paths
    ):
        found.add("pose")
    if has("state", "kinematics") or any(
        p.endswith("/state/kinematics") for p in paths
    ):
        found.add("kinematics")
    if has("state", "motion") or any(p.endswith("/state/motion") for p in paths):
        found.add("motion")
    if (isinstance(catalog.get("navigate"), Mapping) and catalog["navigate"]) or any(
        "/navigate/" in p for p in paths
    ):
        found.add("navigate")
    return frozenset(found)


def profile_from(
    uuid: str,
    *,
    capabilities: Mapping[str, Any] | None = None,
    catalog: Mapping[str, Any] | None = None,
    universal_schema: Mapping[str, Any] | None = None,
    motion_status: Mapping[str, Any] | None = None,
    observed_topics: Iterable[str] = (),
    arm_joints: Iterable[str] | None = None,
    gripper_joints: Iterable[str] | None = None,
    extra_backends: Iterable[str] = (),
) -> TwinProfile:
    """Build the profile the registry plans on, from what the twin reports.

    ``supported_commands`` may stand in for a full catalog: pass
    ``catalog={"commands": {"supported": [...]}}``.
    """
    catalog = catalog or {}
    commands = catalog_commands(catalog)
    arm, gripper = split_joints(
        schema_joints(universal_schema), arm=arm_joints, gripper=gripper_joints
    )
    status = dict(motion_status) if motion_status else None
    default_backend = (status or {}).get("kinematics_backend") or None
    backends: list[str] = []
    for name in (default_backend, *extra_backends):
        if name and name not in backends:
            backends.append(str(name))
    return TwinProfile(
        uuid=str(uuid),
        capabilities=dict(capabilities or {}),
        commands=frozenset(commands),
        command_args=commands,
        topics=catalog_topics(catalog) | frozenset(observed_topics),
        arm_joints=arm,
        gripper_joints=gripper,
        default_backend=default_backend,
        backend_available=bool((status or {}).get("backend_available")),
        backends=tuple(backends),
        motion_status=status,
    )


def discover(
    twin: Any,
    *,
    motion_status: Mapping[str, Any] | None = None,
    observed_topics: Iterable[str] = (),
    arm_joints: Iterable[str] | None = None,
    gripper_joints: Iterable[str] | None = None,
    extra_backends: Iterable[str] = (),
) -> TwinProfile:
    """Read a live twin's capabilities, catalog and universal schema.

    *motion_status* is the latest ``state/motion`` payload, which the caller
    subscribes for (the driver republishes it at least every 5 s while idle).
    """
    from ...twin.capabilities.joints import _local_universal_schema

    try:
        catalog = twin.commands.get_schema()
    except Exception:  # noqa: BLE001 - a twin without a catalog plans no command sections
        catalog = {}
    if not (catalog.get("commands") or {}).get("supported"):
        try:
            names = twin.commands.get_supported_commands()
        except Exception:  # noqa: BLE001
            names = []
        catalog = {**catalog, "commands": {"supported": list(names)}}
    return profile_from(
        twin.uuid,
        capabilities=twin.capabilities,
        catalog=catalog,
        universal_schema=_local_universal_schema(twin),
        motion_status=motion_status,
        observed_topics=observed_topics,
        arm_joints=arm_joints,
        gripper_joints=gripper_joints,
        extra_backends=extra_backends,
    )
