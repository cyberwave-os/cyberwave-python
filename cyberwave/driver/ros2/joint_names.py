"""JointNameMap — bidirectional ros↔platform joint renaming + mimic expansion.

Replaces per-driver hand-rolled joint renaming and mimic-expansion helpers:
the mimic relationship is derived from the arm's ``GripperConfig.mimic`` so
there is a single source of truth. Pure data + dict math; ordered arrays
appear only at the ROS message boundary via ``to_ros``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..kinematics import ArmKinematicsConfig


def _immutable(mapping: Mapping) -> Mapping:
    return MappingProxyType(dict(mapping)) if mapping else MappingProxyType({})


@dataclass(frozen=True)
class JointNameMap:
    """Rename ROS joint names to platform names and expand mimic joints.

    ``mimic`` maps a passive mimic joint to ``(source_platform_joint, factor)``;
    on ``to_platform`` the mimic value is ``factor * source`` whenever the
    source is present.
    """

    ros_to_platform: Mapping[str, str] = field(default_factory=dict)
    mimic: Mapping[str, tuple[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ros_to_platform", _immutable(self.ros_to_platform))
        object.__setattr__(self, "mimic", _immutable(self.mimic))

    @classmethod
    def from_arm_config(
        cls,
        config: "ArmKinematicsConfig",
        ros_to_platform: Mapping[str, str] | None = None,
    ) -> "JointNameMap":
        """Derive mimic expansion from ``config.gripper.mimic`` (factor is of the
        first commanded gripper joint), keeping GripperConfig the single source
        of truth for the mimic relationship."""
        mimic: dict[str, tuple[str, float]] = {}
        g = config.gripper
        if g is not None and g.names:
            source = g.names[0]
            mimic = {m: (source, factor) for m, factor in g.mimic.items()}
        return cls(ros_to_platform=ros_to_platform or {}, mimic=mimic)

    @property
    def _platform_to_ros(self) -> dict[str, str]:
        return {p: r for r, p in self.ros_to_platform.items()}

    def to_platform(
        self, names: list[str], positions: list[float]
    ) -> dict[str, float]:
        """ROS feedback arrays -> platform dict (rename, then expand mimics)."""
        out = {
            self.ros_to_platform.get(n, n): float(p)
            for n, p in zip(names, positions)
        }
        for mimic_joint, (source, factor) in self.mimic.items():
            if source in out:
                out[mimic_joint] = factor * out[source]
        return out

    def to_ros(
        self,
        positions: Mapping[str, float],
        *,
        order: tuple[str, ...],
        default: float = 0.0,
    ) -> tuple[list[str], list[float]]:
        """Platform dict -> ordered ROS name/position arrays (mimics excluded
        unless explicitly listed in *order*)."""
        rename = self._platform_to_ros
        names = [rename.get(j, j) for j in order]
        values = [float(positions.get(j, default)) for j in order]
        return names, values
