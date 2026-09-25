"""Proto mirror: space/cartesian/cartesian_state.proto — CartesianPose."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from ..geometry.primitives import Quaterniond, Vector3d
from .spatial_state import SpatialState


@dataclass(frozen=True, slots=True)
class CartesianPose:
    spatial_state: SpatialState
    position: Vector3d
    orientation: Quaterniond

    def frame_id(self) -> str:
        return self.spatial_state.reference_frame

    def translation_dict(self) -> dict[str, float]:
        return self.position.as_dict()

    def orientation_dict(self) -> dict[str, float]:
        return self.orientation.as_dict()

    def to_legacy_pose(self) -> dict[str, dict[str, float]]:
        return {
            "position": self.translation_dict(),
            "rotation": self.orientation_dict(),
        }

    def copy(self) -> CartesianPose:
        return copy.deepcopy(self)
