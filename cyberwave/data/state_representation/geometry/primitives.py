"""Proto mirror: geometry/primitives.proto."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Vector3d:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass(frozen=True, slots=True)
class Quaterniond:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z, "w": self.w}
