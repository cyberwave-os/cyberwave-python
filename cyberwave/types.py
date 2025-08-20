from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Pose2D:
    x: float
    y: float


@dataclass
class Pose3D:
    x: float
    y: float
    z: float


@dataclass
class JointTargets:
    positions: List[float]


