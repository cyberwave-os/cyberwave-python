"""Config-driven manipulator (robot-arm) kinematics (pinocchio underneath, lazy-imported).

Install pinocchio via the ``drivers`` extra::

    pip install "cyberwave[drivers]"
"""

from __future__ import annotations

from .config import ArmKinematicsConfig, GripperConfig

__all__ = ["ArmKinematicsConfig", "GripperConfig", "BaseKinematicsManipulator"]


def __getattr__(name: str):
    # Lazy re-export so importing the package does not import pinocchio until
    # BaseKinematicsManipulator is actually referenced.
    if name == "BaseKinematicsManipulator":
        from .base import BaseKinematicsManipulator

        return BaseKinematicsManipulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
