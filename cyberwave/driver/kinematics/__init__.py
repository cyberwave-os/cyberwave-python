"""Config-driven kinematics for robot control (pinocchio underneath, lazy-imported).

Manipulator (robot-arm) kinematics live under :mod:`cyberwave.driver.kinematics.arm`
and are re-exported here for convenience. Other robot classes (e.g. mobile
bases) get their own sibling package under :mod:`cyberwave.driver.kinematics`
as support is added — nothing at this top level is manipulator-specific.

Install pinocchio via the ``drivers`` extra::

    pip install "cyberwave[drivers]"
"""

from __future__ import annotations

from .arm import ArmKinematicsConfig, GripperConfig

__all__ = ["ArmKinematicsConfig", "GripperConfig", "BaseKinematicsManipulator"]


def __getattr__(name: str):
    # Lazy re-export so importing the package does not import pinocchio until
    # BaseKinematicsManipulator is actually referenced.
    if name == "BaseKinematicsManipulator":
        from .arm import BaseKinematicsManipulator

        return BaseKinematicsManipulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
