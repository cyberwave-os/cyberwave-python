"""Config-driven manipulator (robot-arm) kinematics (pinocchio underneath, lazy-imported).

Install pinocchio via the ``drivers`` extra::

    pip install "cyberwave[drivers]"
"""

from __future__ import annotations

from .config import ArmKinematicsConfig, GripperConfig, tool_offset_from_xyz_rpy
from .description import (
    CompatibilityDecision,
    CompatibilityVerdict,
    DescriptionSource,
    ResolvedDescription,
    evaluate_hardware_compatibility,
    resolve_runtime_description,
    urdf_fingerprint,
)
from .limits import (
    CartesianLimitBases,
    ControllerVelocityProfile,
    MotionLimitError,
    MotionLimitRefusal,
    PlanningLimits,
    ResolvedMotionLimits,
    SpeedScaleQuantization,
    planning_limits_from,
    resolve_motion_limits,
    urdf_joint_limits,
)
from .trajectory import (
    LocalTrajectory,
    PosePathPreflightError,
    WaypointSampling,
    smooth_trajectory,
    time_parameterise,
    waypoint_count,
)

__all__ = [
    "ArmKinematicsConfig",
    "GripperConfig",
    "BaseKinematicsManipulator",
    "CartesianLimitBases",
    "CollisionAwareKinematics",
    "CompatibilityDecision",
    "CompatibilityVerdict",
    "ControllerVelocityProfile",
    "DescriptionSource",
    "LocalTrajectory",
    "MotionLimitError",
    "MotionLimitRefusal",
    "PlanningLimits",
    "PosePathPreflightError",
    "ResolvedDescription",
    "ResolvedMotionLimits",
    "SpeedScaleQuantization",
    "WaypointSampling",
    "evaluate_hardware_compatibility",
    "planning_limits_from",
    "resolve_motion_limits",
    "resolve_runtime_description",
    "smooth_trajectory",
    "time_parameterise",
    "urdf_fingerprint",
    "urdf_joint_limits",
    "waypoint_count",
    "tool_offset_from_xyz_rpy",
]


def __getattr__(name: str):
    # Lazy re-export so importing the package does not import pinocchio until
    # BaseKinematicsManipulator is actually referenced.
    if name == "BaseKinematicsManipulator":
        from .base import BaseKinematicsManipulator

        return BaseKinematicsManipulator
    if name == "CollisionAwareKinematics":
        from .collision import CollisionAwareKinematics

        return CollisionAwareKinematics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
