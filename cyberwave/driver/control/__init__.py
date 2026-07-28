"""Robot/joint control: pure math plus the driver-facing mixins that use it.

``types.py`` / ``motion.py`` / ``controller.py`` / ``command_vectors.py`` are
pure math (no ROS, no MQTT, no kinematics import at module level — scipy is
lazy inside ``trapezoidal_motion``). ``arm.py`` / ``cartesian.py`` /
``dual_arm.py`` / ``joint_controller.py`` / ``joint_teleop.py`` are the
mixins a driver composes onto a ``BaseDriver`` subclass; they import
``ArmKinematicsConfig`` (numpy-only) and ``interface/`` but never scipy,
rclpy, or pinocchio at module level.
"""

from .arm import ArmCapabilityMixin
from .cartesian import EECartesianPoseMixin, EEPose
from .command_vectors import resolve_command_vector, resolve_effort_vector
from .controller import JointController, waypoint_command
from .dual_arm import DualArmConfig, DualArmMixin
from .joint_controller import JointControllerMixin
from .joint_teleop import HomePositionNotDefinedError, JointCommandBufferMixin
from .motion import trapezoidal_motion
from .types import (
    JointCommand,
    JointControllerConfig,
    MotionGenerator,
    MotionRequest,
    TrajectoryPlan,
    TrajectoryWaypoint,
)

__all__ = [
    "ArmCapabilityMixin",
    "EECartesianPoseMixin",
    "EEPose",
    "DualArmConfig",
    "DualArmMixin",
    "HomePositionNotDefinedError",
    "JointCommand",
    "JointCommandBufferMixin",
    "JointController",
    "JointControllerConfig",
    "JointControllerMixin",
    "MotionGenerator",
    "MotionRequest",
    "TrajectoryPlan",
    "TrajectoryWaypoint",
    "resolve_command_vector",
    "resolve_effort_vector",
    "trapezoidal_motion",
    "waypoint_command",
]
