"""Robot/joint control: pure math plus the driver-facing mixins that use it.

``types.py`` / ``motion.py`` / ``controller.py`` / ``command_vectors.py`` are
pure math (no ROS, no MQTT, no kinematics import at module level — scipy is
lazy inside ``trapezoidal_motion``). ``arm.py`` / ``cartesian.py`` /
``dual_arm.py`` / ``joint_controller.py`` / ``joint_teleop.py`` are the
mixins a driver composes onto a ``BaseDriver`` subclass; they import
``ArmKinematicsConfig`` (numpy-only) and ``interface/`` but never scipy,
rclpy, or pinocchio at module level.

Planned arm motion, shared by every arm driver:

* ``cartesian_request.py`` -- the Cartesian request funnel and its gates;
* ``local_planner.py`` -- the in-process straight-line / PTP planner on
  :class:`~cyberwave.driver.kinematics.arm.ArmController` (pinocchio and scipy
  only when a plan is actually built);
* ``command_boundary.py`` / ``arbiter.py`` -- the pure single-writer command
  arbiter for arms driven through a position-stream interface (its ROS node is
  :mod:`cyberwave.driver.ros2.arbiter_node`);
* ``world_obstacles.py`` -- runtime world obstacles persisted in twin metadata;
* ``planned_motion.py`` -- :class:`PlannedMotionMixin`: ``set_pose`` /
  ``grasp`` / ``scene_set`` / ``attach_payload`` / ``detach_payload``,
  ``ee_pose`` and ``state/motion``, answered by the ``kinematics_backend``
  the driver names (the built-in ``"pinocchio"``, or a backend it registers
  in ``_motion_backend_factories``) or the one a request names.
"""

from .arbiter import (
    GoalState,
    MotionArbiterCore,
    MotionOwner,
    ResetOutcome,
    ResetRefusal,
)
from .arm import ArmCapabilityMixin
from .cartesian import EECartesianPoseMixin, EEPose
from .cartesian_request import (
    CartesianDecision,
    CartesianPlanner,
    CartesianRejection,
    CartesianRequest,
    Preconditions,
    UnavailablePlanner,
    validate_cartesian_request,
)
from .cartesian_state import CartesianStateMixin
from .command_boundary import (
    AdaptationResult,
    CommandRejection,
    CommandScopes,
    MotionSource,
    VendorAdapter,
    VendorJointCommand,
    command_stamp_is_fresh,
    joint_state_adapter,
    validate_command,
)
from .command_vectors import resolve_command_vector, resolve_effort_vector
from .controller import JointController, waypoint_command
from .dual_arm import DualArmConfig, DualArmMixin
from .joint_controller import JointControllerMixin
from .joint_teleop import HomePositionNotDefinedError, JointCommandBufferMixin
from .local_planner import LocalCartesianPlanner
from .motion import trapezoidal_motion
from .planned_motion import PlannedMotionMixin
from .types import (
    JointCommand,
    JointControllerConfig,
    MotionGenerator,
    MotionRequest,
    TrajectoryPlan,
    TrajectoryWaypoint,
)
from .world_obstacles import WorldObstaclesMixin

__all__ = [
    "AdaptationResult",
    "ArmCapabilityMixin",
    "CartesianDecision",
    "CartesianPlanner",
    "CartesianRejection",
    "CartesianRequest",
    "CommandRejection",
    "CommandScopes",
    "GoalState",
    "LocalCartesianPlanner",
    "MotionArbiterCore",
    "MotionOwner",
    "MotionSource",
    "PlannedMotionMixin",
    "Preconditions",
    "ResetOutcome",
    "ResetRefusal",
    "UnavailablePlanner",
    "VendorAdapter",
    "VendorJointCommand",
    "WorldObstaclesMixin",
    "command_stamp_is_fresh",
    "joint_state_adapter",
    "validate_cartesian_request",
    "validate_command",
    "CartesianStateMixin",
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
