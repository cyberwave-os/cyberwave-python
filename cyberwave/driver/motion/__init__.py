"""Robot-agnostic planned-motion layer for arm drivers.

``state/motion`` status, the closed set of motion refusals with the guards that
produce them, the validated scene/pose intermediate representation, and ONE
orchestrator (:class:`PlannedMotionController`) over interchangeable
:class:`MotionBackend` implementations: the built-in in-process one
(:class:`PinocchioMotionBackend`, named :data:`PINOCCHIO_BACKEND`) and any
backend a driver registers for an external planner. A driver normally reaches
all of it through
:class:`~cyberwave.driver.control.planned_motion.PlannedMotionMixin` and one
attribute, ``kinematics_backend``.

Nothing here imports ROS or pinocchio at module scope: pinocchio is reached
through the arm controller's lazily built kinematics, and a backend that talks
to a ROS planner lives in the driver package that owns that dependency.
"""

from .backend import (
    PINOCCHIO_BACKEND,
    GraspCandidate,
    GraspGoal,
    MotionBackend,
    MotionGoal,
    MotionOutcome,
    MotionResult,
    MotionType,
    MotionUpdate,
)
from .guards import (
    MotionRefusal,
    Refusal,
    backend_unavailable,
    check_motion_preconditions,
    check_not_busy,
    check_revision,
    check_start_state,
)
from .pinocchio_backend import PinocchioMotionBackend, TrajectoryExecution
from .planner import (
    MOTION_COMMANDS,
    PlannedMotionConfig,
    PlannedMotionController,
)
from .poses import pose_from_xyz_rpy, pose_to_transform, transform_to_pose
from .requests import (
    DEFAULT_APPROACH_DISTANCE_M,
    DEFAULT_LIFT_DISTANCE_M,
    parse_grasp,
)
from .scene import (
    DEFAULT_PAYLOAD_ID,
    PLANNING_FRAME,
    AttachRequest,
    Pose,
    Primitive,
    SceneDiff,
    SceneError,
    SceneRequest,
    SceneTracker,
    parse_attach,
    parse_pose,
    parse_scene_set,
)
from .status import (
    ACTIVE_HEARTBEAT_S,
    IDLE_HEARTBEAT_S,
    MotionState,
    MotionStatus,
    MotionStatusStream,
    PlannerKind,
    planner_name,
)
from .targets import (
    FrameSource,
    PoseSpec,
    TargetMode,
    parse_pose_spec,
    resolve_target,
)

__all__ = [
    "ACTIVE_HEARTBEAT_S",
    "DEFAULT_APPROACH_DISTANCE_M",
    "DEFAULT_LIFT_DISTANCE_M",
    "DEFAULT_PAYLOAD_ID",
    "IDLE_HEARTBEAT_S",
    "MOTION_COMMANDS",
    "PINOCCHIO_BACKEND",
    "PLANNING_FRAME",
    "AttachRequest",
    "FrameSource",
    "GraspCandidate",
    "GraspGoal",
    "MotionBackend",
    "MotionGoal",
    "MotionOutcome",
    "MotionRefusal",
    "MotionResult",
    "MotionState",
    "MotionStatus",
    "MotionStatusStream",
    "MotionType",
    "MotionUpdate",
    "PinocchioMotionBackend",
    "PlannedMotionConfig",
    "PlannedMotionController",
    "PlannerKind",
    "Pose",
    "PoseSpec",
    "Primitive",
    "Refusal",
    "SceneDiff",
    "SceneError",
    "SceneRequest",
    "SceneTracker",
    "TargetMode",
    "TrajectoryExecution",
    "backend_unavailable",
    "check_motion_preconditions",
    "check_not_busy",
    "check_revision",
    "check_start_state",
    "parse_attach",
    "parse_grasp",
    "parse_pose",
    "parse_pose_spec",
    "parse_scene_set",
    "planner_name",
    "pose_from_xyz_rpy",
    "pose_to_transform",
    "resolve_target",
    "transform_to_pose",
]
