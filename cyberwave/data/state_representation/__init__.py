"""Proto-shaped kinematic types (manual mirror until generated bindings)."""

from .decode import (
    cartesian_pose_from_position_payload,
    cartesian_pose_from_rotation_payload,
    decode_joint_update_payload,
    decode_json_payload,
    decode_kinematics_protobuf_stub,
    merge_cartesian_pose,
    merge_legacy_position_rotation,
)
from .geometry.primitives import Quaterniond, Vector3d
from .robot_state import RobotStateMessage
from .space.cartesian import CartesianPose
from .space.joint import (
    JointState,
    ParsedJointMqttUpdate,
    joint_dict_from_payload,
    parse_joint_mqtt_payload,
)
from .space.spatial_state import SpatialState

__all__ = [
    "CartesianPose",
    "JointState",
    "ParsedJointMqttUpdate",
    "Quaterniond",
    "RobotStateMessage",
    "SpatialState",
    "Vector3d",
    "cartesian_pose_from_position_payload",
    "cartesian_pose_from_rotation_payload",
    "decode_joint_update_payload",
    "decode_json_payload",
    "decode_kinematics_protobuf_stub",
    "joint_dict_from_payload",
    "parse_joint_mqtt_payload",
    "merge_cartesian_pose",
    "merge_legacy_position_rotation",
]
