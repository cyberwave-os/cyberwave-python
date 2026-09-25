"""JSON legacy MQTT payloads → proto-shaped dataclasses."""

from __future__ import annotations

from typing import Any, Mapping

from .geometry.primitives import Quaterniond, Vector3d
from .robot_state import RobotStateMessage
from .space.cartesian import CartesianPose
from .space.joint import joint_dict_from_payload
from .space.spatial_state import SpatialState

_DEFAULT_POSITION = Vector3d()
_DEFAULT_ORIENTATION = Quaterniond()
_DEFAULT_SPATIAL = SpatialState()


def _float_or_zero(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def vector3_from_dict(data: Mapping[str, object] | None) -> Vector3d:
    if not isinstance(data, Mapping):
        return _DEFAULT_POSITION
    return Vector3d(
        x=_float_or_zero(data.get("x")),
        y=_float_or_zero(data.get("y")),
        z=_float_or_zero(data.get("z")),
    )


def quaternion_from_dict(data: Mapping[str, object] | None) -> Quaterniond:
    if not isinstance(data, Mapping):
        return _DEFAULT_ORIENTATION
    return Quaterniond(
        x=_float_or_zero(data.get("x")),
        y=_float_or_zero(data.get("y")),
        z=_float_or_zero(data.get("z")),
        w=_float_or_zero(data.get("w"), 1.0),
    )


def cartesian_pose_from_position_payload(payload: Mapping[str, Any]) -> CartesianPose:
    """Decode ``cyberwave/twin/{uuid}/position`` JSON envelope."""
    position_raw = payload.get("position", payload)
    frame = str(payload.get("frame_id") or payload.get("reference_frame") or "")
    return CartesianPose(
        spatial_state=SpatialState(reference_frame=frame),
        position=vector3_from_dict(position_raw if isinstance(position_raw, Mapping) else payload),
        orientation=_DEFAULT_ORIENTATION,
    )


def cartesian_pose_from_rotation_payload(payload: Mapping[str, Any]) -> CartesianPose:
    """Decode ``cyberwave/twin/{uuid}/rotation`` JSON envelope."""
    rotation_raw = payload.get("rotation", payload)
    frame = str(payload.get("frame_id") or payload.get("reference_frame") or "")
    return CartesianPose(
        spatial_state=SpatialState(reference_frame=frame),
        position=_DEFAULT_POSITION,
        orientation=quaternion_from_dict(
            rotation_raw if isinstance(rotation_raw, Mapping) else payload
        ),
    )


def merge_cartesian_pose(
    base: CartesianPose | None,
    update: CartesianPose,
) -> CartesianPose:
    """Merge partial pose updates into one canonical unit."""
    if base is None:
        return update
    frame = update.spatial_state.reference_frame or base.spatial_state.reference_frame
    pos = update.position if update.position != _DEFAULT_POSITION else base.position
    ori = (
        update.orientation
        if update.orientation != _DEFAULT_ORIENTATION
        else base.orientation
    )
    return CartesianPose(
        spatial_state=SpatialState(reference_frame=frame),
        position=pos,
        orientation=ori,
    )


def merge_legacy_position_rotation(
    position_payload: Mapping[str, Any] | None,
    rotation_payload: Mapping[str, Any] | None,
) -> CartesianPose:
    """One pose unit from split legacy topics."""
    pose: CartesianPose | None = None
    if position_payload is not None:
        pose = merge_cartesian_pose(pose, cartesian_pose_from_position_payload(position_payload))
    if rotation_payload is not None:
        pose = merge_cartesian_pose(pose, cartesian_pose_from_rotation_payload(rotation_payload))
    if pose is None:
        return CartesianPose(
            spatial_state=_DEFAULT_SPATIAL,
            position=_DEFAULT_POSITION,
            orientation=_DEFAULT_ORIENTATION,
        )
    return pose


def decode_joint_update_payload(
    payload: Mapping[str, Any],
    *,
    controllable_names: frozenset[str] | set[str] | None = None,
) -> dict[str, float]:
    from .space.joint import parse_joint_mqtt_payload

    return parse_joint_mqtt_payload(
        payload, controllable_names=controllable_names
    ).positions


def decode_json_payload(
    slug: str,
    payload: Mapping[str, Any],
    *,
    controllable_names: frozenset[str] | set[str] | None = None,
) -> RobotStateMessage | dict[str, float] | CartesianPose:
    """Route catalog slug to the appropriate decoder (v1 JSON only)."""
    if "/position" in slug or slug.endswith("/position"):
        return cartesian_pose_from_position_payload(payload)
    if "/rotation" in slug or slug.endswith("/rotation"):
        return cartesian_pose_from_rotation_payload(payload)
    if "/joint/" in slug or "joint" in slug:
        return decode_joint_update_payload(payload, controllable_names=controllable_names)
    if "kinematics" in slug:
        return decode_kinematics_protobuf_stub(payload)
    raise ValueError(f"No decoder for slug {slug!r}")


def decode_kinematics_protobuf_stub(payload: Mapping[str, Any]) -> RobotStateMessage:
    """Placeholder until generated protobuf bindings land."""
    branch = payload.get("message_type") or payload.get("branch")
    if branch == "cartesian_pose" and isinstance(payload.get("cartesian_pose"), Mapping):
        cp = payload["cartesian_pose"]
        return RobotStateMessage(
            message_type="cartesian_pose",
            cartesian_pose=CartesianPose(
                spatial_state=SpatialState(
                    reference_frame=str(cp.get("reference_frame", ""))
                ),
                position=vector3_from_dict(cp.get("position")),  # type: ignore[arg-type]
                orientation=quaternion_from_dict(cp.get("orientation")),  # type: ignore[arg-type]
            ),
        )
    if branch == "joint_state" or isinstance(payload.get("positions"), Mapping):
        return RobotStateMessage(
            message_type="joint_state",
            joint_positions=decode_joint_update_payload(payload),
            raw=dict(payload),
        )
    return RobotStateMessage(message_type="unknown", raw=dict(payload))
