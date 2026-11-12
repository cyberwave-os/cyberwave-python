"""Serialize ROS 2 messages to JSON-friendly MQTT/Zenoh transport payloads."""

from __future__ import annotations

import math
import time
from typing import Any


def ros_message_to_transport_payload(
    msg: Any, *, source_type: str = "edge"
) -> dict[str, Any]:
    """Serialize a ROS message to a JSON-friendly MQTT/Zenoh payload."""
    data = _message_to_mapping(msg)
    data["source_type"] = source_type
    data["timestamp"] = time.time()
    return data


def ros_joint_state_to_transport_payload(
    msg: Any,
    *,
    source_type: str = "edge",
    aggregated: bool = False,
) -> dict[str, Any] | None:
    """Map ``sensor_msgs/msg/JointState`` to a joint ``/update`` transport payload.

    By default emits the **flat** shape expected by the Cyberwave joint-telemetry
    ingestion pipeline: ``names[i]: position[i]`` as top-level keys, plus optional
    sibling ``velocities`` / ``efforts`` maps and ``timestamp``. This matches
    :meth:`cyberwave.mqtt.CyberwaveMQTTClient.update_joints_state` when only
    positions are passed (flat), while still forwarding velocity/effort per joint.

    Set ``aggregated=True`` for the nested ``positions`` object form.
    """
    names = list(getattr(msg, "name", None) or [])
    if not names:
        return None

    positions = _joint_values_by_name(names, getattr(msg, "position", None) or [])
    if not positions:
        return None

    velocities = _joint_values_by_name(names, getattr(msg, "velocity", None) or [])
    efforts = _joint_values_by_name(names, getattr(msg, "effort", None) or [])
    timestamp = _joint_state_timestamp(msg)

    if aggregated:
        payload: dict[str, Any] = {
            "source_type": source_type,
            "positions": positions,
            "timestamp": timestamp,
        }
        if velocities:
            payload["velocities"] = velocities
        if efforts:
            payload["efforts"] = efforts
        return payload

    payload = {"source_type": source_type, **positions, "timestamp": timestamp}
    if velocities:
        payload["velocities"] = velocities
    if efforts:
        payload["efforts"] = efforts
    return payload


def ros_pose_stamped_to_twin_pose_payload(
    msg: Any,
    *,
    source_type: str = "edge",
    frame_id: str | None = None,
    child_frame_id: str | None = None,
) -> dict[str, Any] | None:
    """Map ``geometry_msgs/msg/PoseStamped`` to the canonical ``twin/pose`` wire form.

    ``pose_xyzw`` is ``[x, y, z, qx, qy, qz, qw]`` (metres, unit quaternion) and
    ``timestamp_ms`` is the message's *source* stamp, not the receive time: a
    vendor end-pose stream is only useful for hand-eye or pick verification when
    it can be matched against the joint state it was measured with.

    ``frame_id`` comes from ``msg.header.frame_id`` unless the caller overrides
    it -- some vendor nodes publish an empty or vendor-internal header frame
    while the driver knows the configured base frame. ``child_frame_id`` is the
    tool frame the pose describes and is omitted when not given.

    Returns ``None`` (drop the sample) rather than publishing something wrong:
    a missing stamp, a non-finite value, a negative or malformed stamp, a zero
    quaternion, or no frame at all. The quaternion is normalised because vendor
    firmware commonly reports one that is only approximately unit length.
    """
    pose = getattr(msg, "pose", None)
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    position = getattr(pose, "position", None)
    orientation = getattr(pose, "orientation", None)
    if position is None or orientation is None or stamp is None:
        return None

    try:
        values = [
            float(position.x),
            float(position.y),
            float(position.z),
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        ]
        stamp_sec = int(stamp.sec)
        stamp_nanosec = int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError):
        return None
    if (
        not all(math.isfinite(value) for value in values)
        or stamp_sec < 0
        or not 0 <= stamp_nanosec < 1_000_000_000
    ):
        return None

    quaternion_norm = math.sqrt(sum(value * value for value in values[3:]))
    if quaternion_norm <= 0.0:
        return None
    values[3:] = [value / quaternion_norm for value in values[3:]]

    frame = frame_id if frame_id is not None else getattr(header, "frame_id", "")
    if not isinstance(frame, str) or not frame:
        return None
    payload: dict[str, Any] = {"source_type": source_type, "frame_id": frame}
    if child_frame_id is not None:
        if not child_frame_id:
            return None
        payload["child_frame_id"] = child_frame_id
    payload["pose_xyzw"] = values
    payload["timestamp_ms"] = stamp_sec * 1_000 + stamp_nanosec // 1_000_000
    return payload


# Top-level keys on joint ``/update`` payloads that are not ``name: position`` pairs
# (aligned with the flat joint-telemetry wire format described above).
_JOINT_UPDATE_METADATA_KEYS = frozenset(
    {
        "source_type",
        "source_subtype",
        "workload_uuid",
        "session_id",
        "timestamp",
        "ts",
        "ts_raw",
        "ts_norm",
        "ts_unix",
        "ts_us",
        "type",
        "mqtt_topic",
        "camera_frame_counters",
        "positions",
        "velocities",
        "efforts",
        "joint_name",
        "joint_state",
        "update_mode",
    }
)


def joint_positions_from_transport_payload(
    payload: dict[str, Any],
) -> dict[str, float]:
    """Extract ``joint_name -> position`` from a flat or aggregated joint ``/update`` payload."""
    nested = payload.get("positions")
    if isinstance(nested, dict):
        return {
            str(name): float(value)
            for name, value in nested.items()
            if isinstance(value, (int, float)) and not math.isnan(float(value))
        }

    out: dict[str, float] = {}
    for key, value in payload.items():
        if key in _JOINT_UPDATE_METADATA_KEYS:
            continue
        if not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isnan(number):
            continue
        out[str(key)] = number
    return out


def _joint_values_by_name(
    names: list[str],
    values: list[Any] | tuple[Any, ...],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for index, joint_name in enumerate(names):
        if index >= len(values):
            break
        value = values[index]
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(number):
            continue
        out[str(joint_name)] = number
    return out


def _joint_state_timestamp(msg: Any) -> float:
    header = getattr(msg, "header", None)
    if header is not None:
        stamp = getattr(header, "stamp", None)
        if stamp is not None:
            sec = int(getattr(stamp, "sec", 0) or 0)
            nanosec = int(getattr(stamp, "nanosec", 0) or 0)
            return float(sec) + float(nanosec) * 1e-9
    return time.time()


def _message_to_mapping(msg: Any) -> dict[str, Any]:
    try:
        from rosidl_runtime_py.convert import message_to_ordereddict

        ordered = message_to_ordereddict(msg)
        return _normalize_ordered_dict(ordered)
    except ImportError:
        pass
    except Exception:
        pass

    try:
        from rosidl_runtime_py.utilities import get_message_slot_types

        fields = get_message_slot_types(type(msg))
    except ImportError:
        fields = {}

    result: dict[str, Any] = {}
    for name in fields or _iter_field_names(msg):
        if not hasattr(msg, name):
            continue
        value = getattr(msg, name)
        result[name] = _serialize_value(value)
    return result


def _iter_field_names(msg: Any) -> list[str]:
    try:
        from rosidl_runtime_py.utilities import get_message_slot_types

        return list(get_message_slot_types(type(msg)).keys())
    except Exception:
        return [name for name in dir(msg) if not name.startswith("_")]


def _normalize_ordered_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_ordered_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_ordered_dict(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if hasattr(value, "get_fields_and_field_types"):
        return _message_to_mapping(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)
