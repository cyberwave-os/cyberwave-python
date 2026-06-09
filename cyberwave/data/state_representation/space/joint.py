"""Proto mirror: joint state helpers for SDK ergonomics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# Metadata keys excluded from flat multi-joint payloads.
_JOINT_MQTT_METADATA_KEYS = frozenset(
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
        "mode",
        "update_mode",
        "velocities",
        "efforts",
        "positions",
        "joint_name",
        "joint_state",
    }
)


@dataclass(frozen=True, slots=True)
class JointState:
    """Ordered joint names with parallel position values (radians)."""

    joint_names: tuple[str, ...]
    positions: tuple[float, ...]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.joint_names, self.positions, strict=True))


@dataclass(frozen=True, slots=True)
class ParsedJointMqttUpdate:
    """Normalized joint kinematics from an MQTT ``/update`` payload."""

    positions: dict[str, float]
    velocities: dict[str, float]
    efforts: dict[str, float]


def _to_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _float_map(data: object) -> dict[str, float]:
    if not isinstance(data, dict):
        return {}
    out: dict[str, float] = {}
    for key, val in data.items():
        parsed = _to_float(val)
        if parsed is not None:
            out[str(key)] = parsed
    return out


def _filter_controllable(
    data: dict[str, float],
    controllable_names: frozenset[str] | set[str] | None,
) -> dict[str, float]:
    if not controllable_names:
        return data
    return {k: v for k, v in data.items() if k in controllable_names}


def _flat_positions_from_payload(payload: Mapping[str, object]) -> dict[str, float]:
    """Backward-compatible flat multi-joint: numeric top-level keys only."""
    raw: dict[str, float] = {}
    for key, val in payload.items():
        if key in _JOINT_MQTT_METADATA_KEYS:
            continue
        parsed = _to_float(val)
        if parsed is not None:
            raw[str(key)] = parsed
    return raw


def _single_joint_from_payload(payload: Mapping[str, object]) -> ParsedJointMqttUpdate | None:
    joint_name = payload.get("joint_name")
    joint_state = payload.get("joint_state")
    if joint_name is None or not isinstance(joint_state, Mapping):
        return None

    name = str(joint_name)
    position = _to_float(joint_state.get("position"))
    if position is None:
        return ParsedJointMqttUpdate(positions={}, velocities={}, efforts={})

    velocity = _to_float(joint_state.get("velocity"))
    effort = _to_float(joint_state.get("effort"))
    return ParsedJointMqttUpdate(
        positions={name: position},
        velocities={name: velocity} if velocity is not None else {},
        efforts={name: effort} if effort is not None else {},
    )


def parse_joint_mqtt_payload(
    payload: Mapping[str, object],
    *,
    controllable_names: frozenset[str] | set[str] | None = None,
) -> ParsedJointMqttUpdate:
    """Parse MQTT joint ``/update`` payloads.

    Supports:

    - Aggregated: ``{"positions": {...}, "velocities": {...}, "efforts": {...}}``
    - Flat multi-joint: ``{"j1": 0.1, "j2": 0.2, "source_type": "edge", ...}``
    - Single joint: ``{"joint_name": "j1", "joint_state": {"position": ..., ...}}``
    """
    single = _single_joint_from_payload(payload)
    if single is not None:
        return ParsedJointMqttUpdate(
            positions=_filter_controllable(single.positions, controllable_names),
            velocities=_filter_controllable(single.velocities, controllable_names),
            efforts=_filter_controllable(single.efforts, controllable_names),
        )

    positions_raw = payload.get("positions")
    if isinstance(positions_raw, dict):
        positions = _float_map(positions_raw)
        velocities = _float_map(payload.get("velocities"))
        efforts = _float_map(payload.get("efforts"))
        return ParsedJointMqttUpdate(
            positions=_filter_controllable(positions, controllable_names),
            velocities=_filter_controllable(velocities, controllable_names),
            efforts=_filter_controllable(efforts, controllable_names),
        )

    positions = _flat_positions_from_payload(payload)
    return ParsedJointMqttUpdate(
        positions=_filter_controllable(positions, controllable_names),
        velocities={},
        efforts={},
    )


def joint_dict_from_payload(
    payload: Mapping[str, object],
    *,
    controllable_names: frozenset[str] | set[str] | None = None,
) -> dict[str, float]:
    """Parse joint MQTT payloads into ``{name: position}`` (positions only)."""
    return parse_joint_mqtt_payload(
        payload, controllable_names=controllable_names
    ).positions
