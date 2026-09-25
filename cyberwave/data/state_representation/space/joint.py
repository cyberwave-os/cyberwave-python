"""Proto mirror: joint state helpers for SDK ergonomics.

MQTT joint contract (see also the controller / sim handlers):

- Measured-state fields (``positions`` / ``velocities`` / ``efforts``) always
  describe *observed* joint state. Any ``source_type`` may publish them.
- Target/command fields (``target_positions`` / ``target_velocities`` /
  ``target_efforts``) always describe a *desired* setpoint the plant is asked
  to track (or the setpoint it is currently tracking). They must never be
  rendered as measured robot state.

A single payload may legitimately carry both buckets (e.g. a plant reporting
both measured ``positions`` and its current ``target_positions``); the parser
keeps them strictly separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

# Measured-state field names (observed joint state).
MEASURED_JOINT_FIELDS = frozenset({"positions", "velocities", "efforts"})

# Target/command field names (desired setpoints a plant tracks).
TARGET_JOINT_FIELDS = frozenset(
    {"target_positions", "target_velocities", "target_efforts"}
)

# Metadata keys excluded from flat multi-joint payloads.
_JOINT_MQTT_METADATA_KEYS = frozenset(
    {
        "source_type",
        "source_subtype",
        "workload_uuid",
        "session_id",
        "controller_session_uuid",
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
        "target_positions",
        "target_velocities",
        "target_efforts",
        "target_joint_state",
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
    """Normalized joint kinematics from an MQTT ``/update`` payload.

    Measured state (``positions`` / ``velocities`` / ``efforts``) and commanded
    targets (``target_positions`` / ``target_velocities`` / ``target_efforts``)
    are kept in separate buckets so a consumer never confuses one for the other.
    """

    positions: dict[str, float]
    velocities: dict[str, float]
    efforts: dict[str, float]
    target_positions: dict[str, float] = field(default_factory=dict)
    target_velocities: dict[str, float] = field(default_factory=dict)
    target_efforts: dict[str, float] = field(default_factory=dict)

    @property
    def has_measured(self) -> bool:
        return bool(self.positions or self.velocities or self.efforts)

    @property
    def has_targets(self) -> bool:
        return bool(
            self.target_positions or self.target_velocities or self.target_efforts
        )


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


def _parse_targets(
    payload: Mapping[str, object],
    controllable_names: frozenset[str] | set[str] | None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Parse target/command buckets (``target_*``) from a payload.

    Target fields are always aggregated dicts. ``target_joint_state`` (single
    joint command) is folded into ``target_positions`` for a uniform shape.
    """
    target_positions = _float_map(payload.get("target_positions"))
    target_velocities = _float_map(payload.get("target_velocities"))
    target_efforts = _float_map(payload.get("target_efforts"))

    joint_name = payload.get("joint_name")
    target_state = payload.get("target_joint_state")
    if joint_name is not None and isinstance(target_state, Mapping):
        name = str(joint_name)
        position = _to_float(target_state.get("position"))
        if position is not None:
            target_positions[name] = position
        velocity = _to_float(target_state.get("velocity"))
        if velocity is not None:
            target_velocities[name] = velocity
        effort = _to_float(target_state.get("effort"))
        if effort is not None:
            target_efforts[name] = effort

    return (
        _filter_controllable(target_positions, controllable_names),
        _filter_controllable(target_velocities, controllable_names),
        _filter_controllable(target_efforts, controllable_names),
    )


def parse_joint_mqtt_payload(
    payload: Mapping[str, object],
    *,
    controllable_names: frozenset[str] | set[str] | None = None,
) -> ParsedJointMqttUpdate:
    """Parse MQTT joint ``/update`` payloads.

    Measured-state shapes:

    - Aggregated: ``{"positions": {...}, "velocities": {...}, "efforts": {...}}``
    - Flat multi-joint: ``{"j1": 0.1, "j2": 0.2, "source_type": "edge", ...}``
    - Single joint: ``{"joint_name": "j1", "joint_state": {"position": ..., ...}}``

    Target/command shapes (kept in the ``target_*`` buckets):

    - Aggregated: ``{"target_positions": {...}, "target_velocities": {...}, ...}``
    - Single joint: ``{"joint_name": "j1", "target_joint_state": {"position": ...}}``

    A payload may contain both measured and target fields; they are routed into
    separate buckets and never merged.
    """
    target_positions, target_velocities, target_efforts = _parse_targets(
        payload, controllable_names
    )

    single = _single_joint_from_payload(payload)
    if single is not None:
        return ParsedJointMqttUpdate(
            positions=_filter_controllable(single.positions, controllable_names),
            velocities=_filter_controllable(single.velocities, controllable_names),
            efforts=_filter_controllable(single.efforts, controllable_names),
            target_positions=target_positions,
            target_velocities=target_velocities,
            target_efforts=target_efforts,
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
            target_positions=target_positions,
            target_velocities=target_velocities,
            target_efforts=target_efforts,
        )

    positions = _flat_positions_from_payload(payload)
    return ParsedJointMqttUpdate(
        positions=_filter_controllable(positions, controllable_names),
        velocities={},
        efforts={},
        target_positions=target_positions,
        target_velocities=target_velocities,
        target_efforts=target_efforts,
    )


def joint_dict_from_payload(
    payload: Mapping[str, object],
    *,
    controllable_names: frozenset[str] | set[str] | None = None,
) -> dict[str, float]:
    """Parse joint MQTT payloads into ``{name: position}`` (measured positions only)."""
    return parse_joint_mqtt_payload(
        payload, controllable_names=controllable_names
    ).positions


def joint_target_dict_from_payload(
    payload: Mapping[str, object],
    *,
    controllable_names: frozenset[str] | set[str] | None = None,
) -> dict[str, float]:
    """Parse joint MQTT payloads into ``{name: position}`` (target positions only)."""
    return parse_joint_mqtt_payload(
        payload, controllable_names=controllable_names
    ).target_positions
