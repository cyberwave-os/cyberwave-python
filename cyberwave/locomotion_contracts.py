# !! GENERATED — do not edit directly. Run python-sdk-gen.sh to regenerate.

"""Locomotion and free-body velocity contracts shared by runtime adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

LOCOMOTION_VELOCITY_COMMAND_CONTRACT = "locomotion.velocity_command.v1"
AERIAL_VELOCITY_COMMAND_CONTRACT = "aerial.velocity_command.v1"
LOCOMOTION_VELOCITY_COMMAND_REQUIRED_FIELDS = (
    "contract",
    "linear_x",
    "angular_z",
    "duration_ms",
    "gait",
    "origin",
)
AERIAL_VELOCITY_COMMAND_REQUIRED_FIELDS = (
    "contract",
    "linear_x",
    "linear_z",
    "angular_z",
    "duration_ms",
    "origin",
)

LocomotionOrigin = Literal["teleop", "ai_policy", "navigation", "workflow"]
LocomotionGait = Literal["walk", "trot", "stand"]

_VALID_ORIGINS = {"teleop", "ai_policy", "navigation", "workflow"}
_VALID_GAITS = {"walk", "trot", "stand"}
_MAX_DURATION_MS = 30_000
MIN_ACTIVE_HOLD_SECONDS = 0.15
MAX_ACTIVE_HOLD_SECONDS = 5.0


class LocomotionVelocityCommandError(ValueError):
    """Raised when a velocity command cannot be consumed safely."""


@dataclass(frozen=True)
class LocomotionVelocityCommand:
    linear_x: float
    linear_y: float
    angular_z: float
    duration_ms: int
    gait: LocomotionGait
    origin: LocomotionOrigin
    warnings: tuple[str, ...] = ()
    clamped_fields: tuple[str, ...] = ()

    @property
    def is_stop(self) -> bool:
        return self.duration_ms == 0 or self.gait == "stand"

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "linear_x": self.linear_x,
            "linear_y": self.linear_y,
            "angular_z": self.angular_z,
            "duration_ms": self.duration_ms,
            "gait": self.gait,
            "origin": self.origin,
            "contract": LOCOMOTION_VELOCITY_COMMAND_CONTRACT,
        }
        if self.warnings:
            payload["warnings"] = list(self.warnings)
        if self.clamped_fields:
            payload["clamped_fields"] = list(self.clamped_fields)
        return payload


@dataclass(frozen=True)
class BodyVelocityCommand:
    linear_x: float
    linear_y: float
    linear_z: float
    angular_z: float
    duration_ms: int
    origin: LocomotionOrigin
    vertical_control: bool = False


def build_locomotion_velocity_command(
    *,
    linear_x: Any = 0.0,
    linear_y: Any = 0.0,
    angular_z: Any = 0.0,
    duration_ms: Any = 500,
    gait: str = "walk",
    origin: str = "teleop",
) -> LocomotionVelocityCommand:
    return normalize_locomotion_velocity_command(
        {
            "contract": LOCOMOTION_VELOCITY_COMMAND_CONTRACT,
            "linear_x": linear_x,
            "linear_y": linear_y,
            "angular_z": angular_z,
            "duration_ms": duration_ms,
            "gait": gait,
            "origin": origin,
        },
    )


def normalize_locomotion_velocity_command(
    raw: Any,
    *,
    defaults: Any = None,
    limits: Any = None,
    require_explicit_fields: bool = True,
    max_linear_x: float | None = None,
    max_linear_y: float | None = None,
    max_angular_z: float | None = None,
    linear_y_supported: bool = True,
) -> LocomotionVelocityCommand:
    if not isinstance(raw, Mapping):
        raise LocomotionVelocityCommandError(
            "Locomotion velocity command must be a mapping",
        )

    raw_dict = raw
    default_dict = defaults if isinstance(defaults, Mapping) else {}
    limit_dict = _limits_dict(
        limits,
        max_linear_x=max_linear_x,
        max_linear_y=max_linear_y,
        max_angular_z=max_angular_z,
        linear_y_supported=linear_y_supported,
    )

    contract = str(raw_dict.get("contract", default_dict.get("contract", ""))).strip()
    if contract != LOCOMOTION_VELOCITY_COMMAND_CONTRACT:
        raise LocomotionVelocityCommandError(
            f"Unsupported locomotion contract: {contract!r}",
        )

    if require_explicit_fields:
        for key in LOCOMOTION_VELOCITY_COMMAND_REQUIRED_FIELDS:
            if key == "contract":
                continue
            if key not in raw_dict:
                raise LocomotionVelocityCommandError(
                    f"Locomotion velocity command missing required {key}",
                )

    warnings: list[str] = []
    clamped_fields: list[str] = []

    linear_x = _clamp_abs(
        _number(_value(raw_dict, default_dict, "linear_x", 0.0), "linear_x"),
        limit_dict.get("max_linear_x"),
        "linear_x",
        warnings,
        clamped_fields,
    )
    linear_y = _clamp_abs(
        _number(_value(raw_dict, default_dict, "linear_y", 0.0), "linear_y"),
        limit_dict.get("max_linear_y"),
        "linear_y",
        warnings,
        clamped_fields,
    )
    if limit_dict.get("linear_y_supported") is False and linear_y != 0.0:
        linear_y = 0.0
        warnings.append("linear_y_unsupported")
        clamped_fields.append("linear_y")
    angular_z = _clamp_abs(
        _number(_value(raw_dict, default_dict, "angular_z", 0.0), "angular_z"),
        limit_dict.get("max_angular_z"),
        "angular_z",
        warnings,
        clamped_fields,
    )

    duration_ms = _integer(
        _value(raw_dict, default_dict, "duration_ms", 500),
        "duration_ms",
    )
    if duration_ms != 0 and duration_ms < 50:
        raise LocomotionVelocityCommandError("duration_ms must be 0 or at least 50")
    if duration_ms > _MAX_DURATION_MS:
        raise LocomotionVelocityCommandError(
            f"duration_ms must be <= {_MAX_DURATION_MS}",
        )

    gait = str(_value(raw_dict, default_dict, "gait", "walk")).strip().lower()
    if gait not in _VALID_GAITS:
        raise LocomotionVelocityCommandError(f"Invalid locomotion gait: {gait!r}")

    origin = str(_value(raw_dict, default_dict, "origin", "teleop")).strip().lower()
    if origin not in _VALID_ORIGINS:
        raise LocomotionVelocityCommandError(f"Invalid locomotion origin: {origin!r}")

    return LocomotionVelocityCommand(
        linear_x=linear_x,
        linear_y=linear_y,
        angular_z=angular_z,
        duration_ms=duration_ms,
        gait=gait,  # type: ignore[arg-type]
        origin=origin,  # type: ignore[arg-type]
        warnings=tuple(warnings),
        clamped_fields=tuple(clamped_fields),
    )


def normalize_body_velocity_command(raw: Any) -> BodyVelocityCommand:
    if not isinstance(raw, Mapping):
        raise LocomotionVelocityCommandError("Velocity command must be a mapping")

    contract = str(raw.get("contract") or "").strip()
    if contract == LOCOMOTION_VELOCITY_COMMAND_CONTRACT:
        command = normalize_locomotion_velocity_command(raw)
        return BodyVelocityCommand(
            linear_x=command.linear_x,
            linear_y=command.linear_y,
            linear_z=0.0,
            angular_z=command.angular_z,
            duration_ms=command.duration_ms,
            origin=command.origin,
            vertical_control=False,
        )
    if contract != AERIAL_VELOCITY_COMMAND_CONTRACT:
        raise LocomotionVelocityCommandError(
            f"Unsupported velocity contract: {contract!r}",
        )

    duration_ms = _integer(_required(raw, "duration_ms"), "duration_ms")
    if duration_ms != 0 and duration_ms < 50:
        raise LocomotionVelocityCommandError("duration_ms must be 0 or at least 50")
    if duration_ms > _MAX_DURATION_MS:
        raise LocomotionVelocityCommandError(
            f"duration_ms must be <= {_MAX_DURATION_MS}",
        )

    origin = str(_required(raw, "origin")).strip().lower()
    if origin not in _VALID_ORIGINS:
        raise LocomotionVelocityCommandError(f"Invalid velocity origin: {origin!r}")

    return BodyVelocityCommand(
        linear_x=_number(_required(raw, "linear_x"), "linear_x"),
        linear_y=_number(raw.get("linear_y", 0.0), "linear_y"),
        linear_z=_number(_required(raw, "linear_z"), "linear_z"),
        angular_z=_number(_required(raw, "angular_z"), "angular_z"),
        duration_ms=duration_ms,
        origin=origin,  # type: ignore[arg-type]
        vertical_control=True,
    )


def stop_locomotion_velocity_command(command: Any = None) -> dict[str, Any]:
    payload = dict(command) if isinstance(command, Mapping) else {}
    payload.update(
        {
            "linear_x": 0.0,
            "linear_y": 0.0,
            "angular_z": 0.0,
            "duration_ms": 0,
            "gait": "stand",
            "contract": LOCOMOTION_VELOCITY_COMMAND_CONTRACT,
        },
    )
    payload.setdefault("origin", "teleop")
    return payload


def stop_aerial_velocity_command(command: Any = None) -> dict[str, Any]:
    payload = dict(command) if isinstance(command, Mapping) else {}
    payload.update(
        {
            "linear_x": 0.0,
            "linear_y": 0.0,
            "linear_z": 0.0,
            "angular_z": 0.0,
            "duration_ms": 0,
            "contract": AERIAL_VELOCITY_COMMAND_CONTRACT,
        },
    )
    payload.setdefault("origin", "teleop")
    return payload


def hold_seconds_for_velocity_command(
    command: LocomotionVelocityCommand | BodyVelocityCommand,
) -> float:
    if command.duration_ms == 0:
        return MIN_ACTIVE_HOLD_SECONDS
    return max(
        MIN_ACTIVE_HOLD_SECONDS,
        min(command.duration_ms / 1000.0, MAX_ACTIVE_HOLD_SECONDS),
    )


def _limits_dict(
    limits: Any,
    *,
    max_linear_x: float | None,
    max_linear_y: float | None,
    max_angular_z: float | None,
    linear_y_supported: bool,
) -> dict[str, Any]:
    limit_dict = dict(limits) if isinstance(limits, Mapping) else {}
    if max_linear_x is not None:
        limit_dict["max_linear_x"] = max_linear_x
    if max_linear_y is not None:
        limit_dict["max_linear_y"] = max_linear_y
    if max_angular_z is not None:
        limit_dict["max_angular_z"] = max_angular_z
    if not linear_y_supported:
        limit_dict["linear_y_supported"] = False
    return limit_dict


def _value(
    raw: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: Any,
) -> Any:
    return raw[key] if key in raw else defaults.get(key, fallback)


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LocomotionVelocityCommandError(
            f"Locomotion velocity command requires numeric {field_name}",
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise LocomotionVelocityCommandError(
            f"Locomotion velocity command requires finite numeric {field_name}",
        )
    return float(value)


def _integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LocomotionVelocityCommandError(
            f"Locomotion velocity command requires numeric {field_name}",
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise LocomotionVelocityCommandError(
            f"Locomotion velocity command requires finite numeric {field_name}",
        )
    if isinstance(value, float) and not value.is_integer():
        raise LocomotionVelocityCommandError(
            f"Locomotion velocity command requires integer {field_name}",
        )
    return int(value)


def _required(raw: Mapping[str, Any], field_name: str) -> Any:
    if field_name not in raw:
        raise LocomotionVelocityCommandError(
            f"Locomotion velocity command missing required {field_name}",
        )
    return raw[field_name]


def _clamp_abs(
    value: float,
    raw_limit: Any,
    field_name: str,
    warnings: list[str],
    clamped_fields: list[str],
) -> float:
    if isinstance(raw_limit, bool) or not isinstance(raw_limit, int | float):
        return value
    limit = abs(float(raw_limit))
    if limit <= 0:
        return value
    clamped = max(-limit, min(value, limit))
    if clamped != value:
        warnings.append(f"{field_name}_clamped_to_max_{field_name}")
        clamped_fields.append(field_name)
    return clamped
