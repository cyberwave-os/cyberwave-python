"""Zenoh key-expression utilities for the Cyberwave data layer.

Key expressions follow the pattern::

    {prefix}/{twin_uuid}/data/{channel}/{sensor_name}

Where:

* **prefix** defaults to ``"cw"`` (configurable via ``BackendConfig.key_prefix``).
* **twin_uuid** is a UUID-4 string identifying the digital twin.
* **channel** is one of the well-known channel names (``frames``, ``depth``,
  ``joint_states``, …) or a custom developer-defined name.
* **sensor_name** is an optional qualifier (e.g. ``"default"``, ``"wrist"``,
  ``"left_arm"``).  Omit it for single-sensor channels.

Stream channels (high-frequency) and latest-value channels (low-frequency)
share the same key space — semantics are determined by the subscriber policy,
not the key itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .exceptions import ChannelError

# ── Well-known channel names ────────────────────────────────────────

STREAM_CHANNELS: frozenset[str] = frozenset(
    {
        "frames",
        "depth",
        "audio",
        "pointcloud",
        "imu",
        "force_torque",
    }
)

LATEST_VALUE_CHANNELS: frozenset[str] = frozenset(
    {
        "joint_states",
        "position",
        "attitude",
        "gps",
        "end_effector_pose",
        "gripper_state",
        "map",
        "battery",
        "temperature",
        "telemetry",
    }
)

# Command channels: published by workers / policy models, consumed by drivers.
# Format: commands/<action_type>  e.g. "commands/velocity", "commands/joint_positions"
# Drivers subscribe to these for closed-loop autonomous actuation.
# Human teleop always takes priority over command-channel messages.
COMMAND_CHANNELS: frozenset[str] = frozenset(
    {
        "commands",  # wildcard parent — subscribe with build_wildcard(channel="commands")
        "commands/velocity",  # {linear_x, linear_y, linear_z, angular_x, angular_y, angular_z}
        "commands/joint_positions",  # {positions: [float], names: [str], velocities?: [float]}
        "commands/end_effector_pose",  # {x, y, z, qx, qy, qz, qw}
        "commands/gripper",  # {opening: float, force?: float}
        "commands/navigate",  # {goal_type: "goto"|"path", position?: {x,y,z}, waypoints?: [...]}
        "commands/pose",  # {pose: "stand"|"sit"|"recovery"}
        "commands/led",  # {action: "toggle"|"on"|"off", color?: str}
    }
)

WELL_KNOWN_CHANNELS: frozenset[str] = (
    STREAM_CHANNELS | LATEST_VALUE_CHANNELS | COMMAND_CHANNELS
)

# ── Privacy / frame-filter pipeline ─────────────────────────────────
# Wire-contract channel for the worker-processed frame filter (see
# ``docs.cyberwave.com/edge/drivers/frame-filters``). The generic-camera
# driver subscribes to this channel only when the per-twin
# ``frame_filter_enabled`` metadata flag is true; workers publish
# anonymised / redacted full frames to it for substitution into the
# WebRTC stream. Per-twin isolation is automatic because the DataBus
# injects the twin UUID into the Zenoh key.
FILTERED_FRAME_CHANNEL: str = "frames/filtered"

# Wire-contract channel for non-substituting overlay metadata (boxes,
# labels, style hints) — JSON, not pixels. The camera driver subscribes
# to this channel unconditionally and composites the overlay onto the
# frame before WebRTC encode, regardless of ``frame_filter_enabled``.
# This is the path the ``annotate`` workflow node publishes to so
# annotation no longer depends on the privacy/substitution gate.
FRAME_OVERLAY_CHANNEL: str = "frames/overlay"

# ── Validation patterns ─────────────────────────────────────────────

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

_CHANNEL_SEGMENT_RE = re.compile(r"^[a-z][a-z0-9_]*(?:/[a-z][a-z0-9_]*)*$")

_SENSOR_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Key prefix: lowercase alphanumeric + underscores, starting with a letter.
_PREFIX_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_uuid(value: str) -> None:
    if not _UUID_RE.match(value):
        raise ChannelError(f"Invalid twin UUID: '{value}'.")


def _validate_channel_segment(segment: str) -> None:
    if not _CHANNEL_SEGMENT_RE.match(segment):
        raise ChannelError(
            f"Invalid channel segment: '{segment}'.  "
            "Must be lowercase alphanumeric + underscores, starting with a letter.  "
            "Hierarchical channels use '/' (e.g. 'commands/velocity')."
        )


def _validate_sensor_name(name: str) -> None:
    if not _SENSOR_NAME_RE.match(name):
        raise ChannelError(
            f"Invalid sensor name: '{name}'.  "
            "Must be lowercase alphanumeric + underscores, starting with a letter."
        )


def _validate_prefix(prefix: str) -> None:
    if not prefix:
        raise ChannelError("Key prefix must not be empty.")
    if not _PREFIX_RE.match(prefix):
        raise ChannelError(
            f"Invalid key prefix: '{prefix}'.  "
            "Must be lowercase alphanumeric + underscores, starting with a letter."
        )


# ── Data types ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class KeyExpression:
    """Parsed key expression."""

    prefix: str
    twin_uuid: str
    channel: str
    sensor_name: str | None = None

    @property
    def is_well_known(self) -> bool:
        """True if the channel is one of the standard well-known names."""
        return self.channel in WELL_KNOWN_CHANNELS

    @property
    def is_stream(self) -> bool:
        return self.channel in STREAM_CHANNELS

    @property
    def is_latest_value(self) -> bool:
        return self.channel in LATEST_VALUE_CHANNELS

    def __str__(self) -> str:
        """Render as a key-expression string."""
        base = f"{self.prefix}/{self.twin_uuid}/data/{self.channel}"
        if self.sensor_name:
            return f"{base}/{self.sensor_name}"
        return base


# ── Builder ─────────────────────────────────────────────────────────


def build_key(
    twin_uuid: str,
    channel: str,
    sensor_name: str | None = None,
    *,
    prefix: str = "cw",
) -> str:
    """Build a validated key-expression string.

    Args:
        twin_uuid: UUID-4 of the digital twin.
        channel: Channel name (e.g. ``"frames"``, ``"joint_states"``).
        sensor_name: Optional sensor qualifier (e.g. ``"default"``).
        prefix: Key prefix (default ``"cw"``).

    Returns:
        The full key expression, e.g. ``"cw/abc.../data/frames/default"``.

    Raises:
        ChannelError: If any component fails validation.
    """
    _validate_prefix(prefix)
    _validate_uuid(twin_uuid)
    _validate_channel_segment(channel)
    if sensor_name is not None:
        _validate_sensor_name(sensor_name)

    ke = KeyExpression(
        prefix=prefix,
        twin_uuid=twin_uuid,
        channel=channel,
        sensor_name=sensor_name,
    )
    return str(ke)


def build_wildcard(
    twin_uuid: str | None = None,
    channel: str | None = None,
    *,
    prefix: str = "cw",
) -> str:
    """Build a Zenoh-compatible wildcard key expression.

    Useful for subscribing to multiple channels or twins at once.

    Examples::

        build_wildcard()                              # "cw/*/data/**"
        build_wildcard(twin_uuid="abc...")             # "cw/abc.../data/**"
        build_wildcard(twin_uuid="abc...", channel="frames")
                                                      # "cw/abc.../data/frames/**"

    Note:
        The channel wildcard uses ``**`` (multi-level) so it matches both bare
        channel keys (``frames``) and keys with a sensor name
        (``frames/default``).

    Raises:
        ChannelError: If provided values fail validation.
    """
    _validate_prefix(prefix)
    if twin_uuid is not None:
        _validate_uuid(twin_uuid)
    if channel is not None:
        _validate_channel_segment(channel)

    twin_part = twin_uuid if twin_uuid else "*"
    if channel is not None:
        return f"{prefix}/{twin_part}/data/{channel}/**"
    return f"{prefix}/{twin_part}/data/**"


# ── Parser ──────────────────────────────────────────────────────────


def parse_key(key: str) -> KeyExpression:
    """Parse a key-expression string into a :class:`KeyExpression`.

    Raises:
        ChannelError: If the key does not match the canonical pattern.
    """
    parts = key.split("/")

    if len(parts) < 4:
        raise ChannelError(
            f"Key expression too short: '{key}'.  "
            "Expected at least '<prefix>/<twin_uuid>/data/<channel>'."
        )

    prefix = parts[0]
    if not prefix:
        raise ChannelError("Key prefix must not be empty.")
    twin_uuid = parts[1]
    data_segment = parts[2]
    channel = parts[3]

    if data_segment != "data":
        raise ChannelError(
            f"Expected 'data' as third segment, got '{data_segment}' in '{key}'."
        )

    _validate_uuid(twin_uuid)
    _validate_channel_segment(channel)

    sensor_name: str | None = None
    if len(parts) == 5:
        sensor_name = parts[4]
        _validate_sensor_name(sensor_name)
    elif len(parts) > 5:
        raise ChannelError(
            f"Key expression has too many segments ({len(parts)}): '{key}'.  "
            "Expected '<prefix>/<twin_uuid>/data/<channel>[/<sensor_name>]'."
        )

    return KeyExpression(
        prefix=prefix,
        twin_uuid=twin_uuid,
        channel=channel,
        sensor_name=sensor_name,
    )


def channel_from_key(key: str) -> str:
    """Extract just the channel name from a full key expression.

    Convenience wrapper around :func:`parse_key`.
    """
    return parse_key(key).channel


# ── Batch helpers ───────────────────────────────────────────────────


def build_keys(
    twin_uuid: str,
    channels: Sequence[str],
    *,
    prefix: str = "cw",
) -> dict[str, str]:
    """Build key expressions for multiple channels at once.

    Returns a ``{channel: key_expression}`` mapping.
    """
    return {ch: build_key(twin_uuid, ch, prefix=prefix) for ch in channels}


def is_valid_key(key: str) -> bool:
    """Return ``True`` if *key* is a valid canonical key expression."""
    try:
        parse_key(key)
        return True
    except ChannelError:
        return False
