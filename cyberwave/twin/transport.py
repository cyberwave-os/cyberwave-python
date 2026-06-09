"""Outbound MQTT transport for twin command handlers."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Literal

DEFAULT_BURST_DURATION_S = 1.0
DEFAULT_BURST_RATE_HZ = 20.0

from typing_extensions import Self

from ..constants import SOURCE_TYPE_SIM_TELE, SOURCE_TYPE_TELE
from ..manifest.driver_config import (
    mqtt_topic_slugs,
    resolve_outbound_topic_slug,
    supported_mqtt_commands,
)

from ._helpers import (
    _default_control_source_type,
    _normalize_locomotion_source_type,
    motion_outbound_requires_policy,
)

logger = logging.getLogger(__name__)

OutboundChannel = Literal["twin_command", "joint_update", "sensor_actuation"]


@dataclass(frozen=True)
class ResolvedOutbound:
    """Resolved MQTT topic and payload for an outbound twin command."""

    topic: str
    payload: dict[str, Any]
    command: str
    source_type: str


class TwinTransportMixin:
    """Outbound command resolution, catalog validation, and MQTT publish."""

    _outbound_log: list[ResolvedOutbound]

    def _init_transport_state(self) -> None:
        if not hasattr(self, "_outbound_log"):
            self._outbound_log = []

    def _ensure_mqtt_connected(self: Self) -> None:
        """Connect to the MQTT broker before outbound publish when disconnected."""
        mqtt = getattr(self.client, "mqtt", None)
        if mqtt is not None and not mqtt.connected:
            mqtt.connect()

    def _resolve_topic_and_payload(
        self: Self,
        *,
        command: str,
        data: dict[str, Any] | None = None,
        channel: OutboundChannel = "twin_command",
        source_type: str | None = None,
        sensor_id: str | None = None,
    ) -> ResolvedOutbound:
        """Map command + catalog to topic + payload."""
        self._init_transport_state()
        resolved_source = self._resolve_outbound_source_type(source_type)
        schema = self.commands.get_schema()
        self._validate_outbound_command(command, schema)

        topic_slug = resolve_outbound_topic_slug(channel=channel, bundle=schema)
        topic_prefix = getattr(getattr(self.client, "config", None), "topic_prefix", None) or ""
        topic = f"{topic_prefix}{topic_slug.format(twin_uuid=self.uuid)}"

        if channel == "joint_update":
            payload = self._legacy_joint_update_payload(
                dict(data or {}),
                source_type=resolved_source,
            )
        else:
            payload = {
                "source_type": resolved_source,
                "command": command,
                "data": dict(data or {}),
                "timestamp": time.time(),
            }
            if sensor_id is not None:
                payload["sensor_id"] = sensor_id

        resolved = ResolvedOutbound(
            topic=topic,
            payload=payload,
            command=command,
            source_type=resolved_source,
        )
        logger.debug(
            "twin outbound resolved: twin=%s command=%s topic=%s",
            self.uuid,
            command,
            topic,
        )
        return resolved

    def _publish_resolved(self: Self, resolved: ResolvedOutbound) -> None:
        """Publish resolved outbound to MQTT and retain a test log entry."""
        if motion_outbound_requires_policy(resolved.command):
            self._prepare_outbound_command()
        self._init_transport_state()
        self._outbound_log.append(resolved)
        mqtt = getattr(self.client, "mqtt", None)
        if mqtt is not None:
            self._ensure_mqtt_connected()
            mqtt.publish(resolved.topic, resolved.payload)
        logger.debug(
            "twin outbound publish: twin=%s command=%s topic=%s",
            self.uuid,
            resolved.command,
            resolved.topic,
        )

    def publish_command(
        self: Self,
        command: str,
        data: dict[str, Any] | None = None,
        *,
        source_type: str | None = None,
    ) -> None:
        """Publish a single catalog command on the twin command topic."""
        resolved = self._resolve_topic_and_payload(
            command=command,
            data=dict(data or {}),
            source_type=source_type,
        )
        self._publish_resolved(resolved)

    def publish_command_burst(
        self: Self,
        command: str,
        data: dict[str, Any] | None = None,
        *,
        duration_s: float = DEFAULT_BURST_DURATION_S,
        rate_hz: float = DEFAULT_BURST_RATE_HZ,
        stop_command: str = "stop",
        source_type: str | None = None,
    ) -> None:
        """Publish *command* repeatedly, then *stop_command* (edge velocity watchdog)."""
        payload = dict(data or {})
        if command == "move_forward" and "linear_x" in payload and "angular_z" not in payload:
            payload = {**payload, "angular_z": 0.0}

        if duration_s <= 0:
            self.publish_command(command, payload, source_type=source_type)
            return

        interval = 1.0 / max(float(rate_hz), 1.0)
        iterations = max(1, math.ceil(float(duration_s) * max(float(rate_hz), 1.0)))
        for index in range(iterations):
            self.publish_command(command, payload, source_type=source_type)
            if index < iterations - 1:
                time.sleep(interval)
        self.publish_command(stop_command, {}, source_type=source_type)

    def _legacy_joint_update_payload(
        self: Self,
        data: dict[str, Any],
        *,
        source_type: str,
    ) -> dict[str, Any]:
        """Build SO101/edge-compatible joint bus payload (no command envelope).

        Matches ``CyberwaveMQTTClient.update_joints_state`` flat/aggregated shapes
        consumed by ``create_joint_state_callback`` on the joint ``/update`` topic.
        The ``mode`` argument on ``joints.set`` is accepted at the API layer but only
        ``update_mode`` is emitted for non-absolute modes; absolute publishes flat keys.
        """
        mode = str(data.get("mode", "absolute"))
        timestamp = data.get("timestamp")
        positions = data.get("positions")
        velocities = data.get("velocities")
        efforts = data.get("efforts")

        if mode != "absolute":
            message: dict[str, Any] = {
                "source_type": source_type,
                "update_mode": mode,
            }
            if isinstance(positions, dict) and positions:
                message["positions"] = positions
            if isinstance(velocities, dict) and velocities:
                message["velocities"] = velocities
            if isinstance(efforts, dict) and efforts:
                message["efforts"] = efforts
            if timestamp is not None:
                message["timestamp"] = timestamp
            return message

        if not isinstance(positions, dict) or not positions:
            raise ValueError("joint_update requires a non-empty positions map")

        if (
            timestamp is not None
            or (isinstance(velocities, dict) and velocities)
            or (isinstance(efforts, dict) and efforts)
        ):
            message = {
                "source_type": source_type,
                "positions": positions,
                "timestamp": timestamp if timestamp is not None else time.time(),
            }
            if isinstance(velocities, dict) and velocities:
                message["velocities"] = velocities
            if isinstance(efforts, dict) and efforts:
                message["efforts"] = efforts
            return message

        return {"source_type": source_type, **positions}

    def _resolve_outbound_source_type(self: Self, source_type: str | None) -> str:
        if source_type is None:
            return _default_control_source_type(self.client)
        normalized = _normalize_locomotion_source_type(source_type)
        if normalized in {SOURCE_TYPE_SIM_TELE, SOURCE_TYPE_TELE}:
            return normalized
        return _default_control_source_type(self.client)

    def _validate_outbound_command(
        self: Self,
        command: str,
        schema: dict[str, Any],
    ) -> None:
        if command in {"joint_update", "stop"}:
            return
        supported = supported_mqtt_commands(schema)
        topic_slugs = mqtt_topic_slugs(schema)
        if topic_slugs and not supported:
            raise ValueError(
                f"Command {command!r} is not in the MQTT catalog for this twin. "
                f"Allowed: {supported}"
            )
        if not supported:
            return
        if command not in supported:
            raise ValueError(
                f"Command {command!r} is not in the MQTT catalog for this twin. "
                f"Allowed: {supported}"
            )
