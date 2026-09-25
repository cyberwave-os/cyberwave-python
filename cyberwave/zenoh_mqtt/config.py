"""Bridge configuration — environment-driven settings for the Zenoh-MQTT bridge.

All fields fall back to environment variables when left at their defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _parse_bool_env(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class BridgeConfig:
    """Configuration for the Zenoh-MQTT bridge.

    Attributes:
        enabled: Master switch. Env: ``CYBERWAVE_BRIDGE_ENABLED``.
        twin_uuids: Twins whose channels are bridged. Env:
            ``CYBERWAVE_BRIDGE_TWIN_UUIDS`` (comma-separated).
        outbound_channels: Zenoh channels forwarded to MQTT.
            Env: ``CYBERWAVE_BRIDGE_OUTBOUND_CHANNELS`` (comma-separated).
            Defaults to ``["model_output", "event", "model_health"]``.
        inbound_topics: MQTT topic suffixes forwarded to Zenoh.
            Env: ``CYBERWAVE_BRIDGE_INBOUND_TOPICS`` (comma-separated).
            Defaults to ``["commands/sync_workflows", "alert"]``.
        mqtt_topic_prefix: Prefix added to MQTT topics (e.g. ``"staging"``).
            Env: ``CYBERWAVE_MQTT_TOPIC_PREFIX``.
        zenoh_key_prefix: Prefix used in Zenoh key expressions.
            Env: ``CYBERWAVE_BRIDGE_KEY_PREFIX``.
        queue_dir: Directory for the persistent offline queue.
            Env: ``CYBERWAVE_BRIDGE_QUEUE_DIR``.
        queue_max_bytes: Maximum total queue size in bytes before oldest
            messages are dropped.  Env: ``CYBERWAVE_BRIDGE_QUEUE_MAX_BYTES``.
        drain_batch_size: Number of queued messages published per drain cycle.
            Env: ``CYBERWAVE_BRIDGE_DRAIN_BATCH_SIZE``.
        mqtt_qos: MQTT QoS level for bridge-published messages.
            Env: ``CYBERWAVE_BRIDGE_MQTT_QOS``.
    """

    enabled: bool | None = None
    twin_uuids: list[str] = field(default_factory=list)
    outbound_channels: list[str] = field(default_factory=list)
    inbound_topics: list[str] = field(default_factory=list)
    mqtt_topic_prefix: str = ""
    zenoh_key_prefix: str = "cw"
    queue_dir: str = ""
    queue_max_bytes: int = 0
    drain_batch_size: int = 64
    mqtt_qos: int = 1

    def __post_init__(self) -> None:
        if self.enabled is None:
            self.enabled = _parse_bool_env(
                os.environ.get("CYBERWAVE_BRIDGE_ENABLED"),
            )

        if not self.twin_uuids:
            raw = os.environ.get("CYBERWAVE_BRIDGE_TWIN_UUIDS", "")
            if raw:
                self.twin_uuids = [u.strip() for u in raw.split(",") if u.strip()]

        if not self.outbound_channels:
            raw = os.environ.get("CYBERWAVE_BRIDGE_OUTBOUND_CHANNELS", "")
            if raw:
                self.outbound_channels = [
                    c.strip() for c in raw.split(",") if c.strip()
                ]
            else:
                self.outbound_channels = ["model_output", "event", "model_health"]

        if not self.inbound_topics:
            raw = os.environ.get("CYBERWAVE_BRIDGE_INBOUND_TOPICS", "")
            if raw:
                self.inbound_topics = [t.strip() for t in raw.split(",") if t.strip()]
            else:
                self.inbound_topics = ["commands/sync_workflows", "alert"]

        if not self.mqtt_topic_prefix:
            self.mqtt_topic_prefix = os.environ.get("CYBERWAVE_MQTT_TOPIC_PREFIX", "")

        if not self.zenoh_key_prefix:
            self.zenoh_key_prefix = os.environ.get("CYBERWAVE_BRIDGE_KEY_PREFIX", "cw")

        if not self.queue_dir:
            self.queue_dir = os.environ.get(
                "CYBERWAVE_BRIDGE_QUEUE_DIR",
                "/tmp/cyberwave_bridge_queue",
            )

        if self.queue_max_bytes <= 0:
            raw_max = os.environ.get("CYBERWAVE_BRIDGE_QUEUE_MAX_BYTES", "")
            self.queue_max_bytes = int(raw_max) if raw_max else 50 * 1024 * 1024

        raw_batch = os.environ.get("CYBERWAVE_BRIDGE_DRAIN_BATCH_SIZE", "")
        if raw_batch:
            self.drain_batch_size = int(raw_batch)

        raw_qos = os.environ.get("CYBERWAVE_BRIDGE_MQTT_QOS", "")
        if raw_qos:
            self.mqtt_qos = int(raw_qos)
