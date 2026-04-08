"""Bidirectional mapping between Zenoh key expressions and MQTT topics.

Outbound (Zenoh → MQTT)::

    cw/{twin}/data/{channel}[/{sensor}]  →  {prefix}cyberwave/twin/{twin}/{channel}

Inbound (MQTT → Zenoh)::

    {prefix}cyberwave/twin/{twin}/{suffix}  →  cw/{twin}/data/{suffix_normalised}

The ``{suffix_normalised}`` replaces ``/`` with ``_`` so the result is a valid
single-segment Zenoh channel name (e.g. ``commands/sync_workflows`` becomes
``commands_sync_workflows``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}$"
)


@dataclass(frozen=True, slots=True)
class OutboundMapping:
    """Result of converting a Zenoh key to an MQTT topic."""

    mqtt_topic: str
    twin_uuid: str
    channel: str


@dataclass(frozen=True, slots=True)
class InboundMapping:
    """Result of converting an MQTT topic to a Zenoh key."""

    zenoh_key: str
    twin_uuid: str
    channel: str


def zenoh_to_mqtt(
    zenoh_key: str,
    *,
    mqtt_prefix: str = "",
) -> OutboundMapping | None:
    """Map a Zenoh key expression to the corresponding MQTT topic.

    Returns ``None`` if the key does not match the canonical pattern.
    """
    parts = zenoh_key.split("/")
    # Minimum: prefix/twin_uuid/data/channel (4 segments)
    if len(parts) < 4 or parts[2] != "data":
        return None

    twin_uuid = parts[1]
    if not _UUID_RE.match(twin_uuid):
        return None

    channel = parts[3]
    base = f"cyberwave/twin/{twin_uuid}/{channel}"
    if mqtt_prefix:
        base = f"{mqtt_prefix}{base}"

    return OutboundMapping(mqtt_topic=base, twin_uuid=twin_uuid, channel=channel)


def mqtt_to_zenoh(
    mqtt_topic: str,
    *,
    mqtt_prefix: str = "",
    zenoh_prefix: str = "cw",
) -> InboundMapping | None:
    """Map an MQTT topic to the corresponding Zenoh key expression.

    Returns ``None`` if the topic does not match the expected pattern.
    """
    topic = mqtt_topic
    if mqtt_prefix and topic.startswith(mqtt_prefix):
        topic = topic[len(mqtt_prefix) :]

    # Expected: cyberwave/twin/{twin_uuid}/{suffix...}
    if not topic.startswith("cyberwave/twin/"):
        return None

    remainder = topic[len("cyberwave/twin/") :]
    slash_idx = remainder.find("/")
    if slash_idx == -1:
        return None

    twin_uuid = remainder[:slash_idx]
    if not _UUID_RE.match(twin_uuid):
        return None

    suffix = remainder[slash_idx + 1 :]
    if not suffix:
        return None

    channel = suffix.replace("/", "_")
    zenoh_key = f"{zenoh_prefix}/{twin_uuid}/data/{channel}"

    return InboundMapping(zenoh_key=zenoh_key, twin_uuid=twin_uuid, channel=channel)


def build_outbound_subscriptions(
    twin_uuids: list[str],
    channels: list[str],
    *,
    zenoh_prefix: str = "cw",
) -> list[str]:
    """Build the list of Zenoh key expressions to subscribe to for outbound bridging."""
    keys: list[str] = []
    for twin in twin_uuids:
        for ch in channels:
            keys.append(f"{zenoh_prefix}/{twin}/data/{ch}")
            # Also subscribe to sensor-qualified keys (e.g. model_output/default)
            keys.append(f"{zenoh_prefix}/{twin}/data/{ch}/*")
    return keys


def build_inbound_subscriptions(
    twin_uuids: list[str],
    topic_suffixes: list[str],
    *,
    mqtt_prefix: str = "",
) -> list[str]:
    """Build the list of MQTT topic filters to subscribe to for inbound bridging."""
    topics: list[str] = []
    for twin in twin_uuids:
        for suffix in topic_suffixes:
            topic = f"cyberwave/twin/{twin}/{suffix}"
            if mqtt_prefix:
                topic = f"{mqtt_prefix}{topic}"
            topics.append(topic)
    return topics
