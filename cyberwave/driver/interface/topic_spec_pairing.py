"""Built-in MQTT ↔ Zenoh channel pairings for :mod:`cyberwave.driver` topic specs."""

from __future__ import annotations

from cyberwave.manifest.driver_config import (
    JOINT_UPDATE_TOPIC_SLUG,
    TWIN_COMMAND_TOPIC_SLUG,
    TWIN_IMU_TOPIC_SLUG,
)

from .args import TopicSpec

_NS_LEAF_TO_ZENOH: dict[tuple[str, str], str] = {
    ("joint", "update"): "joint_states",
}

_SLUG_TO_ZENOH: dict[str, str] = {
    TWIN_IMU_TOPIC_SLUG: "imu",
}


def forbid_zenoh_on_topic(spec: TopicSpec) -> None:
    if not spec.enable_mqtt:
        return
    if spec.topic_slug == TWIN_COMMAND_TOPIC_SLUG:
        raise ValueError("twin/command must stay MQTT-only; enable_zenoh is not allowed")
    if spec.namespace == "twin" and spec.leaf == "command":
        raise ValueError("twin/command must stay MQTT-only; enable_zenoh is not allowed")


def default_zenoh_channel_for_topic(spec: TopicSpec) -> str:
    if spec.topic_slug is not None:
        channel = _SLUG_TO_ZENOH.get(spec.topic_slug)
        if channel is not None:
            return channel
        raise ValueError(
            f"No default Zenoh channel for topic_slug={spec.topic_slug!r}; "
            "set zenoh_channel="
        )
    assert spec.namespace is not None and spec.leaf is not None
    channel = _NS_LEAF_TO_ZENOH.get((spec.namespace, spec.leaf))
    if channel is not None:
        return channel
    raise ValueError(
        f"No default Zenoh channel for {spec.namespace}/{spec.leaf}; set zenoh_channel="
    )


def resolve_zenoh_channel(spec: TopicSpec) -> str:
    if spec.zenoh_channel is not None and spec.zenoh_channel.strip():
        return spec.zenoh_channel.strip()
    if spec.enable_mqtt:
        return default_zenoh_channel_for_topic(spec)
    raise ValueError("zenoh_channel is required when enable_mqtt is False")
