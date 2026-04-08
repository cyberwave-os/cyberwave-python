"""Cyberwave Zenoh-MQTT bridge — bidirectional forwarding between local
Zenoh data bus and cloud MQTT broker.

Usage::

    from cyberwave.zenoh_mqtt import ZenohMqttBridge, BridgeConfig

    bridge = ZenohMqttBridge(
        config=BridgeConfig(
            twin_uuids=["abc-def-..."],
            outbound_channels=["model_output", "event", "model_health"],
        ),
        mqtt_host="<mqtt_host>",
        mqtt_port=8883,
        mqtt_password="<api_key>",
    )
    bridge.start()
"""

from .bridge import ZenohMqttBridge
from .config import BridgeConfig
from .queue import OfflineQueue, QueuedMessage
from .topic_mapping import (
    InboundMapping,
    OutboundMapping,
    build_inbound_subscriptions,
    build_outbound_subscriptions,
    mqtt_to_zenoh,
    zenoh_to_mqtt,
)

__all__ = [
    "ZenohMqttBridge",
    "BridgeConfig",
    "OfflineQueue",
    "QueuedMessage",
    "OutboundMapping",
    "InboundMapping",
    "zenoh_to_mqtt",
    "mqtt_to_zenoh",
    "build_outbound_subscriptions",
    "build_inbound_subscriptions",
]
