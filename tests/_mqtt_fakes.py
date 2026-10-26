"""Fakes shared by the MQTT request/response tests: a paho stand-in that
acknowledges subscriptions like a broker, and v5 messages with properties."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import paho.mqtt.client as paho
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.properties import Properties

from cyberwave.mqtt import CyberwaveMQTTClient


def v5_message(
    topic: str,
    payload: dict[str, Any] | bytes,
    *,
    response_topic: str | None = None,
    correlation: bytes | None = None,
    user_properties: list[tuple[str, str]] | None = None,
) -> paho.MQTTMessage:
    message = paho.MQTTMessage(topic=topic.encode())
    message.payload = (
        payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    )
    properties = Properties(PacketTypes.PUBLISH)
    if response_topic is not None:
        properties.ResponseTopic = response_topic
    if correlation is not None:
        properties.CorrelationData = correlation
    if user_properties:
        properties.UserProperty = user_properties
    message.properties = properties
    return message


def fake_paho(client: CyberwaveMQTTClient) -> MagicMock:
    """A paho stand-in that acknowledges subscriptions like a broker would."""
    paho_client = MagicMock()
    mids = iter(range(1, 1000))

    def subscribe(topic: str, *args: Any, **kwargs: Any) -> tuple[int, int]:
        mid = next(mids)
        threading.Timer(0.01, client._on_subscribe, (None, None, mid, [0])).start()
        return paho.MQTT_ERR_SUCCESS, mid

    paho_client.subscribe.side_effect = subscribe
    paho_client.publish.return_value = SimpleNamespace(rc=paho.MQTT_ERR_SUCCESS)
    return paho_client


def mqtt_client(protocol: int = paho.MQTTv5, **kwargs: Any) -> CyberwaveMQTTClient:
    client = CyberwaveMQTTClient(
        api_key="test-api-key", protocol=protocol, topic_prefix="dev", **kwargs
    )
    client.client = fake_paho(client)
    client.connected = True
    return client
