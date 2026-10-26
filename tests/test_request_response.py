"""Generic MQTT v5 request/response (``CyberwaveMQTTClient.request``).

Nothing here is a twin command: the topics are arbitrary, which is the point.
The twin-command specialisation is covered by ``test_command_response.py``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import paho.mqtt.client as paho
import pytest

from _mqtt_fakes import mqtt_client as _client
from _mqtt_fakes import v5_message as _message
from cyberwave.exceptions import (
    CommandResponseTimeoutError,
    CyberwaveMQTTError,
    RequestTimeoutError,
)
from cyberwave.mqtt import CyberwaveMQTTClient
from cyberwave.mqtt.request_response import (
    message_property,
    new_correlation_id,
    reply_to,
)

REQUEST_TOPIC = "devcyberwave/svc/planner/request"
TWIN = "5f2c7d0e-8a51-4d0f-9e0b-2a9c1f4d6b11"


def _inbox(client_id: str) -> str:
    return f"devcyberwave/svc/planner/{client_id}/reply"


def _responder(client: CyberwaveMQTTClient, reply: Any, *, noise: bool = False) -> list:
    """Answer every request on REQUEST_TOPIC like a responder service would."""
    seen: list[Any] = []

    def publish(topic: str, payload: Any, qos: int = 0, properties: Any = None) -> Any:
        if topic == REQUEST_TOPIC and properties is not None:
            seen.append((json.loads(payload), properties))
            if noise:  # a reply to somebody else's request arrives first
                client._on_message(
                    None,
                    None,
                    _message(
                        properties.ResponseTopic,
                        {"for": "someone else"},
                        correlation=b"other",
                    ),
                )
            client._on_message(
                None,
                None,
                _message(
                    properties.ResponseTopic,
                    reply,
                    correlation=properties.CorrelationData,
                ),
            )
        return SimpleNamespace(rc=paho.MQTT_ERR_SUCCESS)

    client.client.publish.side_effect = publish
    return seen


def test_request_on_an_arbitrary_topic_pair() -> None:
    client = _client()
    seen = _responder(client, {"route": ["a", "b"]})

    reply = client.request(
        REQUEST_TOPIC,
        {"goal": "b"},
        response_topic=_inbox(client.client_id),
        timeout_s=2,
    )

    assert reply == {"route": ["a", "b"]}
    ((body, properties),) = seen
    assert body["goal"] == "b"
    assert properties.ResponseTopic == _inbox(client.client_id)
    assert properties.CorrelationData  # fresh, non-empty
    subscribed = client.client.subscribe.call_args.args[0]
    assert subscribed == _inbox(client.client_id)
    assert client._pending_requests == {}


def test_a_reply_for_another_request_is_ignored() -> None:
    client = _client()
    _responder(client, {"mine": True}, noise=True)

    assert client.request(
        REQUEST_TOPIC, {}, response_topic=_inbox(client.client_id), timeout_s=2
    ) == {"mine": True}


def test_each_request_gets_its_own_correlation_and_reuses_the_inbox() -> None:
    client = _client()
    seen = _responder(client, {"ok": True})

    client.request(
        REQUEST_TOPIC, {}, response_topic=_inbox(client.client_id), timeout_s=2
    )
    client.request(
        REQUEST_TOPIC, {}, response_topic=_inbox(client.client_id), timeout_s=2
    )

    first, second = (properties.CorrelationData for _, properties in seen)
    assert first != second
    client.client.subscribe.assert_called_once()


def test_a_non_json_reply_is_returned_as_text() -> None:
    client = _client()

    def publish(topic: str, payload: Any, qos: int = 0, properties: Any = None) -> Any:
        if topic == REQUEST_TOPIC:
            client._on_message(
                None,
                None,
                _message(
                    properties.ResponseTopic,
                    b"plain text",
                    correlation=properties.CorrelationData,
                ),
            )
        return SimpleNamespace(rc=paho.MQTT_ERR_SUCCESS)

    client.client.publish.side_effect = publish

    assert client.request(
        REQUEST_TOPIC, {}, response_topic=_inbox(client.client_id), timeout_s=2
    ) == ("plain text")


def test_timeout_is_a_request_timeout_not_a_command_one() -> None:
    client = _client()

    with pytest.raises(RequestTimeoutError, match="planner/request") as raised:
        client.request(
            REQUEST_TOPIC, {}, response_topic=_inbox(client.client_id), timeout_s=0.05
        )

    assert not isinstance(raised.value, CommandResponseTimeoutError)
    assert isinstance(raised.value, TimeoutError)
    assert client._pending_requests == {}


def test_the_command_timeout_is_still_catchable_as_a_request_timeout() -> None:
    assert issubclass(CommandResponseTimeoutError, RequestTimeoutError)


def test_a_client_set_to_mqtt_311_cannot_make_requests() -> None:
    client = _client(protocol=paho.MQTTv311)

    with pytest.raises(CyberwaveMQTTError, match="CYBERWAVE_MQTT_PROTOCOL=311"):
        client.request(
            REQUEST_TOPIC, {}, response_topic=_inbox(client.client_id), timeout_s=2
        )
    with pytest.raises(CyberwaveMQTTError, match="MQTT v5"):
        client.request_command(TWIN, "arm", timeout_s=2)

    client.client.publish.assert_not_called()
    client.client.subscribe.assert_not_called()


def test_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError):
        (client := _client()).request(
            REQUEST_TOPIC, {}, response_topic=_inbox(client.client_id), timeout_s=0
        )


def test_an_unpublishable_request_raises_instead_of_waiting() -> None:
    client = _client()
    client.client.publish.return_value = SimpleNamespace(rc=paho.MQTT_ERR_NO_CONN)

    with pytest.raises(CyberwaveMQTTError, match="could not publish"):
        client.request(
            REQUEST_TOPIC, {}, response_topic=_inbox(client.client_id), timeout_s=2
        )


# -- the responder side ----------------------------------------------------------------


def test_reply_to_answers_on_the_response_topic_echoing_the_correlation() -> None:
    paho_client = MagicMock()
    correlation = new_correlation_id()
    request = _message(
        REQUEST_TOPIC,
        {},
        response_topic="devcyberwave/x/reply",
        correlation=correlation,
    )

    assert reply_to(paho_client, request, {"done": True})

    topic, body = paho_client.publish.call_args.args
    assert topic == "devcyberwave/x/reply"
    assert json.loads(body) == {"done": True}
    kwargs = paho_client.publish.call_args.kwargs
    assert kwargs["qos"] == 1
    assert kwargs["properties"].CorrelationData == correlation


def test_reply_to_without_a_response_topic_publishes_nothing() -> None:
    paho_client = MagicMock()

    assert not reply_to(paho_client, _message(REQUEST_TOPIC, {}), {"done": True})
    paho_client.publish.assert_not_called()


def test_reply_to_uses_an_explicit_topic_over_the_requesters() -> None:
    """A responder that validates the Response Topic passes its own choice."""
    paho_client = MagicMock()
    request = _message(
        REQUEST_TOPIC,
        {},
        response_topic="devcyberwave/attacker/chosen",
        correlation=b"c",
    )

    reply_to(paho_client, request, "ok", topic="devcyberwave/x/validated")

    assert paho_client.publish.call_args.args == ("devcyberwave/x/validated", "ok")


def test_message_property_is_none_without_v5_properties() -> None:
    message = paho.MQTTMessage(topic=b"t")
    assert message_property(message, "ResponseTopic") is None
