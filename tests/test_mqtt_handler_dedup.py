"""Regression tests for MQTT handler accumulation (the "answer storm").

Background
----------
``BaseVideoStreamer._subscribe_to_answer()`` is invoked deep inside
``_setup_webrtc()``, which runs on every auto-reconnect cycle. Under the
old ``_add_handler`` semantics each call appended a *new* closure to
``MQTTClient._handlers[answer_topic]``. After N reconnects a single SFU
answer would fan out into N "Processing answer" invocations, and the
matching ``webrtc-candidate`` subscription would call
``self.pc.addIceCandidate(...)`` N times for each candidate — corrupting
aioice's checklist and producing a "connected but no media" zombie
state.

These tests pin the new idempotent semantics so the regression cannot
re-introduce itself.
"""

from unittest.mock import MagicMock, patch

import paho.mqtt.client as mqtt

from cyberwave.mqtt import CyberwaveMQTTClient


def _make_client() -> CyberwaveMQTTClient:
    """Construct a client with the underlying paho client mocked out."""
    with patch("cyberwave.mqtt.mqtt.Client") as mqtt_client_cls:
        mqtt_client = mqtt_client_cls.return_value
        mqtt_client.subscribe.return_value = (mqtt.MQTT_ERR_SUCCESS, 1)
        client = CyberwaveMQTTClient(
            mqtt_broker="localhost",
            mqtt_port=1883,
            mqtt_username="user",
            api_key="api_key_secret",
            auto_connect=False,
        )
        client.connected = True
    return client


def test_resubscribe_same_topic_replaces_handler_not_appends():
    """Two ``subscribe`` calls for one topic must leave a single handler.

    Reproduces the camera storm: every reconnect cycle the streamer
    registered a fresh ``on_answer`` closure for the same topic.
    """
    client = _make_client()

    h1 = MagicMock(name="handler-v1")
    h2 = MagicMock(name="handler-v2")

    client.subscribe("cyberwave/twin/UUID/webrtc-answer", h1)
    client.subscribe("cyberwave/twin/UUID/webrtc-answer", h2)

    handlers = client._handlers["cyberwave/twin/UUID/webrtc-answer"]
    assert len(handlers) == 1, (
        f"Expected idempotent re-subscribe (single handler), "
        f"found {len(handlers)} handlers — handler accumulation bug regressed."
    )
    assert handlers[0] is h2, "Latest registration should win (replace semantics)."


def test_one_message_fires_one_handler_after_many_resubscribes():
    """End-to-end: 5 ``subscribe`` calls → 1 ``on_answer`` invocation per message.

    This is the precise log signature from the production driver
    (``cyberwave-driver-d144a144``): five "Processing answer targeted at
    edge" lines fired for a single SFU publish because five stale
    closures had accumulated on ``_handlers[webrtc-answer]``.
    """
    client = _make_client()
    topic = "cyberwave/twin/UUID/webrtc-answer"

    handler = MagicMock(name="on_answer")
    for _ in range(5):
        client.subscribe(topic, handler)

    client._trigger_handlers(topic, {"type": "answer", "target": "edge"})

    assert handler.call_count == 1, (
        f"Single MQTT message should trigger handler exactly once after "
        f"5 re-subscribes, fired {handler.call_count} times instead."
    )


def test_resubscribe_skips_redundant_broker_subscribe_roundtrip():
    """Re-registering a topic should not generate a fresh SUBACK round-trip.

    The old code issued ``client.subscribe(topic)`` to paho on every
    call regardless of whether the topic was already subscribed —
    visible in driver logs as ``mid=49`` and ``mid=51`` for the same
    topic moments apart. The new code skips the broker call when only
    the handler is being replaced.
    """
    client = _make_client()

    paho_subscribe = client.client.subscribe
    paho_subscribe.reset_mock()

    h1 = MagicMock()
    h2 = MagicMock()
    client.subscribe("cyberwave/twin/UUID/webrtc-answer", h1)
    client.subscribe("cyberwave/twin/UUID/webrtc-answer", h2)

    assert paho_subscribe.call_count == 1, (
        f"Expected single broker SUBSCRIBE for an already-subscribed topic, "
        f"got {paho_subscribe.call_count} round-trips."
    )


def test_distinct_topics_keep_independent_handlers():
    """Replace semantics must scope to a single topic key.

    The frontend's twin updates rely on registering handlers across
    several distinct topics simultaneously (position, joints, generic
    twin wildcard). Idempotent registration on one topic must not
    affect registrations on other topics.
    """
    client = _make_client()

    h_position = MagicMock()
    h_joints = MagicMock()
    h_wildcard = MagicMock()

    client.subscribe("cyberwave/twin/UUID/position", h_position)
    client.subscribe("cyberwave/twin/UUID/joint_states", h_joints)
    client.subscribe("cyberwave/twin/UUID/+", h_wildcard)

    assert client._handlers["cyberwave/twin/UUID/position"] == [h_position]
    assert client._handlers["cyberwave/twin/UUID/joint_states"] == [h_joints]
    assert client._handlers["cyberwave/twin/UUID/+"] == [h_wildcard]


def test_unsubscribe_then_subscribe_treats_topic_as_new():
    """``unsubscribe`` must reset the bookkeeping so the next subscribe
    issues a fresh broker SUBSCRIBE — otherwise an explicit unsubscribe
    would leave the broker out of sync with our local handler map.
    """
    client = _make_client()
    topic = "cyberwave/twin/UUID/webrtc-answer"

    h1 = MagicMock()
    client.subscribe(topic, h1)

    paho_subscribe = client.client.subscribe
    paho_subscribe.reset_mock()

    client.unsubscribe(topic)
    assert topic not in client._handlers

    h2 = MagicMock()
    client.subscribe(topic, h2)

    assert paho_subscribe.call_count == 1, (
        "After unsubscribe(), the next subscribe() must hit the broker again."
    )
    assert client._handlers[topic] == [h2]
