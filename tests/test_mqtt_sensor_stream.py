"""Shared MqttSensorStreamHandle base behavior."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cyberwave.exceptions import TwinStateTimeoutError, TwinStateUnavailableError
from cyberwave.consumers.mqtt_snapshot import MqttSensorStreamHandle


def _identity(payload: dict) -> dict:
    return dict(payload)


def _clientless_twin():
    """Twin whose client has no MQTT — mqtt_client_for() raises."""
    return SimpleNamespace(
        uuid="t1",
        client=SimpleNamespace(mqtt=None, config=SimpleNamespace(topic_prefix="")),
        driver=SimpleNamespace(get_mqtt_schema=lambda: {}),
    )


def _fake_twin():
    callbacks: dict[str, object] = {}

    def subscribe(topic, callback, **kwargs):
        callbacks[topic] = callback

    mqtt = MagicMock()
    mqtt.subscribe = subscribe
    mqtt.connected = True
    client = SimpleNamespace(mqtt=mqtt, config=SimpleNamespace(topic_prefix=""))
    twin = SimpleNamespace(
        uuid="t1",
        client=client,
        driver=SimpleNamespace(get_mqtt_schema=lambda: {}),
        _ensure_mqtt_connected=lambda: None,
    )
    return twin, callbacks


def test_get_latest_returns_decoded_value() -> None:
    twin, callbacks = _fake_twin()
    handle = MqttSensorStreamHandle(twin)
    sub = handle._register_callback("imu", _identity, lambda v: None)
    callbacks["cyberwave/twin/t1/imu"]({"type": "imu", "x": 1})
    assert handle._get_latest("imu", _identity, timeout=0.0) == {"type": "imu", "x": 1}
    sub.cancel()


def test_get_latest_waits_for_a_fresh_message_on_repeat_calls() -> None:
    """Regression test: a call must not return a stale cached value forever —
    it should wait (up to timeout) for a message newer than what's already
    cached, rather than treating "any message ever received" as "always fresh".
    """
    twin, callbacks = _fake_twin()
    handle = MqttSensorStreamHandle(twin)
    handle._register_callback("imu", _identity, lambda v: None)
    topic = "cyberwave/twin/t1/imu"
    callbacks[topic]({"v": 1})
    assert handle._get_latest("imu", _identity, timeout=0.0) == {"v": 1}

    def _publish_later() -> None:
        threading.Event().wait(timeout=0.05)
        callbacks[topic]({"v": 2})

    threading.Thread(target=_publish_later).start()
    assert handle._get_latest("imu", _identity, timeout=2.0) == {"v": 2}


def test_get_latest_falls_back_to_cached_value_without_a_fresh_message() -> None:
    """If nothing new arrives within timeout, still return the last known value
    instead of raising — the fallback that keeps `timeout=0.0` reads fast."""
    twin, callbacks = _fake_twin()
    handle = MqttSensorStreamHandle(twin)
    handle._register_callback("imu", _identity, lambda v: None)
    callbacks["cyberwave/twin/t1/imu"]({"v": 1})
    assert handle._get_latest("imu", _identity, timeout=0.05) == {"v": 1}


def test_get_latest_times_out_without_message() -> None:
    twin, _ = _fake_twin()
    handle = MqttSensorStreamHandle(twin)
    with pytest.raises(TwinStateTimeoutError, match="No MQTT 'imu'"):
        handle._get_latest("imu", _identity, timeout=0.05)


def test_on_update_fires_and_cancel_stops() -> None:
    twin, callbacks = _fake_twin()
    handle = MqttSensorStreamHandle(twin)
    received: list = []
    sub = handle._register_callback("imu", _identity, received.append)
    cb = callbacks["cyberwave/twin/t1/imu"]
    cb({"v": 1})
    sub.cancel()
    cb({"v": 2})
    assert received == [{"v": 1}]


def test_callback_exception_does_not_break_listener() -> None:
    twin, callbacks = _fake_twin()
    handle = MqttSensorStreamHandle(twin)
    handle._register_callback(
        "imu", _identity, lambda v: (_ for _ in ()).throw(ValueError)
    )
    callbacks["cyberwave/twin/t1/imu"]({"v": 1})  # must not raise
    assert handle._get_latest("imu", _identity, timeout=0.0) == {"v": 1}


def test_ensure_stream_attaches_once_under_concurrency() -> None:
    twin, _ = _fake_twin()
    subscribe_calls: list = []

    def counting_subscribe(topic, callback, **kwargs):
        subscribe_calls.append(topic)

    twin.client.mqtt.subscribe = counting_subscribe

    handle = MqttSensorStreamHandle(twin)
    start = threading.Event()

    def worker() -> None:
        start.wait()
        handle._ensure_stream("imu", _identity)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join()

    assert subscribe_calls.count("cyberwave/twin/t1/imu") == 1


def test_get_latest_fails_fast_without_mqtt_client() -> None:
    """No MQTT client -> the actionable unavailable error, not a slow timeout."""
    handle = MqttSensorStreamHandle(_clientless_twin())
    with pytest.raises(TwinStateUnavailableError, match="MQTT client is not available"):
        handle._get_latest("imu", _identity, timeout=5.0)


def test_register_callback_fails_fast_without_mqtt_client() -> None:
    """on_update-style registration must raise, not silently never fire."""
    handle = MqttSensorStreamHandle(_clientless_twin())
    with pytest.raises(TwinStateUnavailableError, match="MQTT client is not available"):
        handle._register_callback("imu", _identity, lambda v: None)
