"""MQTT inbound topic listeners (plane C — decode callbacks, not a separate cache layer)."""

from __future__ import annotations

import threading
from typing import Any, Callable

from ..exceptions import TwinStateTimeoutError, TwinStateUnavailableError

FIRST_READ_TIMEOUT_S = 3.0

PayloadHandler = Callable[[dict[str, Any]], None]


def mqtt_client_for(twin: Any) -> Any:
    client = getattr(twin, "client", None)
    mqtt = getattr(client, "mqtt", None) if client is not None else None
    if mqtt is None:
        raise TwinStateUnavailableError("MQTT client is not available on this twin")
    return mqtt


def ensure_mqtt_connected(twin: Any) -> None:
    connect = getattr(twin, "_ensure_mqtt_connected", None)
    if callable(connect):
        connect()
        return
    mqtt = mqtt_client_for(twin)
    if not getattr(mqtt, "connected", False) and hasattr(mqtt, "connect"):
        mqtt.connect()


def attach_topic_listener(
    twin: Any,
    *,
    topic: str,
    on_payload: PayloadHandler,
    attached_topics: set[str],
) -> None:
    """Subscribe once per resolved topic; ``on_payload`` decodes and updates handle state."""

    if topic in attached_topics:
        return
    ensure_mqtt_connected(twin)

    def _callback(payload: Any) -> None:
        if isinstance(payload, dict):
            on_payload(payload)

    mqtt_client_for(twin).subscribe(topic, _callback)
    attached_topics.add(topic)


def wait_for_first_message(
    ready: threading.Event,
    *,
    timeout: float,
    twin_uuid: str,
    stream: str,
) -> None:
    if ready.is_set():
        return
    if not ready.wait(timeout=timeout):
        raise TwinStateTimeoutError(
            f"No MQTT {stream} update received within {timeout}s for twin {twin_uuid}"
        )
