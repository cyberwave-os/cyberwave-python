"""ZenohMqttBridge — bidirectional forwarder between local Zenoh bus and cloud MQTT.

Outbound path (edge → cloud):
    Subscribes to configured Zenoh channels, encodes as MQTT messages, and
    publishes to the cloud broker.  When MQTT is disconnected, messages are
    persisted to the :class:`~.queue.OfflineQueue` and drained on reconnect.

Inbound path (cloud → edge):
    Subscribes to configured MQTT topics and republishes payloads into the
    local Zenoh session so edge workers can consume commands via ``cw.data``.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .config import BridgeConfig
from .queue import OfflineQueue, QueuedMessage
from .topic_mapping import (
    build_inbound_subscriptions,
    build_outbound_subscriptions,
    mqtt_to_zenoh,
    zenoh_to_mqtt,
)

logger = logging.getLogger(__name__)

try:
    import zenoh

    _has_zenoh = True
except ImportError:
    zenoh = None  # type: ignore[assignment]
    _has_zenoh = False

try:
    import paho.mqtt.client as paho_mqtt
    from paho.mqtt.enums import CallbackAPIVersion

    _has_paho = True
except ImportError:
    paho_mqtt = None  # type: ignore[assignment]
    CallbackAPIVersion = None  # type: ignore[assignment,misc]
    _has_paho = False


class ZenohMqttBridge:
    """Application-level bridge between a Zenoh session and an MQTT broker.

    Typical usage::

        from cyberwave.zenoh_mqtt import ZenohMqttBridge, BridgeConfig

        bridge = ZenohMqttBridge(
            config=BridgeConfig(twin_uuids=["abc-..."]),
            mqtt_host="<mqtt_host>",
            mqtt_port=8883,
            mqtt_password="<api_key>",
        )
        bridge.start()
        # ... bridge runs until stopped ...
        bridge.stop()

    Args:
        config: Bridge configuration.
        zenoh_session: Pre-existing Zenoh session.  If ``None`` the bridge
            opens its own peer-to-peer session.
        mqtt_host: MQTT broker hostname.
        mqtt_port: MQTT broker port.
        mqtt_username: MQTT username.
        mqtt_password: MQTT password (typically the Cyberwave API key).
        mqtt_use_tls: Enable TLS for the MQTT connection.
        mqtt_tls_ca_cert: Path to a CA certificate file for TLS verification.
    """

    def __init__(
        self,
        config: BridgeConfig | None = None,
        *,
        zenoh_session: Any | None = None,
        zenoh_connect: list[str] | None = None,
        mqtt_host: str = "localhost",
        mqtt_port: int = 1883,
        mqtt_username: str = "mqttcyb",
        mqtt_password: str = "",
        mqtt_use_tls: bool = False,
        mqtt_tls_ca_cert: str | None = None,
    ) -> None:
        self._config = config or BridgeConfig()
        self._started = False
        self._stop_event = threading.Event()

        # Zenoh
        self._owns_zenoh = zenoh_session is None
        self._zenoh_connect = zenoh_connect
        self._zenoh_session: Any = zenoh_session
        self._zenoh_subscriptions: list[Any] = []

        # MQTT
        self._mqtt_host = mqtt_host
        self._mqtt_port = mqtt_port
        self._mqtt_username = mqtt_username
        self._mqtt_password = mqtt_password
        self._mqtt_use_tls = mqtt_use_tls
        self._mqtt_tls_ca_cert = mqtt_tls_ca_cert
        self._mqtt_client: Any = None
        self._mqtt_connected = threading.Event()

        # Offline queue
        self._queue = OfflineQueue(
            queue_dir=self._config.queue_dir,
            max_bytes=self._config.queue_max_bytes,
        )

        # Stats
        self._outbound_count = 0
        self._inbound_count = 0
        self._queued_count = 0
        self._drained_count = 0
        self._stats_lock = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the bridge: open connections, subscribe, begin forwarding."""
        if self._started:
            return

        if not _has_zenoh:
            raise RuntimeError(
                "eclipse-zenoh is not installed. "
                "Install with: pip install 'cyberwave[zenoh]'"
            )
        if not _has_paho:
            raise RuntimeError(
                "paho-mqtt is not installed. Install with: pip install paho-mqtt"
            )

        self._started = True
        self._stop_event.clear()

        self._init_zenoh()
        self._init_mqtt()
        self._subscribe_outbound()
        self._subscribe_inbound()
        self._start_drain_thread()

        logger.info(
            "ZenohMqttBridge started — outbound channels: %s, inbound topics: %s, "
            "twins: %d",
            self._config.outbound_channels,
            self._config.inbound_topics,
            len(self._config.twin_uuids),
        )

    def stop(self) -> None:
        """Stop the bridge gracefully."""
        if not self._started:
            return
        self._stop_event.set()

        # Unsubscribe Zenoh
        for sub in self._zenoh_subscriptions:
            try:
                sub.undeclare()
            except Exception:
                pass
        self._zenoh_subscriptions.clear()

        # Close own Zenoh session
        if self._owns_zenoh and self._zenoh_session is not None:
            try:
                self._zenoh_session.close()
            except Exception:
                pass
            self._zenoh_session = None

        # Disconnect MQTT
        if self._mqtt_client is not None:
            try:
                self._mqtt_client.disconnect()
                self._mqtt_client.loop_stop()
            except Exception:
                pass
            self._mqtt_client = None

        self._queue.close()
        self._started = False

        with self._stats_lock:
            logger.info(
                "ZenohMqttBridge stopped — outbound: %d, inbound: %d, "
                "queued: %d, drained: %d",
                self._outbound_count,
                self._inbound_count,
                self._queued_count,
                self._drained_count,
            )

    @property
    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return {
                "outbound": self._outbound_count,
                "inbound": self._inbound_count,
                "queued": self._queued_count,
                "drained": self._drained_count,
            }

    # ── Zenoh setup ──────────────────────────────────────────────────

    def _init_zenoh(self) -> None:
        if self._zenoh_session is not None:
            return
        cfg = zenoh.Config()
        if self._zenoh_connect:
            import json as _json

            cfg.insert_json5("connect/endpoints", _json.dumps(self._zenoh_connect))
        cfg.insert_json5("transport/shared_memory/enabled", "false")
        self._zenoh_session = zenoh.open(cfg)

    # ── MQTT setup ───────────────────────────────────────────────────

    def _init_mqtt(self) -> None:
        import ssl

        self._mqtt_client = paho_mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id=f"bridge_{id(self):x}",
            protocol=paho_mqtt.MQTTv311,
        )
        self._mqtt_client.username_pw_set(self._mqtt_username, self._mqtt_password)

        if self._mqtt_use_tls or self._mqtt_port == 8883:
            self._mqtt_client.tls_set(
                ca_certs=self._mqtt_tls_ca_cert,
                cert_reqs=ssl.CERT_REQUIRED,
            )

        self._mqtt_client.on_connect = self._on_mqtt_connect
        self._mqtt_client.on_disconnect = self._on_mqtt_disconnect
        self._mqtt_client.on_message = self._on_mqtt_message

        self._mqtt_client.connect_async(self._mqtt_host, self._mqtt_port, keepalive=60)
        self._mqtt_client.loop_start()

    def _on_mqtt_connect(
        self,
        client: Any,
        userdata: Any,
        flags: Any,
        rc: Any,
        properties: Any = None,
    ) -> None:
        logger.info("Bridge MQTT connected (rc=%s)", rc)
        self._mqtt_connected.set()
        # Re-subscribe inbound topics on reconnect
        self._subscribe_inbound_mqtt()

    def _on_mqtt_disconnect(
        self,
        client: Any,
        userdata: Any,
        flags: Any = None,
        rc: Any = None,
        properties: Any = None,
    ) -> None:
        logger.warning("Bridge MQTT disconnected (rc=%s)", rc)
        self._mqtt_connected.clear()

    def _on_mqtt_message(
        self,
        client: Any,
        userdata: Any,
        message: Any,
    ) -> None:
        """Forward an inbound MQTT message to the local Zenoh session."""
        mapping = mqtt_to_zenoh(
            message.topic,
            mqtt_prefix=self._config.mqtt_topic_prefix,
            zenoh_prefix=self._config.zenoh_key_prefix,
        )
        if mapping is None:
            logger.debug("Ignoring unmapped MQTT topic: %s", message.topic)
            return

        try:
            self._zenoh_session.put(mapping.zenoh_key, message.payload)
            with self._stats_lock:
                self._inbound_count += 1
            logger.debug(
                "Inbound: MQTT %s → Zenoh %s (%d bytes)",
                message.topic,
                mapping.zenoh_key,
                len(message.payload),
            )
        except Exception:
            logger.warning(
                "Failed to forward MQTT → Zenoh: %s",
                message.topic,
                exc_info=True,
            )

    # ── Outbound subscriptions (Zenoh → MQTT) ────────────────────────

    def _subscribe_outbound(self) -> None:
        keys = build_outbound_subscriptions(
            self._config.twin_uuids,
            self._config.outbound_channels,
            zenoh_prefix=self._config.zenoh_key_prefix,
        )
        for key in keys:
            try:
                sub = self._zenoh_session.declare_subscriber(
                    key, self._make_outbound_callback(key)
                )
                self._zenoh_subscriptions.append(sub)
                logger.debug("Outbound Zenoh subscription: %s", key)
            except Exception:
                logger.warning(
                    "Failed to subscribe to Zenoh key: %s", key, exc_info=True
                )

    def _make_outbound_callback(self, key: str):  # noqa: ANN202
        """Return a Zenoh subscriber callback that forwards to MQTT."""

        def _callback(sample: Any) -> None:
            try:
                raw_key = str(sample.key_expr)
            except Exception:
                raw_key = key

            mapping = zenoh_to_mqtt(raw_key, mqtt_prefix=self._config.mqtt_topic_prefix)
            if mapping is None:
                return

            try:
                payload = bytes(sample.payload)
            except Exception:
                try:
                    payload = sample.payload.to_bytes()
                except Exception:
                    logger.warning("Cannot extract payload for key %s", raw_key)
                    return

            self._publish_or_queue(mapping.mqtt_topic, payload)

        return _callback

    def _publish_or_queue(self, mqtt_topic: str, payload: bytes) -> None:
        """Publish to MQTT, or enqueue if disconnected."""
        if self._mqtt_connected.is_set() and self._queue.is_empty:
            try:
                result = self._mqtt_client.publish(
                    mqtt_topic, payload, qos=self._config.mqtt_qos
                )
                result.wait_for_publish(timeout=2.0)
                with self._stats_lock:
                    self._outbound_count += 1
                logger.debug("Outbound: → MQTT %s (%d bytes)", mqtt_topic, len(payload))
                return
            except Exception:
                logger.debug(
                    "MQTT publish failed, queueing: %s", mqtt_topic, exc_info=True
                )

        self._queue.enqueue(
            QueuedMessage(
                mqtt_topic=mqtt_topic,
                payload=payload,
                qos=self._config.mqtt_qos,
                enqueued_at=time.time(),
            )
        )
        with self._stats_lock:
            self._queued_count += 1

    # ── Inbound subscriptions (MQTT → Zenoh) ─────────────────────────

    def _subscribe_inbound(self) -> None:
        """Called once after start(); actual MQTT subscribes happen in _subscribe_inbound_mqtt."""
        pass

    def _subscribe_inbound_mqtt(self) -> None:
        topics = build_inbound_subscriptions(
            self._config.twin_uuids,
            self._config.inbound_topics,
            mqtt_prefix=self._config.mqtt_topic_prefix,
        )
        for topic in topics:
            try:
                self._mqtt_client.subscribe(topic, qos=self._config.mqtt_qos)
                logger.debug("Inbound MQTT subscription: %s", topic)
            except Exception:
                logger.warning(
                    "Failed to subscribe to MQTT topic: %s", topic, exc_info=True
                )

    # ── Queue drain thread ───────────────────────────────────────────

    def _start_drain_thread(self) -> None:
        t = threading.Thread(target=self._drain_loop, daemon=True)
        t.start()

    def _drain_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._queue.is_empty or not self._mqtt_connected.is_set():
                self._stop_event.wait(0.5)
                continue

            batch = self._queue.drain(self._config.drain_batch_size)
            for msg in batch:
                if self._stop_event.is_set():
                    break
                try:
                    result = self._mqtt_client.publish(
                        msg.mqtt_topic, msg.payload, qos=msg.qos
                    )
                    result.wait_for_publish(timeout=5.0)
                    with self._stats_lock:
                        self._drained_count += 1
                    logger.debug(
                        "Drained queued message → MQTT %s (%d bytes, queued %.1fs ago)",
                        msg.mqtt_topic,
                        len(msg.payload),
                        time.time() - msg.enqueued_at,
                    )
                except Exception:
                    logger.warning(
                        "Failed to drain message to %s, re-queueing",
                        msg.mqtt_topic,
                        exc_info=True,
                    )
                    self._queue.enqueue(msg)
                    break

            if not batch:
                self._stop_event.wait(0.1)
