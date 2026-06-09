"""Catalog-driven multi-topic listen session for twins."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..manifest.driver_config import select_listen_slugs

logger = logging.getLogger(__name__)

TopicHandler = Callable[["MqttMessage"], None]


@dataclass(frozen=True, slots=True)
class MqttMessage:
    topic: str
    slug: str
    payload: dict[str, Any]


@dataclass
class TopicListenSpec:
    slug: str
    topic: str
    direction: str
    description: str
    payload_schema_ref: str | None
    json_schema: dict[str, Any] | None = None
    handler: TopicHandler | None = None


def noop_handler(_message: MqttMessage) -> None:
    """Default handler for slugs without an explicit callback."""


def print_handler(message: MqttMessage) -> None:
    print(f"[{message.slug}] {message.topic}: {message.payload}")


def build_listen_specs(
    twin: Any,
    *,
    handlers: Mapping[str, TopicHandler] | None = None,
    filters: Sequence[str] | None = None,
    include_telemetry: bool = False,
    verbose: bool = False,
    topic_prefix: str | None = None,
) -> dict[str, TopicListenSpec]:
    """Build listen specs from catalog without connecting MQTT."""
    schema = twin.commands.get_schema()
    prefix = topic_prefix
    if prefix is None:
        config = getattr(getattr(twin, "client", None), "config", None)
        prefix = getattr(config, "topic_prefix", None) or ""

    slugs = select_listen_slugs(
        schema,
        filters=list(filters) if filters is not None else None,
        include_telemetry=include_telemetry,
    )
    if handlers:
        slugs = sorted(set(slugs) | set(handlers.keys()))
    topics_meta = schema.get("topics") if isinstance(schema.get("topics"), dict) else {}
    specs: dict[str, TopicListenSpec] = {}

    for slug in slugs:
        entry = topics_meta.get(slug, {}) if isinstance(topics_meta, dict) else {}
        if not isinstance(entry, dict):
            entry = {}
        handler = None
        if handlers and slug in handlers:
            handler = handlers[slug]
        elif verbose:
            handler = print_handler
        else:
            handler = noop_handler
        specs[slug] = TopicListenSpec(
            slug=slug,
            topic=f"{prefix}{slug.format(twin_uuid=twin.uuid)}",
            direction=str(entry.get("direction", "both")),
            description=str(entry.get("description", "")),
            payload_schema_ref=entry.get("payload_schema_ref"),
            json_schema=entry.get("json_schema") if isinstance(entry.get("json_schema"), dict) else None,
            handler=handler,
        )
    return specs


class TwinListenSession:
    """Active MQTT listen session with per-slug dispatch."""

    def __init__(self, twin: Any, specs: dict[str, TopicListenSpec]) -> None:
        self._twin = twin
        self._specs = dict(specs)
        self._started = False

    def set_handler(self, slug: str, handler: TopicHandler) -> None:
        if slug not in self._specs:
            raise KeyError(f"Unknown slug {slug!r}; available: {sorted(self._specs)}")
        spec = self._specs[slug]
        self._specs[slug] = TopicListenSpec(
            slug=spec.slug,
            topic=spec.topic,
            direction=spec.direction,
            description=spec.description,
            payload_schema_ref=spec.payload_schema_ref,
            json_schema=spec.json_schema,
            handler=handler,
        )

    def set_handlers(self, handlers: Mapping[str, TopicHandler]) -> None:
        for slug, handler in handlers.items():
            self.set_handler(slug, handler)

    def start(self) -> None:
        if self._started:
            return
        mqtt = self._twin.client.mqtt
        connect = getattr(self._twin, "_ensure_mqtt_connected", None)
        if callable(connect):
            connect()
        elif not getattr(mqtt, "connected", False):
            mqtt.connect()

        for spec in self._specs.values():
            handler = spec.handler or noop_handler

            def _callback(
                payload: Any,
                *,
                _spec: TopicListenSpec = spec,
                _handler: TopicHandler = handler,
            ) -> None:
                if not isinstance(payload, dict):
                    payload = {"data": payload}
                _handler(
                    MqttMessage(
                        topic=_spec.topic,
                        slug=_spec.slug,
                        payload=payload,
                    )
                )

            mqtt.subscribe(spec.topic, _callback)
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        mqtt = self._twin.client.mqtt
        for spec in self._specs.values():
            if hasattr(mqtt, "unsubscribe"):
                mqtt.unsubscribe(spec.topic)
        self._started = False

    def __enter__(self) -> TwinListenSession:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
