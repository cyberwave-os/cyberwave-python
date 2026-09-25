"""Power / battery handle."""

from __future__ import annotations

import copy
import threading
from typing import TYPE_CHECKING, Any, Callable, Optional

from ...consumers.callback_hub import CallbackHub, StateSubscription
from ...consumers.mqtt_live_view import LiveMappingView
from ...manifest.driver_config import resolve_inbound_topics
from ...mqtt.state import attach_topic_listener

if TYPE_CHECKING:
    from ..base import Twin


class PowerHandle:
    def __init__(self, twin: Twin) -> None:
        self._twin = twin
        self._curr_power: dict[str, Any] | None = None
        self._power_attached_topics: set[str] = set()
        self._power_listeners_attached = False
        self._power_ready = threading.Event()
        self._power_lock = threading.Lock()
        self._power_hub = CallbackHub(label="power")
        self._power_view: LiveMappingView | None = None

    def _topic_prefix(self) -> str:
        config = getattr(getattr(self._twin, "client", None), "config", None)
        return getattr(config, "topic_prefix", None) or ""

    def _on_power_payload(self, payload: dict[str, Any]) -> None:
        with self._power_lock:
            self._curr_power = dict(payload)
            self._power_ready.set()
        self._power_hub.notify(self._power_snapshot)

    def _ensure_power_listeners(self) -> None:
        if self._power_listeners_attached:
            return
        topics = resolve_inbound_topics(
            "power",
            self._twin.driver.get_mqtt_schema(),
            twin_uuid=self._twin.uuid,
            topic_prefix=self._topic_prefix(),
        )
        for _, topic in topics:
            attach_topic_listener(
                self._twin,
                topic=topic,
                on_payload=self._on_power_payload,
                attached_topics=self._power_attached_topics,
            )
        self._power_listeners_attached = True

    def check(self, *, source_type: Optional[str] = None) -> None:
        self._twin._prepare_outbound_command()
        resolved = self._twin._resolve_topic_and_payload(
            command="battery_check",
            data={},
            source_type=source_type,
        )
        self._twin._publish_resolved(resolved)

    def _safe_ensure_power_listeners(self) -> None:
        from ...exceptions import TwinStateUnavailableError

        try:
            self._ensure_power_listeners()
        except (TwinStateUnavailableError, NotImplementedError):
            pass

    def _power_snapshot(self) -> dict[str, Any]:
        with self._power_lock:
            return copy.deepcopy(self._curr_power) if self._curr_power is not None else {}

    def get(self, *, timeout: float = 3.0) -> LiveMappingView:
        """Return a **live** battery/status view.

        Refreshes in place on every inbound status; empty (``{}``) until the first
        message. If empty, waits up to *timeout* for the first message. The same
        view is returned on later calls.
        """
        self._safe_ensure_power_listeners()
        if self._curr_power is None:
            self._power_ready.wait(timeout=timeout)
        if self._power_view is None:
            self._power_view = LiveMappingView(self._power_hub, self._power_snapshot)
        return self._power_view

    def on_update(
        self, callback: "Callable[[dict[str, Any]], None]"
    ) -> StateSubscription:
        self._safe_ensure_power_listeners()
        return self._power_hub.subscribe(callback)
