"""Power / battery handle."""

from __future__ import annotations

import copy
import threading
from typing import TYPE_CHECKING, Any, Optional

from ...manifest.driver_config import resolve_inbound_topics
from ...mqtt.state import attach_topic_listener, wait_for_first_message

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

    def _topic_prefix(self) -> str:
        config = getattr(getattr(self._twin, "client", None), "config", None)
        return getattr(config, "topic_prefix", None) or ""

    def _on_power_payload(self, payload: dict[str, Any]) -> None:
        with self._power_lock:
            self._curr_power = dict(payload)
            self._power_ready.set()

    def _ensure_power_listeners(self) -> None:
        if self._power_listeners_attached:
            return
        topics = resolve_inbound_topics(
            "power",
            self._twin.commands.get_schema(),
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

    def get(self, *, timeout: float = 3.0) -> dict[str, Any]:
        """Read battery/status from MQTT listeners (``_curr_power``)."""
        self._ensure_power_listeners()
        wait_for_first_message(
            self._power_ready,
            timeout=timeout if self._curr_power is None else 0.0,
            twin_uuid=self._twin.uuid,
            stream="power",
        )
        with self._power_lock:
            if self._curr_power is None:
                from ...exceptions import TwinStateTimeoutError

                raise TwinStateTimeoutError(
                    f"No MQTT power/battery update within {timeout}s for twin {self._twin.uuid}"
                )
            return copy.deepcopy(self._curr_power)
