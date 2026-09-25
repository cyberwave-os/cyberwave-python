"""Latest-value MQTT snapshot reader (numpy sensor streams: pointcloud/depth/lidar).

Snapshot path: ``get_*`` returns a fresh *copy* of the latest decoded value — the
counterpart to ``mqtt_live_view.py``, whose views refresh in place. Callback
fan-out is delegated to the shared :class:`CallbackHub` so the snapshot path and
the live-view path use one subscription type.
"""

from __future__ import annotations

import copy
import logging
import threading
from typing import TYPE_CHECKING, Any, Callable

from ..exceptions import TwinStateTimeoutError
from ..manifest.driver_config import resolve_inbound_topics
from ..mqtt.state import (
    FIRST_READ_TIMEOUT_S,  # noqa: F401  (re-exported for handle modules)
    attach_topic_listener,
    mqtt_client_for,
    wait_for_first_message,
)
from .callback_hub import CallbackHub, StateSubscription

if TYPE_CHECKING:
    from ..twin.base import Twin

logger = logging.getLogger(__name__)

DecodeFn = Callable[[dict[str, Any]], Any]


def _copy_value(value: Any) -> Any:
    copier = getattr(value, "copy", None)
    if callable(copier):
        return copier()
    return copy.deepcopy(value)


class _StreamState:
    def __init__(self, decode: DecodeFn) -> None:
        self.decode = decode
        self.curr: Any = None
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.attached_topics: set[str] = set()
        self.attached = False
        self.hub = CallbackHub(label="stream")


class MqttSensorStreamHandle:
    """Caches the latest decoded value per named MQTT stream + on_update fan-out."""

    def __init__(self, twin: "Twin", sensor_id: str | None = None) -> None:
        self._twin = twin
        # Subclasses that also extend TwinCameraHandle expose ``sensor_id`` as a
        # read-only property backed by ``_sensor_id``; don't clobber it.
        if not isinstance(getattr(type(self), "sensor_id", None), property):
            self.sensor_id = sensor_id
        self._streams: dict[str, _StreamState] = {}
        self._streams_lock = threading.Lock()

    def _topic_prefix(self) -> str:
        config = getattr(getattr(self._twin, "client", None), "config", None)
        return getattr(config, "topic_prefix", None) or ""

    def _stream_state(self, stream: str, decode: DecodeFn) -> _StreamState:
        with self._streams_lock:
            state = self._streams.get(stream)
            if state is None:
                state = _StreamState(decode)
                self._streams[stream] = state
            return state

    def _on_payload(
        self, stream: str, state: _StreamState, payload: dict[str, Any]
    ) -> None:
        try:
            value = state.decode(payload)
        except Exception:
            logger.exception("failed to decode %s payload; ignoring", stream)
            return
        if value is None:
            return
        with state.lock:
            state.curr = value
            state.ready.set()
        state.hub.notify(lambda v=value: _copy_value(v))

    def _ensure_stream(self, stream: str, decode: DecodeFn) -> _StreamState:
        state = self._stream_state(stream, decode)
        if state.attached:
            return state
        # Double-checked locking: serialize the attach so two concurrent first
        # readers can't both subscribe (which would double every callback and
        # the ``curr`` update). ``_on_payload`` also takes ``state.lock`` but only
        # after the subscription fires, i.e. after this block releases it.
        with state.lock:
            if state.attached:
                return state
            # Let TwinStateUnavailableError propagate: fail fast with its actionable
            # message (missing client / API key) instead of returning an unattached
            # state, which would surface later as a misleading read timeout or, for
            # on_update(), a subscription that silently never fires.
            mqtt_client_for(self._twin)
            topics = resolve_inbound_topics(
                stream,
                self._twin.driver.get_mqtt_schema(),
                twin_uuid=self._twin.uuid,
                topic_prefix=self._topic_prefix(),
            )
            for _slug, topic in topics:
                attach_topic_listener(
                    self._twin,
                    topic=topic,
                    on_payload=(
                        lambda p, s=state, st=stream: self._on_payload(st, s, p)
                    ),
                    attached_topics=state.attached_topics,
                )
            state.attached = True
        return state

    def _get_latest(self, stream: str, decode: DecodeFn, *, timeout: float) -> Any:
        """Return the latest decoded value, waiting up to *timeout* for a fresh one.

        Every call waits for a new inbound message — not just the very first one
        ever for this stream — so repeated calls don't silently hand back a
        byte-identical stale snapshot when nothing new has arrived since the
        previous call. If *timeout* elapses without a fresher message, falls
        back to the most recently cached value; raises
        :class:`TwinStateTimeoutError` if no message has ever arrived.
        """
        state = self._ensure_stream(stream, decode)
        fresh = threading.Event()
        sub = state.hub.subscribe(lambda _snap: fresh.set())
        try:
            with state.lock:
                had_data = state.curr is not None
            if had_data:
                fresh.wait(timeout=timeout)
            else:
                wait_for_first_message(
                    state.ready,
                    timeout=timeout,
                    twin_uuid=self._twin.uuid,
                    stream=stream,
                )
        finally:
            sub.cancel()
        with state.lock:
            if state.curr is None:
                raise TwinStateTimeoutError(
                    f"No MQTT {stream} update within {timeout}s "
                    f"for twin {self._twin.uuid}"
                )
            return _copy_value(state.curr)

    def _register_callback(
        self, stream: str, decode: DecodeFn, callback: Callable[[Any], None]
    ) -> StateSubscription:
        state = self._ensure_stream(stream, decode)
        return state.hub.subscribe(callback)
