"""Zenoh-based DataBackend — primary high-performance transport.

This backend delegates to `eclipse-zenoh`_ for pub/sub and latest-value
queries.  It supports shared-memory transport for zero-copy delivery between
containers on the same host.

The ``eclipse-zenoh`` package is an **optional** dependency.  If it is not
installed, importing this module still works but instantiating
:class:`ZenohBackend` raises :class:`~.exceptions.BackendUnavailableError`.

.. _eclipse-zenoh: https://pypi.org/project/eclipse-zenoh/
"""

from __future__ import annotations

import collections
import json
import logging
import threading
import time
from typing import Any, Callable

from .backend import DataBackend, Sample, Subscription
from .exceptions import (
    BackendUnavailableError,
    PublishError,
    SubscriptionError,
)

logger = logging.getLogger(__name__)

try:
    import zenoh

    _has_zenoh = True
except ImportError:
    zenoh = None  # type: ignore[assignment]
    _has_zenoh = False


class ZenohSubscription(Subscription):
    """Subscription handle backed by a Zenoh subscriber."""

    def __init__(
        self,
        subscriber: Any,
        *,
        stop_event: threading.Event | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self._subscriber = subscriber
        self._stop_event = stop_event
        self._closed = False
        self._on_close = on_close

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._stop_event is not None:
            self._stop_event.set()
        try:
            self._subscriber.undeclare()
        except Exception:
            pass
        if self._on_close is not None:
            self._on_close()


_WATCHDOG_INTERVAL_S = 5.0
_RECONNECT_BACKOFF_BASE_S = 1.0
_RECONNECT_BACKOFF_MAX_S = 30.0
_RECONNECT_MAX_ATTEMPTS = 20


def extract_sample_key_expr(zenoh_sample: Any) -> str | None:
    """Return the publishing key of a Zenoh sample as a string, or ``None``.

    The Python zenoh bindings have exposed ``key_expr`` on samples at
    different attribute names across releases (``key_expr``, ``keyexpr``,
    ``key``); this helper keeps callers working whether the attribute is
    a plain string or a ``KeyExpr`` object.

    Wildcard subscribers rely on this to recover the actual sensor name
    that published a frame — the CLI ``worker doctor`` probe uses it
    too, which is why this is a public helper rather than private to
    the backend.
    """
    for attr in ("key_expr", "keyexpr", "key"):
        key = getattr(zenoh_sample, attr, None)
        if key is None:
            continue
        try:
            return str(key)
        except Exception:
            continue
    return None


# Backwards-compatible alias — kept because it was the original private
# name used internally within this module before the helper was made
# public.  Prefer :func:`extract_sample_key_expr` in new code.
_zenoh_sample_key = extract_sample_key_expr


class ZenohBackend(DataBackend):
    """Zenoh-backed data bus.

    Args:
        connect: Zenoh router endpoints (e.g. ``["tcp/localhost:7447"]``).
            ``None`` uses peer-to-peer discovery.
        listen: Zenoh listener endpoints (e.g. ``["tcp/0.0.0.0:7447"]``).
            Binds a TCP listener so external peers can connect without
            multicast discovery.
        shared_memory: Enable Zenoh shared-memory transport for same-host
            zero-copy delivery.
    """

    def __init__(
        self,
        *,
        connect: list[str] | None = None,
        listen: list[str] | None = None,
        shared_memory: bool = False,
    ) -> None:
        if not _has_zenoh:
            raise BackendUnavailableError(
                "The 'eclipse-zenoh' package is not installed.  "
                "Install it with:  pip install 'cyberwave[zenoh]'  "
                "or:  pip install eclipse-zenoh"
            )

        self._connect = connect
        self._listen = listen
        self._shared_memory = shared_memory

        self._session: Any = self._open_session()

        self._subscriptions: list[ZenohSubscription] = []
        self._lock = threading.Lock()
        self._closed = False
        self._connected = True

        self._latest_store: dict[str, Sample] = {}
        self._store_lock = threading.Lock()
        self._queryables: dict[str, Any] = {}

        # Per-channel message counters for monitoring.  We use defaultdict
        # and skip locking on the hot path — the GIL makes individual dict
        # operations atomic and occasional lost increments are acceptable
        # for monitoring data.  The _stats_lock is only held during
        # stats()/stats_and_reset() snapshot reads.
        self._stats_lock = threading.Lock()
        self._publish_counts: dict[str, int] = collections.defaultdict(int)
        self._publish_bytes: dict[str, int] = collections.defaultdict(int)
        self._recv_counts: dict[str, int] = collections.defaultdict(int)
        self._recv_bytes: dict[str, int] = collections.defaultdict(int)
        self._stats_start_time: float = time.time()

        # Reconnection machinery
        self._reconnect_event = threading.Event()
        self._watchdog_stop = threading.Event()
        self._active_sub_specs: list[tuple[str, Callable[[Sample], None], str]] = []
        self._active_sub_specs_lock = threading.Lock()

        self._watchdog_thread = threading.Thread(
            target=self._session_watchdog, name="zenoh-watchdog", daemon=True,
        )
        self._watchdog_thread.start()

    # -- session management ---------------------------------------------------

    def _build_config(self) -> Any:
        cfg = zenoh.Config()
        if self._connect:
            cfg.insert_json5("connect/endpoints", json.dumps(self._connect))
        if self._listen:
            cfg.insert_json5("listen/endpoints", json.dumps(self._listen))
        cfg.insert_json5(
            "transport/shared_memory/enabled",
            "true" if self._shared_memory else "false",
        )
        return cfg

    def _open_session(self) -> Any:
        cfg = self._build_config()
        try:
            return zenoh.open(cfg)
        except Exception as exc:
            raise BackendUnavailableError(
                f"Failed to open Zenoh session: {exc}"
            ) from exc

    @property
    def is_connected(self) -> bool:
        """Whether the Zenoh session is believed to be alive."""
        return self._connected and not self._closed

    def _session_watchdog(self) -> None:
        """Periodically probe session liveness; trigger reconnect on failure."""
        while not self._watchdog_stop.wait(_WATCHDOG_INTERVAL_S):
            if self._closed:
                return
            try:
                # .info is a property in zenoh >=1.x, a method in older versions.
                info = self._session.info
                if callable(info):
                    info = info()
                info.zid()
                if not self._connected:
                    logger.info("Zenoh session probe succeeded — marking connected")
                    self._connected = True
            except Exception:
                if self._connected:
                    logger.warning("Zenoh session probe failed — starting reconnect")
                    self._connected = False
                self._reconnect_event.set()
                self._reconnect()

    def _reconnect(self) -> None:
        """Close the old session and open a new one, re-subscribing all active channels.

        Note: existing Zenoh subscription handles held by callers become stale
        (``handle._closed`` is True) after reconnect.  This is acceptable because
        callers interact through callbacks, not handles — ``_resubscribe_all``
        creates fresh handles on the new session transparently.
        """
        # Close old subscription handles so their recv-loop threads exit cleanly.
        with self._lock:
            for handle in self._subscriptions:
                handle.close()
            self._subscriptions.clear()

        delay = _RECONNECT_BACKOFF_BASE_S
        for attempt in range(1, _RECONNECT_MAX_ATTEMPTS + 1):
            if self._closed:
                return
            try:
                try:
                    self._session.close()
                except Exception:
                    pass

                self._session = self._open_session()
                self._connected = True
                self._reconnect_event.clear()
                logger.info("Zenoh session reconnected (attempt %d)", attempt)

                self._resubscribe_all()
                return
            except Exception as exc:
                logger.warning(
                    "Zenoh reconnect attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt, _RECONNECT_MAX_ATTEMPTS, exc, delay,
                )
                self._watchdog_stop.wait(delay)
                delay = min(delay * 2, _RECONNECT_BACKOFF_MAX_S)

        logger.error(
            "Zenoh reconnection failed after %d attempts — session is down",
            _RECONNECT_MAX_ATTEMPTS,
        )

    def _resubscribe_all(self) -> None:
        """Re-declare all tracked subscriptions on the new session.

        Uses the internal ``_subscribe_on_session`` helper to avoid
        re-appending specs to ``_active_sub_specs`` (which would cause
        unbounded growth after repeated reconnects).
        """
        with self._active_sub_specs_lock:
            specs = list(self._active_sub_specs)

        for channel, callback, policy in specs:
            try:
                logger.debug("Re-subscribing to '%s' (policy=%s)", channel, policy)
                self._subscribe_on_session(channel, callback, policy=policy)
            except Exception:
                logger.exception("Failed to re-subscribe to '%s' after reconnect", channel)

    # -- DataBackend implementation -------------------------------------------

    def publish(
        self,
        channel: str,
        payload: bytes,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        try:
            self._session.put(channel, payload)
        except Exception as exc:
            raise PublishError(f"Zenoh publish to '{channel}' failed: {exc}") from exc

        self._publish_counts[channel] += 1
        self._publish_bytes[channel] += len(payload)

        with self._store_lock:
            self._latest_store[channel] = Sample(
                channel=channel,
                payload=payload,
                metadata=metadata,
            )
        self._ensure_queryable(channel)

    def subscribe(
        self,
        channel: str,
        callback: Callable[[Sample], None],
        *,
        policy: str = "latest",
    ) -> Subscription:
        self._validate_policy(policy)

        spec = (channel, callback, policy)
        with self._active_sub_specs_lock:
            self._active_sub_specs.append(spec)

        def _remove_spec() -> None:
            with self._active_sub_specs_lock:
                try:
                    self._active_sub_specs.remove(spec)
                except ValueError:
                    pass

        return self._subscribe_on_session(channel, callback, policy=policy, on_close=_remove_spec)

    def _subscribe_on_session(
        self,
        channel: str,
        callback: Callable[[Sample], None],
        *,
        policy: str = "latest",
        on_close: Callable[[], None] | None = None,
    ) -> Subscription:
        """Declare a Zenoh subscriber on the current session (no spec tracking)."""
        backend_ref = self

        if policy == "latest":
            try:
                sub = self._session.declare_subscriber(
                    channel,
                    zenoh.handlers.RingChannel(1),
                )
            except Exception as exc:
                raise SubscriptionError(
                    f"Zenoh subscribe to '{channel}' failed: {exc}"
                ) from exc

            stop_event = threading.Event()

            def _recv_loop(
                subscriber: Any,
                stop: threading.Event,
                ch: str,
                cb: Callable[[Sample], None],
            ) -> None:
                while not stop.is_set():
                    try:
                        zenoh_sample = subscriber.try_recv()
                    except Exception:
                        logger.debug(
                            "Zenoh recv exception on '%s' — waiting for reconnect", ch,
                        )
                        while not stop.is_set() and not backend_ref._connected:
                            stop.wait(1.0)
                        if stop.is_set():
                            break
                        continue
                    if zenoh_sample is None:
                        stop.wait(0.001)
                        continue
                    if stop.is_set():
                        break
                    try:
                        raw = bytes(zenoh_sample.payload)
                    except Exception:
                        raw = zenoh_sample.payload.to_bytes()
                    backend_ref._recv_counts[ch] += 1
                    backend_ref._recv_bytes[ch] += len(raw)
                    # Use the sample's actual publishing key so wildcard
                    # subscribers can recover the real sensor name
                    # (``color_camera`` vs ``depth_camera``).  Fall back
                    # to the subscribed key for backends that don't carry
                    # a key_expr on the sample.
                    wire_key = _zenoh_sample_key(zenoh_sample) or ch
                    cb(Sample(channel=wire_key, payload=raw, timestamp=time.time()))

            t = threading.Thread(
                target=_recv_loop,
                args=(sub, stop_event, channel, callback),
                daemon=True,
            )
            t.start()
            handle = ZenohSubscription(sub, stop_event=stop_event, on_close=on_close)
        else:

            def _on_sample_fifo(zenoh_sample: Any) -> None:
                try:
                    raw = bytes(zenoh_sample.payload)
                except Exception:
                    raw = zenoh_sample.payload.to_bytes()
                backend_ref._recv_counts[channel] += 1
                backend_ref._recv_bytes[channel] += len(raw)
                wire_key = _zenoh_sample_key(zenoh_sample) or channel
                callback(
                    Sample(
                        channel=wire_key,
                        payload=raw,
                        timestamp=time.time(),
                    )
                )

            try:
                sub = self._session.declare_subscriber(channel, _on_sample_fifo)
            except Exception as exc:
                raise SubscriptionError(
                    f"Zenoh subscribe to '{channel}' failed: {exc}"
                ) from exc
            handle = ZenohSubscription(sub, on_close=on_close)

        with self._lock:
            # Prune handles that were already closed by the caller.  Keeps the
            # list bounded for long-running processes that subscribe/unsubscribe
            # repeatedly without the backend being torn down.
            self._subscriptions = [h for h in self._subscriptions if not h._closed]
            self._subscriptions.append(handle)
        return handle

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of publish/receive counters per channel.

        The returned dict has the structure::

            {
                "publish": {"channel_key": count, ...},
                "publish_bytes": {"channel_key": total_bytes, ...},
                "recv": {"channel_key": count, ...},
                "recv_bytes": {"channel_key": total_bytes, ...},
                "uptime_s": float,
            }
        """
        with self._stats_lock:
            return {
                "publish": dict(self._publish_counts),
                "publish_bytes": dict(self._publish_bytes),
                "recv": dict(self._recv_counts),
                "recv_bytes": dict(self._recv_bytes),
                "uptime_s": time.time() - self._stats_start_time,
            }

    def stats_and_reset(self) -> dict[str, Any]:
        """Return counters then reset them to zero.

        Useful for computing per-interval rates: call periodically and
        divide counts by the elapsed interval.
        """
        now = time.time()
        with self._stats_lock:
            snapshot = {
                "publish": dict(self._publish_counts),
                "publish_bytes": dict(self._publish_bytes),
                "recv": dict(self._recv_counts),
                "recv_bytes": dict(self._recv_bytes),
                "elapsed_s": now - self._stats_start_time,
            }
            self._publish_counts = collections.defaultdict(int)
            self._publish_bytes = collections.defaultdict(int)
            self._recv_counts = collections.defaultdict(int)
            self._recv_bytes = collections.defaultdict(int)
            self._stats_start_time = now
        return snapshot

    def latest(
        self,
        channel: str,
        *,
        timeout_s: float = 1.0,
    ) -> Sample | None:
        with self._store_lock:
            cached = self._latest_store.get(channel)
        if cached is not None:
            return cached

        try:
            replies = self._session.get(channel, timeout=timeout_s)
            for reply in replies:
                try:
                    ok = reply.ok
                    raw = bytes(ok.payload)
                except Exception:
                    try:
                        raw = ok.payload.to_bytes()
                    except Exception:
                        continue
                sample = Sample(
                    channel=channel,
                    payload=raw,
                    timestamp=time.time(),
                )
                self._latest_store[channel] = sample
                return sample
        except Exception:
            logger.debug("Zenoh get('%s') returned no results", channel, exc_info=True)
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._connected = False
        self._watchdog_stop.set()
        with self._active_sub_specs_lock:
            self._active_sub_specs.clear()
        with self._lock:
            for handle in self._subscriptions:
                handle.close()
            self._subscriptions.clear()
        for qable in self._queryables.values():
            try:
                qable.undeclare()
            except Exception:
                pass
        self._queryables.clear()
        try:
            self._session.close()
        except Exception:
            pass

    # -- internal helpers -----------------------------------------------------

    def _ensure_queryable(self, channel: str) -> None:
        """Declare a queryable so ``latest()`` from other sessions can resolve."""
        if channel in self._queryables:
            return
        try:
            qable = self._session.declare_queryable(channel, complete=True)

            def _serve(q_channel: str, queryable: Any) -> None:
                """Serve queries in a background thread."""
                while True:
                    try:
                        query = queryable.recv()
                    except Exception:
                        break
                    with self._store_lock:
                        cached = self._latest_store.get(q_channel)
                    if cached is not None:
                        try:
                            query.reply(q_channel, cached.payload)
                        except Exception:
                            pass

            t = threading.Thread(target=_serve, args=(channel, qable), daemon=True)
            t.start()
            self._queryables[channel] = qable
        except Exception:
            logger.debug("Failed to declare queryable for '%s'", channel, exc_info=True)
