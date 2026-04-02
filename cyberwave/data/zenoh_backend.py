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
    ) -> None:
        self._subscriber = subscriber
        self._stop_event = stop_event
        self._closed = False

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


class ZenohBackend(DataBackend):
    """Zenoh-backed data bus.

    Args:
        connect: Zenoh router endpoints (e.g. ``["tcp/localhost:7447"]``).
            ``None`` uses peer-to-peer discovery.
        shared_memory: Enable Zenoh shared-memory transport for same-host
            zero-copy delivery.
        key_prefix: Prefix prepended to every channel name when constructing
            Zenoh key expressions.
    """

    def __init__(
        self,
        *,
        connect: list[str] | None = None,
        shared_memory: bool = False,
        key_prefix: str = "cw",
    ) -> None:
        if not _has_zenoh:
            raise BackendUnavailableError(
                "The 'eclipse-zenoh' package is not installed.  "
                "Install it with:  pip install 'cyberwave[zenoh]'  "
                "or:  pip install eclipse-zenoh"
            )

        cfg = zenoh.Config()
        if connect:
            cfg.insert_json5("connect/endpoints", json.dumps(connect))
        # Explicitly set SHM in both directions: some Zenoh builds auto-enable
        # SHM by default, which fails on platforms where POSIX SHM is
        # restricted (e.g. certain WSL2 / container environments).
        cfg.insert_json5(
            "transport/shared_memory/enabled",
            "true" if shared_memory else "false",
        )

        try:
            self._session: Any = zenoh.open(cfg)
        except Exception as exc:
            raise BackendUnavailableError(
                f"Failed to open Zenoh session: {exc}"
            ) from exc

        self._key_prefix = key_prefix
        self._subscriptions: list[ZenohSubscription] = []
        self._lock = threading.Lock()
        self._closed = False

        self._latest_store: dict[str, Sample] = {}
        self._store_lock = threading.Lock()
        self._queryables: dict[str, Any] = {}

    def _resolve_key(self, channel: str) -> str:
        if self._key_prefix:
            return f"{self._key_prefix}/{channel}"
        return channel

    # -- DataBackend implementation -------------------------------------------

    def publish(
        self,
        channel: str,
        payload: bytes,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        key = self._resolve_key(channel)
        try:
            self._session.put(key, payload)
        except Exception as exc:
            raise PublishError(f"Zenoh publish to '{key}' failed: {exc}") from exc

        with self._store_lock:
            self._latest_store[channel] = Sample(
                channel=channel,
                payload=payload,
                metadata=metadata,
            )
        self._ensure_queryable(channel, key)

    def subscribe(
        self,
        channel: str,
        callback: Callable[[Sample], None],
        *,
        policy: str = "latest",
    ) -> Subscription:
        self._validate_policy(policy)
        key = self._resolve_key(channel)

        if policy == "latest":
            try:
                sub = self._session.declare_subscriber(
                    key,
                    zenoh.handlers.RingChannel(1),
                )
            except Exception as exc:
                raise SubscriptionError(
                    f"Zenoh subscribe to '{key}' failed: {exc}"
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
                        break
                    if zenoh_sample is None:
                        # 1 ms: low enough to not miss samples on 1 kHz
                        # force/torque streams, short enough to keep CPU idle
                        # between frames on a 60 fps camera channel.
                        stop.wait(0.001)
                        continue
                    if stop.is_set():
                        break
                    try:
                        raw = bytes(zenoh_sample.payload)
                    except Exception:
                        raw = zenoh_sample.payload.to_bytes()
                    cb(Sample(channel=ch, payload=raw, timestamp=time.time()))

            t = threading.Thread(
                target=_recv_loop,
                args=(sub, stop_event, channel, callback),
                daemon=True,
            )
            t.start()
            handle = ZenohSubscription(sub, stop_event=stop_event)
        else:

            def _on_sample_fifo(zenoh_sample: Any) -> None:
                try:
                    raw = bytes(zenoh_sample.payload)
                except Exception:
                    raw = zenoh_sample.payload.to_bytes()
                callback(
                    Sample(
                        channel=channel,
                        payload=raw,
                        timestamp=time.time(),
                    )
                )

            try:
                sub = self._session.declare_subscriber(key, _on_sample_fifo)
            except Exception as exc:
                raise SubscriptionError(
                    f"Zenoh subscribe to '{key}' failed: {exc}"
                ) from exc
            handle = ZenohSubscription(sub)

        with self._lock:
            # Prune handles that were already closed by the caller.  Keeps the
            # list bounded for long-running processes that subscribe/unsubscribe
            # repeatedly without the backend being torn down.
            self._subscriptions = [h for h in self._subscriptions if not h._closed]
            self._subscriptions.append(handle)
        return handle

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

        key = self._resolve_key(channel)
        try:
            replies = self._session.get(key, timeout=timeout_s)
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
            logger.debug("Zenoh get('%s') returned no results", key, exc_info=True)
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
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

    def _ensure_queryable(self, channel: str, key: str) -> None:
        """Declare a queryable so ``latest()`` from other sessions can resolve."""
        if channel in self._queryables:
            return
        try:
            qable = self._session.declare_queryable(key, complete=True)

            def _serve(q_channel: str, q_key: str, queryable: Any) -> None:
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
                            query.reply(q_key, cached.payload)
                        except Exception:
                            pass

            t = threading.Thread(target=_serve, args=(channel, key, qable), daemon=True)
            t.start()
            self._queryables[channel] = qable
        except Exception:
            logger.debug("Failed to declare queryable for '%s'", key, exc_info=True)
