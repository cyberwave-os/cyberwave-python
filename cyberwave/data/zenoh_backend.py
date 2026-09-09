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


_THREAD_JOIN_TIMEOUT_S = 2.0
"""How long ``close()`` waits for each thread it spawned to return.

Returning while one is still inside zenoh's native code lets the interpreter
finalize underneath it.  CPython then kills the daemon thread mid-call, and on
glibc the forced unwind through zenoh's Rust frames aborts the process with
``FATAL: exception not rethrown`` instead of exiting cleanly.
"""


def _join_thread(thread: threading.Thread | None) -> None:
    """Wait for a thread spawned by this backend to finish."""
    if thread is None or thread is threading.current_thread():
        return
    thread.join(timeout=_THREAD_JOIN_TIMEOUT_S)
    if thread.is_alive():
        logger.warning(
            "Zenoh thread '%s' still running %.1fs after close()",
            thread.name,
            _THREAD_JOIN_TIMEOUT_S,
        )


class ZenohSubscription(Subscription):
    """Subscription handle backed by a Zenoh subscriber."""

    def __init__(
        self,
        subscriber: Any,
        *,
        stop_event: threading.Event | None = None,
        recv_thread: threading.Thread | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self._subscriber = subscriber
        self._stop_event = stop_event
        self._recv_thread = recv_thread
        self._closed = False
        self._on_close = on_close

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._stop_event is not None:
            self._stop_event.set()
        # Join before undeclaring: a thread inside ``try_recv()`` holds a
        # borrow on the subscriber, which makes ``undeclare()`` raise.
        _join_thread(self._recv_thread)
        try:
            self._subscriber.undeclare()
        except Exception:
            pass
        if self._on_close is not None:
            self._on_close()


_WATCHDOG_INTERVAL_S = 5.0

_ZERO_PEER_PROBES_BEFORE_WARN = 3
"""Consecutive zero-peer probes before warning that a session never linked up.
Whichever of the driver and worker starts first is alone until the other boots,
so warning on the first probe would fire on every normal start and train
operators to ignore the line. A session that *loses* its last peer skips this
grace period — that one is always worth hearing about."""

_RECONNECT_BACKOFF_BASE_S = 1.0
_RECONNECT_BACKOFF_MAX_S = 30.0
_RECONNECT_MAX_ATTEMPTS = 20

# SHM publish defaults. Pool size is capped by the container's RLIMIT_MEMLOCK
# (default 8 MB) — lift it with ``ulimits: memlock: -1``.
_DEFAULT_SHM_POOL_BYTES = 64 * 1024 * 1024
"""Default SHM pool size. Env: ``ZENOH_SHM_POOL_BYTES``."""

_DEFAULT_SHM_MIN_BYTES = 4096
"""Payloads below this size take the copy path (zero-copy isn't worth it for
small messages). Env: ``ZENOH_SHM_MIN_BYTES``."""

_SHM_GC_INTERVAL = 8
"""Reclaim released pool buffers every N publishes (a full GC + defragment also
runs on any alloc failure). Cheaper than collecting every frame."""

_SHM_DEGRADE_WARN_INTERVAL_S = 30.0
"""Minimum seconds between repeated SHM-degradation warnings."""

_SHM_RECOVERY_CONFIRM_FRAMES = 30
"""Consecutive successes before declaring recovery. One success is not enough —
a flapping pool would then log a pair every other frame."""

SHM_EXHAUSTION_POLICIES = ("copy", "drop")
"""Backpressure policy when the SHM pool can't take a frame. ``"copy"`` falls
back to a copy publish (guaranteed delivery, latency grows under overload);
``"drop"`` skips the frame (lossy, latency stays flat — for live video)."""

_DEFAULT_SHM_ON_EXHAUSTION = "copy"


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
        shared_memory: Enable same-host zero-copy delivery. :meth:`publish`
            then allocates each frame from a POSIX SHM pool so only a descriptor
            crosses the transport. The pool is used per frame regardless of
            whether a subscriber can use zero-copy, so enable this only when a
            same-host SHM-capable consumer exists.
        shm_pool_bytes: SHM pool size (ignored unless ``shared_memory``). Must
            exceed ``max_frame_bytes × in_flight_depth``.
        shm_min_bytes: Payloads smaller than this use the copy path.
        shm_on_exhaustion: Backpressure policy when the pool is full —
            ``"copy"`` (default, guaranteed delivery) or ``"drop"`` (lossy but
            low-latency). See :data:`SHM_EXHAUSTION_POLICIES`.
    """

    def __init__(
        self,
        *,
        connect: list[str] | None = None,
        listen: list[str] | None = None,
        shared_memory: bool = False,
        shm_pool_bytes: int = _DEFAULT_SHM_POOL_BYTES,
        shm_min_bytes: int = _DEFAULT_SHM_MIN_BYTES,
        shm_on_exhaustion: str = _DEFAULT_SHM_ON_EXHAUSTION,
    ) -> None:
        if not _has_zenoh:
            raise BackendUnavailableError(
                "The 'eclipse-zenoh' package is not installed.  "
                "Install it with:  pip install 'cyberwave[zenoh]'  "
                "or:  pip install eclipse-zenoh"
            )
        if shm_on_exhaustion not in SHM_EXHAUSTION_POLICIES:
            raise ValueError(
                f"Invalid shm_on_exhaustion '{shm_on_exhaustion}'. "
                f"Must be one of: {', '.join(SHM_EXHAUSTION_POLICIES)}."
            )

        self._connect = connect
        self._listen = listen
        self._shared_memory = shared_memory
        self._shm_pool_bytes = shm_pool_bytes
        self._shm_min_bytes = shm_min_bytes
        self._shm_on_exhaustion = shm_on_exhaustion

        self._session: Any = self._open_session()

        # Shared-memory pool (created only when shared_memory is on).  The
        # provider is session-independent, so it survives reconnects untouched.
        self._shm_mod: Any = None
        self._shm_provider: Any = None
        self._shm_lock = threading.Lock()
        self._shm_frames = 0
        self._copy_frames = 0
        self._dropped_frames = 0
        self._shm_alloc_count = 0
        self._shm_degraded = False
        self._shm_last_warn = 0.0
        self._shm_last_error = ""
        self._shm_fallbacks_since_warn = 0
        self._shm_consecutive_ok = 0
        self._init_shm_provider()

        self._subscriptions: list[ZenohSubscription] = []
        self._lock = threading.Lock()
        self._closed = False
        self._connected = True

        self._latest_store: dict[str, Sample] = {}
        self._store_lock = threading.Lock()
        self._queryables: dict[str, Any] = {}
        self._queryable_threads: dict[str, threading.Thread] = {}

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
        # Remote-peer connectivity tracking. All three are written only by the
        # watchdog thread. ``_peer_links`` is ``None`` until the first sample.
        self._peer_links: int | None = None
        self._zero_peer_probes = 0
        self._zero_peer_warned = False
        self._active_sub_specs: list[tuple[str, Callable[[Sample], None], str]] = []
        self._active_sub_specs_lock = threading.Lock()

        self._watchdog_thread = threading.Thread(
            target=self._session_watchdog,
            name="zenoh-watchdog",
            daemon=True,
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

    def _init_shm_provider(self) -> None:
        """Create the SHM allocation pool, if SHM is enabled.

        Non-fatal on failure: logs a warning and leaves ``_shm_provider`` as
        ``None`` so :meth:`publish` falls back to copy. On containers the cause
        is usually one of two independent caps — ``RLIMIT_MEMLOCK``
        (``ulimits: memlock: -1``) or a ``/dev/shm`` smaller than the pool.
        ``ipc: host`` covers the latter by inheriting the host's ``/dev/shm``;
        without it Docker defaults to a private 64 MB one.
        """
        if not self._shared_memory:
            return
        try:
            import zenoh.shm as _shm

            self._shm_mod = _shm
            self._shm_provider = _shm.ShmProvider.default_backend(
                _shm.MemoryLayout(self._shm_pool_bytes)
            )
            logger.info(
                "Zenoh SHM zero-copy publish enabled (pool=%d bytes, min=%d bytes)",
                self._shm_pool_bytes,
                self._shm_min_bytes,
            )
        except Exception as exc:
            self._shm_mod = None
            self._shm_provider = None
            logger.warning(
                "Zenoh SHM pool unavailable (%s) — publishing falls back to copy. "
                "On Linux containers this needs `ipc: host`, `ulimits: memlock: -1`, "
                "and a /dev/shm at least as large as the pool (%d bytes). Docker "
                "defaults /dev/shm to 64 MB unless `ipc: host` (which inherits the "
                "host's) or an explicit `shm_size` is set.",
                exc,
                self._shm_pool_bytes,
            )

    @property
    def shm_enabled(self) -> bool:
        """Whether this session can *publish* from an SHM pool — not a statement
        about delivery. The pool creates fine even when no peer can map it (e.g.
        containers without ``ipc: host`` each get a private ``/dev/shm``), so
        this stays ``True`` and ``shm_frames`` climbs while consumers silently
        get wire copies. Only the receiver's ``payload.as_shm()`` proves a
        zero-copy hop.
        """
        return self._shm_provider is not None

    @property
    def is_connected(self) -> bool:
        """Whether the Zenoh session is believed to be alive."""
        return self._connected and not self._closed

    def _session_info(self) -> Any:
        """Return the session info handle.

        ``.info`` is a property in zenoh >=1.x and a method in older versions.
        """
        info = self._session.info
        return info() if callable(info) else info

    def _peer_link_count(self) -> int | None:
        """Remote sessions currently linked to ours (peers + routers).

        ``None`` when the installed bindings don't expose the query — the
        caller must treat that as *unknown*, never as zero, so an old binding
        can't manufacture a spurious "no peers" warning.
        """
        try:
            info = self._session_info()
            return len(list(info.peers_zid())) + len(list(info.routers_zid()))
        except Exception:
            return None

    def _log_peer_link_changes(self) -> None:
        """Log transitions in remote-peer connectivity.

        ``zid()`` succeeding only proves our *own* session object is alive — it
        answers from local state and keeps answering after every remote peer has
        gone away. Traffic on a bus with no peers therefore raises nothing and is
        discarded silently. Sampling the peer count is what makes that visible.

        Losing the last peer warns immediately; never having had one waits
        :data:`_ZERO_PEER_PROBES_BEFORE_WARN` probes, because whichever of the
        driver and worker starts first is legitimately alone until the other
        boots. At most one warning per zero-peer episode.
        """
        peers = self._peer_link_count()
        if peers is None:
            return

        if peers == 0:
            self._zero_peer_probes += 1
            lost_last_peer = bool(self._peer_links)
            if not self._zero_peer_warned and (
                lost_last_peer
                or self._zero_peer_probes >= _ZERO_PEER_PROBES_BEFORE_WARN
            ):
                logger.warning(
                    "Zenoh session has no remote peers — nothing published "
                    "here reaches another process (same-process subscribers "
                    "still receive). Discovery uses multicast scouting; set "
                    "ZENOH_CONNECT on every participant if this host blocks it."
                )
                self._zero_peer_warned = True
            self._peer_links = 0
            return

        self._zero_peer_probes = 0
        self._zero_peer_warned = False
        if self._peer_links in (0, None):
            logger.info("Zenoh session linked to %d remote peer(s)", peers)
        elif peers != self._peer_links:
            logger.debug("Zenoh remote peers %d -> %d", self._peer_links, peers)
        self._peer_links = peers

    def _session_watchdog(self) -> None:
        """Periodically probe session liveness; trigger reconnect on failure."""
        while not self._watchdog_stop.wait(_WATCHDOG_INTERVAL_S):
            if self._closed:
                return
            try:
                self._session_info().zid()
                if not self._connected:
                    logger.info("Zenoh session probe succeeded — marking connected")
                    self._connected = True
                self._log_peer_link_changes()
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
                # Fresh session: forget the old peer state so the next probe
                # reports who we rejoined rather than comparing against a tally
                # that belonged to a session that no longer exists.
                self._peer_links = None
                self._zero_peer_probes = 0
                self._zero_peer_warned = False
                logger.info("Zenoh session reconnected (attempt %d)", attempt)

                self._resubscribe_all()
                return
            except Exception as exc:
                logger.warning(
                    "Zenoh reconnect attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt,
                    _RECONNECT_MAX_ATTEMPTS,
                    exc,
                    delay,
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
                logger.exception(
                    "Failed to re-subscribe to '%s' after reconnect", channel
                )

    # -- DataBackend implementation -------------------------------------------

    def publish(
        self,
        channel: str,
        payload: bytes,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        tried_shm = (
            self._shm_provider is not None and len(payload) >= self._shm_min_bytes
        )
        used_shm = self._publish_shm(channel, payload) if tried_shm else False

        if used_shm:
            self._shm_frames += 1
            self._note_shm_recovered()
        else:
            # A fallback (wanted SHM but the pool couldn't take it) is
            # noteworthy; a sub-threshold small message copying is normal.
            if tried_shm:
                self._note_shm_degraded()
                if self._shm_on_exhaustion == "drop":
                    self._dropped_frames += 1
                    return  # lossy backpressure — skip this frame entirely
            try:
                self._session.put(channel, payload)
            except Exception as exc:
                raise PublishError(
                    f"Zenoh publish to '{channel}' failed: {exc}"
                ) from exc
            self._copy_frames += 1

        self._publish_counts[channel] += 1
        self._publish_bytes[channel] += len(payload)

        with self._store_lock:
            self._latest_store[channel] = Sample(
                channel=channel,
                payload=payload,
                metadata=metadata,
            )
        self._ensure_queryable(channel)

    def _publish_shm(self, channel: str, payload: bytes) -> bool:
        """Publish *payload* as a zero-copy SHM buffer.

        Returns ``True`` on success, ``False`` to let the caller fall back to a
        copy ``put``. Only the descriptor crosses the transport. ``ZShmMut`` has
        no buffer protocol, so the payload is written via slice assignment.

        Concurrency invariant — do not break: allocate a *fresh* buffer per
        publish and never reuse, mutate, or stash it after ``put()``. zenoh's
        allocator is cross-process refcounted and reclaims a segment only once
        no consumer holds it, so fresh-alloc-per-frame is what makes reads safe
        without a lock. ``_shm_lock`` only serializes provider state
        (GC/alloc/defragment); the write and ``put`` need no lock.
        """
        n = len(payload)
        try:
            layout = self._shm_mod.MemoryLayout(n)
            provider = self._shm_provider
            with self._shm_lock:
                self._shm_alloc_count += 1
                # Cadence GC keeps the steady-state path cheap; a failed alloc
                # below escalates to a full GC + defragment.
                if self._shm_alloc_count % _SHM_GC_INTERVAL == 0:
                    provider.garbage_collect()
                try:
                    buf = provider.alloc(layout)
                except zenoh.ZError:
                    provider.garbage_collect()
                    provider.defragment()
                    buf = provider.alloc(layout)
            buf[0:n] = payload
            self._session.put(channel, buf)
            return True
        except zenoh.ZError as exc:
            # ZError = expected operational failure (pool exhausted/oversized,
            # bad layout, transport error mid-reconnect) → fall back to copy.
            # Anything else (e.g. a non-bytes-like payload) is a real defect and
            # is left to propagate rather than masked as backpressure.
            self._shm_last_error = str(exc)
            logger.debug(
                "SHM publish to '%s' failed (%s) — falling back to copy",
                channel,
                exc,
                exc_info=True,
            )
            return False

    def _note_shm_degraded(self) -> None:
        """Warn that SHM fell back to copy, at most once per
        :data:`_SHM_DEGRADE_WARN_INTERVAL_S`, with the count folded in.

        Gated on elapsed time alone: also gating on the ok->degraded transition
        would defeat the limit exactly when it matters, since a flapping pool
        re-enters the "first occurrence" branch every other frame.
        """
        self._shm_consecutive_ok = 0
        self._shm_fallbacks_since_warn += 1
        now = time.time()
        if (now - self._shm_last_warn) < _SHM_DEGRADE_WARN_INTERVAL_S:
            return
        logger.warning(
            "Zenoh SHM publish falling back to copy (%d occurrence(s) since the "
            "last report; %s) — pool may be exhausted (slow/stalled consumer) or "
            "the payload may exceed the pool size. Raise ZENOH_SHM_POOL_BYTES or "
            "check consumers.",
            self._shm_fallbacks_since_warn,
            self._shm_last_error or "unknown",
        )
        self._shm_last_warn = now
        self._shm_fallbacks_since_warn = 0
        self._shm_degraded = True

    def _note_shm_recovered(self) -> None:
        """Log once SHM has resumed for :data:`_SHM_RECOVERY_CONFIRM_FRAMES`
        consecutive frames. ``_shm_last_warn`` is left untouched so the next
        degradation still respects the warning interval.
        """
        if not self._shm_degraded:
            return
        self._shm_consecutive_ok += 1
        if self._shm_consecutive_ok < _SHM_RECOVERY_CONFIRM_FRAMES:
            return
        logger.info(
            "Zenoh SHM zero-copy publish recovered (%d consecutive frames)",
            self._shm_consecutive_ok,
        )
        self._shm_degraded = False
        self._shm_consecutive_ok = 0
        self._shm_fallbacks_since_warn = 0

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

        return self._subscribe_on_session(
            channel, callback, policy=policy, on_close=_remove_spec
        )

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
                            "Zenoh recv exception on '%s' — waiting for reconnect",
                            ch,
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
                name=f"zenoh-recv-{channel}",
                daemon=True,
            )
            t.start()
            handle = ZenohSubscription(
                sub, stop_event=stop_event, recv_thread=t, on_close=on_close
            )
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
                "shm_frames": int,
                "copy_frames": int,
                "dropped_frames": int,
                "uptime_s": float,
            }

        ``publish_bytes`` is logical payload volume, not wire bytes (SHM sends
        only a descriptor). ``shm_frames`` counts frames published *as an SHM
        buffer* — publish-side intent, not a delivery guarantee, since a non-SHM
        or remote subscriber still gets a wire copy. Only the receiver's
        ``payload.as_shm()`` confirms an end-to-end zero-copy hop.
        ``dropped_frames`` counts frames skipped by the ``"drop"`` exhaustion
        policy (never published).
        """
        with self._stats_lock:
            return {
                "publish": dict(self._publish_counts),
                "publish_bytes": dict(self._publish_bytes),
                "recv": dict(self._recv_counts),
                "recv_bytes": dict(self._recv_bytes),
                "shm_frames": self._shm_frames,
                "copy_frames": self._copy_frames,
                "dropped_frames": self._dropped_frames,
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
                "shm_frames": self._shm_frames,
                "copy_frames": self._copy_frames,
                "dropped_frames": self._dropped_frames,
                "elapsed_s": now - self._stats_start_time,
            }
            self._publish_counts = collections.defaultdict(int)
            self._publish_bytes = collections.defaultdict(int)
            self._recv_counts = collections.defaultdict(int)
            self._recv_bytes = collections.defaultdict(int)
            self._shm_frames = 0
            self._copy_frames = 0
            self._dropped_frames = 0
            self._stats_start_time = now
        return snapshot

    def latest(
        self,
        channel: str,
        *,
        timeout_s: float = 1.0,
    ) -> Sample | None:
        # Always do a fresh network query. ``_latest_store`` is only for
        # the publish side (feeds the local queryable); caching replies
        # here has no invalidation and would freeze polling callers on
        # the first-tick frame.
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
                self._recv_counts[channel] += 1
                self._recv_bytes[channel] += len(raw)
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
                # Raises "Already borrowed" whenever _serve is parked in
                # recv() on this queryable, which is the usual case.
                qable.undeclare()
            except Exception:
                pass
        self._queryables.clear()
        _join_thread(self._watchdog_thread)
        try:
            self._session.close()
        except Exception:
            pass
        # Only closing the session unblocks a queryable parked in recv().  Wait
        # for those threads here: letting the caller exit while one is still
        # unwinding out of zenoh is what aborts the process on glibc.
        for thread in self._queryable_threads.values():
            _join_thread(thread)
        self._queryable_threads.clear()

    # -- internal helpers -----------------------------------------------------

    def _ensure_queryable(self, channel: str) -> None:
        """Declare a queryable so ``latest()`` from other sessions can resolve."""
        if channel in self._queryables:
            return
        try:
            qable = self._session.declare_queryable(channel, complete=True)

            def _serve(q_channel: str, queryable: Any) -> None:
                """Serve queries in a background thread.

                Blocks in ``recv()`` -- polling instead would wake this thread
                hundreds of times a second per channel for the whole life of
                the publisher, which perturbs delivery timing and leaves a
                spinning thread behind if the backend is never closed.
                ``close()`` retires it by closing the session under it.
                """
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

            t = threading.Thread(
                target=_serve,
                args=(channel, qable),
                name=f"zenoh-queryable-{channel}",
                daemon=True,
            )
            t.start()
            self._queryables[channel] = qable
            self._queryable_threads[channel] = t
        except Exception:
            logger.debug("Failed to declare queryable for '%s'", channel, exc_info=True)
