"""Zenoh shared-memory zero-copy publish tests.

These exercise :class:`ZenohBackend`'s SHM publish path — the pool is created
when ``shared_memory=True`` and frame-sized payloads are published as
zero-copy ``ZShmMut`` buffers instead of copied over the transport.

The tests are arch-independent (POSIX SHM behaves the same on aarch64 and
x86_64).  They skip gracefully when:

* ``eclipse-zenoh`` is not installed, or
* the SHM pool cannot be created — almost always an un-lifted
  ``RLIMIT_MEMLOCK`` (default 8 MB).  Lift it with ``ulimits: memlock: -1``
  (and ``ipc: host`` for cross-container zero-copy).
"""

from __future__ import annotations

import contextlib
import json
import logging
import socket
import threading
import time
from typing import Any

import pytest

try:
    import zenoh  # noqa: F401

    _has_zenoh = True
except ImportError:
    _has_zenoh = False

pytestmark = pytest.mark.skipif(not _has_zenoh, reason="eclipse-zenoh not installed")

FRAME_BYTES = 1280 * 720 * 3  # ~2.64 MB, one 720p RGB frame
SMALL_BYTES = 256


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _shm_backend(**kwargs: Any) -> Any:
    """A SHM-enabled backend, or ``pytest.skip`` if the pool can't be created."""
    from cyberwave.data.zenoh_backend import ZenohBackend

    be = ZenohBackend(shared_memory=True, **kwargs)
    if not be.shm_enabled:
        be.close()
        pytest.skip(
            "SHM pool unavailable (lift RLIMIT_MEMLOCK: `ulimits: memlock: -1`)"
        )
    return be


class TestShmProvider:
    def test_provider_created_when_enabled(self) -> None:
        be = _shm_backend()
        try:
            assert be.shm_enabled is True
        finally:
            be.close()

    def test_provider_absent_when_disabled(self) -> None:
        from cyberwave.data.zenoh_backend import ZenohBackend

        be = ZenohBackend(shared_memory=False)
        try:
            assert be.shm_enabled is False
        finally:
            be.close()


class TestPublishPathSelection:
    def test_frame_payload_uses_shm(self) -> None:
        be = _shm_backend()
        try:
            be.publish("shm/frame", b"\x00" * FRAME_BYTES)
            stats = be.stats()
            assert stats["shm_frames"] == 1
            assert stats["copy_frames"] == 0
        finally:
            be.close()

    def test_small_payload_uses_copy(self) -> None:
        be = _shm_backend()
        try:
            be.publish("shm/small", b"\x00" * SMALL_BYTES)
            stats = be.stats()
            assert stats["shm_frames"] == 0
            assert stats["copy_frames"] == 1
        finally:
            be.close()

    def test_custom_min_bytes_threshold(self) -> None:
        be = _shm_backend(shm_min_bytes=128)
        try:
            be.publish("shm/small", b"\x00" * SMALL_BYTES)  # 256 >= 128 → SHM
            assert be.stats()["shm_frames"] == 1
        finally:
            be.close()


class TestFallback:
    def test_publish_falls_back_when_provider_none(self) -> None:
        """A backend whose pool failed to create still publishes (via copy)."""
        from cyberwave.data.zenoh_backend import ZenohBackend

        be = ZenohBackend(shared_memory=False)
        try:
            be.publish("shm/fallback", b"\x00" * FRAME_BYTES)
            stats = be.stats()
            assert stats["shm_frames"] == 0
            assert stats["copy_frames"] == 1
        finally:
            be.close()

    def test_oversized_payload_falls_back_to_copy(self) -> None:
        """A payload larger than the pool degrades to copy without crashing —
        the alloc-failure → GC/defragment → copy-fallback path."""
        be = _shm_backend(shm_pool_bytes=2 * 1024 * 1024, shm_min_bytes=1024)
        try:
            be.publish("shm/oversized", b"\x00" * (8 * 1024 * 1024))  # 8 MB > 2 MB pool
            stats = be.stats()
            assert stats["shm_frames"] == 0
            assert stats["copy_frames"] == 1
        finally:
            be.close()

    def test_unexpected_error_propagates(self) -> None:
        """A non-ZError from the SHM path is a real defect and must propagate,
        not get swallowed into a copy fallback."""
        be = _shm_backend()

        class BadProvider:
            def garbage_collect(self) -> None: ...
            def defragment(self) -> None: ...

            def alloc(self, layout: Any) -> Any:
                raise ValueError("boom")  # not a zenoh.ZError

        be._shm_provider = BadProvider()
        try:
            with pytest.raises(ValueError, match="boom"):
                be.publish("shm/bug", b"\x00" * FRAME_BYTES)
        finally:
            be.close()

    def test_recovers_after_fallback(self) -> None:
        """After a forced fallback, a fitting frame publishes via SHM again."""
        be = _shm_backend(shm_pool_bytes=8 * 1024 * 1024, shm_min_bytes=1024)
        try:
            be.publish("shm/mix", b"\x00" * (16 * 1024 * 1024))  # too big → copy
            be.publish("shm/mix", b"\x00" * (1024 * 1024))  # fits → shm
            stats = be.stats()
            assert stats["copy_frames"] == 1
            assert stats["shm_frames"] == 1
        finally:
            be.close()


class _FlappingProvider:
    """Alloc fails or succeeds per the ``fail`` flag; the GC/defragment retry
    reads the same flag, so a frame is cleanly one or the other."""

    def __init__(self, real: Any) -> None:
        self._real = real
        self.fail = False

    def garbage_collect(self) -> None: ...

    def defragment(self) -> None: ...

    def alloc(self, layout: Any) -> Any:
        if self.fail:
            raise zenoh.ZError("out of memory")
        return self._real.alloc(layout)


@contextlib.contextmanager
def _capture_backend_logs(caplog: Any) -> Any:
    """Capture backend logs regardless of the logger's propagate setting."""
    lg = logging.getLogger("cyberwave.data.zenoh_backend")
    prev_level, prev_propagate = lg.level, lg.propagate
    lg.setLevel(logging.INFO)
    lg.addHandler(caplog.handler)
    try:
        yield
    finally:
        lg.removeHandler(caplog.handler)
        lg.setLevel(prev_level)
        lg.propagate = prev_propagate


class TestDegradationLogging:
    def test_flapping_pool_does_not_spam_warnings(self, caplog: Any) -> None:
        """Regression: a transition-based gate re-armed the "first occurrence"
        branch every other frame, so an alternating pool logged per failure
        (~60 lines/s at 30 fps) despite the documented 30 s limit."""
        be = _shm_backend(shm_min_bytes=1024)
        provider = _FlappingProvider(be._shm_provider)
        be._shm_provider = provider
        payload = b"\x00" * (1024 * 1024)
        try:
            with _capture_backend_logs(caplog):
                for i in range(40):
                    provider.fail = i % 2 == 0  # 20 failures, 20 successes
                    be.publish("shm/flap", payload)
            warns = [r for r in caplog.records if r.levelno == logging.WARNING]
            recovered = [r for r in caplog.records if "recovered" in r.getMessage()]
            assert len(warns) == 1, f"expected 1 warning, got {len(warns)}"
            # 20 successes never reach the consecutive-frames threshold.
            assert not recovered, f"expected no recovery log, got {len(recovered)}"
            assert be.stats()["shm_frames"] == 20
            assert be.stats()["copy_frames"] == 20
        finally:
            be._shm_provider = provider._real
            be.close()

    def test_sustained_failure_warns_once_then_reports_count(self, caplog: Any) -> None:
        """40 sustained failures warn once; the suppressed ones are folded into
        the count reported by the next warning after the interval elapses."""
        from cyberwave.data.zenoh_backend import _SHM_DEGRADE_WARN_INTERVAL_S

        be = _shm_backend(shm_min_bytes=1024)
        provider = _FlappingProvider(be._shm_provider)
        provider.fail = True
        be._shm_provider = provider
        payload = b"\x00" * (1024 * 1024)
        try:
            with _capture_backend_logs(caplog):
                for _ in range(40):
                    be.publish("shm/sustained", payload)
                warns = [r for r in caplog.records if r.levelno == logging.WARNING]
                assert len(warns) == 1, f"expected 1 warning, got {len(warns)}"
                assert "1 occurrence" in warns[0].getMessage()

                # Simulate the interval elapsing; the next fallback reports the
                # 39 suppressed so far plus itself.
                be._shm_last_warn -= _SHM_DEGRADE_WARN_INTERVAL_S + 1
                be.publish("shm/sustained", payload)
            warns = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert len(warns) == 2, f"expected 2 warnings, got {len(warns)}"
            assert "40 occurrence" in warns[1].getMessage()
        finally:
            be._shm_provider = provider._real
            be.close()

    def test_recovery_needs_sustained_success(self, caplog: Any) -> None:
        """Recovery takes a run of successes, not one lucky frame."""
        from cyberwave.data.zenoh_backend import _SHM_RECOVERY_CONFIRM_FRAMES

        be = _shm_backend(shm_min_bytes=1024)
        provider = _FlappingProvider(be._shm_provider)
        be._shm_provider = provider
        payload = b"\x00" * (1024 * 1024)
        try:
            with _capture_backend_logs(caplog):
                provider.fail = True
                be.publish("shm/recover", payload)
                provider.fail = False
                for _ in range(_SHM_RECOVERY_CONFIRM_FRAMES - 1):
                    be.publish("shm/recover", payload)
                assert not [r for r in caplog.records if "recovered" in r.getMessage()]
                be.publish("shm/recover", payload)  # crosses the threshold
            recovered = [r for r in caplog.records if "recovered" in r.getMessage()]
            assert len(recovered) == 1, f"expected 1 recovery log, got {len(recovered)}"
        finally:
            be._shm_provider = provider._real
            be.close()


class TestExhaustionPolicy:
    def test_drop_policy_drops_on_exhaustion(self) -> None:
        """With ``drop``, an unfittable frame is skipped, not copied."""
        be = _shm_backend(
            shm_pool_bytes=2 * 1024 * 1024,
            shm_min_bytes=1024,
            shm_on_exhaustion="drop",
        )
        try:
            be.publish("shm/drop", b"\x00" * (8 * 1024 * 1024))  # too big → dropped
            be.publish("shm/drop", b"\x00" * (1024 * 1024))  # fits → shm
            stats = be.stats()
            assert stats["dropped_frames"] == 1
            assert stats["copy_frames"] == 0
            assert stats["shm_frames"] == 1
            assert stats["publish"].get("shm/drop") == 1  # only the SHM one published
        finally:
            be.close()

    def test_invalid_policy_rejected(self) -> None:
        from cyberwave.data.zenoh_backend import ZenohBackend

        with pytest.raises(ValueError, match="shm_on_exhaustion"):
            ZenohBackend(shared_memory=False, shm_on_exhaustion="bogus")


class TestReconnect:
    def test_provider_survives_session_swap(self) -> None:
        """The session-independent pool must survive a reconnect: ``_reconnect``
        swaps ``_session`` but must not touch the provider, else zero-copy would
        silently drop. Swaps the session the way ``_reconnect`` does."""
        be = _shm_backend()
        original = be._session
        try:
            provider = be._shm_provider
            # Swap in a fresh session, as _reconnect does. (original is closed
            # in finally, not here — this build's zenoh close path can block.)
            be._session = be._open_session()
            assert be._shm_provider is provider
            be.publish("shm/reconnect", b"\x00" * FRAME_BYTES)
            assert be.stats()["shm_frames"] == 1
        finally:
            try:
                original.close()
            except Exception:
                pass
            be.close()


class TestStatsAndLatest:
    def test_stats_and_reset_zeroes_counters(self) -> None:
        be = _shm_backend()
        try:
            be.publish("shm/r", b"\x00" * FRAME_BYTES)
            snap = be.stats_and_reset()
            assert snap["shm_frames"] == 1
            after = be.stats()
            assert after["shm_frames"] == 0
            assert after["copy_frames"] == 0
            assert after["dropped_frames"] == 0
        finally:
            be.close()

    def test_latest_intact_after_shm_publish(self) -> None:
        """``latest()`` serves the stored bytes, not a dangling ``ZShmMut``."""
        be = _shm_backend()
        try:
            payload = bytes((i % 251) for i in range(FRAME_BYTES))
            be.publish("shm/latest", payload)
            assert be.stats()["shm_frames"] == 1
            sample = be.latest("shm/latest", timeout_s=2.0)
            assert sample is not None
            assert sample.payload == payload
        finally:
            be.close()


class TestEndToEnd:
    def test_payload_intact_over_shm(self) -> None:
        """A frame published via SHM is received byte-for-byte by a subscriber."""
        port = _find_free_port()
        pub = _shm_backend(listen=[f"tcp/127.0.0.1:{port}"])

        from cyberwave.data.zenoh_backend import ZenohBackend

        sub_be = ZenohBackend(connect=[f"tcp/127.0.0.1:{port}"], shared_memory=True)
        received: list[bytes] = []
        done = threading.Event()

        def on_sample(s: Any) -> None:
            received.append(s.payload)
            done.set()

        sub = sub_be.subscribe("shm/e2e", on_sample, policy="fifo")
        time.sleep(0.3)  # let the subscriber declare + SHM negotiate

        payload = bytes((i % 251) for i in range(FRAME_BYTES))
        try:
            pub.publish("shm/e2e", payload)
            assert done.wait(timeout=5.0), "subscriber received no frame"
            assert received[0] == payload
            assert pub.stats()["shm_frames"] == 1
        finally:
            sub.close()
            sub_be.close()
            pub.close()

    def test_many_frames_intact_over_shm(self) -> None:
        """Streaming many distinct frames through a small pool forces buffer
        recycling (GC/reuse); each must still arrive byte-exact — the real
        proof that fresh-alloc-per-frame has no torn reads."""
        port = _find_free_port()
        # 8 MB pool with 1 MB frames → buffers must be recycled across the run.
        pub = _shm_backend(
            listen=[f"tcp/127.0.0.1:{port}"], shm_pool_bytes=8 * 1024 * 1024
        )

        from cyberwave.data.zenoh_backend import ZenohBackend

        sub_be = ZenohBackend(connect=[f"tcp/127.0.0.1:{port}"], shared_memory=True)
        n = 50
        received: list[bytes] = []
        done = threading.Event()

        def on_sample(s: Any) -> None:
            received.append(s.payload)
            if len(received) >= n:
                done.set()

        sub = sub_be.subscribe("shm/many", on_sample, policy="fifo")
        time.sleep(0.3)

        frames = [bytes([i]) * (1024 * 1024) for i in range(n)]  # each distinct
        try:
            for f in frames:
                pub.publish("shm/many", f)
                time.sleep(0.005)
            assert done.wait(timeout=10.0), f"received {len(received)}/{n}"
            assert received == frames  # exact content + order, no torn reads
            assert pub.stats()["shm_frames"] == n
            assert pub.stats()["dropped_frames"] == 0
        finally:
            sub.close()
            sub_be.close()
            pub.close()

    def test_receiver_gets_zero_copy_buffer(self) -> None:
        """A direct peer receives the payload as an SHM-backed sample.

        Uses a raw zenoh subscriber because the SDK ``Sample`` exposes only
        decoded bytes — ``payload.as_shm()`` is the definitive per-sample
        zero-copy detector.
        """
        port = _find_free_port()
        pub = _shm_backend(listen=[f"tcp/127.0.0.1:{port}"])

        cfg = zenoh.Config()
        cfg.insert_json5("mode", '"peer"')
        cfg.insert_json5("scouting/multicast/enabled", "false")
        cfg.insert_json5("scouting/gossip/enabled", "false")
        cfg.insert_json5("connect/endpoints", json.dumps([f"tcp/127.0.0.1:{port}"]))
        cfg.insert_json5("transport/shared_memory/enabled", "true")
        sub_session = zenoh.open(cfg)

        shm_hits = {"n": 0, "total": 0}
        done = threading.Event()

        def on_sample(s: Any) -> None:
            shm_hits["total"] += 1
            if s.payload.as_shm() is not None:
                shm_hits["n"] += 1
            done.set()

        sub = sub_session.declare_subscriber("shm/zc", on_sample)
        time.sleep(0.4)  # SHM negotiation needs the link fully up

        try:
            pub.publish("shm/zc", b"\x00" * FRAME_BYTES)
            assert done.wait(timeout=5.0), "raw subscriber received no frame"
            assert shm_hits["n"] == shm_hits["total"] == 1, (
                f"expected 1 SHM-backed sample, got {shm_hits}"
            )
        finally:
            sub.undeclare()
            sub_session.close()
            pub.close()
