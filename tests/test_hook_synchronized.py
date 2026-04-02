"""Tests for ``@cw.on_synchronized`` multi-channel time-window dispatch.

These tests verify the dispatch mechanism in
``WorkerRuntime._subscribe_synchronized_group`` without requiring a live
data backend.  A lightweight ``FakeDataBus`` delivers samples directly to
the subscription callbacks, simulating what the real data bus does.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import pytest

from cyberwave.data.backend import Sample, Subscription
from cyberwave.workers.context import HookContext
from cyberwave.workers.hooks import HookRegistry
from cyberwave.workers.runtime import WorkerRuntime

TWIN_UUID = "550e8400-e29b-41d4-a716-446655440000"


# ── Lightweight fakes ────────────────────────────────────────────────


class FakeSubscription(Subscription):
    def close(self) -> None:
        pass


class FakeBackend:
    """Minimal backend stand-in that records raw-key subscriptions."""

    def __init__(self) -> None:
        self._callbacks: dict[str, list[Callable[[Sample], None]]] = {}

    def subscribe(
        self,
        key: str,
        callback: Callable[[Sample], None],
        **_kwargs: Any,
    ) -> FakeSubscription:
        self._callbacks.setdefault(key, []).append(callback)
        return FakeSubscription()

    def inject(self, key: str, sample: Sample) -> None:
        """Push *sample* to all callbacks registered on *key*."""
        for cb in self._callbacks.get(key, []):
            cb(sample)


class FakeDataBus:
    """Minimal stand-in for ``DataBus`` that exposes a ``FakeBackend``."""

    KEY_PREFIX = "cw"

    def __init__(self) -> None:
        self._backend = FakeBackend()

    @property
    def backend(self) -> FakeBackend:
        return self._backend

    @property
    def key_prefix(self) -> str:
        return self.KEY_PREFIX

    def inject(self, key: str, sample: Sample) -> None:
        """Convenience delegate to the backend."""
        self._backend.inject(key, sample)


class FakeCyberwave:
    """Minimal stand-in for the ``Cyberwave`` client, just enough to
    construct a ``WorkerRuntime``.
    """

    def __init__(self, data_bus: FakeDataBus) -> None:
        self._hook_registry = HookRegistry()
        self._data_bus = data_bus

    @property
    def data(self) -> FakeDataBus:
        return self._data_bus


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def bus() -> FakeDataBus:
    return FakeDataBus()


@pytest.fixture()
def cw(bus: FakeDataBus) -> FakeCyberwave:
    return FakeCyberwave(bus)


@pytest.fixture()
def registry(cw: FakeCyberwave) -> HookRegistry:
    return cw._hook_registry


def _make_sample(channel: str, payload: bytes, ts: float) -> Sample:
    return Sample(channel=channel, payload=payload, timestamp=ts)


def _chan(channel_str: str) -> str:
    """Build the Zenoh key the runtime subscribes to for a channel."""
    # channel_str may be "frames/front" (base/sensor) or "joint_states" (bare).
    parts = channel_str.split("/", 1)
    base = parts[0]
    sensor = parts[1] if len(parts) > 1 else None
    suffix = f"/{sensor}" if sensor else ""
    return f"cw/{TWIN_UUID}/data/{base}{suffix}"


def _start_runtime(cw: FakeCyberwave) -> WorkerRuntime:
    rt = WorkerRuntime(cw)
    rt.start()
    return rt


# ── Tests ────────────────────────────────────────────────────────────


class TestSynchronizedFiresWhenAllChannelsArrive:
    """test_synchronized_fires_when_all_channels_arrive"""

    def test_callback_fires(
        self, cw: FakeCyberwave, registry: HookRegistry, bus: FakeDataBus
    ) -> None:
        received: list[dict[str, Sample]] = []

        @registry.on_synchronized(
            TWIN_UUID,
            ["frames/front", "depth/default", "joint_states"],
            tolerance_ms=100,
        )
        def handler(samples: dict[str, Sample], ctx: HookContext) -> None:
            received.append(samples)

        _start_runtime(cw)

        now = time.time()
        bus.inject(
            _chan("frames/front"),
            _make_sample("frames/front", b"frame", now),
        )
        bus.inject(
            _chan("depth/default"),
            _make_sample("depth/default", b"depth", now + 0.01),
        )
        bus.inject(
            _chan("joint_states"),
            _make_sample("joint_states", b"joints", now + 0.02),
        )

        assert len(received) == 1
        assert set(received[0].keys()) == {
            "frames/front",
            "depth/default",
            "joint_states",
        }


class TestSynchronizedDoesNotFireWhenPartial:
    """test_synchronized_does_not_fire_when_partial"""

    def test_partial_channels_do_not_fire(
        self, cw: FakeCyberwave, registry: HookRegistry, bus: FakeDataBus
    ) -> None:
        received: list[dict[str, Sample]] = []

        @registry.on_synchronized(
            TWIN_UUID,
            ["frames/front", "depth/default", "joint_states"],
            tolerance_ms=100,
        )
        def handler(samples: dict[str, Sample], ctx: HookContext) -> None:
            received.append(samples)

        _start_runtime(cw)

        now = time.time()
        bus.inject(
            _chan("frames/front"),
            _make_sample("frames/front", b"frame", now),
        )
        bus.inject(
            _chan("depth/default"),
            _make_sample("depth/default", b"depth", now + 0.01),
        )

        assert len(received) == 0


class TestSynchronizedDoesNotFireWhenStale:
    """test_synchronized_does_not_fire_when_stale"""

    def test_stale_timestamps_do_not_fire(
        self, cw: FakeCyberwave, registry: HookRegistry, bus: FakeDataBus
    ) -> None:
        received: list[dict[str, Sample]] = []

        @registry.on_synchronized(
            TWIN_UUID,
            ["frames/front", "depth/default"],
            tolerance_ms=50,
        )
        def handler(samples: dict[str, Sample], ctx: HookContext) -> None:
            received.append(samples)

        _start_runtime(cw)

        now = time.time()
        bus.inject(
            _chan("frames/front"),
            _make_sample("frames/front", b"frame", now),
        )
        bus.inject(
            _chan("depth/default"),
            _make_sample("depth/default", b"depth", now + 0.2),
        )

        assert len(received) == 0


class TestSynchronizedFiresOnUpdate:
    """test_synchronized_fires_on_update — stale channel refreshes and all align."""

    def test_fires_after_refresh(
        self, cw: FakeCyberwave, registry: HookRegistry, bus: FakeDataBus
    ) -> None:
        received: list[dict[str, Sample]] = []

        @registry.on_synchronized(
            TWIN_UUID,
            ["frames/front", "depth/default"],
            tolerance_ms=50,
        )
        def handler(samples: dict[str, Sample], ctx: HookContext) -> None:
            received.append(samples)

        _start_runtime(cw)

        now = time.time()
        bus.inject(
            _chan("frames/front"),
            _make_sample("frames/front", b"frame_old", now),
        )
        bus.inject(
            _chan("depth/default"),
            _make_sample("depth/default", b"depth", now + 0.2),
        )
        assert len(received) == 0

        bus.inject(
            _chan("frames/front"),
            _make_sample("frames/front", b"frame_new", now + 0.19),
        )
        assert len(received) == 1


class TestSynchronizedCtxTimestampIsMax:
    """test_synchronized_ctx_timestamp_is_max"""

    def test_ctx_has_max_timestamp(
        self, cw: FakeCyberwave, registry: HookRegistry, bus: FakeDataBus
    ) -> None:
        contexts: list[HookContext] = []

        @registry.on_synchronized(
            TWIN_UUID,
            ["frames/front", "depth/default"],
            tolerance_ms=100,
        )
        def handler(samples: dict[str, Sample], ctx: HookContext) -> None:
            contexts.append(ctx)

        _start_runtime(cw)

        t1 = 1000.0
        t2 = 1000.05
        bus.inject(
            _chan("frames/front"),
            _make_sample("frames/front", b"f", t1),
        )
        bus.inject(
            _chan("depth/default"),
            _make_sample("depth/default", b"d", t2),
        )

        assert len(contexts) == 1
        assert contexts[0].timestamp == t2
        assert contexts[0].twin_uuid == TWIN_UUID
        assert "synchronized_channels" in contexts[0].metadata
        assert set(contexts[0].metadata["synchronized_channels"]) == {
            "frames/front",
            "depth/default",
        }


class TestSynchronizedCallbackReceivesAllPayloads:
    """test_synchronized_callback_receives_all_payloads"""

    def test_all_payloads_present(
        self, cw: FakeCyberwave, registry: HookRegistry, bus: FakeDataBus
    ) -> None:
        received: list[dict[str, Sample]] = []

        @registry.on_synchronized(
            TWIN_UUID,
            ["frames/front", "depth/default", "joint_states"],
            tolerance_ms=100,
        )
        def handler(samples: dict[str, Sample], ctx: HookContext) -> None:
            received.append(samples)

        _start_runtime(cw)

        now = 2000.0
        bus.inject(
            _chan("frames/front"),
            _make_sample("frames/front", b"FRAME_DATA", now),
        )
        bus.inject(
            _chan("depth/default"),
            _make_sample("depth/default", b"DEPTH_DATA", now + 0.01),
        )
        bus.inject(
            _chan("joint_states"),
            _make_sample("joint_states", b"JOINT_DATA", now + 0.02),
        )

        assert len(received) == 1
        assert received[0]["frames/front"].payload == b"FRAME_DATA"
        assert received[0]["depth/default"].payload == b"DEPTH_DATA"
        assert received[0]["joint_states"].payload == b"JOINT_DATA"


class TestSynchronizedErrorDoesNotCrash:
    """test_synchronized_error_does_not_crash"""

    def test_exception_is_logged_runtime_continues(
        self, cw: FakeCyberwave, registry: HookRegistry, bus: FakeDataBus
    ) -> None:
        call_count = 0

        @registry.on_synchronized(
            TWIN_UUID,
            ["frames/front", "depth/default"],
            tolerance_ms=100,
        )
        def handler(samples: dict[str, Sample], ctx: HookContext) -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("boom")

        _start_runtime(cw)

        now = 3000.0
        bus.inject(
            _chan("frames/front"),
            _make_sample("frames/front", b"f1", now),
        )
        bus.inject(
            _chan("depth/default"),
            _make_sample("depth/default", b"d1", now + 0.01),
        )
        assert call_count == 1

        now2 = 4000.0
        bus.inject(
            _chan("frames/front"),
            _make_sample("frames/front", b"f2", now2),
        )
        bus.inject(
            _chan("depth/default"),
            _make_sample("depth/default", b"d2", now2 + 0.01),
        )
        assert call_count == 2


# ── Key-building sanity checks ───────────────────────────────────────


class TestKeyBuilding:
    """Verify that the runtime subscribes to the expected Zenoh key expressions."""

    def test_backend_subscribes_to_correct_keys(
        self, cw: FakeCyberwave, registry: HookRegistry, bus: FakeDataBus
    ) -> None:
        @registry.on_synchronized(
            TWIN_UUID,
            ["frames/front", "depth/default"],
            tolerance_ms=100,
        )
        def handler(samples: dict[str, Sample], ctx: HookContext) -> None:
            pass

        _start_runtime(cw)

        expected_keys = {
            f"cw/{TWIN_UUID}/data/frames/front",
            f"cw/{TWIN_UUID}/data/depth/default",
        }
        assert set(bus.backend._callbacks.keys()) == expected_keys
