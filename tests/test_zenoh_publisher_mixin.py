"""Unit tests for :class:`ZenohPublisherMixin` frame publishing.

Focus on the ``frame_metadata`` contract: metadata is bound to the frame channel
on the *first* frame only (``HeaderTemplate`` makes it immutable afterwards), and
it is configurable so non-RGB channels (e.g. uint16 depth) are not mislabeled as
``rgb24``.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
from cyberwave.driver.transports.zenoh_publisher import _FrameSlot, ZenohPublisherMixin


class _FakeBus:
    """Records publish calls and stops the loop once *target* frames arrive."""

    def __init__(self, stop: threading.Event, target: int) -> None:
        self.calls: list[tuple[str, Any]] = []
        self._stop = stop
        self._target = target
        self._backend = object()  # for the init-time log line

    def publish(self, channel: str, frame: Any, *, metadata: Any = None) -> None:
        self.calls.append((channel, metadata))
        if len(self.calls) >= self._target:
            self._stop.set()


class _Pub(ZenohPublisherMixin):
    pass


def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met within timeout")


def test_frame_metadata_bound_on_first_frame_only() -> None:
    pub = _Pub()
    pub._zenoh_stop = threading.Event()
    bus = _FakeBus(pub._zenoh_stop, target=2)
    pub._zenoh_data_bus = bus  # type: ignore[assignment]
    pub._zenoh_frame_slot = _FrameSlot()
    pub._zenoh_channel = "depth"
    pub._zenoh_frame_metadata = {"content": "depth16"}

    thread = threading.Thread(target=pub._zenoh_frame_loop, daemon=True)
    thread.start()
    try:
        # Feed frames one at a time so the single-slot buffer does not coalesce
        # them (latest-wins), guaranteeing two distinct publishes.
        pub._zenoh_frame_slot.put(np.zeros((2, 2), dtype=np.uint16))
        _wait_for(lambda: len(bus.calls) >= 1)
        pub._zenoh_frame_slot.put(np.ones((2, 2), dtype=np.uint16))
        _wait_for(lambda: len(bus.calls) >= 2)
    finally:
        pub._zenoh_stop.set()
        thread.join(timeout=2.0)

    assert bus.calls[0] == ("depth", {"content": "depth16"})
    assert bus.calls[1] == ("depth", None)


def test_init_zenoh_bus_defaults_to_rgb24(monkeypatch) -> None:
    import cyberwave.data.config as data_config

    monkeypatch.setenv("CYBERWAVE_DATA_BACKEND", "zenoh")
    monkeypatch.setattr(data_config, "is_zenoh_publish_enabled", lambda: True)

    stop = threading.Event()

    class _Client:
        def data_bus_for(self, _uuid: str) -> _FakeBus:
            return _FakeBus(stop, target=99)

    pub = _Pub()
    pub._init_zenoh_bus(twin_uuid="abcd1234", cw_client=_Client())
    try:
        assert pub._zenoh_frame_metadata == {"color_format": "rgb24"}
    finally:
        pub._close_zenoh_bus()


def test_init_zenoh_bus_allows_none_metadata(monkeypatch) -> None:
    import cyberwave.data.config as data_config

    monkeypatch.setenv("CYBERWAVE_DATA_BACKEND", "zenoh")
    monkeypatch.setattr(data_config, "is_zenoh_publish_enabled", lambda: True)

    stop = threading.Event()

    class _Client:
        def data_bus_for(self, _uuid: str) -> _FakeBus:
            return _FakeBus(stop, target=99)

    pub = _Pub()
    pub._init_zenoh_bus(
        twin_uuid="abcd1234",
        cw_client=_Client(),
        frame_channel="depth",
        frame_metadata=None,
    )
    try:
        assert pub._zenoh_channel == "depth"
        assert pub._zenoh_frame_metadata is None
    finally:
        pub._close_zenoh_bus()
