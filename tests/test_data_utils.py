"""``cyberwave.data.utils`` — Zenoh frame subscribe cache."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from cyberwave.data.backend import Sample
from cyberwave.data.keys import build_key
from cyberwave.data.utils import (
    _TwinFrameSubscribeCache,
    sensor_name_from_frame_channel,
)

SAMPLE_UUID = "00000000-0000-4000-8000-000000000001"


class _FakeSub:
    def close(self) -> None:
        pass


class FakeSubscribeBackend:
    def __init__(self) -> None:
        self._callbacks: dict[str, list] = {}

    def subscribe(self, key: str, callback, *, policy: str = "latest"):
        self._callbacks.setdefault(key, []).append(callback)
        return _FakeSub()

    @staticmethod
    def _matches(pattern: str, key: str) -> bool:
        if pattern.endswith("/**"):
            return key.startswith(pattern[:-3])
        return pattern == key

    def inject(self, key: str, payload: bytes) -> None:
        sample = Sample(channel=key, payload=payload, timestamp=time.time())
        for pattern, callbacks in self._callbacks.items():
            if not self._matches(pattern, key):
                continue
            for cb in callbacks:
                cb(sample)


def _numpy_wire_payload() -> bytes:
    from cyberwave.data.header import CONTENT_TYPE_NUMPY, HeaderTemplate

    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    tmpl = HeaderTemplate(CONTENT_TYPE_NUMPY, shape=arr.shape, dtype=str(arr.dtype))
    return tmpl.pack(arr.tobytes())


def test_sensor_name_from_frame_channel() -> None:
    key = build_key(SAMPLE_UUID, "frames", "default")
    assert sensor_name_from_frame_channel(key) == "default"


def test_frame_subscribe_cache_receives_wildcard_frames() -> None:
    backend = FakeSubscribeBackend()
    cache = _TwinFrameSubscribeCache(backend, SAMPLE_UUID)
    key = build_key(SAMPLE_UUID, "frames", "default")

    def _publish_later() -> None:
        time.sleep(0.05)
        backend.inject(key, _numpy_wire_payload())

    threading.Thread(target=_publish_later, daemon=True).start()

    frame = cache.fetch(sensor_name="default", timeout_s=2.0, max_age_ms=None)
    cache.close()

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (4, 4, 3)
