from __future__ import annotations

import time

from cyberwave.driver.base import BaseDriver
from cyberwave.driver.interface.stream_publish_rate import StreamPublishRateLimiter


class _ProbeDriver(BaseDriver):
    REGISTRY_ID = "test/probe"
    driver_family = "python"

    @classmethod
    def create(cls) -> _ProbeDriver:
        return cls()

    async def on_configure(self) -> None:
        return None

    async def on_connect_to_device(self) -> None:
        return None

    async def on_register_callbacks(self) -> None:
        return None

    async def on_activate(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None


def test_stream_publish_rate_limiter_blocks_until_interval() -> None:
    limiter = StreamPublishRateLimiter()
    assert limiter.acquire("joints", max_hz=50.0) is True
    assert limiter.acquire("joints", max_hz=50.0) is False
    time.sleep(0.021)
    assert limiter.acquire("joints", max_hz=50.0) is True


def test_stream_publish_rate_limiter_unlimited_when_max_hz_zero() -> None:
    limiter = StreamPublishRateLimiter()
    assert limiter.acquire("x", max_hz=0.0) is True
    assert limiter.acquire("x", max_hz=0.0) is True


def test_base_driver_acquire_stream_publish_slot() -> None:
    driver = _ProbeDriver()
    assert driver.acquire_stream_publish_slot("telemetry") is True
    assert driver.acquire_stream_publish_slot("telemetry") is False


def test_base_driver_per_stream_override() -> None:
    driver = _ProbeDriver()
    assert driver.acquire_stream_publish_slot("a", max_hz=10.0) is True
    assert driver.acquire_stream_publish_slot("b", max_hz=10.0) is True
    assert driver.acquire_stream_publish_slot("a", max_hz=10.0) is False
    assert driver.acquire_stream_publish_slot("b", max_hz=10.0) is False


def test_stream_publish_max_hz_override() -> None:
    class _FastDriver(_ProbeDriver):
        def stream_publish_max_hz(self, stream_key: str) -> float:
            if stream_key == "fast":
                return 100.0
            return 25.0

    driver = _FastDriver()
    assert driver.stream_publish_max_hz("fast") == 100.0
    assert driver.stream_publish_max_hz("slow") == 25.0
