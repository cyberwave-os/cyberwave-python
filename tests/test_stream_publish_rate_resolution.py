"""Publish-rate resolution: manifest override > class default (30 Hz)."""
from cyberwave.driver.base import BaseDriver


class _Probe(BaseDriver):
    REGISTRY_ID = "test/probe"

    async def on_configure(self): ...
    async def on_connect_to_device(self): ...
    async def on_register_callbacks(self): ...
    async def on_activate(self): ...
    async def on_shutdown(self): ...

    @classmethod
    def create(cls): return cls()


def _probe() -> BaseDriver:
    return _Probe(twin=None, client=None, auto_register_interface=False)


def test_default_stream_rate_is_30hz():
    assert _probe().stream_publish_max_hz("any") == 30.0


def test_mqtt_max_hz_override_wins_over_default():
    d = _probe()
    d._mqtt_max_hz = 15.0
    assert d.stream_publish_max_hz("any") == 15.0
