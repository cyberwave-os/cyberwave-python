"""BaseDriver edge_health lifecycle: start/stop once, extras/streams seams."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import cyberwave.driver.base as base_module
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


def _connected_probe() -> BaseDriver:
    mqtt = MagicMock()
    client = SimpleNamespace(mqtt=mqtt)
    twin = SimpleNamespace(uuid="twin-1")
    return _Probe(twin=twin, client=client, auto_register_interface=False)


def test_default_extras_and_streams_are_empty():
    d = _connected_probe()
    assert d.edge_health_extras() == {}
    assert d.edge_health_streams() == []


def test_start_edge_health_creates_and_starts_once(monkeypatch):
    fake_instance = MagicMock()
    fake_cls = MagicMock(return_value=fake_instance)
    monkeypatch.setattr(base_module, "EdgeHealthCheck", fake_cls)

    d = _connected_probe()
    d._start_edge_health()

    fake_cls.assert_called_once()
    _, kwargs = fake_cls.call_args
    assert kwargs["twin_uuids"] == ["twin-1"]
    assert kwargs["edge_id"] == "twin-1"
    assert kwargs["interval"] == 5
    assert kwargs["host_metrics_provider"] == d.edge_health_extras
    fake_instance.start.assert_called_once()

    # Idempotent: a second call does not recreate the publisher.
    d._start_edge_health()
    fake_cls.assert_called_once()


def test_start_edge_health_registers_declared_streams(monkeypatch):
    fake_instance = MagicMock()
    monkeypatch.setattr(base_module, "EdgeHealthCheck", MagicMock(return_value=fake_instance))

    class _WithStreams(_Probe):
        def edge_health_streams(self):
            return [("arm-joints", {"kind": "imu", "source": "joint_states", "rate_hz": 10})]

    d = _WithStreams(
        twin=SimpleNamespace(uuid="twin-2"),
        client=SimpleNamespace(mqtt=MagicMock()),
        auto_register_interface=False,
    )
    d._start_edge_health()

    fake_instance.register_stream_config.assert_called_once_with(
        "arm-joints", {"kind": "imu", "source": "joint_states", "rate_hz": 10}
    )


def test_start_edge_health_honors_class_interval(monkeypatch):
    fake_instance = MagicMock()
    fake_cls = MagicMock(return_value=fake_instance)
    monkeypatch.setattr(base_module, "EdgeHealthCheck", fake_cls)

    class _FastHeartbeat(_Probe):
        EDGE_HEALTH_INTERVAL_S = 1

    d = _FastHeartbeat(
        twin=SimpleNamespace(uuid="twin-3"),
        client=SimpleNamespace(mqtt=MagicMock()),
        auto_register_interface=False,
    )
    d._start_edge_health()
    assert fake_cls.call_args.kwargs["interval"] == 1


def test_stop_edge_health_stops_and_clears(monkeypatch):
    fake_instance = MagicMock()
    monkeypatch.setattr(base_module, "EdgeHealthCheck", MagicMock(return_value=fake_instance))

    d = _connected_probe()
    d._start_edge_health()
    d._stop_edge_health()

    fake_instance.stop.assert_called_once()
    assert d._edge_health is None

    # Idempotent: stopping again without a live instance is a no-op.
    d._stop_edge_health()
    fake_instance.stop.assert_called_once()


def test_touch_edge_health_marks_alive_only_when_started(monkeypatch):
    fake_instance = MagicMock()
    monkeypatch.setattr(base_module, "EdgeHealthCheck", MagicMock(return_value=fake_instance))

    d = _connected_probe()
    d._touch_edge_health()  # not started yet — must not raise
    fake_instance.mark_alive.assert_not_called()

    d._start_edge_health()
    d._touch_edge_health()
    fake_instance.mark_alive.assert_called_once()
