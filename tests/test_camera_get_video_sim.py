"""get_video() sim-mode stream-identity resolution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("aiortc", reason="aiortc not installed (install with extras: camera)")

from cyberwave.exceptions import SimulationNotRunningError
from cyberwave.twin.base import Twin
from cyberwave.twin.sensors.camera import TwinCameraHandle


class _FakeStream:
    def __init__(self, mqtt, twin_uuid, *, sensor_id=None, stream_source=None,
                 stream_instance_id=None, timeout=30.0):
        self.sensor_id = sensor_id
        self.stream_source = stream_source
        self.stream_instance_id = stream_instance_id
        self.started = False

    def start(self):
        self.started = True


def _make_handle(monkeypatch, *, runtime_mode, active_sim, stream_cls=_FakeStream):
    monkeypatch.setattr(
        "cyberwave.consumers.video.IncomingVideoStream", stream_cls
    )
    simulations = SimpleNamespace(
        get_active=lambda env: active_sim,
        start=lambda env, backend="mujoco": pytest.fail("must not auto-start"),
    )
    client = SimpleNamespace(
        config=SimpleNamespace(runtime_mode=runtime_mode),
        environments=SimpleNamespace(simulations=simulations),
        mqtt=object(),
    )
    twin = SimpleNamespace(
        client=client,
        uuid="twin-1",
        environment_id="env-1",
        _ensure_mqtt_connected=lambda: None,
        _resolve_sensor_id=lambda sid: sid or "default",
    )
    # Bind the real preflight so the @simulation_level(MUJOCO) decorator runs.
    twin._ensure_simulation_support = Twin._ensure_simulation_support.__get__(twin)
    return TwinCameraHandle(twin, sensor_id="wrist")


def test_get_video_sim_mode_sends_simulation_identity(monkeypatch) -> None:
    sim = SimpleNamespace(
        simulation_id="sim-42", status="running", backend="mujoco", raw={},
        total_duration_s=None,
    )
    handle = _make_handle(monkeypatch, runtime_mode="simulation", active_sim=sim)

    stream = handle.get_video()

    assert stream.stream_source == "simulation"
    assert stream.stream_instance_id == "sim-42"
    assert stream.started is True


def test_get_video_sim_mode_raises_when_no_active_sim(monkeypatch) -> None:
    handle = _make_handle(monkeypatch, runtime_mode="simulation", active_sim=None)
    with pytest.raises(SimulationNotRunningError):
        handle.get_video()


def test_get_video_live_mode_sends_no_sim_identity(monkeypatch) -> None:
    handle = _make_handle(monkeypatch, runtime_mode="live", active_sim=None)
    stream = handle.get_video()
    assert stream.stream_source is None
    assert stream.stream_instance_id is None


def test_get_video_sim_mode_waits_for_producer(monkeypatch) -> None:
    """A just-started sim reaches 'running' before its producer registers, so
    get_video should poll through the initial 'no producer' window in sim mode."""
    from cyberwave.exceptions import NoOngoingVideoStreamAvailable

    attempts = {"n": 0}

    class _FlakyStream(_FakeStream):
        def start(self):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise NoOngoingVideoStreamAvailable("waiting for stream producer")
            self.started = True

    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *_: None)
    sim = SimpleNamespace(
        simulation_id="sim-42", status="running", backend="mujoco", raw={},
        total_duration_s=None,
    )
    handle = _make_handle(
        monkeypatch, runtime_mode="simulation", active_sim=sim, stream_cls=_FlakyStream
    )

    stream = handle.get_video()

    assert attempts["n"] == 3
    assert stream.started is True


def test_get_video_live_mode_does_not_wait_for_producer(monkeypatch) -> None:
    """Live mode preserves the pure-consumer contract: raise on the first
    'no producer' reply instead of polling."""
    from cyberwave.exceptions import NoOngoingVideoStreamAvailable

    attempts = {"n": 0}

    class _AlwaysMissingStream(_FakeStream):
        def start(self):
            attempts["n"] += 1
            raise NoOngoingVideoStreamAvailable("no producer")

    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *_: None)
    handle = _make_handle(
        monkeypatch,
        runtime_mode="live",
        active_sim=None,
        stream_cls=_AlwaysMissingStream,
    )

    with pytest.raises(NoOngoingVideoStreamAvailable):
        handle.get_video()
    assert attempts["n"] == 1
