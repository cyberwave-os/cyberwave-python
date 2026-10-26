"""MuJoCo camera streamers stamp their ``edge_health`` heartbeat like the client.

A heartbeat carries the provenance of the process that sends it, the same
``source_type`` its joints and poses carry: ``sim`` for a client in simulation
mode, ``edge`` for a scene that stands in for an edge device (the so101 MuJoCo
example). ``CyberwaveSimStreaming.from_env`` builds a simulation-mode client.
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

pytest.importorskip("av", reason="pyav not installed")
pytest.importorskip("aiortc", reason="aiortc not installed")

_CAMERA_SIM = "cyberwave.sensor.camera_sim"


@pytest.fixture
def camera_sim(monkeypatch: pytest.MonkeyPatch):
    """Import ``camera_sim`` even where mujoco is not installed.

    ``camera_sim`` imports ``mujoco`` at module scope but only touches it
    inside methods these tests never reach. CI installs neither extra that
    pulls mujoco, so without a stand-in these tests would always skip there.
    A module imported against the stand-in is dropped afterwards so no other
    test sees it.
    """
    already_loaded = _CAMERA_SIM in sys.modules
    try:
        importlib.import_module("mujoco")
    except ImportError:
        monkeypatch.setitem(sys.modules, "mujoco", types.ModuleType("mujoco"))
    module = importlib.import_module(_CAMERA_SIM)
    yield module
    if not already_loaded:
        sys.modules.pop(_CAMERA_SIM, None)
        sensor_pkg = sys.modules.get("cyberwave.sensor")
        if sensor_pkg is not None and hasattr(sensor_pkg, "camera_sim"):
            delattr(sensor_pkg, "camera_sim")


class _RecordingHealthCheck:
    instances: List["_RecordingHealthCheck"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs: Dict[str, Any] = kwargs
        _RecordingHealthCheck.instances.append(self)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> List[_RecordingHealthCheck]:
    from cyberwave.edge import health as health_module
    from cyberwave.sensor import base_video

    _RecordingHealthCheck.instances.clear()
    monkeypatch.setattr(health_module, "EdgeHealthCheck", _RecordingHealthCheck)
    # base_video caches the class in a module global on first use.
    monkeypatch.setattr(base_video, "EdgeHealthCheck", _RecordingHealthCheck)
    return _RecordingHealthCheck.instances


def _wrapper_client(source_type: str) -> MagicMock:
    """A ``CyberwaveMQTTClient`` wrapper: provenance lives on its config."""
    client = MagicMock()
    client.config.source_type = source_type
    return client


@pytest.mark.parametrize("source_type", ["sim", "edge"])
def test_cyberwave_sim_streaming_heartbeat_follows_the_client(
    camera_sim, recorded, monkeypatch, source_type
):
    monkeypatch.setattr(camera_sim, "MujocoMultiCameraStreamer", MagicMock())

    camera_sim.CyberwaveSimStreaming(
        client=_wrapper_client(source_type), twin_uuid="twin-sim"
    ).start(model=None)

    assert len(recorded) == 1
    assert recorded[0].kwargs["source_type"] == source_type
    assert recorded[0].kwargs["twin_uuids"] == ["twin-sim"]


@pytest.mark.parametrize("source_type", ["sim", "edge"])
def test_standalone_sim_camera_streamer_heartbeat_follows_the_client(
    camera_sim, recorded, monkeypatch, source_type
):
    from cyberwave.sensor import base_video

    # No running loop in a sync test: drop the frame-count monitor coroutine.
    monkeypatch.setattr(base_video.asyncio, "create_task", lambda coro: coro.close())
    streamer = camera_sim.SimCameraStreamer(
        client=_wrapper_client(source_type),
        frame_buffer=camera_sim.ThreadSafeFrameBuffer(fps=15),
        twin_uuid="twin-sim",
    )

    streamer._start_health_check()

    assert len(recorded) == 1
    assert recorded[0].kwargs["source_type"] == source_type


def test_from_env_builds_a_simulation_mode_client(camera_sim, monkeypatch):
    import cyberwave

    built: List[Dict[str, Any]] = []

    class _FakeCyberwave:
        def __init__(self, **kwargs: Any) -> None:
            built.append(kwargs)
            self.mqtt = _wrapper_client("sim")
            self.config = SimpleNamespace(base_url="http://test")

    monkeypatch.setenv("CYBERWAVE_STREAM_CAMERAS", "1")
    monkeypatch.setattr(cyberwave, "Cyberwave", _FakeCyberwave)
    monkeypatch.setattr(camera_sim, "MujocoMultiCameraStreamer", MagicMock())

    streaming = camera_sim.CyberwaveSimStreaming.from_env(fallback_twin_uuid="twin-sim")

    assert streaming is not None
    assert built == [{"mode": "simulation"}]


def test_client_source_type_reads_either_client_shape() -> None:
    from cyberwave.edge.health import client_source_type

    # The base MQTT client carries it directly, the wrapper on its config.
    assert client_source_type(SimpleNamespace(source_type="sim")) == "sim"
    assert client_source_type(_wrapper_client("sim")) == "sim"
    assert client_source_type(_wrapper_client("edge")) == "edge"
    # No provenance, or one the platform does not know: the ``edge`` default.
    assert client_source_type(MagicMock()) == "edge"
    assert client_source_type(SimpleNamespace(source_type="bogus")) == "edge"
