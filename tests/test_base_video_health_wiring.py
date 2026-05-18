"""Pin the wiring between ``BaseVideoStreamer._start_health_check`` and ``EdgeHealthCheck``.

The integration here is small but easy to silently regress: a refactor
that switched the constructor kwarg name or stopped passing the
provider would compile cleanly, pass every unit test in
``test_edge_health.py``, and ship a publisher that never picks up
post-negotiation ``actual_fps``.  These tests catch that class of bug
by exercising the actual ``_start_health_check`` codepath with a
recording stub in place of ``EdgeHealthCheck``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

pytest.importorskip("cv2", reason="OpenCV not installed")
pytest.importorskip("av", reason="pyav not installed")

from cyberwave.sensor import base_video  # noqa: E402
from cyberwave.sensor.camera_cv2 import CV2CameraStreamer  # noqa: E402
from cyberwave.sensor.config import Resolution  # noqa: E402


class _RecordingHealthCheck:
    """Drop-in stub for ``EdgeHealthCheck`` that captures init kwargs.

    We want to assert that ``_start_health_check`` constructs the real
    publisher with ``stream_config_provider=self._collect_stream_configs``;
    intercepting at the class level is the cheapest way to do that
    without spinning up a background thread.
    """

    instances: List["_RecordingHealthCheck"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = False
        _RecordingHealthCheck.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False


class _FakeMQTT:
    topic_prefix = ""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def publish(self, topic: str, payload: Dict[str, Any], qos: int = 0) -> None:
        del qos
        self.calls.append((topic, dict(payload)))


@pytest.fixture
def patched_health_check(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[], List[_RecordingHealthCheck]]:
    """Install the recording stub for the duration of the test."""
    _RecordingHealthCheck.instances.clear()
    monkeypatch.setattr(base_video, "EdgeHealthCheck", _RecordingHealthCheck)

    def _no_op_create_task(coro: Any, *args: Any, **kwargs: Any) -> None:
        # _start_health_check schedules a monitor coroutine; we just
        # close it so we don't leak an un-awaited coroutine warning.
        try:
            coro.close()
        except Exception:
            pass
        return None

    monkeypatch.setattr(base_video.asyncio, "create_task", _no_op_create_task)
    return lambda: _RecordingHealthCheck.instances


def test_start_health_check_wires_provider_not_static_register(
    patched_health_check: Callable[[], List[_RecordingHealthCheck]],
) -> None:
    """``EdgeHealthCheck`` is constructed with ``stream_config_provider``.

    The provider path is what makes ``actual_fps`` and other
    runtime-negotiated values flow to the wire on every heartbeat; if
    we regress to ``register_stream_config`` (one-shot snapshot at
    startup) the dashboard would freeze on the requested fps forever.
    """
    streamer = CV2CameraStreamer(
        client=_FakeMQTT(),
        camera_id=0,
        fps=30,
        resolution=Resolution.HD,
        twin_uuid="twin-a",
    )

    streamer._start_health_check()

    instances = patched_health_check()
    assert len(instances) == 1
    kwargs = instances[0].kwargs
    assert "stream_config_provider" in kwargs
    assert kwargs["stream_config_provider"] == streamer._collect_stream_configs
    assert instances[0].started, "health check must be started after wiring"


def test_provider_returns_stream_config_dict_keyed_by_canonical_stream_id(
    patched_health_check: Callable[[], List[_RecordingHealthCheck]],
) -> None:
    """The captured provider returns ``{"stream": {kind: "camera", ...}}``.

    Pins the cross-language wire contract: single-stream publishers
    use the literal ``"stream"`` id (matching the C++ ``CameraStreamer``
    default) so the frontend doesn't have to guess.
    """
    streamer = CV2CameraStreamer(
        client=_FakeMQTT(),
        camera_id=0,
        fps=20,
        resolution=Resolution.VGA,
        twin_uuid="twin-a",
    )

    streamer._start_health_check()

    provider: Callable[[], Dict[str, Dict[str, Any]]] = patched_health_check()[
        0
    ].kwargs["stream_config_provider"]
    snapshot = provider()
    assert set(snapshot) == {"stream"}
    block = snapshot["stream"]
    assert block["kind"] == "camera"
    assert block["resolution"] == "640x480"
    assert block["fps"] == 20
    assert block["camera_type"] == "cv2"


def test_provider_returns_empty_dict_when_build_returns_none(
    patched_health_check: Callable[[], List[_RecordingHealthCheck]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subclasses that don't override ``_build_stream_config`` get the legacy shape.

    ``EdgeHealthCheck.get_health_data`` treats an empty provider dict
    as "no registered streams", which falls back to the historical
    single ``"stream"`` entry without a ``stream_config`` block.
    Without this, a subclass that hasn't been touched for CYB-2004
    would ship an unexpected wire shape change.
    """
    streamer = CV2CameraStreamer(
        client=_FakeMQTT(),
        camera_id=0,
        fps=15,
        resolution=Resolution.HD,
        twin_uuid="twin-a",
    )
    # Simulate a subclass that doesn't override the hook by returning None.
    monkeypatch.setattr(streamer, "_build_stream_config", lambda: None)

    streamer._start_health_check()

    provider = patched_health_check()[0].kwargs["stream_config_provider"]
    assert provider() == {}


def test_provider_swallows_build_exception(
    patched_health_check: Callable[[], List[_RecordingHealthCheck]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A buggy ``_build_stream_config`` must not break the heartbeat.

    Subclass override raising on the publisher thread would otherwise
    kill the entire publish cycle, silently marking the edge offline
    in the dashboard.  Empty-dict-on-exception keeps liveness flowing.
    """
    streamer = CV2CameraStreamer(
        client=_FakeMQTT(),
        camera_id=0,
        fps=15,
        resolution=Resolution.HD,
        twin_uuid="twin-a",
    )

    def _explode() -> Optional[Dict[str, Any]]:
        raise RuntimeError("subclass override regression")

    monkeypatch.setattr(streamer, "_build_stream_config", _explode)

    streamer._start_health_check()
    provider = patched_health_check()[0].kwargs["stream_config_provider"]

    assert provider() == {}


def test_start_health_check_no_ops_when_disabled(
    patched_health_check: Callable[[], List[_RecordingHealthCheck]],
) -> None:
    """``enable_health_check=False`` skips the wiring entirely.

    A regression that ignored this flag would start publishing
    ``edge_health`` for streamers that opted out (e.g. local
    debugging tracks), polluting MQTT and confusing dashboards.
    """
    streamer = CV2CameraStreamer(
        client=_FakeMQTT(),
        camera_id=0,
        fps=15,
        resolution=Resolution.HD,
        twin_uuid="twin-a",
    )
    # CV2CameraStreamer doesn't surface ``enable_health_check`` in its
    # own kwargs; the flag lives on ``BaseVideoStreamer`` and is set
    # by sub-streamer factories.  Toggle it directly to exercise the
    # guard in ``_start_health_check``.
    streamer.enable_health_check = False

    streamer._start_health_check()

    assert patched_health_check() == []
