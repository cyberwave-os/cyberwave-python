"""CameraTwin simulation-mode streaming branches.

Streaming is MuJoCo-level: it consumes an already-running simulation (started via
cw.affect("sim") or cw.environments.simulations.start) and never starts one.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cyberwave.exceptions import SimulationNotRunningError
from cyberwave.twin.classes import CameraTwin


def _running_sim(status="running"):
    sim = SimpleNamespace(
        simulation_id="sim-1", status=status, backend="mujoco", raw={},
        total_duration_s=None, stop=MagicMock(),
    )
    sim.wait_until_active = MagicMock(return_value=sim)
    return sim


def _sim_twin(*, runtime_mode="simulation", active_sim="__running__", cls=CameraTwin):
    """Build a camera twin bypassing __init__, wired for sim-branch tests."""
    twin = object.__new__(cls)
    twin._camera_streamer = None
    twin._active_simulation = None
    twin._data = {"environment_uuid": "env-1"}
    sim = _running_sim()
    if active_sim == "__running__":
        active_sim = sim
    simulations = SimpleNamespace(
        start=MagicMock(side_effect=AssertionError("streaming must not start a sim")),
        get_active=MagicMock(return_value=active_sim),
    )
    twin.client = SimpleNamespace(
        config=SimpleNamespace(runtime_mode=runtime_mode),
        environments=SimpleNamespace(simulations=simulations),
        video_stream=MagicMock(),
    )
    return twin, active_sim


def test_streaming_methods_are_mujoco_level() -> None:
    from cyberwave.twin.simulation_support import SimLevel

    assert CameraTwin.start_streaming.__cw_sim_level__ == SimLevel.MUJOCO
    assert CameraTwin.stream_video_background.__cw_sim_level__ == SimLevel.MUJOCO


def test_resolve_simulation_stream_stores_running_sim() -> None:
    twin, sim = _sim_twin()
    out = twin._resolve_simulation_stream()
    assert out is sim
    twin.client.environments.simulations.start.assert_not_called()
    sim.wait_until_active.assert_not_called()  # already running
    assert twin._active_simulation is sim


def test_resolve_simulation_stream_waits_for_loading_sim() -> None:
    loading = _running_sim(status="loading")
    twin, _ = _sim_twin(active_sim=loading)
    out = twin._resolve_simulation_stream()
    assert out is loading
    loading.wait_until_active.assert_called_once()
    assert twin._active_simulation is loading


def test_stream_video_background_sim_returns_sim_without_local_camera() -> None:
    twin, sim = _sim_twin()
    out = asyncio.run(twin.stream_video_background())
    assert out is sim
    twin.client.video_stream.assert_not_called()


def test_stream_video_background_raises_without_running_sim() -> None:
    twin, _ = _sim_twin(active_sim=None)
    with pytest.raises(SimulationNotRunningError):
        asyncio.run(twin.stream_video_background())


def test_stop_streaming_stops_active_simulation() -> None:
    twin, sim = _sim_twin()
    twin._active_simulation = sim
    asyncio.run(twin.stop_streaming())
    sim.stop.assert_called_once()
    assert twin._active_simulation is None


def test_stream_video_background_sim_does_not_block_event_loop() -> None:
    """The sim branch offloads the blocking resolve to a thread, so awaiting it
    must not stall other coroutines on the loop."""
    import time as _time

    twin, sim = _sim_twin()
    # _resolve_simulation_stream() may block (wait_until_active); simulate that.
    twin._resolve_simulation_stream = lambda: (_time.sleep(0.2), sim)[1]  # type: ignore[method-assign]

    async def _scenario() -> int:
        ticks = 0

        async def _ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        ticker = asyncio.create_task(_ticker())
        out = await twin.stream_video_background()
        ticker.cancel()
        assert out is sim
        return ticks

    ticks = asyncio.run(_scenario())
    assert ticks >= 3


def test_depth_camera_twin_background_sim_returns_sim() -> None:
    from cyberwave.twin.classes import DepthCameraTwin

    twin, sim = _sim_twin(cls=DepthCameraTwin)
    out = asyncio.run(twin.stream_video_background())
    assert out is sim
    twin.client.video_stream.assert_not_called()
