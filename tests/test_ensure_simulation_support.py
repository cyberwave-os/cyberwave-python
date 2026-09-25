"""Twin._ensure_simulation_support: the three error paths + no-op cases."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cyberwave.exceptions import (
    NotSimulatedError,
    SimulationLevelError,
    SimulationNotRunningError,
)
from cyberwave.twin.base import Twin
from cyberwave.twin.simulation_support import SimLevel


def _twin(runtime_mode, *, active=None):
    sims = SimpleNamespace(
        get_active=lambda env: active,
        start=lambda env, backend="mujoco": pytest.fail("must not start"),
    )
    twin = object.__new__(Twin)
    twin.client = SimpleNamespace(
        config=SimpleNamespace(runtime_mode=runtime_mode),
        environments=SimpleNamespace(simulations=sims),
    )
    twin._data = {"environment_uuid": "env-1"}
    return twin


def _sim(status="running", backend="mujoco"):
    return SimpleNamespace(simulation_id="s1", status=status, backend=backend,
                           total_duration_s=None)


def test_live_mode_is_noop_for_all_levels() -> None:
    twin = _twin("live")
    twin._ensure_simulation_support(SimLevel.UNSUPPORTED, method="x")
    twin._ensure_simulation_support(SimLevel.MUJOCO, method="x")  # no start, no raise


def test_playground_level_is_noop_in_sim_mode() -> None:
    twin = _twin("simulation")
    twin._ensure_simulation_support(SimLevel.PLAYGROUND, method="x")


def test_unsupported_raises_in_sim_mode() -> None:
    twin = _twin("simulation")
    with pytest.raises(NotSimulatedError):
        twin._ensure_simulation_support(SimLevel.UNSUPPORTED, method="imu.get")


def test_mujoco_without_running_sim_raises_not_running() -> None:
    twin = _twin("simulation", active=None)
    with pytest.raises(SimulationNotRunningError) as exc:
        twin._ensure_simulation_support(SimLevel.MUJOCO, method="camera.get_frame")
    assert "simulations.start" in str(exc.value)


def test_mujoco_against_playground_backend_raises_level_error() -> None:
    twin = _twin("simulation", active=_sim(backend="playground"))
    with pytest.raises(SimulationLevelError):
        twin._ensure_simulation_support(SimLevel.MUJOCO, method="camera.get_frame")


def test_mujoco_against_mujoco_backend_passes() -> None:
    twin = _twin("simulation", active=_sim(backend="mujoco"))
    twin._ensure_simulation_support(SimLevel.MUJOCO, method="camera.get_frame")


def test_mujoco_against_loading_sim_raises_not_running() -> None:
    """A sim that exists but hasn't finished starting must not be treated as ready."""
    twin = _twin("simulation", active=_sim(status="loading"))
    with pytest.raises(SimulationNotRunningError) as exc:
        twin._ensure_simulation_support(SimLevel.MUJOCO, method="camera.get_frame")
    assert "loading" in str(exc.value)


def test_both_without_running_sim_raises_not_running() -> None:
    twin = _twin("simulation", active=None)
    with pytest.raises(SimulationNotRunningError):
        twin._ensure_simulation_support(SimLevel.BOTH, method="twin.something")


def test_both_against_playground_backend_passes() -> None:
    twin = _twin("simulation", active=_sim(backend="playground"))
    twin._ensure_simulation_support(SimLevel.BOTH, method="twin.something")


def test_both_against_mujoco_backend_passes() -> None:
    twin = _twin("simulation", active=_sim(backend="mujoco"))
    twin._ensure_simulation_support(SimLevel.BOTH, method="twin.something")
