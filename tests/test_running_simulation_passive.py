"""running_simulation() is passive — it never starts a simulation."""

from __future__ import annotations

from types import SimpleNamespace

from cyberwave.managers.simulations import _SimReadyCache, running_simulation


def _twin(runtime_mode, *, active):
    started = {"count": 0}

    def _start(env, backend="mujoco"):
        started["count"] += 1
        return SimpleNamespace(simulation_id="new", status="running", backend=backend,
                               total_duration_s=None)

    sims = SimpleNamespace(
        get_active=lambda env: active,
        start=_start,
    )
    twin = SimpleNamespace(
        client=SimpleNamespace(
            config=SimpleNamespace(runtime_mode=runtime_mode),
            environments=SimpleNamespace(simulations=sims),
        ),
        environment_id="env-1",
    )
    return twin, started


def test_returns_none_in_live_mode() -> None:
    twin, started = _twin("live", active=None)
    assert running_simulation(twin) is None
    assert started["count"] == 0


def test_returns_none_when_nothing_active_and_never_starts() -> None:
    twin, started = _twin("simulation", active=None)
    assert running_simulation(twin) is None
    assert started["count"] == 0


def test_returns_active_running_sim_without_starting() -> None:
    running = SimpleNamespace(simulation_id="s1", status="running", backend="mujoco",
                              total_duration_s=None)
    twin, started = _twin("simulation", active=running)
    assert running_simulation(twin) is running
    assert started["count"] == 0


def _running_sim(sim_id="sim-1", duration=None):
    return SimpleNamespace(
        simulation_id=sim_id, status="running", total_duration_s=duration
    )


def test_cache_get_valid_expires_after_duration() -> None:
    cache = _SimReadyCache()
    sim = _running_sim(duration=0.0)  # duration 0 -> immediately elapsed
    cache.put("env-x", sim)
    assert cache.get_valid("env-x") is None


def test_cache_get_valid_drops_non_running() -> None:
    cache = _SimReadyCache()
    sim = _running_sim()
    cache.put("env-x", sim)
    sim.status = "stopped"
    assert cache.get_valid("env-x") is None
