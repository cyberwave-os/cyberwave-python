"""cw.affect(sim profile) auto-starts a MuJoCo simulation; playground/live do not."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from cyberwave.client import _affect_autostart_backend


def _client():
    from cyberwave.client import Cyberwave

    client = object.__new__(Cyberwave)
    client.config = SimpleNamespace(
        runtime_mode="live", source_type="edge", environment_id=None
    )
    client._mqtt_client = None
    simulations = SimpleNamespace(
        get_active=MagicMock(return_value=None),
        start=MagicMock(
            return_value=SimpleNamespace(simulation_id="sim-1", status="running")
        ),
    )
    client.environments = SimpleNamespace(simulations=simulations)
    client._resolve_environment_id = lambda e: e
    return client, simulations


def test_backend_resolution() -> None:
    assert _affect_autostart_backend("sim") == "mujoco"
    assert _affect_autostart_backend("simulation") == "mujoco"
    assert _affect_autostart_backend("mujoco") == "mujoco"
    assert _affect_autostart_backend("playground") is None
    assert _affect_autostart_backend("live") is None
    assert _affect_autostart_backend("real-world") is None


def test_affect_sim_starts_mujoco_when_env_set() -> None:
    client, simulations = _client()
    client.config.environment_id = "env-1"
    client.affect("sim")
    simulations.start.assert_called_once_with("env-1", backend="mujoco", duration=None)


def test_affect_sim_reuses_active_simulation() -> None:
    client, simulations = _client()
    client.config.environment_id = "env-1"
    simulations.get_active.return_value = SimpleNamespace(
        simulation_id="existing", status="running"
    )
    client.affect("simulation")
    simulations.start.assert_not_called()


def test_affect_sim_without_env_does_not_start(caplog) -> None:
    client, simulations = _client()
    with caplog.at_level("WARNING"):
        client.affect("sim")
    simulations.start.assert_not_called()
    assert any("no environment is set" in r.message for r in caplog.records)


def test_affect_sim_warns_about_credits_when_starting(caplog) -> None:
    client, simulations = _client()
    client.config.environment_id = "env-1"
    with caplog.at_level("WARNING"):
        client.affect("sim")
    simulations.start.assert_called_once()
    assert any(
        "credits/hour" in r.message and r.levelname == "WARNING"
        for r in caplog.records
    )


def test_affect_playground_does_not_start() -> None:
    client, simulations = _client()
    client.config.environment_id = "env-1"
    client.affect("playground")
    simulations.start.assert_not_called()
    assert client.config.runtime_mode == "simulation"


def test_affect_live_does_not_start() -> None:
    client, simulations = _client()
    client.config.environment_id = "env-1"
    client.affect("live")
    simulations.start.assert_not_called()
    assert client.config.runtime_mode == "live"


def test_affect_passes_duration() -> None:
    client, simulations = _client()
    client.affect("mujoco", environment_id="env-9", duration=120.0)
    simulations.start.assert_called_once_with("env-9", backend="mujoco", duration=120.0)


def test_affect_sim_waits_for_newly_started_sim_to_become_active() -> None:
    """affect() must not return while the sim it just started is still 'loading'."""
    client, simulations = _client()
    client.config.environment_id = "env-1"
    loading_sim = SimpleNamespace(simulation_id="sim-1", status="loading")
    simulations.start.return_value = loading_sim

    def _wait_until_active():
        loading_sim.status = "running"
        return loading_sim

    loading_sim.wait_until_active = MagicMock(side_effect=_wait_until_active)

    client.affect("sim")

    loading_sim.wait_until_active.assert_called_once()
    assert loading_sim.status == "running"


def test_affect_sim_waits_for_reused_sim_still_loading() -> None:
    """Reusing a sim that another caller just started must also wait for it."""
    client, simulations = _client()
    client.config.environment_id = "env-1"
    loading_sim = SimpleNamespace(simulation_id="existing", status="loading")
    loading_sim.wait_until_active = MagicMock(
        side_effect=lambda: setattr(loading_sim, "status", "running")
    )
    simulations.get_active.return_value = loading_sim

    client.affect("sim")

    loading_sim.wait_until_active.assert_called_once()
    simulations.start.assert_not_called()


def test_affect_sim_skips_wait_when_already_running() -> None:
    client, simulations = _client()
    client.config.environment_id = "env-1"
    running_sim = SimpleNamespace(simulation_id="existing", status="running")
    running_sim.wait_until_active = MagicMock()
    simulations.get_active.return_value = running_sim

    client.affect("sim")

    running_sim.wait_until_active.assert_not_called()
