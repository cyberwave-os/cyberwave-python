"""cw.affect(sim profile) auto-starts a MuJoCo or Isaac simulation; playground/live do not."""

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
    assert _affect_autostart_backend("isaac") == "isaac"
    assert _affect_autostart_backend(" Isaac ") == "isaac"
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
        simulation_id="existing", status="running", backend="mujoco"
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
    loading_sim = SimpleNamespace(
        simulation_id="existing", status="loading", backend="mujoco"
    )
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
    running_sim = SimpleNamespace(
        simulation_id="existing", status="running", backend="mujoco"
    )
    running_sim.wait_until_active = MagicMock()
    simulations.get_active.return_value = running_sim

    client.affect("sim")

    running_sim.wait_until_active.assert_not_called()


def test_affect_isaac_starts_isaac_and_targets_the_simulation() -> None:
    client, simulations = _client()
    client.config.environment_id = "env-1"
    client.affect("isaac", duration=60.0)
    simulations.start.assert_called_once_with("env-1", backend="isaac", duration=60.0)
    assert client.config.runtime_mode == "simulation"
    assert client.config.source_type == "sim"


def test_affect_isaac_warns_the_isaac_rate(caplog) -> None:
    client, simulations = _client()
    client.config.environment_id = "env-1"
    with caplog.at_level("WARNING"):
        client.affect("isaac")
    started = [r.getMessage() for r in caplog.records if "Started" in r.getMessage()]
    # The backend bills Isaac as ``isaac-sim``: 0.05 credits/min.
    assert started == [
        "Started Isaac Sim simulation sim-1 for environment env-1 — this is a "
        "billable cloud instance consuming credits at ~3.00 credits/hour "
        "(0.05 credits/min). Stop it with sim.stop() when done."
    ]


def test_affect_isaac_reuses_an_active_isaac_run() -> None:
    client, simulations = _client()
    client.config.environment_id = "env-1"
    simulations.get_active.return_value = SimpleNamespace(
        simulation_id="existing", status="running", backend="isaac"
    )
    client.affect("isaac")
    simulations.start.assert_not_called()


def test_affect_isaac_replaces_an_active_mujoco_run(caplog) -> None:
    client, simulations = _client()
    client.config.environment_id = "env-1"
    simulations.get_active.return_value = SimpleNamespace(
        simulation_id="existing", status="running", backend=None
    )
    with caplog.at_level("WARNING"):
        client.affect("isaac")
    simulations.start.assert_called_once_with("env-1", backend="isaac", duration=None)
    assert any("runs on mujoco, not isaac" in r.getMessage() for r in caplog.records)


def test_affect_mujoco_replaces_an_active_isaac_run() -> None:
    client, simulations = _client()
    client.config.environment_id = "env-1"
    simulations.get_active.return_value = SimpleNamespace(
        simulation_id="existing", status="running", backend="isaac"
    )
    client.affect("sim")
    simulations.start.assert_called_once_with("env-1", backend="mujoco", duration=None)


def test_affect_mujoco_profiles_reuse_an_active_mujoco_warp_run() -> None:
    # A MuJoCo Warp run is a MuJoCo plant: the MuJoCo profiles keep it, as
    # they did before ``affect("isaac")`` existed.
    for mode, backend in (("sim", "mujoco_warp"), ("mujoco", "mjwarp")):
        client, simulations = _client()
        client.config.environment_id = "env-1"
        simulations.get_active.return_value = SimpleNamespace(
            simulation_id="warp", status="running", backend=backend
        )
        client.affect(mode)
        simulations.start.assert_not_called()


def test_affect_isaac_replaces_an_active_mujoco_warp_run() -> None:
    client, simulations = _client()
    client.config.environment_id = "env-1"
    simulations.get_active.return_value = SimpleNamespace(
        simulation_id="warp", status="running", backend="mujoco_warp"
    )
    client.affect("isaac")
    simulations.start.assert_called_once_with("env-1", backend="isaac", duration=None)


def test_affect_replaces_a_run_on_an_engine_the_sdk_does_not_know(caplog) -> None:
    # A newer server may list an engine this SDK predates: that run is not
    # the one asked for, so it is replaced rather than failing the call.
    client, simulations = _client()
    client.config.environment_id = "env-1"
    simulations.get_active.return_value = SimpleNamespace(
        simulation_id="other", status="running", backend="future_engine"
    )
    with caplog.at_level("WARNING"):
        client.affect("sim")
    simulations.start.assert_called_once_with("env-1", backend="mujoco", duration=None)
    assert any(
        "runs on future_engine, not mujoco" in r.getMessage() for r in caplog.records
    )


def test_affect_isaac_without_env_names_the_isaac_start(caplog) -> None:
    client, simulations = _client()
    with caplog.at_level("WARNING"):
        client.affect("isaac")
    simulations.start.assert_not_called()
    [message] = [r.getMessage() for r in caplog.records]
    assert message.startswith(
        "affect('isaac') selected the Isaac Sim simulation runtime, but no "
        "environment is set"
    )
    assert message.endswith(
        "cw.environments.simulations.start(environment_id, backend='isaac')."
    )
