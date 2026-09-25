"""Tests for live/simulation runtime bucket helpers."""

from types import SimpleNamespace

from cyberwave.twin.runtime_state import (
    RUNTIME_MODE_LIVE,
    RUNTIME_MODE_SIMULATION,
    active_runtime_mode,
    runtime_mode_from_mqtt_source_type,
)


def test_active_runtime_mode_from_config() -> None:
    live_client = SimpleNamespace(config=SimpleNamespace(runtime_mode="live"))
    sim_client = SimpleNamespace(config=SimpleNamespace(runtime_mode="simulation"))
    assert active_runtime_mode(live_client) == RUNTIME_MODE_LIVE
    assert active_runtime_mode(sim_client) == RUNTIME_MODE_SIMULATION


def test_runtime_mode_from_mqtt_source_type() -> None:
    client = SimpleNamespace(config=SimpleNamespace(runtime_mode="live"))
    assert runtime_mode_from_mqtt_source_type("sim_tele") == RUNTIME_MODE_SIMULATION
    assert runtime_mode_from_mqtt_source_type("tele") == RUNTIME_MODE_LIVE
    assert runtime_mode_from_mqtt_source_type("edge") == RUNTIME_MODE_LIVE
    assert runtime_mode_from_mqtt_source_type(None, client=client) == RUNTIME_MODE_LIVE
