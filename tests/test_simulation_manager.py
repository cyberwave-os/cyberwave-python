"""Tests for cyberwave.managers.simulations (SimulationManager + Simulation)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cyberwave.exceptions import CyberwaveError
from cyberwave.managers.simulations import Simulation, SimulationManager


def _api_returning(payload: dict) -> MagicMock:
    """Build a fake DefaultApi whose raw-call chain yields `payload`."""
    api = MagicMock()
    api.api_client.param_serialize.return_value = ()  # call_api(*()) -> call_api()
    api.api_client.response_deserialize.return_value.data = payload
    return api


def test_start_posts_body_and_returns_simulation() -> None:
    api = _api_returning(
        {"success": True, "simulation": {"simulation_id": "sim-1", "status": "loading",
                                         "stream_data": True, "backend": "mujoco"}}
    )
    mgr = SimulationManager(api)

    sim = mgr.start("env-1", backend="mujoco", duration=30.0)

    assert isinstance(sim, Simulation)
    assert sim.simulation_id == "sim-1"
    assert sim.status == "loading"
    assert sim.environment_id == "env-1"
    call = api.api_client.param_serialize.call_args
    assert call.kwargs["method"] == "POST"
    assert call.kwargs["resource_path"] == "/api/v1/environments/{uuid}/simulations"
    assert call.kwargs["path_params"] == {"uuid": "env-1"}
    body = call.kwargs["body"]
    assert body["backend"] == "mujoco"
    assert body["stream_data"] is True
    assert body["duration"] == 30.0


def test_start_omits_duration_when_none() -> None:
    api = _api_returning({"simulation": {"simulation_id": "s", "status": "loading"}})
    SimulationManager(api).start("env-1")
    assert "duration" not in api.api_client.param_serialize.call_args.kwargs["body"]


def test_list_maps_active_simulations() -> None:
    api = _api_returning({"active_simulations": [
        {"simulation_id": "a", "status": "loading"},
        {"simulation_id": "b", "status": "running"},
    ]})
    sims = SimulationManager(api).list("env-1")
    assert [s.simulation_id for s in sims] == ["a", "b"]
    assert api.api_client.param_serialize.call_args.kwargs["method"] == "GET"


def test_get_active_prefers_running_over_loading() -> None:
    api = _api_returning({"active_simulations": [
        {"simulation_id": "a", "status": "loading"},
        {"simulation_id": "b", "status": "running"},
    ]})
    sim = SimulationManager(api).get_active("env-1")
    assert sim is not None and sim.simulation_id == "b"


def test_get_active_returns_none_when_empty() -> None:
    api = _api_returning({"active_simulations": []})
    assert SimulationManager(api).get_active("env-1") is None


def test_stop_posts_to_stop_endpoint() -> None:
    api = _api_returning({"success": True})
    SimulationManager(api).stop("env-1", "sim-1")
    call = api.api_client.param_serialize.call_args
    assert call.kwargs["method"] == "POST"
    assert call.kwargs["resource_path"] == (
        "/api/v1/environments/{uuid}/simulations/{simulation_id}/stop"
    )
    assert call.kwargs["path_params"] == {"uuid": "env-1", "simulation_id": "sim-1"}


def test_wait_until_active_returns_when_running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cyberwave.managers.simulations.time.sleep", lambda *_: None)
    mgr = MagicMock()
    sim = Simulation(environment_id="env-1", simulation_id="s", status="loading",
                     stream_data=True, backend="mujoco", raw={})
    sim._manager = mgr
    mgr._get_by_id.side_effect = [
        Simulation("env-1", "s", "loading", True, "mujoco", {}),
        Simulation("env-1", "s", "running", True, "mujoco", {}),
    ]
    out = sim.wait_until_active(timeout=10.0, poll=0.0)
    assert out.status == "running"


def test_wait_until_active_raises_on_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cyberwave.managers.simulations.time.sleep", lambda *_: None)
    mgr = MagicMock()
    sim = Simulation("env-1", "s", "loading", True, "mujoco", {})
    sim._manager = mgr
    mgr._get_by_id.return_value = Simulation("env-1", "s", "failed", True, "mujoco", {})
    with pytest.raises(CyberwaveError):
        sim.wait_until_active(timeout=10.0, poll=0.0)


def test_wait_until_active_raises_when_sim_vanishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reused 'loading' sim that drops off the active list (refresh -> 'stopped')
    is terminal: fail fast instead of polling until the full timeout."""
    monkeypatch.setattr("cyberwave.managers.simulations.time.sleep", lambda *_: None)
    mgr = MagicMock()
    sim = Simulation("env-1", "s", "loading", True, "mujoco", {})
    sim._manager = mgr
    mgr._get_by_id.return_value = None  # no longer in the active list
    with pytest.raises(CyberwaveError) as excinfo:
        sim.wait_until_active(timeout=10.0, poll=0.0)
    assert "stopped" in str(excinfo.value)


def _api_raising(exc: Exception) -> MagicMock:
    """Build a fake DefaultApi whose response_deserialize raises `exc`."""
    api = MagicMock()
    api.api_client.param_serialize.return_value = ()
    api.api_client.response_deserialize.side_effect = exc
    return api


def test_start_surfaces_backend_detail_on_bad_request() -> None:
    from cyberwave.rest.exceptions import BadRequestException

    exc = BadRequestException(
        status=400,
        reason="Bad Request",
        body='{"detail": "Mujoco simulation is not enabled for: Logitech C920."}',
    )
    exc.headers = {"content-type": "application/json"}
    mgr = SimulationManager(_api_raising(exc))

    with pytest.raises(CyberwaveError) as excinfo:
        mgr.start("env-1", backend="mujoco")

    msg = str(excinfo.value)
    assert "Mujoco simulation is not enabled for: Logitech C920." in msg
    assert "HTTP response headers" not in msg
    assert "Request headers" not in msg


def test_start_surfaces_backend_detail_from_data_dict() -> None:
    from cyberwave.rest.exceptions import BadRequestException

    exc = BadRequestException(status=400, reason="Bad Request", body=None)
    exc.data = {"detail": "Environment is not simulatable."}
    mgr = SimulationManager(_api_raising(exc))

    with pytest.raises(CyberwaveError) as excinfo:
        mgr.start("env-1", backend="mujoco")

    assert "Environment is not simulatable." in str(excinfo.value)


def test_start_falls_back_to_generic_error_without_detail() -> None:
    mgr = SimulationManager(_api_raising(RuntimeError("boom")))
    with pytest.raises(CyberwaveError) as excinfo:
        mgr.start("env-1", backend="mujoco")
    assert "start simulation for environment env-1" in str(excinfo.value)
