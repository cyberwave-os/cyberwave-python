"""Tests for cyberwave.managers.simulations (SimulationManager + Simulation)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from cyberwave.exceptions import CyberwaveAPIError, CyberwaveError
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
    mgr._get_detail.return_value = None  # backend without the detail route
    with pytest.raises(CyberwaveError) as excinfo:
        sim.wait_until_active(timeout=10.0, poll=0.0)
    assert "stopped" in str(excinfo.value)


@pytest.mark.parametrize("status", ["completed", "failed", "stopped"])
def test_wait_until_active_raises_on_every_terminal_status(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    monkeypatch.setattr("cyberwave.managers.simulations.time.sleep", lambda *_: None)
    mgr = MagicMock()
    sim = Simulation("env-1", "s", "loading", True, "mujoco", {})
    sim._manager = mgr
    mgr._get_by_id.return_value = None  # dropped off the listing
    mgr._get_detail.return_value = Simulation("env-1", "s", status, True, None, {})
    with pytest.raises(CyberwaveError) as excinfo:
        sim.wait_until_active(timeout=10.0, poll=0.0)
    assert f"status: {status}" in str(excinfo.value)
    assert mgr._get_by_id.call_count == 1  # failed fast, no polling


def test_refresh_reads_the_detail_when_sim_leaves_the_listing() -> None:
    mgr = MagicMock()
    sim = Simulation("env-1", "s", "running", True, "mujoco", {})
    sim._manager = mgr
    mgr._get_by_id.return_value = None
    mgr._get_detail.return_value = Simulation(
        "env-1", "s", "completed", False, "mujoco", {"status": "completed"}
    )

    assert sim.refresh().status == "completed"
    assert sim.is_terminal and not sim.is_active
    assert sim.raw == {"status": "completed"}
    mgr._get_detail.assert_called_once_with("env-1", "s")


def test_refresh_skips_the_detail_while_listed() -> None:
    mgr = MagicMock()
    sim = Simulation("env-1", "s", "loading", True, "mujoco", {})
    sim._manager = mgr
    mgr._get_by_id.return_value = Simulation("env-1", "s", "running", True, None, {})

    assert sim.refresh().status == "running"
    mgr._get_detail.assert_not_called()


def _api_with_response(http_status: int, body: object) -> MagicMock:
    """Build a fake DefaultApi whose raw call returns `body` with `http_status`."""
    api = MagicMock()
    api.api_client.param_serialize.return_value = ()
    response = api.api_client.call_api.return_value
    response.status = http_status
    response.data = json.dumps(body).encode()
    return api


@pytest.mark.parametrize("status", ["completed", "failed", "stopped"])
def test_get_detail_takes_the_backend_status_as_is(status: str) -> None:
    api = _api_with_response(
        200, {"simulation_id": "sim-1", "status": status, "backend": "mujoco"}
    )

    sim = SimulationManager(api)._get_detail("env-1", "sim-1")

    assert sim is not None
    assert (sim.simulation_id, sim.status, sim.backend) == ("sim-1", status, "mujoco")
    call = api.api_client.param_serialize.call_args
    assert call.kwargs["method"] == "GET"
    assert call.kwargs["resource_path"] == (
        "/api/v1/environments/{uuid}/simulations/{simulation_id}"
    )
    assert call.kwargs["path_params"] == {"uuid": "env-1", "simulation_id": "sim-1"}


@pytest.mark.parametrize("http_status", [404, 405])
def test_get_detail_is_none_when_backend_cannot_say(http_status: int) -> None:
    # Unknown run, or a backend that predates the detail route.
    api = _api_with_response(http_status, {"detail": "Not Found"})
    assert SimulationManager(api)._get_detail("env-1", "sim-1") is None


@pytest.mark.parametrize("http_status", [401, 403, 500, 503])
def test_get_detail_raises_when_the_backend_fails(http_status: int) -> None:
    # A failed read says nothing about how the run ended: refresh() must not
    # turn it into "stopped".
    api = _api_with_response(http_status, {"detail": "boom"})
    with pytest.raises(CyberwaveAPIError):
        SimulationManager(api)._get_detail("env-1", "sim-1")


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


# ---------------------------------------------------------------------------
# Backend selection: isaac + execution profiles
# ---------------------------------------------------------------------------


def _started(api: MagicMock) -> dict:
    return api.api_client.param_serialize.call_args.kwargs["body"]


def _sim_api() -> MagicMock:
    return _api_returning({"simulation": {"simulation_id": "s", "status": "loading"}})


def test_start_default_backend_is_still_mujoco_without_profile() -> None:
    api = _sim_api()
    SimulationManager(api).start("env-1")
    body = _started(api)
    assert body["backend"] == "mujoco"
    assert "execution_profile" not in body


def test_start_backend_none_omits_field_for_server_default() -> None:
    api = _sim_api()
    SimulationManager(api).start("env-1", backend=None)
    assert "backend" not in _started(api)


def test_start_isaac_sends_backend_and_profile() -> None:
    api = _sim_api()
    SimulationManager(api).start(
        "env-1", backend="isaac", execution_profile="isaac-gpu-physics"
    )
    body = _started(api)
    assert body["backend"] == "isaac"
    assert body["execution_profile"] == "isaac-gpu-physics"


def test_start_isaac_accepts_experimental_newton_profile() -> None:
    from cyberwave.managers.simulations import (
        EXPERIMENTAL_ISAAC_EXECUTION_PROFILES,
        ISAAC_EXECUTION_PROFILES,
    )

    assert ISAAC_EXECUTION_PROFILES == (
        "isaac-cpu-physics", "isaac-gpu-physics", "isaac-newton-physics",
    )
    assert EXPERIMENTAL_ISAAC_EXECUTION_PROFILES == ("isaac-newton-physics",)
    api = _sim_api()
    SimulationManager(api).start(
        "env-1", backend="isaac", execution_profile="isaac-newton-physics"
    )
    assert _started(api)["execution_profile"] == "isaac-newton-physics"


def test_start_isaac_without_profile_leaves_default_to_server() -> None:
    api = _sim_api()
    SimulationManager(api).start("env-1", backend="isaac")
    body = _started(api)
    assert body["backend"] == "isaac"
    assert "execution_profile" not in body


def test_start_normalizes_alias_and_case() -> None:
    api = _sim_api()
    SimulationManager(api).start("env-1", backend="MJWarp")
    assert _started(api)["backend"] == "mujoco_warp"
    SimulationManager(api).start("env-1", backend=" Isaac ", execution_profile="ISAAC-CPU-PHYSICS")
    body = _started(api)
    assert body["backend"] == "isaac"
    assert body["execution_profile"] == "isaac-cpu-physics"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "gazebo"},
        {"backend": "isaac_sim"},
        {"backend": ""},
        {"backend": "mujoco", "execution_profile": "isaac-cpu-physics"},
        {"backend": None, "execution_profile": "isaac-cpu-physics"},
        {"backend": "isaac", "execution_profile": "isaac-rtx"},
    ],
)
def test_start_rejects_unsupported_selection_before_any_request(kwargs: dict) -> None:
    from cyberwave.exceptions import CyberwaveValidationError

    api = _sim_api()
    with pytest.raises(CyberwaveValidationError):
        SimulationManager(api).start("env-1", **kwargs)
    api.api_client.param_serialize.assert_not_called()
    api.api_client.call_api.assert_not_called()


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (400, "simulation_backend_unsupported"),
        (403, "simulation_backend_not_enabled"),
        (409, "simulation_backend_incompatible"),
        (409, "simulation_backend_runtime_mismatch"),
        (503, "simulation_backend_capacity_unavailable"),
        (400, "simulation_parameter_unsupported"),
        (400, "simulation_parameter_invalid"),
    ],
)
def test_start_raises_simulation_backend_error_for_stable_codes(status: int, code: str) -> None:
    import json

    from cyberwave.exceptions import SimulationBackendError
    from cyberwave.rest.exceptions import ApiException

    blocking = [
        {"asset_uuid": "a-1", "twin_uuid": "t-1", "name": "Soft gripper",
         "reasons": ["Tendon actuators are not supported."]},
    ]
    exc = ApiException(
        status=status,
        reason="Error",
        body=json.dumps({
            "code": code, "detail": "Isaac refused.", "backend": "isaac",
            "blocking_assets": blocking, "reasons": ["Scene has a deformable body."],
        }),
    )
    mgr = SimulationManager(_api_raising(exc))

    with pytest.raises(SimulationBackendError) as excinfo:
        mgr.start("env-1", backend="isaac")

    err = excinfo.value
    assert isinstance(err, CyberwaveError)
    assert err.code == code
    assert err.backend == "isaac"
    assert err.status_code == status
    assert err.blocking_assets == blocking
    assert err.reasons == ["Scene has a deformable body."]
    msg = str(err)
    assert "Isaac refused." in msg
    assert code in msg
    assert "Soft gripper: Tendon actuators are not supported." in msg
    assert "Scene has a deformable body." in msg


def test_start_accepts_legacy_error_code_field() -> None:
    from cyberwave.exceptions import SimulationBackendError
    from cyberwave.rest.exceptions import ApiException

    exc = ApiException(status=403, reason="Forbidden", body=None)
    exc.data = {"error_code": "simulation_backend_not_enabled", "detail": "Not enabled."}
    with pytest.raises(SimulationBackendError) as excinfo:
        SimulationManager(_api_raising(exc)).start("env-1", backend="isaac")
    assert excinfo.value.code == "simulation_backend_not_enabled"
    assert excinfo.value.blocking_assets == []


def test_unrelated_error_codes_stay_plain_cyberwave_errors() -> None:
    from cyberwave.exceptions import SimulationBackendError
    from cyberwave.rest.exceptions import BadRequestException

    exc = BadRequestException(
        status=400, reason="Bad Request", body='{"code": "other", "detail": "Nope."}'
    )
    with pytest.raises(CyberwaveError) as excinfo:
        SimulationManager(_api_raising(exc)).start("env-1")
    assert not isinstance(excinfo.value, SimulationBackendError)
    assert "Nope." in str(excinfo.value)


def test_simulation_exposes_execution_profile_and_runtime_backend() -> None:
    api = _api_returning({"simulation": {
        "simulation_id": "s", "status": "running", "backend": "isaac",
        "execution_profile": "isaac-cpu-physics", "runtime_backend": "isaac",
    }})
    sim = SimulationManager(api).start("env-1", backend="isaac")
    assert sim.backend == "isaac"
    assert sim.execution_profile == "isaac-cpu-physics"
    assert sim.runtime_backend == "isaac"


def test_simulation_profile_fields_absent_on_older_servers() -> None:
    sim = SimulationManager(_sim_api()).start("env-1")
    assert sim.execution_profile is None
    assert sim.runtime_backend is None


def test_preflight_gets_endpoint_with_query() -> None:
    payload = {"backend": "isaac", "enabled": True, "eligible": False,
               "blocking_assets": [], "reasons": ["x"], "bundle": {"status": "missing", "digest": None}}
    api = _api_returning(payload)
    result = SimulationManager(api).preflight("env-1", execution_profile="isaac-gpu-physics")
    assert result == payload
    call = api.api_client.param_serialize.call_args
    assert call.kwargs["method"] == "GET"
    assert call.kwargs["resource_path"] == "/api/v1/environments/{uuid}/simulation/preflight"
    assert call.kwargs["query_params"] == [
        ("backend", "isaac"), ("execution_profile", "isaac-gpu-physics"),
    ]


def test_preflight_rejects_unknown_backend_locally() -> None:
    from cyberwave.exceptions import CyberwaveValidationError

    api = _sim_api()
    with pytest.raises(CyberwaveValidationError):
        SimulationManager(api).preflight("env-1", backend="gazebo")
    api.api_client.param_serialize.assert_not_called()


def test_isaac_backend_provides_physics_sim_level() -> None:
    from cyberwave.twin.simulation_support import SimLevel, backend_sim_level

    assert backend_sim_level("isaac") == SimLevel.PHYSICS
    assert backend_sim_level("mujoco") == SimLevel.PHYSICS
    assert backend_sim_level("playground") == SimLevel.PLAYGROUND


@pytest.mark.parametrize("backend", ["mujoco_warp", "mjwarp", "MuJoCo_Warp"])
def test_mujoco_warp_backend_provides_physics_sim_level(backend: str) -> None:
    from cyberwave.twin.simulation_support import SimLevel, backend_sim_level

    # Every startable engine is physics-backed; an unmapped one would make
    # cameras and depth refuse while a Warp run is live.
    assert backend_sim_level(backend) == SimLevel.PHYSICS
