"""Tests for EnvironmentManager.export_usd_scene.

USD comes from the Isaac Sim scene bundle: the export reads the bundle state
from the preflight, requests a build only when there is none in flight, and
downloads the tar from the archive endpoint once the bundle is ready.
"""

from unittest.mock import MagicMock

import pytest

from cyberwave.exceptions import CyberwaveAPIError, SimulationBackendError
from cyberwave.managers.simulations import SimulationManager
from cyberwave.resources import EnvironmentManager

DIGEST = "sha256:" + "a" * 64
ARCHIVE_URL = (
    "https://api.cyberwave.com/api/v1/environments/env-1/simulation/"
    "scene-bundles/sha256%3A" + "a" * 64 + "/archive"
)


def _make_manager(*preflight_bundles, requested=None):
    mock_api = MagicMock()
    mock_api.api_client.configuration.host = "https://api.cyberwave.com"
    mock_api.api_client.configuration.auth_settings.return_value = {
        "CustomTokenAuthentication": {
            "type": "api_key",
            "in": "header",
            "key": "Authorization",
            "value": "Token secret",
        }
    }
    manager = EnvironmentManager(mock_api)
    simulations = MagicMock(spec=SimulationManager)
    simulations.preflight.side_effect = [
        {"enabled": True, "bundle": bundle} for bundle in preflight_bundles
    ]
    simulations.request_scene_bundle.return_value = requested or {
        "status": "pending",
        "digest": None,
        "bundle_id": "b-1",
    }
    manager._simulations = simulations
    return manager, mock_api, simulations


def _download(mock_api, *, status: int = 200, data: bytes = b"tar-bytes"):
    download = MagicMock()
    download.status = status
    download.read.return_value = data
    mock_api.api_client.rest_client.request.return_value = download
    return download


def test_downloads_a_ready_bundle_without_requesting_a_build():
    manager, mock_api, simulations = _make_manager({"status": "ready", "digest": DIGEST})
    _download(mock_api)

    result = manager.export_usd_scene("env-1")

    assert result == b"tar-bytes"
    simulations.preflight.assert_called_once_with("env-1", backend="isaac")
    simulations.request_scene_bundle.assert_not_called()
    method, url = mock_api.api_client.rest_client.request.call_args.args
    assert (method, url) == ("GET", ARCHIVE_URL)
    # The archive is served by the API itself, so it gets the API credentials.
    assert mock_api.api_client.rest_client.request.call_args.kwargs["headers"] == {
        "Authorization": "Token secret"
    }


def test_requests_a_missing_bundle_and_polls_until_ready(monkeypatch, tmp_path):
    manager, mock_api, simulations = _make_manager(
        {"status": "missing", "digest": None},
        {"status": "pending", "digest": None},
        {"status": "ready", "digest": DIGEST},
    )
    _download(mock_api)
    sleeps: list[float] = []
    monkeypatch.setattr("cyberwave.resources.time.sleep", sleeps.append)
    target = tmp_path / "scene.tar"

    result = manager.export_usd_scene("env-1", str(target), poll_interval=0.5)

    assert result == b"tar-bytes"
    assert target.read_bytes() == b"tar-bytes"
    # One request: a build in flight is joined, not re-requested.
    simulations.request_scene_bundle.assert_called_once_with("env-1")
    assert sleeps == [0.5, 0.5]


def test_downloads_straight_away_when_the_request_finds_a_ready_bundle():
    manager, mock_api, simulations = _make_manager(
        {"status": "missing", "digest": None},
        requested={"status": "ready", "digest": DIGEST, "bundle_id": "b-1"},
    )
    _download(mock_api)

    manager.export_usd_scene("env-1")

    _, url = mock_api.api_client.rest_client.request.call_args.args
    assert url == ARCHIVE_URL
    assert simulations.preflight.call_count == 1


def test_a_refused_scene_keeps_the_simulation_backend_error():
    manager, _mock_api, simulations = _make_manager({"status": "missing", "digest": None})
    refusal = SimulationBackendError(
        "Failed to request the scene bundle: mesh missing",
        code="simulation_backend_incompatible",
        reasons=["mesh_not_resolved: mesh base.stl not found"],
        blocking_assets=[{"name": "Arm"}],
        status_code=409,
    )
    simulations.request_scene_bundle.side_effect = refusal

    with pytest.raises(SimulationBackendError) as excinfo:
        manager.export_usd_scene("env-1")

    assert excinfo.value is refusal
    assert excinfo.value.code == "simulation_backend_incompatible"


def test_reports_why_the_build_failed_after_its_request(monkeypatch):
    manager, _mock_api, simulations = _make_manager(
        {"status": "failed", "digest": None},
        {
            "status": "failed",
            "digest": None,
            "error": {"code": "physics_parameters_invalid", "detail": "substeps must be >= 1"},
        },
    )
    monkeypatch.setattr("cyberwave.resources.time.sleep", lambda _s: None)

    with pytest.raises(CyberwaveAPIError, match="substeps must be >= 1") as excinfo:
        manager.export_usd_scene("env-1")

    # The stale failure is retried once; the retry's failure is final.
    simulations.request_scene_bundle.assert_called_once_with("env-1")
    # The environment is named once, by the export's own prefix.
    assert excinfo.value.message == (
        "Failed to export USD scene for environment env-1: substeps must be >= 1"
    )


def test_a_recent_failure_returned_by_the_request_raises_without_polling(monkeypatch):
    manager, _mock_api, simulations = _make_manager(
        {
            "status": "failed",
            "digest": None,
            "error": {"code": "physics_parameters_invalid", "detail": "substeps must be >= 1"},
        },
        requested={"status": "failed", "digest": None, "bundle_id": "b-1"},
    )
    sleeps = []
    monkeypatch.setattr("cyberwave.resources.time.sleep", sleeps.append)

    with pytest.raises(CyberwaveAPIError, match="substeps must be >= 1.*recent build failed"):
        manager.export_usd_scene("env-1")

    assert sleeps == []
    simulations.preflight.assert_called_once()


def test_raises_when_the_wait_budget_runs_out(monkeypatch):
    manager, _mock_api, simulations = _make_manager({"status": "pending", "digest": None})
    monkeypatch.setattr("cyberwave.resources.time.sleep", lambda _s: None)

    with pytest.raises(CyberwaveAPIError, match="Timed out"):
        manager.export_usd_scene("env-1", timeout=0.0)

    simulations.request_scene_bundle.assert_not_called()


def test_raises_when_the_archive_download_fails():
    manager, mock_api, _simulations = _make_manager({"status": "ready", "digest": DIGEST})
    _download(mock_api, status=404)

    with pytest.raises(CyberwaveAPIError, match="HTTP 404"):
        manager.export_usd_scene("env-1")


def test_request_scene_bundle_posts_to_the_scene_bundles_endpoint():
    api = MagicMock()
    api.api_client.param_serialize.return_value = ()
    api.api_client.response_deserialize.return_value.data = {
        "status": "pending",
        "digest": None,
        "bundle_id": "b-1",
    }

    result = SimulationManager(api).request_scene_bundle("env-1")

    assert result == {"status": "pending", "digest": None, "bundle_id": "b-1"}
    call = api.api_client.param_serialize.call_args
    assert call.kwargs["method"] == "POST"
    assert call.kwargs["resource_path"] == (
        "/api/v1/environments/{uuid}/simulation/scene-bundles"
    )
    assert call.kwargs["path_params"] == {"uuid": "env-1"}


def test_request_scene_bundle_raises_the_servers_refusal():
    api = MagicMock()
    api.api_client.param_serialize.return_value = ()
    error = Exception("409 Conflict")
    error.status = 409
    error.data = {
        "code": "simulation_backend_incompatible",
        "detail": "The environment cannot be exported to an Isaac scene bundle.",
        "backend": "isaac",
        "reasons": ["mesh_not_resolved: mesh base.stl not found"],
        "blocking_assets": [{"name": "Arm", "reasons": ["mesh"]}],
    }
    api.api_client.response_deserialize.side_effect = error

    with pytest.raises(SimulationBackendError) as excinfo:
        SimulationManager(api).request_scene_bundle("env-1")

    assert excinfo.value.code == "simulation_backend_incompatible"
    assert excinfo.value.status_code == 409
    assert excinfo.value.reasons == ["mesh_not_resolved: mesh base.stl not found"]
