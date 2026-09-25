"""Tests for EnvironmentManager.export_mujoco_scene.

The export goes through the descriptor endpoint rather than the direct
``mujoco-scene.zip`` route: the archive is built by a worker and served from
object storage, so it is not bounded by the API's response size limit.
"""

import json
from unittest.mock import MagicMock

import pytest

from _rest_probe import rest_module_is_real
from cyberwave.exceptions import CyberwaveAPIError
from cyberwave.resources import EnvironmentManager


def _mock_api_spec() -> list[str] | None:
    """Restrict the stub to names the generated client actually exposes.

    A bare ``MagicMock`` materialises any attribute on access, so these tests
    would pass against any spelling of the operation — and ``cyberwave/rest`` is
    generated at release time rather than committed, so nothing else in this
    file could tell the difference. ``api_client`` is set in ``DefaultApi``'s
    ``__init__`` rather than declared on the class, so it has to be named here
    for the spec to admit it.

    Returns ``None`` (no spec, permissive stub) when the generated client is
    absent, so a checkout that has not run the generator still exercises the
    behaviour below. ``test_generated_client_operation_names`` is the check that
    does not degrade that way.
    """
    if not rest_module_is_real():
        return None
    from cyberwave.rest import DefaultApi

    return [*dir(DefaultApi), "api_client"]


def _make_manager(host: str = "https://api.cyberwave.com"):
    mock_api = MagicMock(spec=_mock_api_spec())
    mock_api.api_client = MagicMock()
    mock_api.api_client.configuration.host = host
    mock_api.api_client.configuration.auth_settings.return_value = {
        "CustomTokenAuthentication": {
            "type": "api_key",
            "in": "header",
            "key": "Authorization",
            "value": "Token secret",
        }
    }
    return EnvironmentManager(mock_api), mock_api


def _descriptor(mock_api, *payloads):
    responses = []
    for payload in payloads:
        response = MagicMock()
        response.raw_data = json.dumps(payload).encode("utf-8")
        responses.append(response)
    mock_api.src_app_api_environments_exports_get_environment_mujoco_scene_with_http_info.side_effect = responses


def _download(mock_api, *, status: int = 200, data: bytes = b"PK\x03\x04zip"):
    download = MagicMock()
    download.status = status
    download.read.return_value = data
    mock_api.api_client.rest_client.request.return_value = download
    return download


def test_export_withholds_token_from_absolute_storage_url():
    manager, mock_api = _make_manager()
    _descriptor(
        mock_api,
        {
            "status": "completed",
            "url": "https://storage.googleapis.com/bucket/env.zip?signature=abc",
        },
    )
    _download(mock_api)

    result = manager.export_mujoco_scene("env-1")

    assert result == b"PK\x03\x04zip"
    method, url = mock_api.api_client.rest_client.request.call_args.args
    assert method == "GET"
    assert url == "https://storage.googleapis.com/bucket/env.zip?signature=abc"
    assert mock_api.api_client.rest_client.request.call_args.kwargs["headers"] == {}


def test_export_sends_token_for_relative_url_on_the_api_host():
    manager, mock_api = _make_manager()
    _descriptor(mock_api, {"status": "completed", "url": "/media/environments/env.zip"})
    _download(mock_api)

    manager.export_mujoco_scene("env-1")

    _, url = mock_api.api_client.rest_client.request.call_args.args
    assert url == "https://api.cyberwave.com/media/environments/env.zip"
    assert mock_api.api_client.rest_client.request.call_args.kwargs["headers"] == {
        "Authorization": "Token secret"
    }


def test_export_polls_until_the_url_appears(monkeypatch):
    manager, mock_api = _make_manager()
    _descriptor(
        mock_api,
        {"status": "pending", "url": ""},
        {"status": "completed", "url": "https://storage.googleapis.com/bucket/env.zip"},
    )
    _download(mock_api)
    sleeps: list[float] = []
    monkeypatch.setattr("cyberwave.resources.time.sleep", sleeps.append)

    manager.export_mujoco_scene("env-1", poll_interval=0.25)

    assert sleeps == [0.25]
    assert (
        mock_api.src_app_api_environments_exports_get_environment_mujoco_scene_with_http_info.call_count
        == 2
    )


def test_export_writes_to_output_path(tmp_path):
    manager, mock_api = _make_manager()
    _descriptor(mock_api, {"status": "completed", "url": "https://storage.googleapis.com/b/env.zip"})
    _download(mock_api, data=b"zip-bytes")
    target = tmp_path / "scene.zip"

    result = manager.export_mujoco_scene("env-1", str(target))

    assert result == b"zip-bytes"
    assert target.read_bytes() == b"zip-bytes"


def test_export_raises_when_the_build_failed():
    manager, mock_api = _make_manager()
    _descriptor(mock_api, {"status": "failed", "url": ""})

    with pytest.raises(CyberwaveAPIError, match="MuJoCo scene export failed"):
        manager.export_mujoco_scene("env-1")


def test_export_raises_when_the_wait_budget_runs_out(monkeypatch):
    manager, mock_api = _make_manager()
    mock_api.src_app_api_environments_exports_get_environment_mujoco_scene_with_http_info.side_effect = None
    pending = MagicMock()
    pending.raw_data = json.dumps({"status": "pending", "url": ""}).encode("utf-8")
    mock_api.src_app_api_environments_exports_get_environment_mujoco_scene_with_http_info.return_value = pending
    monkeypatch.setattr("cyberwave.resources.time.sleep", lambda _s: None)

    with pytest.raises(CyberwaveAPIError, match="Timed out"):
        manager.export_mujoco_scene("env-1", timeout=0.0)


def test_export_raises_when_the_download_fails():
    manager, mock_api = _make_manager()
    _descriptor(mock_api, {"status": "completed", "url": "https://storage.googleapis.com/b/env.zip"})
    _download(mock_api, status=403)

    with pytest.raises(CyberwaveAPIError, match="HTTP 403"):
        manager.export_mujoco_scene("env-1")
