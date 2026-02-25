"""Tests for the EdgeManager resource and related client attributes.

Regression test: the Cyberwave client must expose `client.edges` (an
`EdgeManager` instance).  The edge-core startup failed with
  AttributeError: 'Cyberwave' object has no attribute 'edges'
because an older SDK build was installed on the device.  These tests
guard against that regression and exercise the EdgeManager methods.
"""

from unittest.mock import MagicMock, patch

import pytest

from cyberwave import Cyberwave
from cyberwave.resources import EdgeManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client() -> Cyberwave:
    """Create a Cyberwave client without a live API connection."""
    return Cyberwave(base_url="http://localhost:8000", api_key="test_key")


def _make_manager() -> tuple[EdgeManager, MagicMock]:
    """Return an EdgeManager backed by a fully-mocked DefaultApi."""
    mock_api = MagicMock()
    manager = EdgeManager(mock_api)
    return manager, mock_api


def _mock_api_call(mock_api: MagicMock, return_value) -> None:
    """Wire mock_api so that param_serialize / call_api / response_deserialize
    returns *return_value* in the .data attribute."""
    response_data = MagicMock()
    mock_api.api_client.param_serialize.return_value = (
        "GET", "/api/v1/edges", {}, None, [], {}, [], {}
    )
    mock_api.api_client.call_api.return_value = response_data
    deserialized = MagicMock()
    deserialized.data = return_value
    mock_api.api_client.response_deserialize.return_value = deserialized


# ---------------------------------------------------------------------------
# Client attribute tests (regression for the AttributeError)
# ---------------------------------------------------------------------------

def test_client_has_edges_attribute():
    """Cyberwave client must expose a .edges attribute."""
    client = _make_client()
    assert hasattr(client, "edges"), (
        "'Cyberwave' object has no attribute 'edges' — "
        "EdgeManager was not wired up in Cyberwave.__init__"
    )


def test_client_edges_is_edge_manager():
    """client.edges must be an EdgeManager instance."""
    client = _make_client()
    assert isinstance(client.edges, EdgeManager)


def test_client_has_all_expected_managers():
    """Smoke-test all resource managers that startup.py depends on."""
    client = _make_client()
    for attr in ("workspaces", "projects", "environments", "assets", "edges", "twins"):
        assert hasattr(client, attr), f"client.{attr} is missing"


# ---------------------------------------------------------------------------
# EdgeManager.create
# ---------------------------------------------------------------------------

def test_edge_manager_create_returns_edge_schema():
    manager, mock_api = _make_manager()
    fake_edge = MagicMock()
    fake_edge.uuid = "edge-uuid-1"
    fake_edge.fingerprint = "linux-abc123"
    _mock_api_call(mock_api, fake_edge)

    result = manager.create(fingerprint="linux-abc123")

    assert result is fake_edge
    mock_api.api_client.param_serialize.assert_called_once()
    call_kwargs = mock_api.api_client.param_serialize.call_args
    assert call_kwargs.kwargs.get("method") == "POST" or call_kwargs.args[0] == "POST"


def test_edge_manager_create_includes_optional_fields():
    manager, mock_api = _make_manager()
    _mock_api_call(mock_api, MagicMock())

    manager.create(
        fingerprint="linux-abc123",
        name="My Edge",
        workspace_id="ws-uuid",
        metadata={"location": "lab"},
    )

    call_kwargs = mock_api.api_client.param_serialize.call_args
    body = call_kwargs.kwargs.get("body") or call_kwargs.args[3]
    assert body["fingerprint"] == "linux-abc123"
    assert body["name"] == "My Edge"
    assert body["workspace_uuid"] == "ws-uuid"
    assert body["metadata"] == {"location": "lab"}


def test_edge_manager_create_raises_on_api_error():
    manager, mock_api = _make_manager()
    mock_api.api_client.param_serialize.side_effect = Exception("network error")

    with pytest.raises(Exception):
        manager.create(fingerprint="linux-abc123")


# ---------------------------------------------------------------------------
# EdgeManager.list
# ---------------------------------------------------------------------------

def test_edge_manager_list_returns_list():
    manager, mock_api = _make_manager()
    fake_edges = [MagicMock(), MagicMock()]
    _mock_api_call(mock_api, fake_edges)

    result = manager.list()

    assert result is fake_edges


# ---------------------------------------------------------------------------
# EdgeManager.get
# ---------------------------------------------------------------------------

def test_edge_manager_get_returns_edge():
    manager, mock_api = _make_manager()
    fake_edge = MagicMock()
    fake_edge.uuid = "edge-uuid-42"
    _mock_api_call(mock_api, fake_edge)

    result = manager.get("edge-uuid-42")

    assert result is fake_edge
    call_kwargs = mock_api.api_client.param_serialize.call_args
    path_params = call_kwargs.kwargs.get("path_params") or {}
    assert path_params.get("uuid") == "edge-uuid-42"


# ---------------------------------------------------------------------------
# EdgeManager.update
# ---------------------------------------------------------------------------

def test_edge_manager_update_sends_data():
    manager, mock_api = _make_manager()
    fake_edge = MagicMock()
    _mock_api_call(mock_api, fake_edge)

    result = manager.update("edge-uuid-1", {"name": "Updated Name"})

    assert result is fake_edge
    call_kwargs = mock_api.api_client.param_serialize.call_args
    body = call_kwargs.kwargs.get("body") or call_kwargs.args[3]
    assert body == {"name": "Updated Name"}


# ---------------------------------------------------------------------------
# EdgeManager.delete
# ---------------------------------------------------------------------------

def test_edge_manager_delete_calls_api():
    manager, mock_api = _make_manager()
    _mock_api_call(mock_api, {"success": True})

    result = manager.delete("edge-uuid-1")

    assert result == {"success": True}
    call_kwargs = mock_api.api_client.param_serialize.call_args
    method = call_kwargs.kwargs.get("method") or call_kwargs.args[0]
    assert method == "DELETE"


# ---------------------------------------------------------------------------
# EdgeManager import
# ---------------------------------------------------------------------------

def test_edge_manager_importable_from_resources():
    """EdgeManager must be importable from cyberwave.resources."""
    from cyberwave.resources import EdgeManager as EM  # noqa: F401

    assert EM is not None
    assert callable(EM)
