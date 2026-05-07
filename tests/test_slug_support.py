"""Tests for unified slug support across the SDK.

These tests verify that:
- Entity managers accept slugs as identifiers (not just UUIDs)
- Console messages use slug-based URLs when available
- Slug check / availability works
- Backward compatibility with UUID identifiers is preserved
"""

from types import SimpleNamespace
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from cyberwave import Cyberwave
from cyberwave.config import CyberwaveConfig
from cyberwave.resources import _is_uuid, _looks_like_slug


# ── Helper utilities ──────────────────────────────────────────────────


def _make_client_with_mocked_managers():
    """Return a Cyberwave client with REST setup and managers replaced by mocks."""
    with patch.object(Cyberwave, "_setup_rest_client", lambda self: None), patch.object(
        Cyberwave, "_wrap_api_methods", lambda self: None
    ):
        client = Cyberwave.__new__(Cyberwave)
        client.config = CyberwaveConfig(
            base_url="http://localhost:8000",
            api_key="test_key",
        )
        client.api = MagicMock()
        client._mqtt_client = None
        client._data_bus = None
        from cyberwave.workers.hooks import HookRegistry
        from cyberwave.models.manager import ModelManager

        client._hook_registry = HookRegistry()
        client.models = ModelManager()

    client.workspaces = MagicMock()
    client.projects = MagicMock()
    client.environments = MagicMock()
    client.twins = MagicMock()
    client.assets = MagicMock()
    client.workflows = MagicMock()
    client.workflow_runs = MagicMock()
    return client


# ── _is_uuid / _looks_like_slug helpers ──────────────────────────────


def test_is_uuid_with_valid_uuid():
    assert _is_uuid("550e8400-e29b-41d4-a716-446655440000") is True


def test_is_uuid_with_slug():
    assert _is_uuid("acme/catalog/my-robot") is False


def test_is_uuid_with_empty_string():
    assert _is_uuid("") is False


def test_looks_like_slug_with_slug():
    assert _looks_like_slug("acme/catalog/my-robot") is True


def test_looks_like_slug_with_uuid():
    assert _looks_like_slug("550e8400-e29b-41d4-a716-446655440000") is False


def test_looks_like_slug_with_simple_alias():
    assert _looks_like_slug("camera") is False


# ── Twin slug property ───────────────────────────────────────────────


def test_twin_slug_property_from_schema():
    """Twin.slug reads from the underlying data."""
    from cyberwave.twin import Twin

    data = SimpleNamespace(
        uuid="twin-uuid",
        name="my-twin",
        slug="acme/twins/my-twin",
        asset_uuid="asset-uuid",
        environment_uuid="env-uuid",
    )
    client = MagicMock()
    twin = Twin(client, data)
    assert twin.slug == "acme/twins/my-twin"


def test_twin_slug_property_from_dict():
    """Twin.slug reads from dict data."""
    from cyberwave.twin import Twin

    data = {
        "uuid": "twin-uuid",
        "name": "my-twin",
        "slug": "acme/twins/my-twin",
        "asset_uuid": "asset-uuid",
        "environment_uuid": "env-uuid",
    }
    client = MagicMock()
    twin = Twin(client, data)
    assert twin.slug == "acme/twins/my-twin"


def test_twin_slug_property_missing():
    """Twin.slug returns '' when no slug is available."""
    from cyberwave.twin import Twin

    data = SimpleNamespace(uuid="twin-uuid", name="my-twin", asset_uuid="a", environment_uuid="e")
    client = MagicMock()
    twin = Twin(client, data)
    assert twin.slug == ""


# ── Workflow slug property ───────────────────────────────────────────


def test_workflow_slug_property():
    """Workflow.slug reads from the underlying data."""
    from cyberwave.workflows import Workflow

    data = SimpleNamespace(
        uuid="wf-uuid",
        name="pick-and-place",
        slug="acme/workflows/pick-and-place",
        is_active=True,
    )
    client = MagicMock()
    wf = Workflow(client, data)
    assert wf.slug == "acme/workflows/pick-and-place"


# ── Console URL uses slug ────────────────────────────────────────────


def test_build_environment_url_with_slug():
    """_build_environment_url returns slug-based URL when available."""
    client = _make_client_with_mocked_managers()
    env = SimpleNamespace(uuid="env-uuid", slug="acme/envs/production-floor")
    client.environments.get.return_value = env
    url = client._build_environment_url("env-uuid")
    assert url == "https://cyberwave.com/acme/envs/production-floor"


def test_build_environment_url_fallback_to_uuid():
    """_build_environment_url falls back to UUID when slug is not available."""
    client = _make_client_with_mocked_managers()
    env = SimpleNamespace(uuid="env-uuid", slug=None)
    client.environments.get.return_value = env
    url = client._build_environment_url("env-uuid")
    assert url == "https://cyberwave.com/environments/env-uuid"


def test_build_twin_url_with_slug():
    """_build_twin_url returns slug-based URL when available."""
    client = _make_client_with_mocked_managers()
    twin_data = SimpleNamespace(uuid="twin-uuid", slug="acme/twins/my-robot")
    url = client._build_twin_url(twin_data)
    assert url == "https://cyberwave.com/acme/twins/my-robot"


def test_build_twin_url_fallback_to_uuid():
    """_build_twin_url falls back to UUID when slug is not available."""
    client = _make_client_with_mocked_managers()
    twin_data = SimpleNamespace(uuid="twin-uuid", slug=None)
    url = client._build_twin_url(twin_data)
    assert url == "https://cyberwave.com/twins/twin-uuid"


def test_build_twin_url_with_dict():
    """_build_twin_url works with dict data."""
    client = _make_client_with_mocked_managers()
    twin_data = {"uuid": "twin-uuid", "slug": "acme/twins/arm-1"}
    url = client._build_twin_url(twin_data)
    assert url == "https://cyberwave.com/acme/twins/arm-1"


# ── Quickstart env message uses slug URL ─────────────────────────────


def test_quickstart_env_message_uses_slug_url(capsys):
    """Quickstart env message prints slug-based URL when available."""
    client = _make_client_with_mocked_managers()

    existing_env = SimpleNamespace(
        uuid="env-uuid",
        name="Quickstart Environment",
        slug="acme/envs/quickstart-environment",
    )
    existing_project = SimpleNamespace(uuid="proj-uuid", name="Quickstart Project")
    existing_workspace = SimpleNamespace(uuid="ws-uuid", name="Quickstart Workspace")
    existing_twin = SimpleNamespace(
        uuid="twin-uuid", asset_uuid="asset-uuid", name="twin"
    )
    asset = SimpleNamespace(uuid="asset-uuid", registry_id="vendor/robot")

    client.workspaces.list.return_value = [existing_workspace]
    client.projects.list.return_value = [existing_project]
    client.environments.list.return_value = [existing_env]
    client.environments.get.return_value = existing_env
    client.twins.list.return_value = [existing_twin]
    client.assets.get_by_registry_id.return_value = asset

    with patch("cyberwave.client.create_twin", return_value=MagicMock()):
        client.twin("vendor/robot")

    captured = capsys.readouterr()
    assert "acme/envs/quickstart-environment" in captured.out


# ── Resolve environment by slug ──────────────────────────────────────


def test_resolve_environment_id_with_slug():
    """_resolve_environment_id resolves slug to UUID."""
    client = _make_client_with_mocked_managers()
    env = SimpleNamespace(uuid="env-uuid-resolved", slug="acme/envs/my-env")
    client.environments.get_by_slug.return_value = env
    result = client._resolve_environment_id("acme/envs/my-env")
    assert result == "env-uuid-resolved"


def test_resolve_environment_id_with_uuid():
    """_resolve_environment_id returns UUID unchanged."""
    client = _make_client_with_mocked_managers()
    result = client._resolve_environment_id("550e8400-e29b-41d4-a716-446655440000")
    assert result == "550e8400-e29b-41d4-a716-446655440000"


# ── client.twin() with slug identifiers ──────────────────────────────


def test_twin_method_accepts_twin_slug():
    """client.twin(twin_id='acme/twins/my-robot') resolves via slug."""
    client = _make_client_with_mocked_managers()
    twin_data = SimpleNamespace(
        uuid="twin-uuid",
        name="my-robot",
        slug="acme/twins/my-robot",
        asset_uuid="asset-uuid",
        environment_uuid="env-uuid",
    )
    client.twins.get_raw.return_value = twin_data

    with patch("cyberwave.client.create_twin", return_value=MagicMock()) as mock_create:
        client.twin(twin_id="acme/twins/my-robot")

    client.twins.get_raw.assert_called_once_with("acme/twins/my-robot")
    mock_create.assert_called_once()


def test_twin_method_accepts_environment_slug():
    """client.twin(environment_id='acme/envs/prod') resolves env slug to UUID."""
    client = _make_client_with_mocked_managers()
    client.config.environment_id = None

    env = SimpleNamespace(uuid="resolved-env-uuid", slug="acme/envs/prod")
    client.environments.get_by_slug.return_value = env

    asset = SimpleNamespace(uuid="asset-uuid", registry_id="vendor/robot")
    client.assets.get_by_registry_id.return_value = asset

    twin_data = SimpleNamespace(
        uuid="twin-uuid", asset_uuid="asset-uuid", name="twin"
    )
    client.twins.list.return_value = [twin_data]

    with patch("cyberwave.client.create_twin", return_value=MagicMock()):
        client.twin("vendor/robot", environment_id="acme/envs/prod")

    client.twins.list.assert_called_once_with(environment_id="resolved-env-uuid")


# ── Workflow get by slug ─────────────────────────────────────────────


def test_workflow_get_by_slug_full_unified():
    """WorkflowManager.get_by_slug uses /by-slug for full unified slugs."""
    from cyberwave.workflows import WorkflowManager, _get_workflow_by_slug

    client = MagicMock()
    wm = WorkflowManager(client)

    with patch("cyberwave.workflows._get_workflow_by_slug") as mock_get:
        mock_get.return_value = SimpleNamespace(
            uuid="wf-uuid", name="pick", slug="acme/workflows/pick"
        )
        result = wm.get_by_slug("acme/workflows/pick")

    assert result is not None
    assert result.uuid == "wf-uuid"


def test_workflow_get_by_slug_legacy_workspace_scoped():
    """WorkflowManager.get_by_slug supports legacy (workspace_id, slug) form."""
    from cyberwave.workflows import WorkflowManager

    client = MagicMock()
    wm = WorkflowManager(client)

    with patch("cyberwave.workflows._list_workflows") as mock_list:
        mock_list.return_value = [
            SimpleNamespace(uuid="wf-uuid", name="pick", slug="pick")
        ]
        result = wm.get_by_slug("pick", workspace_id="ws-uuid")

    assert result is not None
    mock_list.assert_called_once_with(client, workspace_id="ws-uuid", slug="pick")


# ── Workflow trigger by slug ─────────────────────────────────────────


def test_workflow_trigger_by_slug():
    """WorkflowManager.trigger resolves slug to UUID."""
    from cyberwave.workflows import WorkflowManager, Workflow

    client = MagicMock()
    wm = WorkflowManager(client)

    mock_wf = MagicMock(spec=Workflow)
    mock_wf.uuid = "wf-uuid-resolved"

    with patch.object(wm, "get", return_value=mock_wf):
        with patch("cyberwave.workflows._trigger_workflow") as mock_trigger:
            mock_trigger.return_value = SimpleNamespace(
                uuid="run-uuid", status="running"
            )
            run = wm.trigger("acme/workflows/pick", inputs={"speed": 1.0})

    mock_trigger.assert_called_once_with(client, "wf-uuid-resolved", {"speed": 1.0})
    assert run.uuid == "run-uuid"


def test_workflow_trigger_by_uuid_no_resolution():
    """WorkflowManager.trigger uses UUID directly when given a valid UUID."""
    from cyberwave.workflows import WorkflowManager

    client = MagicMock()
    wm = WorkflowManager(client)

    with patch("cyberwave.workflows._trigger_workflow") as mock_trigger:
        mock_trigger.return_value = SimpleNamespace(
            uuid="run-uuid", status="running"
        )
        run = wm.trigger("550e8400-e29b-41d4-a716-446655440000")

    mock_trigger.assert_called_once_with(
        client, "550e8400-e29b-41d4-a716-446655440000", None
    )
