from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from cyberwave import Cyberwave
from cyberwave.config import CyberwaveConfig


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
    return client


def test_client_twin_preserves_list_errors_instead_of_masking_them():
    client = _make_client_with_mocked_managers()
    client.config.environment_id = "env-uuid"
    client.assets.get_by_registry_id.return_value = SimpleNamespace(
        uuid="asset-uuid",
        registry_id="the-robot-studio/so101",
    )
    expected_error = RuntimeError("list twins failed")
    client.twins.list = lambda environment_id=None: (_ for _ in ()).throw(expected_error)

    with patch("cyberwave.client.create_twin") as mock_create_twin:
        with pytest.raises(RuntimeError, match="list twins failed"):
            client.twin("the-robot-studio/so101")

    mock_create_twin.assert_not_called()


def test_quickstart_env_reuses_existing_environment():
    """When an env named 'Quickstart Environment' already exists, reuse it."""
    client = _make_client_with_mocked_managers()

    existing_env = SimpleNamespace(uuid="existing-env-uuid", name="Quickstart Environment")
    existing_project = SimpleNamespace(uuid="proj-uuid", name="Quickstart Project")
    existing_workspace = SimpleNamespace(uuid="ws-uuid", name="Quickstart Workspace")

    client.workspaces.list.return_value = [existing_workspace]
    client.projects.list.return_value = [existing_project]
    client.environments.list.return_value = [existing_env]

    env_id, created = client._get_or_create_quickstart_env()

    assert env_id == "existing-env-uuid"
    assert created is False
    client.environments.create.assert_not_called()


def test_quickstart_env_creates_new_environment_when_none_exists():
    """When no 'Quickstart Environment' exists under the project, create one."""
    client = _make_client_with_mocked_managers()

    existing_project = SimpleNamespace(uuid="proj-uuid", name="Quickstart Project")
    existing_workspace = SimpleNamespace(uuid="ws-uuid", name="Quickstart Workspace")
    new_env = SimpleNamespace(uuid="new-env-uuid", name="Quickstart Environment")

    client.workspaces.list.return_value = [existing_workspace]
    client.projects.list.return_value = [existing_project]
    client.environments.list.return_value = []
    client.environments.create.return_value = new_env

    env_id, created = client._get_or_create_quickstart_env()

    assert env_id == "new-env-uuid"
    assert created is True
    client.environments.create.assert_called_once()


def test_quickstart_env_logs_reused_message(capsys):
    """twin() without environment_id prints a 'reusing' message when env exists."""
    client = _make_client_with_mocked_managers()

    existing_env = SimpleNamespace(
        uuid="existing-env-uuid",
        name="Quickstart Environment",
        slug="ws/envs/quickstart-environment",
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
    assert "reusing existing" in captured.out.lower()
    assert "ws/envs/quickstart-environment" in captured.out


def test_quickstart_env_logs_created_message(capsys):
    """twin() without environment_id prints a 'created' message when new env is made."""
    client = _make_client_with_mocked_managers()

    existing_project = SimpleNamespace(uuid="proj-uuid", name="Quickstart Project")
    existing_workspace = SimpleNamespace(uuid="ws-uuid", name="Quickstart Workspace")
    new_env = SimpleNamespace(
        uuid="new-env-uuid",
        name="Quickstart Environment",
        slug="ws/envs/quickstart-environment",
    )
    new_twin = SimpleNamespace(uuid="twin-uuid", asset_uuid="asset-uuid", name="twin")
    asset = SimpleNamespace(uuid="asset-uuid", registry_id="vendor/robot")

    client.workspaces.list.return_value = [existing_workspace]
    client.projects.list.return_value = [existing_project]
    client.environments.list.return_value = []
    client.environments.create.return_value = new_env
    client.environments.get.return_value = new_env
    client.twins.list.return_value = []
    client.twins.create.return_value = new_twin
    client.assets.get_by_registry_id.return_value = asset

    with patch("cyberwave.client.create_twin", return_value=MagicMock()):
        client.twin("vendor/robot")

    captured = capsys.readouterr()
    assert "created a new" in captured.out.lower()
    assert "ws/envs/quickstart-environment" in captured.out


def test_quickstart_env_no_log_when_env_id_specified(capsys):
    """twin() with explicit environment_id does NOT print any quickstart message."""
    client = _make_client_with_mocked_managers()

    existing_twin = SimpleNamespace(
        uuid="twin-uuid", asset_uuid="asset-uuid", name="twin"
    )
    asset = SimpleNamespace(uuid="asset-uuid", registry_id="vendor/robot")

    client.twins.list.return_value = [existing_twin]
    client.assets.get_by_registry_id.return_value = asset

    with patch("cyberwave.client.create_twin", return_value=MagicMock()):
        client.twin("vendor/robot", environment_id="explicit-env-uuid")

    captured = capsys.readouterr()
    assert "[Cyberwave]" not in captured.out
