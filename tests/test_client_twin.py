from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from cyberwave import Cyberwave
from cyberwave.config import CyberwaveConfig
from cyberwave.data.api import DataBus
from cyberwave.data.filesystem_backend import FilesystemBackend


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
        client._data_twin_uuid_override = None
        client._data_sensor_name_override = None
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

    env_id, created = client.get_or_create_quickstart_environment()

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

    env_id, created = client.get_or_create_quickstart_environment()

    assert env_id == "new-env-uuid"
    assert created is True
    client.environments.create.assert_called_once()


def test_twin_create_reuses_quickstart_on_second_call():
    """Second client.twin() call must not create another Quickstart Environment."""
    client = _make_client_with_mocked_managers()

    existing_env = SimpleNamespace(
        uuid="existing-env-uuid",
        name="Quickstart Environment",
        slug="ws/envs/quickstart-environment",
    )
    existing_project = SimpleNamespace(
        uuid="proj-uuid", name="Quickstart Project", workspace_uuid="ws-uuid"
    )
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
        client.twin("vendor/robot")

    client.environments.create.assert_not_called()


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


def test_get_or_create_quickstart_environment_delegates_to_helper():
    client = _make_client_with_mocked_managers()
    with patch.object(
        client,
        "_get_or_create_quickstart_env",
        return_value=("env-uuid", False),
    ) as helper:
        env_id, created = client.get_or_create_quickstart_environment()

    assert env_id == "env-uuid"
    assert created is False
    helper.assert_called_once()


def test_quickstart_env_prefers_project_in_active_workspace():
    """Do not attach quickstart env to the first global project from another workspace."""
    client = _make_client_with_mocked_managers()

    other_project = SimpleNamespace(
        uuid="other-proj", name="Edge Project", workspace_uuid="other-ws"
    )
    quickstart_project = SimpleNamespace(
        uuid="proj-uuid", name="Quickstart Project", workspace_uuid="ws-uuid"
    )
    existing_workspace = SimpleNamespace(uuid="ws-uuid", name="Quickstart Workspace")
    existing_env = SimpleNamespace(uuid="existing-env-uuid", name="Quickstart Environment")

    client.config.workspace_id = "ws-uuid"
    client.workspaces.list.return_value = [existing_workspace]
    client.projects.list.return_value = [other_project, quickstart_project]
    client.environments.list.return_value = [existing_env]

    env_id, created = client._get_or_create_quickstart_env()

    assert env_id == "existing-env-uuid"
    assert created is False
    client.environments.list.assert_called_once_with(project_id="proj-uuid")
    client.environments.create.assert_not_called()


def test_quickstart_env_creates_project_when_workspace_has_no_matching_projects():
    """When workspace_id is set, never attach to a global project from another workspace."""
    client = _make_client_with_mocked_managers()

    other_project = SimpleNamespace(
        uuid="other-proj", name="Edge Project", workspace_uuid="other-ws"
    )
    new_project = SimpleNamespace(uuid="new-proj", name="Quickstart Project")
    new_env = SimpleNamespace(uuid="new-env-uuid", name="Quickstart Environment")

    client.config.workspace_id = "ws-uuid"
    client.projects.list.return_value = [other_project]
    client.projects.create.return_value = new_project
    client.environments.list.return_value = []
    client.environments.create.return_value = new_env

    env_id, created = client.get_or_create_quickstart_environment()

    assert env_id == "new-env-uuid"
    assert created is True
    client.projects.create.assert_called_once_with(
        name="Quickstart Project",
        workspace_id="ws-uuid",
    )
    client.environments.create.assert_called_once_with(
        name="Quickstart Environment",
        project_id="new-proj",
    )


def test_quickstart_does_not_overwrite_preset_workspace_id():
    client = _make_client_with_mocked_managers()

    client.config.workspace_id = "preset-ws"
    client.projects.list.return_value = []
    new_project = SimpleNamespace(uuid="proj-uuid", name="Quickstart Project")
    new_env = SimpleNamespace(uuid="env-uuid", name="Quickstart Environment")
    client.projects.create.return_value = new_project
    client.environments.list.return_value = []
    client.environments.create.return_value = new_env

    client.get_or_create_quickstart_environment()

    assert client.config.workspace_id == "preset-ws"


def test_use_data_bus_for_binds_and_rebinds(tmp_path):
    client = _make_client_with_mocked_managers()
    client._data_backend = FilesystemBackend(base_dir=tmp_path)

    twin_a = "11111111-1111-4111-8111-111111111111"
    twin_b = "22222222-2222-4222-8222-222222222222"

    client.use_data_bus_for(twin_a)
    bus_a = client.data
    assert isinstance(bus_a, DataBus)
    assert bus_a.twin_uuid == twin_a

    client.use_data_bus_for(twin_b)
    bus_b = client.data
    assert bus_a is not bus_b
    assert bus_b.twin_uuid == twin_b


def test_use_data_bus_for_is_lazy_and_idempotent(tmp_path):
    """Repeated calls with the same twin must not rebuild the bus.

    Preserves per-channel ``HeaderTemplate.seq`` counters across
    multi-trigger hook invocations that re-seed on every frame.
    """
    client = _make_client_with_mocked_managers()
    client._data_backend = FilesystemBackend(base_dir=tmp_path)

    twin = "11111111-1111-4111-8111-111111111111"

    client.use_data_bus_for(twin)
    assert client._data_bus is None, "pinning must not eagerly create a bus"

    bus = client.data
    client.use_data_bus_for(twin)
    assert client.data is bus, "same-twin re-seed must reuse the existing bus"


def test_use_data_bus_for_overrides_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("CYBERWAVE_TWIN_UUID", "env-twin-uuid")
    client = _make_client_with_mocked_managers()
    client._data_backend = FilesystemBackend(base_dir=tmp_path)

    pinned = "11111111-1111-4111-8111-111111111111"
    client.use_data_bus_for(pinned)
    assert client.data.twin_uuid == pinned
