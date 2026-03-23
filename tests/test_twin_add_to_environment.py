from types import SimpleNamespace
from unittest.mock import MagicMock

from cyberwave.twin import Twin


def _make_twin(client=None):
    source_twin_data = SimpleNamespace(
        uuid="twin-old",
        name="Robot",
        description="Original twin",
        asset_uuid="asset-uuid",
        environment_uuid="env-old",
        position_x=1.0,
        position_y=2.0,
        position_z=3.0,
        rotation_w=1.0,
        rotation_x=0.0,
        rotation_y=0.0,
        rotation_z=0.0,
        scale_x=1.0,
        scale_y=1.0,
        scale_z=1.0,
        metadata={"nested": {"k": "v"}},
        kinematics_override={"limits": {"joint1": 1.0}},
        joint_calibration={"joint1": {"id": "1", "range_min": 0, "range_max": 10}},
        fixed_base=False,
    )
    if client is None:
        client = SimpleNamespace(
            twins=MagicMock(),
            environments=MagicMock(),
            _api_client=MagicMock(),
        )
    return Twin(client, source_twin_data)


def test_add_to_environment_recreates_and_rebinds_twin():
    twins_manager = MagicMock()
    twins_manager.create.return_value = SimpleNamespace(
        uuid="twin-new",
        name="Robot",
        asset_uuid="asset-uuid",
        environment_uuid="env-new",
    )
    twins_manager.list.return_value = [SimpleNamespace(uuid="another-twin")]
    client = SimpleNamespace(
        twins=twins_manager,
        environments=MagicMock(),
        _api_client=MagicMock(),
    )
    twin = _make_twin(client=client)
    original_metadata = twin._data.metadata

    result = twin.add_to_environment("env-new")

    assert result is twin
    assert twin.uuid == "twin-new"
    assert twin.environment_id == "env-new"
    twins_manager.delete.assert_called_once_with("twin-old")
    twins_manager.list.assert_called_once_with(environment_id="env-old")

    create_kwargs = twins_manager.create.call_args.kwargs
    assert create_kwargs["asset_id"] == "asset-uuid"
    assert create_kwargs["environment_id"] == "env-new"
    assert create_kwargs["name"] == "Robot"
    assert create_kwargs["metadata"] == {"nested": {"k": "v"}}
    assert create_kwargs["metadata"] is not original_metadata


def test_add_to_environment_deletes_source_environment_when_empty():
    twins_manager = MagicMock()
    twins_manager.create.return_value = SimpleNamespace(
        uuid="twin-new",
        name="Robot",
        asset_uuid="asset-uuid",
        environment_uuid="env-new",
    )
    twins_manager.list.return_value = []

    environments_manager = MagicMock()
    environments_manager.get.return_value = SimpleNamespace(project_uuid="project-uuid")

    client = SimpleNamespace(
        twins=twins_manager,
        environments=environments_manager,
        _api_client=MagicMock(),
    )
    twin = _make_twin(client=client)

    twin.add_to_environment("env-new")

    environments_manager.delete.assert_called_once_with("env-old", "project-uuid")


def test_add_to_environment_uses_standalone_delete_fallback_when_project_missing():
    twins_manager = MagicMock()
    twins_manager.create.return_value = SimpleNamespace(
        uuid="twin-new",
        name="Robot",
        asset_uuid="asset-uuid",
        environment_uuid="env-new",
    )
    twins_manager.list.return_value = []

    environments_manager = MagicMock()
    environments_manager.get.return_value = SimpleNamespace(project_uuid=None)

    api_client = MagicMock()
    api_client.param_serialize.return_value = ("DELETE", "/api/v1/environments/{uuid}")
    response = MagicMock()
    api_client.call_api.return_value = response

    client = SimpleNamespace(
        twins=twins_manager,
        environments=environments_manager,
        _api_client=api_client,
    )
    twin = _make_twin(client=client)

    twin.add_to_environment("env-new")

    api_client.param_serialize.assert_called_once()
    api_client.call_api.assert_called_once_with("DELETE", "/api/v1/environments/{uuid}")
    response.read.assert_called_once()


def test_add_to_environment_noops_for_same_environment():
    twins_manager = MagicMock()
    client = SimpleNamespace(
        twins=twins_manager,
        environments=MagicMock(),
        _api_client=MagicMock(),
    )
    twin = _make_twin(client=client)

    result = twin.add_to_environment("env-old")

    assert result is twin
    twins_manager.create.assert_not_called()
    twins_manager.delete.assert_not_called()
    twins_manager.list.assert_not_called()
