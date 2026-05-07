from unittest.mock import MagicMock

from cyberwave.resources import EnvironmentManager
from cyberwave.rest import EnvironmentWaypointBulkCreateSchema


def _make_manager() -> tuple[EnvironmentManager, MagicMock]:
    mock_api = MagicMock()
    manager = EnvironmentManager(mock_api)
    return manager, mock_api


def test_environment_manager_create_preview_calls_preview_endpoint():
    manager, mock_api = _make_manager()
    fake_attachment = MagicMock()

    response_data = MagicMock()
    mock_api.api_client.param_serialize.return_value = (
        "POST",
        "/api/v1/environments/{uuid}/preview",
        {},
        None,
        [],
        {},
        [],
        {},
    )
    mock_api.api_client.call_api.return_value = response_data

    deserialized = MagicMock()
    deserialized.data = fake_attachment
    mock_api.api_client.response_deserialize.return_value = deserialized

    result = manager.create_preview("env-uuid-1")

    assert result is fake_attachment
    mock_api.api_client.param_serialize.assert_called_once_with(
        method="POST",
        resource_path="/api/v1/environments/{uuid}/preview",
        path_params={"uuid": "env-uuid-1"},
        auth_settings=["CustomTokenAuthentication"],
    )
    response_data.read.assert_called_once()


def test_environment_manager_get_waypoints_calls_waypoints_endpoint():
    manager, mock_api = _make_manager()
    expected_waypoints = [{"id": "dock-a", "name": "Dock A"}]
    mock_api.src_app_api_environments_list_environment_waypoints.return_value = (
        expected_waypoints
    )

    result = manager.get_waypoints("env-uuid-1")

    assert result == expected_waypoints
    mock_api.src_app_api_environments_list_environment_waypoints.assert_called_once_with(
        "env-uuid-1"
    )


def test_environment_manager_create_waypoint_posts_single_waypoint_payload():
    manager, mock_api = _make_manager()
    expected_waypoints = [{"id": "dock-a", "name": "Dock A"}]
    mock_api.src_app_api_environments_add_environment_waypoints.return_value = (
        expected_waypoints
    )

    result = manager.create_waypoint(
        "env-uuid-1",
        name="Dock A",
        position={"x": 1.0, "y": 2.0, "z": 0.0},
        rotation={"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        waypoint_id="dock-a",
        collection="docks",
        metadata={"priority": "high"},
    )

    assert result == expected_waypoints
    mock_api.src_app_api_environments_add_environment_waypoints.assert_called_once()
    environment_id, payload = (
        mock_api.src_app_api_environments_add_environment_waypoints.call_args.args
    )
    assert environment_id == "env-uuid-1"
    assert isinstance(payload, EnvironmentWaypointBulkCreateSchema)
    assert payload.to_dict() == {
        "waypoints": [
            {
                "id": "dock-a",
                "name": "Dock A",
                "collection": "docks",
                "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                "metadata": {"priority": "high"},
            }
        ]
    }


def test_environment_manager_delete_waypoint_calls_waypoint_delete_endpoint():
    manager, mock_api = _make_manager()
    expected_waypoints = [{"id": "dock-b", "name": "Dock B"}]
    mock_api.src_app_api_environments_delete_environment_waypoint.return_value = (
        expected_waypoints
    )

    result = manager.delete_waypoint("env-uuid-1", "dock-a")

    assert result == expected_waypoints
    mock_api.src_app_api_environments_delete_environment_waypoint.assert_called_once_with(
        "env-uuid-1", "dock-a"
    )
