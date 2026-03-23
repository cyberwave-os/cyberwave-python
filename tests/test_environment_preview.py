from unittest.mock import MagicMock

from cyberwave.resources import EnvironmentManager


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

