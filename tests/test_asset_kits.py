from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cyberwave.resources import AssetManager


def manager_with_transport(result):
    client = Mock()
    client.param_serialize.return_value = ("serialized",)
    client.response_deserialize.return_value = SimpleNamespace(data=result)
    return AssetManager(SimpleNamespace(api_client=client)), client


def test_create_kit_preserves_order_and_docking_contract():
    manager, transport = manager_with_transport({"asset": {"uuid": "kit"}})
    components = [
        {"key": "base", "asset_uuid": "robot"},
        {
            "key": "camera",
            "asset_uuid": "sensor",
            "parent_key": "base",
            "position": [0, 0, 0.25],
        },
    ]
    result = manager.create_kit("Inspection", components, workspace_uuid="workspace")
    assert result["asset"]["uuid"] == "kit"
    params = transport.param_serialize.call_args.kwargs
    assert params["resource_path"] == "/api/v1/asset-kits"
    assert params["auth_settings"] == ["CustomTokenAuthentication"]
    assert params["body"]["components"] == components
    assert params["body"]["visibility"] == "private"


def test_deploy_returns_every_created_twin():
    manager, transport = manager_with_transport([{"uuid": "root"}, {"uuid": "child"}])
    assert len(manager.instantiate_kit("kit", "environment", position=[1, 2, 3])) == 2
    params = transport.param_serialize.call_args.kwargs
    assert params["resource_path"] == "/api/v1/asset-kits/kit/instantiate"
    assert params["body"] == {
        "environment_uuid": "environment",
        "position_x": 1,
        "position_y": 2,
        "position_z": 3,
    }


def test_bad_position_does_not_send_request():
    manager, transport = manager_with_transport([])
    with pytest.raises(ValueError, match="x, y, z"):
        manager.instantiate_kit("kit", "environment", position=[1, 2])
    transport.call_api.assert_not_called()


def test_read_and_update_use_kit_routes():
    manager, transport = manager_with_transport({})
    manager.get_kit("kit")
    assert transport.param_serialize.call_args.kwargs["method"] == "GET"
    manager.update_kit("kit", name="Revised", components=[], visibility="workspace")
    params = transport.param_serialize.call_args.kwargs
    assert params["method"] == "PUT"
    assert params["body"]["visibility"] == "workspace"


def test_update_kit_omits_unspecified_sharing_fields():
    manager, transport = manager_with_transport({})

    manager.update_kit("kit", name="Revised", components=[])

    body = transport.param_serialize.call_args.kwargs["body"]
    assert body == {"name": "Revised", "components": []}
