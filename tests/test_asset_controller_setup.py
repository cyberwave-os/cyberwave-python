from types import SimpleNamespace
from typing import get_type_hints

import cyberwave
from cyberwave.resources import (
    AssetControllerSetupView,
    AssetManager,
    ControlRuntimeTargetPayload,
    PolicyRefPayload,
)


class _Response:
    def __init__(self) -> None:
        self.read_called = False

    def read(self) -> None:
        self.read_called = True


class _ApiClient:
    def __init__(self, payload):
        self.payload = payload
        self.response = _Response()
        self.serialize_kwargs = None
        self.call_args = None
        self.response_types_map = None

    def param_serialize(self, **kwargs):
        self.serialize_kwargs = kwargs
        return ("GET", kwargs["resource_path"], {}, None, [], kwargs)

    def call_api(self, *args):
        self.call_args = args
        return self.response

    def response_deserialize(self, *, response_data, response_types_map):
        assert response_data is self.response
        self.response_types_map = response_types_map
        return SimpleNamespace(data=self.payload)


def test_get_controller_setup_calls_backend_setup_view() -> None:
    payload = {
        "asset_uuid": "asset-uuid",
        "controller_configs": [
            {
                "id": "keyboard",
                "controller_key": "controller:keyboard-locomotion:v1",
                "label": "Keyboard",
                "is_default": True,
                "mode": "locomotion",
            }
        ],
        "primary_controller_key": "controller:keyboard-locomotion:v1",
        "runtime_policies": [
            {
                "key": "physical:edge:controller:keyboard-locomotion:v1",
                "runtime_kind": "physical",
                "backend": "edge",
                "controller_key": "controller:keyboard-locomotion:v1",
                "policy_ref": {
                    "kind": "catalog_seed_id",
                    "value": "controller:keyboard-locomotion:v1",
                },
                "controller_policy_uuid": "policy-uuid",
                "available": True,
                "runtime_enabled": True,
                "runtime_target": {
                    "enabled": True,
                    "runtime_kind": "physical",
                    "backend": "edge",
                },
                "artifact_readiness": "not_required",
            }
        ],
        "runtime_options": [
            {
                "runtime_kind": "physical",
                "backend": "edge",
                "controller_key": "controller:keyboard-locomotion:v1",
                "policy_ref": {
                    "kind": "catalog_seed_id",
                    "value": "controller:keyboard-locomotion:v1",
                },
                "controller_policy_uuid": "policy-uuid",
                "controller_name": "Keyboard Locomotion",
                "supports_runtime": True,
                "runtime_enabled": True,
                "runtime_target": {
                    "enabled": True,
                    "runtime_kind": "physical",
                    "backend": "edge",
                },
            }
        ],
        "recommended_setup": {
            "primary_controller_key": "controller:keyboard-locomotion:v1",
            "primary_policy_ref": {
                "kind": "catalog_seed_id",
                "value": "controller:keyboard-locomotion:v1",
            },
            "default_policy_refs": {
                "physical": {
                    "edge": {
                        "kind": "catalog_seed_id",
                        "value": "controller:keyboard-locomotion:v1",
                    }
                }
            },
        },
    }
    api_client = _ApiClient(payload)
    manager = AssetManager(SimpleNamespace(api_client=api_client))

    result = manager.get_controller_setup("asset-uuid")

    assert result == payload
    assert result["runtime_policies"][0]["policy_ref"] == {
        "kind": "catalog_seed_id",
        "value": "controller:keyboard-locomotion:v1",
    }
    assert result["recommended_setup"]["default_policy_refs"]["physical"]["edge"] == {
        "kind": "catalog_seed_id",
        "value": "controller:keyboard-locomotion:v1",
    }
    assert api_client.response.read_called is True
    assert api_client.serialize_kwargs == {
        "method": "GET",
        "resource_path": "/api/v1/assets/{uuid}/controller-setup",
        "path_params": {"uuid": "asset-uuid"},
        "auth_settings": ["CustomTokenAuthentication"],
    }
    assert api_client.call_args == (
        "GET",
        "/api/v1/assets/{uuid}/controller-setup",
        {},
        None,
        [],
        api_client.serialize_kwargs,
    )
    assert api_client.response_types_map == {"200": "object"}


def test_controller_setup_sdk_types_are_exported() -> None:
    setup_hints = get_type_hints(AssetControllerSetupView)
    policy_ref_hints = get_type_hints(PolicyRefPayload)
    runtime_target_hints = get_type_hints(ControlRuntimeTargetPayload)

    assert cyberwave.AssetControllerSetupView is AssetControllerSetupView
    assert cyberwave.PolicyRefPayload is PolicyRefPayload
    assert cyberwave.ControlRuntimeTargetPayload is ControlRuntimeTargetPayload
    assert "runtime_policies" in setup_hints
    assert "runtime_options" in setup_hints
    assert "recommended_setup" in setup_hints
    assert set(policy_ref_hints) == {"kind", "value"}
    assert {"enabled", "runtime_kind", "backend"}.issubset(runtime_target_hints)
