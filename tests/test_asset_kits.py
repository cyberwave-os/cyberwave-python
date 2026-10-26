import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cyberwave.exceptions import BadRequestException, CyberwaveAPIError, CyberwaveError
from cyberwave.resources import AssetKitManager, AssetManager


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


# --- AssetKitManager (replay-feed README §13) --------------------------------

KIT = "cyberwave/server-rack-42u-20x2u"
ASSETS = {
    "cyberwave/server-rack-42u": SimpleNamespace(uuid="rack-asset"),
    "cyberwave/server-2u": SimpleNamespace(uuid="server-asset"),
}


def kit_manager(result, workspace_id="ws-1"):
    api_client = Mock()
    api_client.param_serialize.return_value = ("serialized",)
    api_client.response_deserialize.return_value = SimpleNamespace(data=result)
    client = SimpleNamespace(
        assets=Mock(), config=SimpleNamespace(workspace_id=workspace_id)
    )
    client.assets.get_by_registry_id.side_effect = lambda registry_id: ASSETS.get(
        registry_id
    )
    return (
        AssetKitManager(SimpleNamespace(api_client=api_client), client=client),
        api_client,
        client,
    )


def test_kit_create_resolves_registry_ids_once_and_sets_the_kit_registry_id():
    manager, api_client, client = kit_manager(
        {"asset": {"uuid": "kit-uuid", "name": "Rack"}, "components": []}
    )
    components = [
        {
            "key": "rack",
            "asset_registry_id": "cyberwave/server-rack-42u",
            "name": "rack",
        },
        {
            "key": "s00",
            "asset_registry_id": "cyberwave/server-2u",
            "parent_key": "rack",
            "attach_to_link": "server_rack_42u_link",
            "position": (0.105, 0, 0.14445),
            "rotation": (1, 0, 0, 0),
        },
        {
            "key": "s01",
            "asset_uuid": "server-asset",
            "parent_key": "rack",
            "attach_to_link": "server_rack_42u_link",
            "position": [0.105, 0, 0.23335],
        },
    ]

    kit = manager.create("Rack 42U", KIT, components, description="rack + servers")

    request = api_client.param_serialize.call_args.kwargs
    assert request["method"] == "POST"
    assert request["resource_path"] == "/api/v1/asset-kits"
    assert request["auth_settings"] == ["CustomTokenAuthentication"]
    assert request["body"] == {
        "name": "Rack 42U",
        "description": "rack + servers",
        "workspace_uuid": "ws-1",
        "components": [
            {"key": "rack", "asset_uuid": "rack-asset", "name": "rack"},
            {
                "key": "s00",
                "asset_uuid": "server-asset",
                "parent_key": "rack",
                "attach_to_link": "server_rack_42u_link",
                "position": [0.105, 0.0, 0.14445],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            },
            {
                "key": "s01",
                "asset_uuid": "server-asset",
                "parent_key": "rack",
                "attach_to_link": "server_rack_42u_link",
                "position": [0.105, 0.0, 0.23335],
            },
        ],
    }
    assert client.assets.get_by_registry_id.call_count == 2
    client.assets.update.assert_called_once_with("kit-uuid", {"registry_id": KIT})
    assert kit["asset"]["registry_id"] == KIT


def test_kit_create_without_workspace_or_description():
    manager, api_client, client = kit_manager(
        {"asset": {"uuid": "kit-uuid"}, "components": []}, workspace_id=None
    )

    manager.create(
        "Kit",
        KIT,
        [
            {"key": "a", "asset_uuid": "x"},
            {"key": "b", "asset_uuid": "y", "parent_key": "a"},
        ],
    )

    body = api_client.param_serialize.call_args.kwargs["body"]
    assert body["description"] == ""
    assert "workspace_uuid" not in body


def test_kit_create_rejects_bad_components_before_sending():
    manager, api_client, _ = kit_manager({})
    with pytest.raises(ValueError, match="asset_uuid or asset_registry_id"):
        manager.create("Kit", KIT, [{"key": "rack"}])
    with pytest.raises(ValueError, match="x, y, z"):
        manager.create(
            "Kit", KIT, [{"key": "rack", "asset_uuid": "a", "position": [1, 2]}]
        )
    with pytest.raises(ValueError, match="w, x, y, z"):
        manager.create(
            "Kit", KIT, [{"key": "rack", "asset_uuid": "a", "rotation": [1, 0, 0]}]
        )
    with pytest.raises(CyberwaveError, match="'acme/unknown' not found"):
        manager.create(
            "Kit", KIT, [{"key": "rack", "asset_registry_id": "acme/unknown"}]
        )
    api_client.call_api.assert_not_called()


def test_kit_create_surfaces_backend_detail():
    manager, api_client, client = kit_manager(None)
    api_client.response_deserialize.side_effect = BadRequestException(
        status=400,
        reason="Bad Request",
        body=json.dumps({"detail": "A kit must have exactly one root component"}),
    )
    with pytest.raises(CyberwaveAPIError, match="exactly one root component"):
        manager.create("Kit", KIT, [{"key": "a", "asset_uuid": "x"}])
    client.assets.update.assert_not_called()


def test_kit_instantiate_accepts_registry_id_and_position():
    twins = [{"uuid": "root"}, {"uuid": "child"}]
    ASSETS[KIT] = SimpleNamespace(uuid="kit-uuid")
    try:
        manager, api_client, _ = kit_manager(twins)
        assert manager.instantiate(KIT, "env-1", position=[1, 2, 3]) == twins
    finally:
        del ASSETS[KIT]
    request = api_client.param_serialize.call_args.kwargs
    assert request["method"] == "POST"
    assert request["resource_path"] == "/api/v1/asset-kits/{uuid}/instantiate"
    assert request["path_params"] == {"uuid": "kit-uuid"}
    assert request["body"] == {
        "environment_uuid": "env-1",
        "position_x": 1.0,
        "position_y": 2.0,
        "position_z": 3.0,
    }


def test_kit_instantiate_uses_a_uuid_directly_and_checks_position():
    manager, api_client, client = kit_manager([])
    kit_uuid = "0f3c6a2e-5b1d-4c7e-9a8f-1b2c3d4e5f60"

    manager.instantiate(kit_uuid, "env-1")

    client.assets.get_by_registry_id.assert_not_called()
    request = api_client.param_serialize.call_args.kwargs
    assert request["path_params"] == {"uuid": kit_uuid}
    assert request["body"] == {"environment_uuid": "env-1"}
    with pytest.raises(ValueError, match="x, y, z"):
        manager.instantiate(kit_uuid, "env-1", position=[1])
    with pytest.raises(CyberwaveError, match="Asset kit 'acme/none' not found"):
        manager.instantiate("acme/none", "env-1")


def test_kit_manager_without_client_builds_its_own_asset_manager():
    manager = AssetKitManager(SimpleNamespace(api_client=Mock()))
    assert isinstance(manager._assets, AssetManager)


# --- build_rack planning (replay-feed README §14) ----------------------------

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "datacenter_rack"


@pytest.fixture(scope="module")
def build_rack():
    spec = importlib.util.spec_from_file_location(
        "build_rack", EXAMPLE_DIR / "build_rack.py"
    )
    module = importlib.util.module_from_spec(spec)
    # No __pycache__ in the examples folder.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def test_rack_names_nodes_and_offsets(build_rack):
    servers = build_rack.plan_servers(48)
    assert build_rack.rack_name(48) == "rack-48"
    assert [s["name"] for s in servers][:2] == ["rack-48-s00", "rack-48-s01"]
    assert servers[-1]["name"] == "rack-48-s19"
    assert [s["node"] for s in servers] == list(range(960, 980))
    assert (
        servers[0]["thing_id"] == "node=960" and servers[19]["thing_id"] == "node=979"
    )
    assert servers[0]["offset"] == (0.105, 0.0, 0.14445)
    assert servers[1]["offset"] == (0.105, 0.0, 0.23335)
    assert servers[19]["offset"] == (0.105, 0.0, 1.83355)
    assert build_rack.node_number(0, 5) == 5
    assert build_rack.server_name(7, 3) == "rack-7-s03"


def test_marconi_template_matches_the_contract_and_the_recipe(build_rack):
    template = build_rack.load_template("marconi100")
    assert template == {
        "inlet_temperature": "ipmi/ambient",
        "total_power": "ipmi/total_power",
        "cpu0_power": "ipmi/p0_power",
        "cpu1_power": "ipmi/p1_power",
        "cpu0_temperature": "ipmi/p0_vdd_temp",
        "cpu1_temperature": "ipmi/p1_vdd_temp",
        "gpu0_temperature": "ipmi/gpu0_core_temp",
        "gpu1_temperature": "ipmi/gpu1_core_temp",
        "gpu2_temperature": "ipmi/gpu3_core_temp",
        "gpu3_temperature": "ipmi/gpu4_core_temp",
        "fan_speed": "ipmi/fan0_0",
        "psu0_input_power": "ipmi/ps0_input_power",
        "psu1_input_power": "ipmi/ps1_input_power",
    }
    assert "outlet_temperature" not in template
    assert (
        build_rack.load_template(str(EXAMPLE_DIR / "templates" / "marconi100.json"))
        == template
    )
    assert build_rack.load_template("templates/marconi100.json") == template
    recipe = EXAMPLE_DIR.parent / "timeseries_replay" / "recipes" / "marconi100.json"
    assert json.loads(recipe.read_text(encoding="utf-8"))["slots"] == template


def test_load_template_reports_bad_input(build_rack, tmp_path):
    with pytest.raises(SystemExit, match="bundled: marconi100"):
        build_rack.load_template("nope")
    bad = tmp_path / "bad.json"
    bad.write_text('{"inlet_temperature": 3}', encoding="utf-8")
    with pytest.raises(SystemExit, match="slot ids to signal names"):
        build_rack.load_template(str(bad))


def test_kit_components_mirror_the_build_offsets(build_rack):
    components = build_rack.kit_components("rack-asset", "server-asset")
    assert len(components) == 21
    assert components[0] == {"key": "rack", "asset_uuid": "rack-asset", "name": "rack"}
    for i, component in enumerate(components[1:]):
        assert component["key"] == f"s{i:02d}"
        assert component["asset_uuid"] == "server-asset"
        assert component["parent_key"] == "rack"
        assert component["attach_to_link"] == "server_rack_42u_link"
        assert component["position"] == list(build_rack.server_offset(i))
        assert component["rotation"] == [1.0, 0.0, 0.0, 0.0]


def test_dry_run_prints_the_plan_for_rack_48_without_the_api(
    build_rack, capsys, monkeypatch
):
    def no_client(*args, **kwargs):
        raise AssertionError("dry run must not build a client")

    monkeypatch.setattr(build_rack, "Cyberwave", no_client)

    build_rack.main(
        [
            "--environment",
            "ba7e2341-b3a9-4fdf-97e9-808a234443a2",
            "--rack",
            "48",
            "--dataset",
            "a1985343-53c2-45b2-b366-09a0606dc649",
            "--template",
            "marconi100",
            "--as-kit",
            "--dry-run",
        ]
    )

    out = capsys.readouterr().out
    assert "rack-48: cyberwave/server-rack-42u at (0.0, 0.0, 0.0), fixed base" in out
    assert (
        "rack-48-s00: cyberwave/server-2u docked to server_rack_42u_link at (0.105, 0.0, 0.14445), thing_id node=960"
        in out
    )
    assert (
        "rack-48-s19: cyberwave/server-2u docked to server_rack_42u_link at (0.105, 0.0, 1.83355), thing_id node=979"
        in out
    )
    assert "template (13 slots): inlet_temperature <- ipmi/ambient" in out
    assert "auto-bind: use_recorded on" in out
    assert "kit cyberwave/server-rack-42u-20x2u: rack + 20 servers" in out
    assert out.rstrip().endswith("dry run: nothing was sent")
