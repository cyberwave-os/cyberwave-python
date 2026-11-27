"""Public catalog assets must resolve by registry id even when a workspace is configured.

With CYBERWAVE_WORKSPACE_ID set (or after get_or_create_quickstart_environment saves
one), every request carries ``workspace_uuid``, which on /api/v1/assets limits results
to that workspace's own assets. ``cw.twin("dji/tello")`` then failed with
"Asset 'dji/tello' not found".
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from cyberwave.client import _with_workspace_query_param
from cyberwave.resources import AssetManager


def test_catalog_wide_lookup_sends_explicit_empty_workspace():
    api_client = MagicMock()
    api_client.param_serialize.return_value = ("GET", "url", {}, None, None)
    api_client.response_deserialize.return_value = SimpleNamespace(data=[])
    manager = AssetManager(SimpleNamespace(api_client=api_client))

    manager._list_by_registry_id_catalog_wide("dji/tello")

    params = api_client.param_serialize.call_args.kwargs["query_params"]
    assert ("registry_id", "dji/tello") in params
    assert ("workspace_uuid", "") in params


def test_explicit_empty_workspace_is_not_overwritten_by_the_configured_one():
    url = _with_workspace_query_param("https://x/api/v1/assets?registry_id=a&workspace_uuid=", "ws-1")
    assert "workspace_uuid=ws-1" not in url
    assert "workspace_uuid=" in url


def test_get_by_registry_id_falls_back_to_catalog_wide(monkeypatch):
    manager = AssetManager(SimpleNamespace())
    tello = SimpleNamespace(uuid="a1", registry_id="dji/tello", registry_id_alias=None)
    monkeypatch.setattr(manager, "list_by_registry_id", lambda rid: [])  # workspace-scoped: nothing
    monkeypatch.setattr(manager, "_list_by_registry_id_catalog_wide", lambda rid: [tello])
    monkeypatch.setattr(manager, "search", lambda q: [])
    monkeypatch.setattr(manager, "list", lambda: [])

    assert manager.get_by_registry_id("dji/tello") is tello


def test_workspace_asset_still_wins(monkeypatch):
    manager = AssetManager(SimpleNamespace())
    mine = SimpleNamespace(uuid="mine", registry_id="dji/tello", registry_id_alias=None)
    catalog_wide = MagicMock(side_effect=AssertionError("should not be called"))
    monkeypatch.setattr(manager, "list_by_registry_id", lambda rid: [mine])
    monkeypatch.setattr(manager, "_list_by_registry_id_catalog_wide", catalog_wide)

    assert manager.get_by_registry_id("dji/tello") is mine
