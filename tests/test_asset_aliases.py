from types import SimpleNamespace

import pytest

from cyberwave.resources import AssetManager
from cyberwave.stubs_generator import (
    generate_asset_registry,
    generate_capabilities_cache,
    generate_client_stubs,
)


def _make_manager():
    manager = AssetManager(SimpleNamespace())
    return manager


def test_get_by_registry_id_matches_registry_alias(monkeypatch):
    manager = _make_manager()
    alias_asset = SimpleNamespace(
        uuid="asset-1",
        registry_id="cyberwave/standard-camera",
        registry_id_alias="camera",
    )

    monkeypatch.setattr(manager, "search", lambda query: [alias_asset])
    monkeypatch.setattr(manager, "list", lambda: [alias_asset])
    monkeypatch.setattr(manager, "list_by_registry_id", lambda rid: [alias_asset])

    result = manager.get_by_registry_id("camera")

    assert result is alias_asset


def test_get_by_alias_prefers_first_class_registry_alias(monkeypatch):
    manager = _make_manager()
    list_asset = SimpleNamespace(
        uuid="asset-1",
        name="Standard Camera",
        registry_id="cyberwave/standard-camera",
        registry_id_alias="camera",
        metadata={},
    )
    full_asset = SimpleNamespace(
        uuid="asset-1",
        name="Standard Camera",
        registry_id="cyberwave/standard-camera",
        registry_id_alias="camera",
        metadata={},
    )

    monkeypatch.setattr(manager, "search", lambda query: [list_asset])
    monkeypatch.setattr(manager, "get", lambda asset_id: full_asset)
    monkeypatch.setattr(manager, "list_by_registry_id", lambda rid: [full_asset])

    result = manager.get_by_alias("camera")

    assert result is full_asset


def test_get_by_registry_id_with_slash_uses_list_by_registry_id(monkeypatch):
    """Regression test for CYB-1701.

    When an asset has both a full registry_id (e.g. ``"cyberwave/standard-cam"``)
    and a registry_id_alias (e.g. ``"camera"``), looking it up by the full
    registry_id must succeed.

    The bug: the previous implementation tried ``GET /assets/cyberwave/standard-cam``
    which Django's URL router cannot handle (the slash splits the path), so it
    always returned 404 and then fell through to the search fallback.  The
    search fallback could succeed in some cases but was unreliable.

    The fix: ``get_by_registry_id`` now calls ``list_by_registry_id`` (which
    uses the ``?registry_id=`` query parameter) as the primary resolution
    strategy for identifiers that contain a slash.
    """
    manager = _make_manager()
    asset = SimpleNamespace(
        uuid="asset-1",
        registry_id="cyberwave/standard-cam",
        registry_id_alias="camera",
    )

    list_by_registry_id_calls: list[str] = []
    get_calls: list[str] = []

    def _list_by_registry_id(rid: str):
        list_by_registry_id_calls.append(rid)
        return [asset]

    def _get(asset_id: str):
        get_calls.append(asset_id)
        raise Exception("not found")

    monkeypatch.setattr(manager, "list_by_registry_id", _list_by_registry_id)
    monkeypatch.setattr(manager, "get", _get)
    monkeypatch.setattr(manager, "search", lambda query: [])
    monkeypatch.setattr(manager, "list", lambda: [asset])

    result = manager.get_by_registry_id("cyberwave/standard-cam")

    assert result is asset, "get_by_registry_id must find the asset by its full registry_id"
    assert list_by_registry_id_calls == [
        "cyberwave/standard-cam"
    ], "list_by_registry_id must be called with the full identifier"
    assert get_calls == [], (
        "direct GET must NOT be attempted for identifiers containing a slash"
    )


def test_get_by_registry_id_with_slash_skips_direct_get(monkeypatch):
    """Identifiers containing a slash must never be passed to the direct GET endpoint.

    Django URL routing would interpret the slash as a path separator and return
    a 404, making the direct-GET shortcut useless (and potentially dangerous if
    another route happens to match the prefix).
    """
    manager = _make_manager()
    asset = SimpleNamespace(
        uuid="asset-1",
        registry_id="vendor/my-robot",
        registry_id_alias=None,
    )

    get_calls: list[str] = []

    def _get(asset_id: str):
        get_calls.append(asset_id)
        raise Exception("should not be called")

    monkeypatch.setattr(manager, "get", _get)
    monkeypatch.setattr(manager, "list_by_registry_id", lambda rid: [asset])
    monkeypatch.setattr(manager, "search", lambda query: [])
    monkeypatch.setattr(manager, "list", lambda: [asset])

    result = manager.get_by_registry_id("vendor/my-robot")

    assert result is asset
    assert get_calls == [], "direct GET must not be attempted for a 'vendor/name' identifier"


def test_get_by_registry_id_plain_alias_tries_direct_get_first(monkeypatch):
    """Plain aliases (no slash) should still try the direct GET endpoint first.

    This preserves the fast-path behaviour for the common ``cw.twin("camera")``
    use-case where the alias resolves without an extra round-trip.
    """
    manager = _make_manager()
    asset = SimpleNamespace(
        uuid="asset-1",
        registry_id="cyberwave/standard-cam",
        registry_id_alias="camera",
    )

    get_calls: list[str] = []

    def _get(asset_id: str):
        get_calls.append(asset_id)
        return asset

    monkeypatch.setattr(manager, "get", _get)
    monkeypatch.setattr(manager, "list_by_registry_id", lambda rid: [])
    monkeypatch.setattr(manager, "search", lambda query: [])
    monkeypatch.setattr(manager, "list", lambda: [])

    result = manager.get_by_registry_id("camera")

    assert result is asset
    assert get_calls == ["camera"], "direct GET must be attempted for a plain alias"


def test_stubs_generator_includes_registry_aliases(tmp_path):
    assets = [
        {
            "uuid": "asset-1",
            "name": "Standard Camera",
            "registry_id": "cyberwave/standard-camera",
            "registry_id_alias": "camera",
            "capabilities": {"sensors": [{"id": "cam", "type": "rgb"}]},
        }
    ]

    expected_class_name = "CyberwavestandardCameraTwin"

    registry_text = generate_asset_registry(assets)
    assert f'"cyberwave/standard-camera": {expected_class_name},' in registry_text
    assert f'"camera": {expected_class_name},' in registry_text

    cache_path = tmp_path / "assets_capabilities.json"
    generate_capabilities_cache(assets, cache_path)
    cache_text = cache_path.read_text()
    assert '"cyberwave/standard-camera"' in cache_text
    assert '"camera"' in cache_text

    client_stub_path = tmp_path / "client.pyi"
    generate_client_stubs(assets, client_stub_path)
    client_stub_text = client_stub_path.read_text()
    assert 'Literal["cyberwave/standard-camera"]' in client_stub_text
    assert 'Literal["camera"]' in client_stub_text
