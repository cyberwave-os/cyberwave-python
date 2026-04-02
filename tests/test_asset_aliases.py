from types import SimpleNamespace

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

    result = manager.get_by_alias("camera")

    assert result is full_asset


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
