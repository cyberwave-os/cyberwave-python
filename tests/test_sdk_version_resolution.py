from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cyberwave._version as version_module


def test_get_version_uses_installed_package_metadata(monkeypatch):
    monkeypatch.setattr(version_module, "metadata_version", lambda _name: "0.3.46rc12")

    assert version_module.get_version() == "0.3.46rc12"


def test_get_version_falls_back_to_static_version_when_metadata_missing(monkeypatch):
    def _raise(_name: str) -> str:
        raise version_module.PackageNotFoundError

    monkeypatch.setattr(version_module, "metadata_version", _raise)

    assert version_module.get_version() == version_module.STATIC_VERSION
