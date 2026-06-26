"""cw-driver.yml SDK helpers (compile happens on backend)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cyberwave.driver.interface.cw_driver import (
    load_cw_driver_yml,
    resolve_driver_config_dict,
)

_REPO_SO101 = (
    Path(__file__).resolve().parents[3]
    / "cyberwave-edge-nodes"
    / "cyberwave-edge-so101"
    / "cw-driver.yml"
)


def test_resolve_driver_config_dict_from_mapping() -> None:
    root = {
        "registry_id": "acme/test-arm",
        "mqtt": {
            "schema_version": 1,
            "twin": {
                "command": {
                    "direction": "both",
                    "payload_schema_ref": "TwinCommandPayload",
                }
            },
        },
    }
    resolved = resolve_driver_config_dict(root)
    assert resolved["registry_id"] == "acme/test-arm"
    assert "twin" in resolved["mqtt"]


def test_resolve_driver_config_dict_rejects_compiled_bundle() -> None:
    with pytest.raises(ValueError, match="compiled"):
        resolve_driver_config_dict(
            {
                "topics": {"cyberwave/twin/{twin_uuid}/command": {}},
                "commands": {"supported": ["stop"]},
            }
        )


def test_resolve_driver_config_dict_rejects_compiled_zenoh_bundle() -> None:
    with pytest.raises(ValueError, match="compiled"):
        resolve_driver_config_dict(
            {
                "schema_version": 1,
                "channels": {"imu": {"payload_schema_ref": "ImuPayload"}},
            }
        )


def test_resolve_driver_config_dict_accepts_zenoh_authoring_shape() -> None:
    root = {
        "registry_id": "intel/realsensed455",
        "mqtt": {
            "schema_version": 1,
            "twin": {
                "imu": {
                    "direction": "publish",
                    "payload_schema_ref": "ImuPayload",
                }
            },
        },
        "zenoh": {
            "schema_version": 1,
            "channels": {
                "imu": {"payload_schema_ref": "ImuPayload"},
            },
        },
    }
    resolved = resolve_driver_config_dict(root)
    assert resolved["zenoh"]["channels"]["imu"]["payload_schema_ref"] == "ImuPayload"


def test_resolve_driver_config_dict_from_fake_imu_driver_class() -> None:
    from examples.fake_imu_driver import FakeImu6dDriver

    resolved = resolve_driver_config_dict(FakeImu6dDriver.get_manifest())
    assert "rotate" in resolved["mqtt"]["commands"]["supported"]
    assert "imu" in resolved["mqtt"]["twin"]
    assert "imu" in resolved["zenoh"]["channels"]


@pytest.mark.skipif(not _REPO_SO101.is_file(), reason="monorepo edge-nodes not present")
def test_load_so101_reference_manifest() -> None:
    raw = load_cw_driver_yml(_REPO_SO101)
    resolved = resolve_driver_config_dict(raw)
    assert "joint" in resolved["mqtt"]
    assert "twin" in resolved["mqtt"]
