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


def test_kinematics_slug_is_readable_for_the_pose_stream() -> None:
    """A publish-only kinematics declaration must still reach twin.pose."""
    from cyberwave.manifest.driver_config import (
        TWIN_KINEMATICS_TOPIC_SLUG,
        _slug_readable,
    )

    entry = {"direction": "publish", "payload_schema_ref": "TwinKinematicsPayload"}
    assert _slug_readable(TWIN_KINEMATICS_TOPIC_SLUG, entry, stream="pose") is True


def test_publish_only_slug_stays_unreadable_for_other_streams() -> None:
    from cyberwave.manifest.driver_config import (
        TWIN_KINEMATICS_TOPIC_SLUG,
        _slug_readable,
    )

    entry = {"direction": "publish"}
    assert _slug_readable(TWIN_KINEMATICS_TOPIC_SLUG, entry, stream="joints") is False


def test_kinematics_payload_round_trips_through_the_decoder() -> None:
    """Guards the exact payload shape Task 7's publisher must emit."""
    from cyberwave.data.state_representation import decode_kinematics_protobuf_stub

    payload = {
        "message_type": "cartesian_pose",
        "cartesian_pose": {
            "reference_frame": "base_link",
            "position": {"x": 0.1, "y": 0.2, "z": 0.3},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    msg = decode_kinematics_protobuf_stub(payload)
    assert msg.message_type == "cartesian_pose"
    assert msg.cartesian_pose is not None
    assert msg.cartesian_pose.position.x == 0.1
    assert msg.cartesian_pose.position.z == 0.3
    assert msg.cartesian_pose.orientation.w == 1.0
    assert msg.cartesian_pose.spatial_state.reference_frame == "base_link"
