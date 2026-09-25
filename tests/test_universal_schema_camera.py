"""Tests for camera sensor id discovery from universal schema."""

import pytest

from types import SimpleNamespace
from unittest.mock import MagicMock

from cyberwave.twin import CameraTwin, Twin
from cyberwave.universal_schema_camera import camera_sensor_ids_from_schema


def test_camera_sensor_ids_empty_non_dict_schema():
    assert camera_sensor_ids_from_schema(None) == []
    assert camera_sensor_ids_from_schema([]) == []
    assert camera_sensor_ids_from_schema("x") == []


def test_camera_sensor_ids_top_level_sensors():
    schema = {
        "sensors": [
            {"type": "camera", "id": "front_rgb"},
            {"type": "imu", "id": "imu0"},
        ]
    }
    assert camera_sensor_ids_from_schema(schema) == ["front_rgb"]


def test_camera_sensor_ids_capabilities_nested_sensors():
    schema = {
        "capabilities": {
            "sensors": [
                {"type": "rgb_camera", "sensor_id": "wrist"},
            ]
        }
    }
    assert camera_sensor_ids_from_schema(schema) == ["wrist"]


def test_camera_sensor_ids_merges_top_and_capabilities_order():
    schema = {
        "sensors": [
            {"type": "camera", "name": "cam_a"},
        ],
        "capabilities": {
            "sensors": [
                {"type": "depth_camera", "id": "cam_b"},
            ]
        },
    }
    assert camera_sensor_ids_from_schema(schema) == ["cam_a", "cam_b"]


def test_camera_sensor_ids_dedupes_preserves_first():
    schema = {
        "sensors": [
            {"type": "camera", "id": "same"},
        ],
        "capabilities": {
            "sensors": [
                {"type": "camera", "id": "same"},
            ]
        },
    }
    assert camera_sensor_ids_from_schema(schema) == ["same"]


def test_camera_sensor_ids_prefers_id_over_name():
    schema = {
        "sensors": [
            {"type": "camera", "name": "full_name", "id": "short_id"},
        ]
    }
    assert camera_sensor_ids_from_schema(schema) == ["short_id"]


def test_camera_sensor_ids_rgb_depth_aliases():
    schema = {
        "sensors": [
            {"type": "rgb", "id": "r1"},
            {"type": "depth", "id": "d1"},
        ]
    }
    assert camera_sensor_ids_from_schema(schema) == ["r1", "d1"]


def test_camera_sensor_ids_skips_non_camera_types():
    schema = {
        "sensors": [
            {"type": "lidar", "id": "l0"},
            {"type": "camera", "id": "c0"},
        ]
    }
    assert camera_sensor_ids_from_schema(schema) == ["c0"]


def test_camera_sensor_ids_max_ids():
    schema = {
        "sensors": [
            {"type": "camera", "id": f"c{i}"} for i in range(20)
        ]
    }
    assert len(camera_sensor_ids_from_schema(schema, max_ids=3)) == 3
    assert camera_sensor_ids_from_schema(schema, max_ids=3) == ["c0", "c1", "c2"]


def test_camera_sensor_ids_skips_entry_without_id():
    schema = {
        "sensors": [
            {"type": "camera"},
        ]
    }
    assert camera_sensor_ids_from_schema(schema) == []


def test_twin_list_camera_sensor_ids_delegates_to_get_schema():
    full_schema = {
        "sensors": [{"type": "camera", "id": "main"}],
    }
    twins_manager = MagicMock()
    twins_manager.get_universal_schema_at_path.return_value = {"value": full_schema}
    client = SimpleNamespace(twins=twins_manager)
    twin = Twin(client, SimpleNamespace(uuid="twin-uuid", name="Twin"))

    ids = twin.list_camera_sensor_ids()

    assert ids == ["main"]
    twins_manager.get_universal_schema_at_path.assert_called_once_with("twin-uuid", "")


def test_twin_list_camera_sensor_ids_passes_max_ids():
    schema = {
        "sensors": [{"type": "camera", "id": f"c{i}"} for i in range(5)],
    }
    twins_manager = MagicMock()
    twins_manager.get_universal_schema_at_path.return_value = {"value": schema}
    client = SimpleNamespace(twins=twins_manager)
    twin = Twin(client, SimpleNamespace(uuid="u", name="T"))

    assert twin.list_camera_sensor_ids(max_ids=2) == ["c0", "c1"]


def test_camera_twin_inherits_list_camera_sensor_ids():
    schema = {"sensors": [{"type": "camera", "id": "cam0"}]}
    twins_manager = MagicMock()
    twins_manager.get_universal_schema_at_path.return_value = {"value": schema}
    client = SimpleNamespace(twins=twins_manager)
    cam = CameraTwin(client, SimpleNamespace(uuid="ct", name="CT"))

    assert cam.list_camera_sensor_ids() == ["cam0"]


@pytest.mark.parametrize("name", [None, "Front camera"])
def test_camera_sensor_ids_uses_schema_parameter_id(name):
    sensor = {"type": "camera", "parameters": {"id": "front"}}
    if name is not None:
        sensor["name"] = name
    assert camera_sensor_ids_from_schema({"sensors": [sensor]}) == ["front"]


@pytest.mark.parametrize("key", ["id", "sensor_id"])
def test_camera_sensor_ids_preserves_explicit_id_precedence(key):
    sensor = {"type": "camera", key: "explicit", "parameters": {"id": "nested"}}
    assert camera_sensor_ids_from_schema({"sensors": [sensor]}) == ["explicit"]


@pytest.mark.parametrize("parameters", [None, [], "invalid", {}, {"id": " "}])
def test_camera_sensor_ids_handles_missing_or_malformed_parameters(parameters):
    sensor = {"type": "camera", "name": "front", "parameters": parameters}
    assert camera_sensor_ids_from_schema({"sensors": [sensor]}) == ["front"]


def test_camera_sensor_ids_normalizes_exported_names_before_deduplication():
    schema = {
        "sensors": [{"type": "camera", "name": "1c04c936aaa64a41bfc347e9b5f3cf51__front_camera"}],
        "capabilities": {"sensors": [
            {"type": "rgb", "id": "front_camera"},
            {"type": "rgb", "id": "rear_camera"},
        ]},
    }
    assert camera_sensor_ids_from_schema(schema, max_ids=2) == ["front_camera", "rear_camera"]


def test_camera_sensor_ids_preserves_non_uuid_names_and_explicit_ids():
    schema = {"sensors": [
        {"type": "camera", "name": "rig__front"},
        {"type": "camera", "id": "1c04c936aaa64a41bfc347e9b5f3cf51__explicit"},
    ]}
    assert camera_sensor_ids_from_schema(schema) == [
        "rig__front", "1c04c936aaa64a41bfc347e9b5f3cf51__explicit",
    ]


@pytest.mark.parametrize("schema", [None, {}, {"sensors": [{"type": "camera", "id": "front"}]}])
def test_camera_sensor_ids_zero_limit(schema):
    assert camera_sensor_ids_from_schema(schema, max_ids=0) == []


@pytest.mark.parametrize("schema", [None, {}, {"sensors": [{"type": "camera", "id": "front"}]}])
def test_camera_sensor_ids_negative_limit(schema):
    with pytest.raises(ValueError, match="max_ids must be nonnegative"):
        camera_sensor_ids_from_schema(schema, max_ids=-1)
