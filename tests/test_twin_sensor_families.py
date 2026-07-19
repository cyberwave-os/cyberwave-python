"""Always-singular indexable sensor families on Twin."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cyberwave.twin.classes import LocomoteCameraTwin
from cyberwave.twin.sensors.family import SensorFamily

# (family attr, plural name that must now be gone, single entry, second entry)
_FAMILY_CASES = (
    ("lidar", "lidars", {"id": "lidar_main", "type": "lidar_4d"}, {"id": "lidar_aux", "type": "lidar_3d"}),
    ("gps", "gpss", {"id": "gps_main", "type": "gps"}, {"id": "gps_aux", "type": "gps"}),
    ("compass", "compasses", {"id": "cmp_main", "type": "compass"}, {"id": "cmp_aux", "type": "compass"}),
    ("imu", "imus", {"id": "imu_main", "type": "imu"}, {"id": "imu_aux", "type": "imu"}),
    ("camera", "cameras", {"id": "cam_main", "type": "rgb"}, {"id": "cam_aux", "type": "depth"}),
    ("flashlight", "flashlights", {"id": "torch", "type": "flashlight"}, {"id": "torch2", "type": "flashlight"}),
)


def _twin(sensors):
    return LocomoteCameraTwin(
        SimpleNamespace(twins=SimpleNamespace()),
        SimpleNamespace(
            uuid="rover",
            name="Rover",
            capabilities={"can_locomote": True, "sensors": sensors},
        ),
    )


@pytest.mark.parametrize("attr,plural,one,two", _FAMILY_CASES, ids=[c[0] for c in _FAMILY_CASES])
def test_single_sensor_exposes_singular_family(attr, plural, one, two):
    twin = _twin([one])
    fam = getattr(twin, attr)
    assert isinstance(fam, SensorFamily)
    assert fam.keys() == [one["id"]]
    assert len(fam) == 1
    assert fam[0].sensor_id == one["id"]
    assert attr in dir(twin)


@pytest.mark.parametrize("attr,plural,one,two", _FAMILY_CASES, ids=[c[0] for c in _FAMILY_CASES])
def test_multi_sensor_uses_same_singular_family(attr, plural, one, two):
    twin = _twin([one, two])
    fam = getattr(twin, attr)
    assert isinstance(fam, SensorFamily)
    assert fam.keys() == [one["id"], two["id"]]
    assert fam[0].sensor_id == one["id"]
    assert fam[one["id"]].sensor_id == one["id"]
    assert getattr(fam, two["id"]).sensor_id == two["id"]
    assert attr in dir(twin)


@pytest.mark.parametrize("attr,plural,one,two", _FAMILY_CASES, ids=[c[0] for c in _FAMILY_CASES])
def test_plural_attr_no_longer_exists(attr, plural, one, two):
    twin = _twin([one, two])
    with pytest.raises(AttributeError) as exc:
        getattr(twin, plural)
    msg = str(exc.value)
    # Directed migration hint: names what was typed and the singular replacement.
    assert plural in msg
    assert f"twin.{attr}" in msg
    assert plural not in dir(twin)


@pytest.mark.parametrize("attr,plural,one,two", _FAMILY_CASES, ids=[c[0] for c in _FAMILY_CASES])
def test_absent_family_raises_and_not_in_dir(attr, plural, one, two):
    twin = _twin([])
    with pytest.raises(AttributeError, match=f"no attribute '{attr}'"):
        getattr(twin, attr)
    assert attr not in dir(twin)


def test_family_is_cached_per_attr():
    twin = _twin([{"id": "imu_main", "type": "imu"}])
    assert twin.imu is twin.imu


def test_camera_family_proxies_default_and_indexes():
    twin = _twin([{"id": "cam_a", "type": "rgb"}, {"id": "cam_b", "type": "depth"}])
    fam = twin.camera
    assert fam.keys() == ["cam_a", "cam_b"]
    # proxy: attribute access on the family hits the first sensor's handle
    assert fam[0]._sensor_id == "cam_a"
    assert fam["cam_b"]._sensor_id == "cam_b"
    assert hasattr(fam, "get_frame")  # proxied from cam_a handle


def test_describe_emits_singular_family_sections():
    twin = _twin(
        [
            {"id": "cam_a", "type": "rgb"},
            {"id": "cam_b", "type": "depth"},
            {"id": "imu_main", "type": "imu"},
        ]
    )
    info = twin.describe()
    handles = info["handles"]
    assert "cameras" not in handles and "imus" not in handles
    assert handles["camera"]["keys"] == ["cam_a", "cam_b"]
    assert handles["camera"]["default_sensor_id"] == "cam_a"
    assert handles["camera"]["per_sensor"] == "camera.describe()"
    assert "camera[0]" in handles["camera"]["access"]
    assert handles["imu"]["keys"] == ["imu_main"]
    assert handles["imu"]["methods"] == ["metadata", "get", "get_sample", "on_update"]
    assert "get_frame" in info["flat_methods"]
