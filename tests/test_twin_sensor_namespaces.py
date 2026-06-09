"""Namespace derivation: singular vs plural vs absent per sensor family."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cyberwave.twin.base import Twin
from cyberwave.twin.classes import LocomoteTwin
from cyberwave.twin.namespaces import (
    CompassesNamespace,
    FlashlightsNamespace,
    GpssNamespace,
    ImusNamespace,
    LidarsNamespace,
)
from cyberwave.twin.namespaces.camera import CamerasNamespace

# (handler_key, singular attr, plural attr, single-sensor entry, second entry for multi)
_SENSOR_FAMILY_CASES: tuple[
    tuple[str, str, str, dict[str, str], dict[str, str]],
    ...,
] = (
    (
        "lidar",
        "lidar",
        "lidars",
        {"id": "lidar_main", "type": "lidar_4d"},
        {"id": "lidar_aux", "type": "lidar_3d"},
    ),
    (
        "gps",
        "gps",
        "gpss",
        {"id": "gps_main", "type": "gps"},
        {"id": "gps_aux", "type": "gps"},
    ),
    (
        "compass",
        "compass",
        "compasses",
        {"id": "compass_main", "type": "compass"},
        {"id": "compass_aux", "type": "compass"},
    ),
    (
        "imu",
        "imu",
        "imus",
        {"id": "imu_main", "type": "imu"},
        {"id": "imu_aux", "type": "imu"},
    ),
    (
        "camera",
        "camera",
        "cameras",
        {"id": "cam_main", "type": "rgb"},
        {"id": "cam_aux", "type": "depth"},
    ),
    (
        "flashlight",
        "flashlight",
        "flashlights",
        {"id": "torch", "type": "flashlight"},
        {"id": "torch_aux", "type": "flashlight"},
    ),
)

_PLURAL_NAMESPACE_TYPES: dict[str, type] = {
    "lidars": LidarsNamespace,
    "gpss": GpssNamespace,
    "compasses": CompassesNamespace,
    "imus": ImusNamespace,
    "cameras": CamerasNamespace,
    "flashlights": FlashlightsNamespace,
}


def _twin(*, sensors: list[dict[str, str]] | None = None) -> Twin:
    caps: dict = {"can_locomote": True}
    if sensors is not None:
        caps["sensors"] = sensors
    return LocomoteTwin(
        SimpleNamespace(twins=SimpleNamespace()),
        SimpleNamespace(uuid="rover", name="Rover", capabilities=caps),
    )


@pytest.mark.parametrize(
    "handler,singular,plural,single_entry,_multi_extra",
    _SENSOR_FAMILY_CASES,
    ids=[c[0] for c in _SENSOR_FAMILY_CASES],
)
def test_single_sensor_exposes_singular_not_plural(
    handler: str,
    singular: str,
    plural: str,
    single_entry: dict[str, str],
    _multi_extra: dict[str, str],
) -> None:
    twin = _twin(sensors=[single_entry])
    assert twin.resolve_handler_from_capabilities(handler).available
    assert not twin.resolve_handler_from_capabilities(handler).multi_sensor

    handle = getattr(twin, singular)
    assert handle is not None
    if handler == "camera":
        assert handle._sensor_id == single_entry["id"]
    else:
        assert handle.sensor_id == single_entry["id"]
    if handler == "flashlight":
        assert hasattr(handle, "set")

    with pytest.raises(AttributeError, match=f"no attribute '{plural}'"):
        getattr(twin, plural)

    assert singular in dir(twin)
    assert plural not in dir(twin)


@pytest.mark.parametrize(
    "handler,singular,plural,single_entry,multi_extra",
    _SENSOR_FAMILY_CASES,
    ids=[c[0] for c in _SENSOR_FAMILY_CASES],
)
def test_multiple_sensors_expose_plural_not_singular(
    handler: str,
    singular: str,
    plural: str,
    single_entry: dict[str, str],
    multi_extra: dict[str, str],
) -> None:
    twin = _twin(sensors=[single_entry, multi_extra])
    resolution = twin.resolve_handler_from_capabilities(handler)
    assert resolution.available
    assert resolution.multi_sensor
    assert resolution.sensor_ids == (single_entry["id"], multi_extra["id"])

    ns = getattr(twin, plural)
    assert isinstance(ns, _PLURAL_NAMESPACE_TYPES[plural])
    assert ns.keys() == [single_entry["id"], multi_extra["id"]]

    with pytest.raises(AttributeError, match=f"no attribute '{singular}'"):
        getattr(twin, singular)

    assert plural in dir(twin)
    assert singular not in dir(twin)


@pytest.mark.parametrize(
    "handler,singular,plural,single_entry,_multi_extra",
    _SENSOR_FAMILY_CASES,
    ids=[c[0] for c in _SENSOR_FAMILY_CASES],
)
def test_missing_sensor_family_has_no_namespace_attrs(
    handler: str,
    singular: str,
    plural: str,
    single_entry: dict[str, str],
    _multi_extra: dict[str, str],
) -> None:
    twin = _twin(sensors=[])
    assert not twin.resolve_handler_from_capabilities(handler).available

    with pytest.raises(AttributeError, match=f"no attribute '{singular}'"):
        getattr(twin, singular)
    with pytest.raises(AttributeError, match=f"no attribute '{plural}'"):
        getattr(twin, plural)

    assert singular not in dir(twin)
    assert plural not in dir(twin)


@pytest.mark.parametrize(
    "handler,singular,plural,single_entry,_multi_extra",
    _SENSOR_FAMILY_CASES,
    ids=[c[0] for c in _SENSOR_FAMILY_CASES],
)
def test_other_families_do_not_affect_namespace_derivation(
    handler: str,
    singular: str,
    plural: str,
    single_entry: dict[str, str],
    _multi_extra: dict[str, str],
) -> None:
    """Only the configured family is present; unrelated sensors do not enable it."""
    other = {"id": "other", "type": "audio"}
    twin = _twin(sensors=[other])
    assert not twin.resolve_handler_from_capabilities(handler).available
    with pytest.raises(AttributeError):
        getattr(twin, singular)


def test_locomote_twin_without_flashlight_sensor_has_no_flashlight_attr() -> None:
    twin = _twin(sensors=[{"id": "lidar_4d", "type": "lidar_4d"}])
    assert twin.has_sensor("lidar")
    assert not twin.has_sensor("flashlight")
    assert "flashlight" not in dir(twin)
    with pytest.raises(AttributeError, match="no attribute 'flashlight'"):
        twin.flashlight


def test_no_sensors_key_means_no_read_namespaces() -> None:
    twin = _twin()
    for singular, plural in (
        ("lidar", "lidars"),
        ("gps", "gpss"),
        ("compass", "compasses"),
        ("imu", "imus"),
        ("flashlight", "flashlights"),
    ):
        with pytest.raises(AttributeError):
            getattr(twin, singular)
        with pytest.raises(AttributeError):
            getattr(twin, plural)
