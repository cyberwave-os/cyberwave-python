"""fake_imu_driver.py — 6-DOF IMU example."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from examples.fake_imu_driver import (
    DEFAULT_REGISTRY_ID,
    DEFAULT_SENSOR_ID,
    IMU_PAYLOAD_SCHEMA_REF,
    FakeImu6dDriver,
    build_imu_payload,
    d455_imu_capability_entry,
    ensure_imu_capability,
    random_imu_vectors,
)
from cyberwave.twin._helpers import motion_outbound_requires_policy


@pytest.fixture
def imu_twin() -> SimpleNamespace:
    return SimpleNamespace(
        uuid="00000000-0000-0000-0000-000000000099",
        environment_id="env-00000000-0000-0000-0000-000000000001",
        name="fake-imu",
        client=None,
    )


def test_imu_payload_is_six_dof() -> None:
    payload = build_imu_payload(
        gyro={"x": 0.1, "y": 0.2, "z": 0.3},
        accel={"x": 0.0, "y": 0.0, "z": 9.81},
    )
    assert "type" not in payload
    assert payload["dof"] == 6
    assert payload["gyro"]["z"] == pytest.approx(0.3)
    assert payload["accel"]["z"] == pytest.approx(9.81)
    assert "angular_velocity" not in payload
    assert "linear_acceleration" not in payload


def test_manifest_property_matches_cw_driver_root(imu_twin: SimpleNamespace) -> None:
    driver = FakeImu6dDriver(imu_twin)
    assert driver.manifest == driver.cw_driver
    assert driver.manifest == driver.get_driver_manifest(compiled=False)


def test_get_manifest_classmethod_without_twin(tmp_path) -> None:
    from cyberwave.driver.interface.cw_driver import load_cw_driver_yml

    yml_path = tmp_path / "cw-driver.yml"
    root = FakeImu6dDriver.get_manifest(path=yml_path)
    assert "rotate" in root["mqtt"]["commands"]["supported"]
    assert "imu" in root["mqtt"]["twin"]
    raw = load_cw_driver_yml(yml_path)
    assert "rotate" in raw["mqtt"]["commands"]["supported"]


def test_write_cw_driver_yml_roundtrip(imu_twin: SimpleNamespace, tmp_path) -> None:
    from cyberwave.driver.interface.cw_driver import load_cw_driver_yml

    driver = FakeImu6dDriver(imu_twin)
    path = driver.write_cw_driver_yml(tmp_path / "cw-driver.yml")
    raw = load_cw_driver_yml(path)
    assert "rotate" in raw["mqtt"]["commands"]["supported"]
    assert "imu" in raw["mqtt"]["twin"]
    assert raw == FakeImu6dDriver.get_manifest()


def test_manifest_includes_imu_and_rotate(imu_twin: SimpleNamespace) -> None:
    driver = FakeImu6dDriver(imu_twin)
    root = driver.manifest
    assert "command" in root["mqtt"]["twin"]
    assert "rotate" in root["mqtt"]["commands"]["supported"]
    assert "imu" in root["mqtt"]["twin"]
    assert root["mqtt"]["twin"]["imu"]["payload_schema_ref"] == IMU_PAYLOAD_SCHEMA_REF


def test_cw_driver_includes_zenoh_imu_channel(imu_twin: SimpleNamespace) -> None:
    driver = FakeImu6dDriver(imu_twin)
    raw = driver.cw_driver
    assert "zenoh" in raw
    assert (
        raw["zenoh"]["channels"]["imu"]["payload_schema_ref"] == IMU_PAYLOAD_SCHEMA_REF
    )


def test_publish_imu_dual_mqtt_and_zenoh(
    imu_twin: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from cyberwave.driver import DriverLifecycleState, DriverOperationMode

    mqtt_published: list[tuple[str, dict]] = []
    zenoh_published: list[tuple[str, dict]] = []

    driver = FakeImu6dDriver(imu_twin)
    driver._lifecycle_state = DriverLifecycleState.ACTIVE
    driver._operation_mode = DriverOperationMode.NO_OP
    driver._publisher_tick_counter = 0
    driver._registry_zenoh_subscriptions = []
    driver._cw = SimpleNamespace(
        mqtt=SimpleNamespace(
            publish=lambda path, payload: mqtt_published.append((path, payload)),
            topic_prefix="",
        ),
        data=SimpleNamespace(
            publish=lambda channel, payload, **_: zenoh_published.append(
                (channel, payload)
            ),
        ),
    )
    driver._registry_zenoh_bus = driver._cw.data

    asyncio.run(driver._run_registry_publishers())

    assert len(mqtt_published) == 1
    assert len(zenoh_published) == 1
    assert zenoh_published[0][0] == "imu"
    assert mqtt_published[0][1]["dof"] == 6
    assert zenoh_published[0][1]["dof"] == 6


def test_rotate_does_not_require_teleop_policy() -> None:
    assert motion_outbound_requires_policy("rotate") is False


def test_ensure_imu_capability_adds_sensor() -> None:
    caps: dict = {"sensors": [{"id": "color_camera", "type": "camera"}]}
    twin = SimpleNamespace(_data=SimpleNamespace(capabilities=caps))
    ensure_imu_capability(twin)
    assert any(
        s["id"] == DEFAULT_SENSOR_ID and s["type"] == "imu" for s in caps["sensors"]
    )


def test_ensure_imu_capability_when_property_returns_ephemeral_empty() -> None:
    twin = SimpleNamespace(_data=SimpleNamespace(capabilities=None))
    ensure_imu_capability(twin)
    assert twin._data.capabilities is not None
    assert any(s["type"] == "imu" for s in twin._data.capabilities["sensors"])


def test_d455_registry_and_capability_helper() -> None:
    assert DEFAULT_REGISTRY_ID == "intel/realsensed455"
    assert FakeImu6dDriver.REGISTRY_ID == "intel/realsensed455"
    entry = d455_imu_capability_entry()
    assert entry["id"] == DEFAULT_SENSOR_ID
    assert entry["type"] == "imu"


def test_publish_imu_sample_uses_random_values(imu_twin: SimpleNamespace) -> None:
    driver = FakeImu6dDriver(imu_twin)
    a = driver._publish_imu_sample()
    b = driver._publish_imu_sample()
    assert "type" not in a and a["dof"] == 6
    assert (a["gyro"], a["accel"]) != (b["gyro"], b["accel"])


def test_random_imu_vectors_jitters_around_base() -> None:
    gyro, accel = random_imu_vectors(gyro_base={"x": 1.0, "y": 0.0, "z": 0.0})
    assert abs(gyro["x"] - 1.0) <= 0.15
    assert abs(accel["z"] - 9.81) <= 0.35


def test_rotate_sets_gyro_rate(
    imu_twin: SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    driver = FakeImu6dDriver(imu_twin)
    with caplog.at_level("INFO"):
        driver._on_rotate(
            {
                "command": "rotate",
                "data": {"axis": "yaw", "amount_deg": 90.0},
                "source_type": "tele",
            }
        )
    assert driver._imu.gyro_z == pytest.approx(math.radians(90.0))
    assert driver._imu.gyro_x == pytest.approx(0.0)
    assert any("command received, rotating sensor" in r.message for r in caplog.records)


def test_rotate_zero_stops_gyro(imu_twin: SimpleNamespace) -> None:
    driver = FakeImu6dDriver(imu_twin)
    driver._on_rotate(
        {"command": "rotate", "data": {"axis": "roll", "amount_deg": 45.0}}
    )
    driver._on_rotate(
        {"command": "rotate", "data": {"axis": "roll", "amount_deg": 0.0}}
    )
    assert driver._imu.gyro_x == pytest.approx(0.0)
