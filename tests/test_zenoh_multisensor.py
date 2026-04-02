"""Multi-sensor isolation tests — parametrized over FilesystemBackend and ZenohBackend.

Verifies that ``DataBus(backend, twin_uuid, sensor_name=...)`` namespaces
channels correctly so that left-sensor data never bleeds into right-sensor
channels and vice-versa.

The test file is parametrized the same way as
``test_data_backend_contract.py`` so coverage stays consistent across both
backend implementations.
"""

from __future__ import annotations

import threading
import time

import pytest

from cyberwave.data.api import DataBus
from cyberwave.data.config import BackendConfig, get_backend
from cyberwave.data.keys import build_key

try:
    import zenoh  # noqa: F401

    _has_zenoh = True
except ImportError:
    _has_zenoh = False

BACKEND_PARAMS = ["filesystem"]
if _has_zenoh:
    BACKEND_PARAMS.append("zenoh")

TWIN_UUID = "cccccccc-0000-4000-c000-000000000003"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=BACKEND_PARAMS)
def backend(request, tmp_path):
    """Yield a fresh DataBackend for each parametrized backend type."""
    name = request.param
    if name == "filesystem":
        cfg = BackendConfig(
            backend="filesystem",
            filesystem_base_dir=str(tmp_path / "data"),
            filesystem_ring_buffer_size=50,
        )
    else:
        cfg = BackendConfig(backend="zenoh")
    be = get_backend(cfg)
    yield be
    be.close()


# ---------------------------------------------------------------------------
# TestSensorNameIsolation
# ---------------------------------------------------------------------------


class TestSensorNameIsolation:
    """Publishing on ``sensor_name="left"`` must not appear on ``"right"``."""

    def test_publish_does_not_bleed_across_sensors(self, backend) -> None:
        bus_left = DataBus(backend, TWIN_UUID, sensor_name="left")
        bus_right = DataBus(backend, TWIN_UUID, sensor_name="right")

        received_right: list[object] = []
        received_left: list[object] = []
        got_left = threading.Event()

        def cb_left(v: object) -> None:
            received_left.append(v)
            got_left.set()

        def cb_right(v: object) -> None:
            received_right.append(v)

        sub_left = bus_left.subscribe("frames", cb_left, policy="fifo")
        sub_right = bus_right.subscribe("frames", cb_right, policy="fifo")
        time.sleep(0.2)

        bus_left.publish("frames", {"sensor": "left", "value": 42})
        assert got_left.wait(timeout=3.0), "left subscriber never received the message"

        # Give the right subscriber a window to erroneously receive the message.
        time.sleep(0.3)

        sub_left.close()
        sub_right.close()
        bus_left.close()
        bus_right.close()

        assert len(received_left) == 1
        assert len(received_right) == 0, (
            f"right sensor received {len(received_right)} message(s) "
            "that were published on the left sensor channel"
        )

    def test_latest_is_isolated_per_sensor(self, backend) -> None:
        bus_left = DataBus(backend, TWIN_UUID, sensor_name="left")
        bus_right = DataBus(backend, TWIN_UUID, sensor_name="right")

        bus_left.publish("joint_states", {"j1": 1.0, "j2": 2.0})
        bus_right.publish("joint_states", {"j1": 99.0, "j2": 99.0})
        time.sleep(0.4)

        sample_left = bus_left.latest("joint_states")
        sample_right = bus_right.latest("joint_states")

        bus_left.close()
        bus_right.close()

        assert sample_left is not None, "left latest() returned None"
        assert sample_right is not None, "right latest() returned None"
        assert sample_left["j1"] == pytest.approx(1.0)  # type: ignore[index]
        assert sample_right["j1"] == pytest.approx(99.0)  # type: ignore[index]

    def test_keys_include_sensor_name(self) -> None:
        left_key = build_key(TWIN_UUID, "frames", "left")
        right_key = build_key(TWIN_UUID, "frames", "right")

        assert left_key == f"cw/{TWIN_UUID}/data/frames/left"
        assert right_key == f"cw/{TWIN_UUID}/data/frames/right"
        assert left_key != right_key

    def test_multiple_sensors_coexist(self, backend) -> None:
        """Three sensors publish on the same channel; each gets independent latest."""
        buses = {
            name: DataBus(backend, TWIN_UUID, sensor_name=name)
            for name in ("cam0", "cam1", "cam2")
        }

        for idx, (name, bus) in enumerate(buses.items()):
            bus.publish("frames", {"sensor": name, "idx": idx})

        time.sleep(0.4)

        for idx, (name, bus) in enumerate(buses.items()):
            sample = bus.latest("frames")
            assert sample is not None, f"latest() is None for sensor '{name}'"
            assert sample["sensor"] == name  # type: ignore[index]
            assert sample["idx"] == idx  # type: ignore[index]

        for bus in buses.values():
            bus.close()
