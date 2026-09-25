"""IMU/GPS/compass getters raise NotSimulatedError in simulation mode (SimLevel.UNSUPPORTED)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cyberwave.exceptions import NotSimulatedError
from cyberwave.twin.base import Twin
from cyberwave.twin.sensors.compass import CompassSensorHandle
from cyberwave.twin.sensors.gps import GpsSensorHandle
from cyberwave.twin.sensors.imu import ImuSensorHandle
from cyberwave.twin.simulation_support import SimLevel


def _twin(runtime_mode):
    twin = SimpleNamespace(
        client=SimpleNamespace(config=SimpleNamespace(runtime_mode=runtime_mode)),
        capabilities={"sensors": []},
    )
    twin._ensure_simulation_support = Twin._ensure_simulation_support.__get__(twin)
    return twin


def test_levels_are_unsupported() -> None:
    assert ImuSensorHandle.get.__cw_sim_level__ == SimLevel.UNSUPPORTED
    assert GpsSensorHandle.get_fix.__cw_sim_level__ == SimLevel.UNSUPPORTED
    assert CompassSensorHandle.get_heading.__cw_sim_level__ == SimLevel.UNSUPPORTED


def test_imu_get_raises_in_sim_mode() -> None:
    with pytest.raises(NotSimulatedError):
        ImuSensorHandle(_twin("simulation"), "imu0").get()


def test_gps_get_fix_raises_in_sim_mode() -> None:
    with pytest.raises(NotSimulatedError):
        GpsSensorHandle(_twin("simulation"), "gps0").get_fix()


def test_compass_get_heading_raises_in_sim_mode() -> None:
    with pytest.raises(NotSimulatedError):
        CompassSensorHandle(_twin("simulation"), "compass0").get_heading()


def test_compass_get_heading_not_implemented_in_live_mode() -> None:
    with pytest.raises(NotImplementedError):
        CompassSensorHandle(_twin("live"), "compass0").get_heading()
