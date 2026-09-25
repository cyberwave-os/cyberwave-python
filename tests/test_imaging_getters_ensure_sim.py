"""Imaging getters require a running MuJoCo simulation (SimLevel.MUJOCO)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cyberwave.exceptions import SimulationNotRunningError
from cyberwave.twin.sensors.camera import TwinCameraHandle
from cyberwave.twin.sensors.pointcloud import PointCloudCapableMixin
from cyberwave.twin.simulation_support import SimLevel


def test_camera_getters_are_mujoco_level() -> None:
    # get_frame validates its source arg before the preflight, so it calls
    # _ensure_simulation_support(MUJOCO) manually rather than via the decorator.
    assert TwinCameraHandle.get_frames.__cw_sim_level__ == SimLevel.MUJOCO
    assert TwinCameraHandle.get_video.__cw_sim_level__ == SimLevel.MUJOCO


def test_pointcloud_getter_is_mujoco_level() -> None:
    assert PointCloudCapableMixin.get_pointcloud.__cw_sim_level__ == SimLevel.MUJOCO


def test_camera_get_frame_raises_when_no_sim_running() -> None:
    twin = SimpleNamespace(
        client=SimpleNamespace(
            config=SimpleNamespace(runtime_mode="simulation"),
            environments=SimpleNamespace(
                simulations=SimpleNamespace(get_active=lambda env: None)
            ),
        ),
        environment_id="env-1",
    )
    # Bind the real _ensure_simulation_support so the decorator's check runs.
    from cyberwave.twin.base import Twin

    twin._ensure_simulation_support = Twin._ensure_simulation_support.__get__(twin)
    handle = object.__new__(TwinCameraHandle)
    handle._twin = twin
    with pytest.raises(SimulationNotRunningError):
        handle.get_frame()
