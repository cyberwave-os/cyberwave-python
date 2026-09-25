"""State getters: pose/joints are level 0 (ungated); pointcloud gates via MUJOCO."""

from __future__ import annotations

import pytest

from cyberwave.twin.capabilities.joints import JointsHandle
from cyberwave.twin.capabilities.pose import PoseHandle
from cyberwave.twin.sensors.depth import DepthSensorHandle


class _Boom(Exception):
    pass


class _SentinelTwin:
    """Its _ensure_simulation_support raises, proving the getter calls it first."""

    def _ensure_simulation_support(self, level, *, method):
        raise _Boom()


def test_pose_get_is_level_0_ungated() -> None:
    # Level 0 methods carry no simulation-level tag and never preflight.
    assert not hasattr(PoseHandle.get, "__cw_sim_level__")


def test_joints_get_is_level_0_ungated() -> None:
    assert not hasattr(JointsHandle.get, "__cw_sim_level__")


def test_pointcloud_get_gates_via_ensure_simulation_support() -> None:
    handle = object.__new__(DepthSensorHandle)
    handle._twin = _SentinelTwin()
    with pytest.raises(_Boom):
        handle.get_pointcloud()
