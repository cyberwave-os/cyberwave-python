"""JointTwin and joints handle command tests."""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from cyberwave.twin.capabilities import joints as _joints

import pytest

from cyberwave.twin import Twin
from cyberwave.twin.classes import JointTwin, LocomoteJointTwin
from cyberwave.twin.factory import create_twin, _is_joint_manipulator


def test_base_twin_has_no_joints_property() -> None:
    assert getattr(Twin, "joints", None) is None


def test_joint_controller_import_removed() -> None:
    import importlib

    twin_pkg = importlib.import_module("cyberwave.twin")
    assert "JointController" not in twin_pkg.__all__


def test_go2_twin_is_not_joint_manipulator_but_inherits_joint_twin() -> None:
    """A legged robot with joints is not a manipulator (no gripper), but its
    twin must still expose ``.joints``. We achieve this with a dedicated
    ``LocomoteJointTwin`` subtree that mixes ``LocomoteTwin`` and ``JointTwin``
    so wheeled AMRs (no joints) keep the lean ``LocomoteTwin`` interface."""
    caps = {"has_joints": True, "can_locomote": True, "can_fly": False, "can_grip": False}
    assert not _is_joint_manipulator(caps)
    assert create_twin.__module__
    from cyberwave.twin.factory import _select_twin_class

    cls = _select_twin_class(caps)
    assert cls is LocomoteJointTwin
    assert issubclass(cls, JointTwin)


def test_joint_twin_set_publishes_joint_update() -> None:
    client = SimpleNamespace(
        mqtt=MagicMock(),
        assets=MagicMock(),
        config=SimpleNamespace(runtime_mode="live", topic_prefix="", source_type="tele"),
        twins=SimpleNamespace(api=None),
    )
    client.assets.get.return_value = SimpleNamespace(
        metadata={
            "mqtt": {
                "topics": {
                    "cyberwave/joint/{twin_uuid}/update": {},
                    "cyberwave/twin/{twin_uuid}/command": {},
                },
                "commands": {"supported": []},
            }
        }
    )
    twin = JointTwin(
        client, SimpleNamespace(uuid="arm-1", name="Arm", asset_uuid="asset-1")
    )
    with patch.object(_joints, "controllable_joint_names", return_value=["j1"]):
        with patch.object(twin, "_prepare_outbound_command"):
            twin.joints.set({"j1": 90.0}, degrees=True)
    assert len(twin._outbound_log) == 1
    assert twin._outbound_log[0].command == "joint_update"
    payload = twin._outbound_log[0].payload
    assert "command" not in payload
    assert payload["source_type"] == "tele"
    assert "j1" in payload
    client.mqtt.publish.assert_called_once()


def test_joints_get_default_and_subset() -> None:
    assets = MagicMock()
    assets.get.return_value = SimpleNamespace(
        metadata={
            "mqtt": {
                "topics": {
                    "cyberwave/joint/{twin_uuid}/update": {},
                    "cyberwave/twin/{twin_uuid}/command": {},
                },
                "commands": {"supported": []},
            }
        }
    )
    client = SimpleNamespace(
        mqtt=MagicMock(),
        assets=assets,
        config=SimpleNamespace(runtime_mode="live", topic_prefix="", source_type="tele"),
        twins=SimpleNamespace(api=None),
    )
    twin = JointTwin(
        client, SimpleNamespace(uuid="arm-1", name="Arm", asset_uuid="asset-1")
    )
    with patch.object(_joints, "controllable_joint_names", return_value=["j1", "j2"]):
        with patch.object(twin, "_prepare_outbound_command"):
            twin.joints.set({"j1": 1.0, "j2": 2.0})
        all_pos = twin.joints.get()
        assert all_pos == {"j1": 1.0, "j2": 2.0}
        one = twin.joints.get(what_joints=["j2"])
        assert one == {"j2": 2.0}
        multi = twin.joints.get(what_data=["position", "velocity"])
        assert multi["position"]["j1"] == 1.0
        assert multi["velocity"]["j2"] == 0.0


def test_joints_list_matches_controllable_names() -> None:
    client = SimpleNamespace(twins=SimpleNamespace())
    twin = JointTwin(client, SimpleNamespace(uuid="arm-1", name="Arm"))
    with patch.object(_joints, "controllable_joint_names", return_value=["a", "b"]):
        assert twin.joints.list() == ["a", "b"]
