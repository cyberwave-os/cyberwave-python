import asyncio
import math
from types import SimpleNamespace

import pytest

from cyberwave.compact_api import JointController


class FakeTwinsAPI:
    def __init__(self, *, kinematics=None, joint_states=None):
        self._kinematics = kinematics or {}
        self._joint_states = joint_states or {}
        self.calls = []

    async def get_kinematics(self, twin_uuid):
        await asyncio.sleep(0)
        self.calls.append(("get_kinematics", twin_uuid))
        return self._kinematics

    async def get_joint_states(self, twin_uuid):
        await asyncio.sleep(0)
        self.calls.append(("get_joint_states", twin_uuid))
        return self._joint_states

    async def set_joint(self, twin_uuid, joint_name, position):
        await asyncio.sleep(0)
        self.calls.append(("set_joint", twin_uuid, joint_name, position))
        return {"success": True}

    async def set_joints(self, twin_uuid, joint_states):
        await asyncio.sleep(0)
        self.calls.append(("set_joints", twin_uuid, joint_states))
        return {"success": True}


class DummyTwin:
    def __init__(self, *, uuid="twin-123", api=None, name="Test Twin"):
        self._twin_uuid = uuid
        self._client = SimpleNamespace(twins=api or FakeTwinsAPI())
        self.name = name

    async def _ensure_twin_exists(self):
        return None


def build_controller(kinematics, joint_states):
    api = FakeTwinsAPI(kinematics=kinematics, joint_states=joint_states)
    twin = DummyTwin(api=api)
    controller = JointController(twin)
    return controller, api


def test_initializes_aliases_and_cache():
    controller, api = build_controller(
        {"joints": [{"name": "Shoulder Pan"}, {"name": "wrist-joint"}]},
        {"name": ["Shoulder Pan", "wrist-joint"], "position": [10, -20]},
    )

    assert "shoulder_pan" in dir(controller)
    assert "wrist_joint" in dir(controller)
    assert controller.shoulder_pan == 10.0
    assert controller.wrist_joint == -20.0
    assert controller.all() == {"shoulder_pan": 10.0, "wrist_joint": -20.0}

    # Ensure bootstrap hit both endpoints
    assert ("get_kinematics", "twin-123") in api.calls
    assert ("get_joint_states", "twin-123") in api.calls


def test_setting_single_joint_updates_backend_and_cache():
    controller, api = build_controller(
        {"joints": [{"name": "shoulder_pan"}]},
        {"name": ["shoulder_pan"], "position": [0.0]},
    )

    controller.shoulder_pan = 15.5

    assert math.isclose(controller.shoulder_pan, 15.5)
    assert math.isclose(controller.all()["shoulder_pan"], 15.5)

    set_calls = [c for c in api.calls if c[0] == "set_joint"]
    assert set_calls
    _, twin_uuid, joint_name, value = set_calls[-1]
    assert twin_uuid == "twin-123"
    assert joint_name == "shoulder_pan"
    assert math.isclose(value, 15.5)


def test_set_many_uses_bulk_update():
    controller, api = build_controller(
        {"joints": [{"name": "shoulder_pan"}, {"name": "wrist_joint"}]},
        {"name": ["shoulder_pan", "wrist_joint"], "position": [1.0, 2.0]},
    )

    controller.set_many({"shoulder_pan": 5, "wrist_joint": -3})

    bulk_calls = [c for c in api.calls if c[0] == "set_joints"]
    assert bulk_calls
    _, twin_uuid, payload = bulk_calls[-1]
    assert twin_uuid == "twin-123"
    assert payload == {
        "shoulder_pan": {"position": 5.0},
        "wrist_joint": {"position": -3.0},
    }
    assert controller.all() == {"shoulder_pan": 5.0, "wrist_joint": -3.0}


def test_allows_dynamic_alias_when_no_kinematics():
    controller, api = build_controller({}, {})

    controller.joint_1 = 1.23

    assert controller.all() == {"joint_1": 1.23}
    # No backend call because alias unknown remotely
    assert not [c for c in api.calls if c[0] == "set_joint"]
