import asyncio
import math
from types import SimpleNamespace
from typing import Optional, Dict, Any

import pytest

import cyberwave
from cyberwave.twin_capabilities import SO101Twin


class StubTwinsAPI:
    def __init__(self):
        self.calls = []
        self.kin = {"joints": [{"name": "shoulder_pan"}, {"name": "elbow"}]}
        self.state = {"name": ["shoulder_pan", "elbow"], "position": [0.0, 0.0]}

    async def get_kinematics(self, twin_uuid):
        await asyncio.sleep(0)
        self.calls.append(("get_kinematics", twin_uuid))
        return self.kin

    async def get_joint_states(self, twin_uuid):
        await asyncio.sleep(0)
        self.calls.append(("get_joint_states", twin_uuid))
        return self.state

    async def set_joint(self, twin_uuid, joint_name, position):
        await asyncio.sleep(0)
        self.calls.append(("set_joint", twin_uuid, joint_name, position))
        idx = self.state["name"].index(joint_name)
        self.state["position"][idx] = position
        return {"success": True}

    async def set_joints(self, twin_uuid, joint_states):
        await asyncio.sleep(0)
        self.calls.append(("set_joints", twin_uuid, joint_states))
        for joint_name, payload in joint_states.items():
            if joint_name in self.state["name"]:
                idx = self.state["name"].index(joint_name)
                self.state["position"][idx] = payload.get("position", 0.0)
        return {"success": True}

    async def command(self, twin_uuid, name, payload):
        await asyncio.sleep(0)
        self.calls.append(("command", twin_uuid, name, payload))
        return {"success": True, "command": name, "payload": payload}


class StubClient:
    def __init__(self, twins: Optional[object] = None, uuid: str = "test-twin"):
        self.twins = twins or StubTwinsAPI()
        self.uuid = uuid

    def is_authenticated(self):
        return False


class DummySpec(SimpleNamespace):
    pass


@pytest.fixture(autouse=True)
def reset_globals(monkeypatch):
    import cyberwave.compact_api as compact_api

    compact_api._global_client = None
    compact_api._twin_registry.clear()
    yield
    compact_api._global_client = None
    compact_api._twin_registry.clear()


def setup_twin_environment(monkeypatch, stub_client: StubClient, spec_map: Dict[str, Any]):
    import cyberwave.compact_api as compact_api

    async def fake_ensure(self):
        self._twin_uuid = getattr(self, "_twin_uuid", None) or getattr(stub_client, "uuid", "test-twin")
        return None

    monkeypatch.setattr(compact_api, "_global_client", stub_client, raising=False)
    monkeypatch.setattr(compact_api, "_get_client", lambda: stub_client)
    monkeypatch.setattr(compact_api.CompactTwin, "_ensure_twin_exists", fake_ensure, raising=False)

    def fake_get(device_id: str):
        return spec_map.get(device_id)

    monkeypatch.setattr(
        "cyberwave.device_specs.registry.DeviceSpecRegistry.get",
        staticmethod(fake_get),
        raising=False,
    )


def make_capability(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def test_so101_twin_factory_and_capabilities(monkeypatch):
    spec_map = {
        "cyberwave/so101": DummySpec(asset_class="cyberwave.assets.SO101Robot"),
    }
    stub_client = StubClient()
    setup_twin_environment(monkeypatch, stub_client, spec_map)

    twin = cyberwave.twin("cyberwave/so101")

    assert isinstance(twin, SO101Twin)

    # move joints should delegate to joint controller and twin API
    twin.move_joints({"shoulder_pan": 15.0, "elbow": -5.0})
    assert any(
        call[0] == "set_joints" and call[1] == "test-twin" and call[2]["shoulder_pan"]["position"] == 15.0
        for call in stub_client.twins.calls
    )
    assert twin.joints_live()["shoulder_pan"] == 15.0

    labels = twin.joint_labels()
    assert labels[6] == "gripper"
    twin.grip_percent(80)
    assert math.isclose(twin.joints.get("gripper"), 80.0, rel_tol=1e-3)

    # gripper command
    twin.open_gripper()
    assert any(call for call in stub_client.twins.calls if call[:3] == ("command", "test-twin", "open_gripper"))

    # scripted pick routine
    result = twin.pick({"x": 0.4, "y": 0.1, "z": 0.2}, {"x": 0.4, "y": 0.1, "z": 0.0})
    assert result["success"] is True
    assert any(call for call in stub_client.twins.calls if call[:3] == ("command", "test-twin", "close_gripper"))


def test_tello_twin_exposes_flight_mixins(monkeypatch):
    drone_spec = DummySpec(
        id="dji/tello",
        name="DJI Tello",
        category="drone",
        model="Tello",
        capabilities=[make_capability("flight"), make_capability("video_streaming")],
        asset_class=None,
    )
    spec_map = {"dji/tello": drone_spec}
    stub_client = StubClient(uuid="tello-twin")
    setup_twin_environment(monkeypatch, stub_client, spec_map)

    twin = cyberwave.twin("dji/tello")

    assert hasattr(twin, "takeoff") and hasattr(twin, "start_video_stream")

    twin.takeoff()
    assert ("command", "tello-twin", "takeoff", {}) in stub_client.twins.calls

    twin.navigate_to(100, 0, 50, speed_cm_s=40)
    assert any(
        call[0] == "command" and call[1] == "tello-twin" and call[2] == "go"
        for call in stub_client.twins.calls
    )

    twin.start_video_stream()
    assert ("command", "tello-twin", "streamon", {}) in stub_client.twins.calls


def test_spot_twin_exposes_quadruped_mixins(monkeypatch):
    spot_spec = DummySpec(
        id="boston-dynamics/spot",
        name="Spot",
        category="quadruped",
        model="Spot",
        capabilities=[make_capability("mobility"), make_capability("perception")],
        asset_class=None,
    )
    spec_map = {"boston-dynamics/spot": spot_spec}
    stub_client = StubClient(uuid="spot-twin")
    setup_twin_environment(monkeypatch, stub_client, spec_map)

    twin = cyberwave.twin("boston-dynamics/spot")

    assert hasattr(twin, "walk") and hasattr(twin, "navigate_to")

    twin.walk(0.8)
    assert ("command", "spot-twin", "walk", {"speed": 0.8}) in stub_client.twins.calls

    twin.navigate_to({"x": 1.0, "y": 0.0})
    assert ("command", "spot-twin", "navigate_to", {"x": 1.0, "y": 0.0}) in stub_client.twins.calls
