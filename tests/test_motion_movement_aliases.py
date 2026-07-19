import json
from unittest.mock import MagicMock

from cyberwave.motion import TwinMotionHandle
from cyberwave.twin import Twin


class FakeResponse:
    def __init__(self, payload):
        self.data = json.dumps(payload).encode("utf-8")

    def read(self):
        return None


class FakeApiClient:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.serialized = []

    def param_serialize(self, **kwargs):
        self.serialized.append(kwargs)
        return (kwargs,)

    def call_api(self, *_args):
        payload = self.payloads.pop(0)
        return FakeResponse(payload)


class FakeApi:
    def __init__(self, api_client):
        self.api_client = api_client


class FakeClient:
    def __init__(self, api_client):
        self.api = FakeApi(api_client)


class FakeTwin:
    uuid = "twin-uuid"

    def __init__(self, api_client):
        self.client = FakeClient(api_client)


def test_run_movement_posts_existing_animation_action_payload():
    api_client = FakeApiClient({"action_id": "action-1", "status": "queued"})
    motion = TwinMotionHandle(FakeTwin(api_client))

    result = motion.run_movement(
        "Wave",
        scope="asset",
        environment_uuid="env-uuid",
        source_type="sim",
        transition_ms=500,
    )

    assert result["status"] == "queued"
    assert api_client.serialized[0]["method"] == "POST"
    assert api_client.serialized[0]["resource_path"] == (
        "/api/v1/twins/twin-uuid/actions"
    )
    assert api_client.serialized[0]["body"] == {
        "action_type": "movement",
        "name": "Wave",
        "scope": "asset",
        "execution": "async",
        "preview": False,
        "environment_uuid": "env-uuid",
        "source_type": "sim",
        "transition_ms": 500,
    }


def test_move_to_pose_posts_existing_pose_action_payload():
    api_client = FakeApiClient({"action_id": "action-1", "status": "queued"})
    motion = TwinMotionHandle(FakeTwin(api_client))

    motion.move_to_pose("Stand", scope="asset", environment_uuid="env-uuid")

    assert api_client.serialized[0]["body"] == {
        "action_type": "pose",
        "scope": "asset",
        "execution": "async",
        "preview": False,
        "name": "Stand",
        "environment_uuid": "env-uuid",
    }


def test_list_movements_returns_existing_animations_list():
    api_client = FakeApiClient(
        {
            "keyframes": [],
            "animations": [
                {"name": "Asset Wave", "scope": "asset"},
                {"name": "Twin Wave", "scope": "twin"},
            ],
        }
    )
    motion = TwinMotionHandle(FakeTwin(api_client))

    assert motion.list_movements(scope="asset") == [
        {"name": "Asset Wave", "scope": "asset"}
    ]
    assert api_client.serialized[0]["method"] == "GET"
    assert api_client.serialized[0]["resource_path"] == (
        "/api/v1/twins/twin-uuid/motions"
    )


def test_twin_exposes_top_level_movement_aliases():
    api_client = FakeApiClient({"action_id": "action-1", "status": "queued"})
    twin = Twin(FakeClient(api_client), {"uuid": "twin-uuid", "name": "Go2"})

    twin.run_movement("Wave", scope="asset")

    assert api_client.serialized[0]["body"]["action_type"] == "movement"
    assert api_client.serialized[0]["body"]["name"] == "Wave"


def test_motion_handle_run_movement_defaults_to_auto_scope():
    api_client = FakeApiClient({"action_id": "action-1", "status": "queued"})
    motion = TwinMotionHandle(FakeTwin(api_client))

    motion.run_movement("Wave")

    assert api_client.serialized[0]["body"]["scope"] == "auto"


def test_motion_handle_move_to_pose_defaults_to_auto_scope():
    api_client = FakeApiClient({"action_id": "action-1", "status": "queued"})
    motion = TwinMotionHandle(FakeTwin(api_client))

    motion.move_to_pose("Stand")

    assert api_client.serialized[0]["body"]["scope"] == "auto"


def test_pose_with_joints_uses_joint_mqtt_path_instead_of_actions_endpoint():
    api_client = FakeApiClient({"unused": True})
    twin = FakeTwin(api_client)
    twin.joints = MagicMock()
    motion = TwinMotionHandle(twin)

    result = motion.pose(joints={"joint_1": 0.4}, source_type="sim")

    twin.joints.set.assert_called_once_with({"joint_1": 0.4}, source_type="sim")
    assert api_client.serialized == []
    assert result == {
        "status": "published",
        "transport": "mqtt",
        "command": "joint_update",
    }


def test_pose_with_joints_requires_joint_capable_twin():
    api_client = FakeApiClient({"unused": True})
    motion = TwinMotionHandle(FakeTwin(api_client))

    try:
        motion.pose(joints={"joint_1": 0.4})
        raise AssertionError("Expected ValueError for twin without joints handle")
    except ValueError as exc:
        assert "joint-capable twin" in str(exc)


def test_motion_handle_list_movements_defaults_to_auto_scope():
    api_client = FakeApiClient(
        {
            "keyframes": [],
            "animations": [
                {"name": "Asset Wave", "scope": "asset"},
                {"name": "Twin Wave", "scope": "twin"},
                {"name": "Env Wave", "scope": "environment"},
            ],
        }
    )
    motion = TwinMotionHandle(FakeTwin(api_client))

    movements = motion.list_movements()

    assert movements == [
        {"name": "Asset Wave", "scope": "asset"},
        {"name": "Twin Wave", "scope": "twin"},
        {"name": "Env Wave", "scope": "environment"},
    ]


def test_twin_run_movement_defaults_to_auto_scope():
    api_client = FakeApiClient({"action_id": "action-1", "status": "queued"})
    twin = Twin(FakeClient(api_client), {"uuid": "twin-uuid", "name": "Go2"})

    twin.run_movement("Wave")

    assert api_client.serialized[0]["body"]["scope"] == "auto"


def test_twin_move_to_pose_defaults_to_auto_scope():
    api_client = FakeApiClient({"action_id": "action-1", "status": "queued"})
    twin = Twin(FakeClient(api_client), {"uuid": "twin-uuid", "name": "Go2"})

    twin.move_to_pose("Stand")

    assert api_client.serialized[0]["body"]["scope"] == "auto"
