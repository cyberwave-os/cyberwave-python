"""Capability-scoped handles: only present on the matching twin subclass."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.twin_patch import patch_twin

from cyberwave.twin import CameraTwin, LocomoteTwin, create_twin
from cyberwave.twin.classes import FlyingTwin, GripperJointTwin, JointTwin, LocomoteCameraTwin


def test_camera_twin_exposes_camera_not_locomotion() -> None:
    twin = CameraTwin(
        SimpleNamespace(twins=SimpleNamespace()),
        SimpleNamespace(
            uuid="cam",
            name="Webcam",
            capabilities={
                "can_locomote": False,
                "can_fly": False,
                "can_grip": False,
                "has_joints": False,
                "sensors": [{"id": "color_camera", "type": "rgb", "name": "color_camera"}],
            },
        ),
    )
    assert hasattr(twin, "camera")
    assert hasattr(twin, "get_frame")
    assert not hasattr(twin, "locomotion")
    assert not hasattr(twin, "flight")
    assert not hasattr(twin, "gripper")
    assert not hasattr(twin, "joints")
    assert not hasattr(twin, "position")
    assert not hasattr(twin, "rotation")
    assert not hasattr(twin, "policy")


def test_locomote_twin_exposes_locomotion_not_camera() -> None:
    twin = LocomoteTwin(
        SimpleNamespace(twins=SimpleNamespace()),
        SimpleNamespace(
            uuid="go2",
            name="Go2",
            capabilities={"can_locomote": True, "sensors": []},
        ),
    )
    assert hasattr(twin, "locomotion")
    assert hasattr(twin, "policy")
    assert hasattr(twin, "get_pose")
    assert hasattr(twin, "set_pose")
    assert not hasattr(twin, "camera")
    assert not hasattr(twin, "get_frame")
    assert not hasattr(twin, "position")
    assert not hasattr(twin, "rotation")


def test_factory_returns_camera_twin_for_sensor_only_asset() -> None:
    client = SimpleNamespace(twins=SimpleNamespace())
    twin = create_twin(
        client,
        SimpleNamespace(
            uuid="cam",
            name="Webcam",
            capabilities={
                "sensors": [{"id": "c", "type": "rgb"}],
            },
        ),
    )
    assert type(twin) is CameraTwin


def test_flying_twin_has_flight_and_locomotion() -> None:
    twin = FlyingTwin(
        SimpleNamespace(twins=SimpleNamespace()),
        SimpleNamespace(
            uuid="drone",
            name="Drone",
            capabilities={"can_fly": True, "can_locomote": True},
        ),
    )
    assert hasattr(twin, "flight")
    assert hasattr(twin, "locomotion")


def test_joint_twin_get_pose_set_pose() -> None:
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
        client,
        SimpleNamespace(uuid="arm", name="SO101", asset_uuid="asset-1"),
    )
    with patch_twin("capabilities.joints.controllable_joint_names", return_value=["_1", "_2"]):
        with patch.object(twin, "_prepare_outbound_command"):
            twin.set_pose({"_1": -1.5, "_2": 1.5})
        assert twin.get_joints()["_1"] != 0.0
        assert twin.get_pose() == twin.get_joints()


def test_joints_index_access_uses_list_order() -> None:
    assets = MagicMock()
    assets.get.return_value = SimpleNamespace(
        metadata={
            "mqtt": {
                "topics": {"cyberwave/joint/{twin_uuid}/update": {}},
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
        client,
        SimpleNamespace(uuid="arm", name="Arm", asset_uuid="asset-1"),
    )
    with patch(
        "cyberwave.twin.capabilities.joints.controllable_joint_names",
        return_value=["alpha", "beta"],
    ):
        assert twin.joints.list() == ["alpha", "beta"]
        with patch.object(twin, "_prepare_outbound_command"):
            twin.joints[1] = 0.25
        assert twin.joints[1] == 0.25
        assert twin.get_joints()["beta"] == 0.25


def test_locomote_camera_twin_exposes_lidar_and_camera_for_go2() -> None:
    twin = LocomoteCameraTwin(
        SimpleNamespace(twins=SimpleNamespace()),
        SimpleNamespace(
            uuid="go2",
            name="Go2",
            capabilities={
                "can_locomote": True,
                "has_joints": True,
                "sensors": [
                    {
                        "id": "lidar_4d",
                        "name": "lidar_4d",
                        "type": "lidar_4d",
                    },
                    {
                        "id": "front_camera",
                        "name": "front_camera",
                        "type": "rgb",
                    },
                ],
            },
        ),
    )
    assert hasattr(twin, "lidar")
    assert twin.resolve_handler_from_capabilities("lidar").available
    assert twin.lidar.sensor_id == "lidar_4d"
    assert twin.lidar.metadata()["type"] == "lidar_4d"
    assert twin.default_camera_name == "front_camera"
    assert not twin.resolve_handler_from_capabilities("lidar").multi_sensor
    assert not twin.resolve_handler_from_capabilities("camera").multi_sensor


def test_gripper_joint_twin_has_gripper_and_joints() -> None:
    twin = GripperJointTwin(
        SimpleNamespace(twins=SimpleNamespace()),
        SimpleNamespace(
            uuid="arm",
            name="SO101",
            capabilities={"can_grip": True, "has_joints": True, "can_locomote": False},
        ),
    )
    assert hasattr(twin, "gripper")
    assert hasattr(twin, "joints")
    assert not hasattr(twin, "locomotion")
