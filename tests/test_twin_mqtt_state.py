"""PR3 MQTT inbound: topic listeners and transport planes."""

from __future__ import annotations

import threading
import time
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from cyberwave.twin.capabilities import joints as _joints

import pytest

from cyberwave.data.state_representation import CartesianPose
from cyberwave.exceptions import TwinStateTimeoutError
from cyberwave.manifest.driver_config import (
    JOINT_UPDATE_TOPIC_SLUG,
    TWIN_COMMAND_TOPIC_SLUG,
    TWIN_POSITION_TOPIC_SLUG,
    TWIN_ROTATION_TOPIC_SLUG,
)
from cyberwave.twin import LocomoteTwin
from cyberwave.twin.classes import JointTwin


def _fake_mqtt() -> MagicMock:
    subs: dict[str, object] = {}

    def _subscribe(topic: str, callback: object, **kwargs: object) -> None:
        subs[topic] = callback

    mqtt = MagicMock()
    mqtt.connected = True
    mqtt.subscribe = MagicMock(side_effect=_subscribe)
    mqtt._subs = subs
    return mqtt


def _locomote_metadata() -> dict:
    return {
        "mqtt": {
            "topics": {
                TWIN_COMMAND_TOPIC_SLUG: {},
                TWIN_POSITION_TOPIC_SLUG: {"direction": "publish"},
                TWIN_ROTATION_TOPIC_SLUG: {"direction": "publish"},
            },
            "commands": {
                "supported": [
                    "move_forward",
                    "move_backward",
                    "turn_left",
                    "turn_right",
                    "stop",
                    "move",
                ]
            },
        }
    }


def _make_locomote_twin(*, metadata: dict | None = None) -> LocomoteTwin:
    mqtt = _fake_mqtt()
    assets = MagicMock()
    assets.get.return_value = SimpleNamespace(metadata=metadata or _locomote_metadata())
    _active_sim = SimpleNamespace(
        simulation_id="sim-1", status="running", raw={}, total_duration_s=None
    )
    client = SimpleNamespace(
        mqtt=mqtt,
        assets=assets,
        config=SimpleNamespace(runtime_mode="live", source_type="tele", topic_prefix=""),
        twins=SimpleNamespace(api=None, update_state=MagicMock(), get_raw=MagicMock()),
        environments=SimpleNamespace(
            simulations=SimpleNamespace(get_active=lambda env: _active_sim)
        ),
    )
    return LocomoteTwin(
        client,
        SimpleNamespace(
            uuid="twin-uuid",
            name="Go2",
            asset_uuid="asset-uuid",
            environment_uuid="env-1",
            capabilities={"can_locomote": True},
            position_x=0.0,
            position_y=0.0,
            position_z=0.0,
            rotation_w=1.0,
            rotation_x=0.0,
            rotation_y=0.0,
            rotation_z=0.0,
        ),
    )


def _inject(mqtt: MagicMock, topic: str, payload: dict) -> None:
    mqtt._subs[topic](payload)


def _read_pose_async(twin: LocomoteTwin) -> tuple[list[CartesianPose], threading.Thread]:
    result: list[CartesianPose] = []
    thread = threading.Thread(target=lambda: result.append(twin.pose.get(timeout=3.0)))
    thread.start()
    time.sleep(0.05)
    return result, thread


def test_pose_get_single_canonical_state() -> None:
    twin = _make_locomote_twin()
    pos_topic = TWIN_POSITION_TOPIC_SLUG.format(twin_uuid=twin.uuid)
    rot_topic = TWIN_ROTATION_TOPIC_SLUG.format(twin_uuid=twin.uuid)
    result, thread = _read_pose_async(twin)
    _inject(
        twin.client.mqtt,
        pos_topic,
        {"type": "position", "position": {"x": 1.0, "y": 2.0, "z": 3.0}},
    )
    _inject(
        twin.client.mqtt,
        rot_topic,
        {"type": "rotation", "rotation": {"x": 0.0, "y": 0.0, "z": 0.1, "w": 0.99}},
    )
    thread.join(timeout=2.0)
    pose = result[0]
    assert pose.position.x == 1.0
    assert pose.position.y == 2.0
    assert pose.orientation.w == 0.99


def test_pose_get_returns_same_live_view() -> None:
    twin = _make_locomote_twin()
    pos_topic = TWIN_POSITION_TOPIC_SLUG.format(twin_uuid=twin.uuid)
    rot_topic = TWIN_ROTATION_TOPIC_SLUG.format(twin_uuid=twin.uuid)
    result, thread = _read_pose_async(twin)
    _inject(twin.client.mqtt, pos_topic, {"position": {"x": 1.0, "y": 0.0, "z": 0.0}})
    _inject(twin.client.mqtt, rot_topic, {"rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}})
    thread.join(timeout=2.0)

    first = twin.pose.get(timeout=0.0)
    second = twin.pose.get(timeout=0.0)
    assert first is second  # cached live view per mode
    assert first.position.x == 1.0


def test_pose_get_first_read_waits_for_mqtt_message() -> None:
    twin = _make_locomote_twin()
    pos_topic = TWIN_POSITION_TOPIC_SLUG.format(twin_uuid=twin.uuid)
    result, thread = _read_pose_async(twin)
    _inject(twin.client.mqtt, pos_topic, {"position": {"x": 4.0, "y": 5.0, "z": 6.0}})
    thread.join(timeout=2.0)
    assert result[0].position.x == 4.0


def test_pose_get_first_read_timeout_returns_empty_view() -> None:
    twin = _make_locomote_twin()
    view = twin.pose.get(timeout=0.05)
    assert view.pose is None
    assert view.position is None


def test_pose_get_returns_empty_view_without_mqtt_transport() -> None:
    twin = _make_locomote_twin()
    twin.client.mqtt = None
    view = twin.pose.get(timeout=0.05)
    assert view.pose is None
    assert twin.get_pose() is None
    assert view.frame_id() is None


def test_pose_view_auto_refreshes_and_fires_callback() -> None:
    twin = _make_locomote_twin()
    pos_topic = TWIN_POSITION_TOPIC_SLUG.format(twin_uuid=twin.uuid)
    view = twin.pose.get(timeout=0.0)
    seen: list[object] = []
    view.on_update(lambda p: seen.append(p))
    _inject(twin.client.mqtt, pos_topic, {"position": {"x": 5.0, "y": 0.0, "z": 0.0}})
    assert view.position.x == 5.0  # same object refreshed in place
    assert seen and seen[-1].position.x == 5.0


def test_pose_get_does_not_call_prepare_outbound_command() -> None:
    twin = _make_locomote_twin()
    with patch.object(twin, "_prepare_outbound_command") as gate:
        pos_topic = TWIN_POSITION_TOPIC_SLUG.format(twin_uuid=twin.uuid)
        result, thread = _read_pose_async(twin)
        _inject(twin.client.mqtt, pos_topic, {"position": {"x": 0.0, "y": 0.0, "z": 0.0}})
        thread.join(timeout=2.0)
        result[0]
    gate.assert_not_called()


def test_pose_set_raises_not_implemented() -> None:
    twin = _make_locomote_twin()
    with patch.object(twin, "_prepare_outbound_command") as gate:
        with pytest.raises(NotImplementedError, match="pose.set"):
            twin.pose.set(x=1.0)
    gate.assert_not_called()
    twin.client.mqtt.publish.assert_not_called()


def test_set_pose_does_not_call_edit_or_rest() -> None:
    twin = _make_locomote_twin()
    with patch.object(twin, "edit_position") as edit_pos:
        with patch.object(twin, "edit_rotation") as edit_rot:
            with patch.object(twin, "_update_state") as update_state:
                with pytest.raises(NotImplementedError):
                    twin.set_pose(x=1.0, y=2.0)
    edit_pos.assert_not_called()
    edit_rot.assert_not_called()
    update_state.assert_not_called()


def test_get_pose_uses_mqtt_not_rest_refresh() -> None:
    twin = _make_locomote_twin()
    with patch.object(twin, "refresh") as refresh:
        pos_topic = TWIN_POSITION_TOPIC_SLUG.format(twin_uuid=twin.uuid)
        rot_topic = TWIN_ROTATION_TOPIC_SLUG.format(twin_uuid=twin.uuid)
        result, thread = _read_pose_async(twin)
        _inject(twin.client.mqtt, pos_topic, {"position": {"x": 7.0, "y": 8.0, "z": 9.0}})
        _inject(twin.client.mqtt, rot_topic, {"rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}})
        thread.join(timeout=2.0)
        pose = twin.get_pose()
    refresh.assert_not_called()
    assert pose["position"]["x"] == 7.0


def test_edit_position_does_not_publish_mqtt() -> None:
    twin = _make_locomote_twin()
    twin.edit_position(x=1.0, y=2.0, z=3.0)
    twin.client.mqtt.publish.assert_not_called()
    twin.client.twins.update_state.assert_called_once()


def test_subscribe_position_emits_deprecation() -> None:
    twin = _make_locomote_twin()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        twin.subscribe_position(lambda _: None)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def _make_joint_twin(*, metadata: dict) -> JointTwin:
    mqtt = _fake_mqtt()
    assets = MagicMock()
    assets.get.return_value = SimpleNamespace(metadata=metadata)
    _active_sim = SimpleNamespace(
        simulation_id="sim-1", status="running", raw={}, total_duration_s=None
    )
    client = SimpleNamespace(
        mqtt=mqtt,
        assets=assets,
        config=SimpleNamespace(runtime_mode="live", topic_prefix="", source_type="tele"),
        twins=SimpleNamespace(api=None),
        environments=SimpleNamespace(
            simulations=SimpleNamespace(get_active=lambda env: _active_sim)
        ),
    )
    return JointTwin(
        client,
        SimpleNamespace(
            uuid="arm-1",
            name="Arm",
            asset_uuid="a",
            environment_uuid="env-1",
            capabilities={
                "has_joints": True,
                "joints": [
                    {"name": "j1", "type": "revolute"},
                    {"name": "j2", "type": "revolute"},
                ],
            },
        ),
    )


@patch.object(
    _joints, "controllable_joint_names",
    return_value=["j1", "j2"],
)
def test_joints_state_get_parses_joint_update_payload(_mock_names: MagicMock) -> None:
    twin = _make_joint_twin(
        metadata={
            "mqtt": {
                "topics": {
                    JOINT_UPDATE_TOPIC_SLUG: {},
                },
                "commands": {"supported": []},
            }
        }
    )
    topic = JOINT_UPDATE_TOPIC_SLUG.format(twin_uuid="arm-1")
    result: list[dict[str, float]] = []
    thread = threading.Thread(target=lambda: result.append(twin.joints.get(what_joints=["j1", "j2"])))
    thread.start()
    time.sleep(0.05)
    _inject(twin.client.mqtt, topic, {"j1": 0.5, "j2": 1.0, "source_type": "edge"})
    thread.join(timeout=2.0)
    assert result[0]["j1"] == 0.5
    assert result[0]["j2"] == 1.0


@patch.object(
    _joints, "controllable_joint_names",
    return_value=["j1"],
)
def test_joints_state_get_falls_back_when_catalog_has_no_joint_slug(
    _mock_names: MagicMock,
) -> None:
    twin = _make_joint_twin(
        metadata={
            "mqtt": {
                "topics": {TWIN_COMMAND_TOPIC_SLUG: {}},
                "commands": {"supported": []},
            }
        }
    )
    topic = JOINT_UPDATE_TOPIC_SLUG.format(twin_uuid="arm-1")
    result: list[dict[str, float]] = []
    thread = threading.Thread(target=lambda: result.append(twin.joints.get(what_joints=["j1"])))
    thread.start()
    time.sleep(0.05)
    _inject(twin.client.mqtt, topic, {"j1": 0.25})
    thread.join(timeout=2.0)
    assert result[0]["j1"] == 0.25
    twin.client.mqtt.subscribe.assert_called()


@patch.object(
    _joints, "controllable_joint_names",
    return_value=["j1"],
)
def test_joints_listener_starts_when_handle_created(_mock_names: MagicMock) -> None:
    twin = _make_joint_twin(
        metadata={
            "mqtt": {
                "topics": {JOINT_UPDATE_TOPIC_SLUG: {}},
                "commands": {"supported": []},
            }
        }
    )
    _ = twin.joints
    twin.client.mqtt.subscribe.assert_called()


@patch.object(
    _joints, "controllable_joint_names",
    return_value=["j1"],
)
def test_joints_get_reflects_continuous_mqtt_updates(_mock_names: MagicMock) -> None:
    twin = _make_joint_twin(
        metadata={
            "mqtt": {
                "topics": {JOINT_UPDATE_TOPIC_SLUG: {}},
                "commands": {"supported": []},
            }
        }
    )
    topic = JOINT_UPDATE_TOPIC_SLUG.format(twin_uuid="arm-1")
    first: list[dict[str, float]] = []
    thread = threading.Thread(target=lambda: first.append(twin.joints.get(what_joints=["j1"])))
    thread.start()
    time.sleep(0.05)
    _inject(twin.client.mqtt, topic, {"j1": 0.1})
    thread.join(timeout=2.0)
    assert first[0]["j1"] == 0.1

    _inject(
        twin.client.mqtt,
        topic,
        {
            "positions": {"j1": 0.9},
            "velocities": {"j1": 0.01},
            "source_type": "edge",
        },
    )
    got = twin.joints.get(timeout=0.0)
    assert got["j1"] == 0.9


@patch.object(
    _joints, "controllable_joint_names",
    return_value=["shoulder_pan"],
)
def test_joints_inbound_drops_names_outside_local_schema(_mock_names: MagicMock) -> None:
    """Joint names not in joints.list() must be silently dropped, even on a
    total mismatch between the payload and the schema."""
    twin = _make_joint_twin(
        metadata={
            "mqtt": {
                "topics": {JOINT_UPDATE_TOPIC_SLUG: {}},
                "commands": {"supported": []},
            }
        }
    )
    topic = JOINT_UPDATE_TOPIC_SLUG.format(twin_uuid="arm-1")
    _ = twin.joints  # attach listener before updates arrive

    # Robot sends motor indices that don't match the schema → must be dropped.
    _inject(twin.client.mqtt, topic, {"positions": {"1": 0.1}, "source_type": "edge"})
    got = twin.joints.get(timeout=0.0)
    assert "1" not in dict(got)
    # Schema joint is still present (seeded at 0).
    assert "shoulder_pan" in dict(got)


@patch.object(_joints, "controllable_joint_names", return_value=["j1"])
def test_joints_get_view_is_real_dict(_n: MagicMock) -> None:
    """JointStateView is a dict subclass: json.dumps, .copy(), isinstance all work."""
    import json

    twin = _make_joint_twin(
        metadata={
            "mqtt": {
                "topics": {JOINT_UPDATE_TOPIC_SLUG: {}},
                "commands": {"supported": []},
            }
        }
    )
    topic = JOINT_UPDATE_TOPIC_SLUG.format(twin_uuid="arm-1")
    _ = twin.joints  # attach listener
    _inject(twin.client.mqtt, topic, {"positions": {"j1": 0.5}, "source_type": "edge"})
    view = twin.joints.get(timeout=0.0)

    assert isinstance(view, dict)
    assert json.dumps(view) == '{"j1": 0.5}'
    snapshot = view.copy()
    assert isinstance(snapshot, dict)
    assert snapshot == {"j1": 0.5}


@patch.object(_joints, "controllable_joint_names", return_value=["j1"])
def test_joints_get_returns_live_view(_n: MagicMock) -> None:
    twin = _make_joint_twin(
        metadata={
            "mqtt": {
                "topics": {JOINT_UPDATE_TOPIC_SLUG: {}},
                "commands": {"supported": []},
            }
        }
    )
    topic = JOINT_UPDATE_TOPIC_SLUG.format(twin_uuid="arm-1")
    _ = twin.joints
    _inject(twin.client.mqtt, topic, {"positions": {"j1": 0.1}, "source_type": "edge"})
    view = twin.joints.get(timeout=0.0)
    assert view["j1"] == 0.1
    # Same object reflects later updates without re-calling get().
    _inject(twin.client.mqtt, topic, {"positions": {"j1": 0.8}, "source_type": "edge"})
    assert view["j1"] == 0.8
    assert dict(view) == {"j1": 0.8}


@patch.object(_joints, "controllable_joint_names", return_value=["j1"])
def test_joints_get_after_update_callback_and_view_stop(_n: MagicMock) -> None:
    twin = _make_joint_twin(
        metadata={
            "mqtt": {
                "topics": {JOINT_UPDATE_TOPIC_SLUG: {}},
                "commands": {"supported": []},
            }
        }
    )
    topic = JOINT_UPDATE_TOPIC_SLUG.format(twin_uuid="arm-1")
    seen: list[dict] = []
    view = twin.joints.get(timeout=0.0, after_update_callback=lambda s: seen.append(s))
    _inject(twin.client.mqtt, topic, {"positions": {"j1": 0.4}, "source_type": "edge"})
    assert seen[-1]["j1"] == 0.4
    view.stop()
    _inject(twin.client.mqtt, topic, {"positions": {"j1": 0.6}, "source_type": "edge"})
    assert len(seen) == 1
    # stop() cancels all subscriptions and freezes the dict at its last value.
    assert view["j1"] == 0.4


@patch.object(_joints, "controllable_joint_names", return_value=["j1"])
def test_joints_on_update_fires_snapshot_and_cancels(_n: MagicMock) -> None:
    twin = _make_joint_twin(
        metadata={
            "mqtt": {
                "topics": {JOINT_UPDATE_TOPIC_SLUG: {}},
                "commands": {"supported": []},
            }
        }
    )
    topic = JOINT_UPDATE_TOPIC_SLUG.format(twin_uuid="arm-1")
    seen: list[dict] = []
    sub = twin.joints.on_update(lambda s: seen.append(s))
    _inject(twin.client.mqtt, topic, {"positions": {"j1": 0.2}, "source_type": "edge"})
    assert seen and seen[-1]["j1"] == 0.2
    sub.cancel()
    _inject(twin.client.mqtt, topic, {"positions": {"j1": 0.7}, "source_type": "edge"})
    assert len(seen) == 1  # no delivery after cancel


@patch.object(_joints, "controllable_joint_names", return_value=["j1"])
def test_joints_on_update_callback_exception_does_not_kill_listener(_n: MagicMock) -> None:
    twin = _make_joint_twin(
        metadata={
            "mqtt": {
                "topics": {JOINT_UPDATE_TOPIC_SLUG: {}},
                "commands": {"supported": []},
            }
        }
    )
    topic = JOINT_UPDATE_TOPIC_SLUG.format(twin_uuid="arm-1")

    def boom(_s):
        raise RuntimeError("bad callback")

    twin.joints.on_update(boom)
    _inject(twin.client.mqtt, topic, {"positions": {"j1": 0.5}, "source_type": "edge"})
    # Listener survived: cache still updated and readable.
    assert twin.joints.get(timeout=0.0)["j1"] == 0.5


@patch.object(_joints, "controllable_joint_names", return_value=["j1", "j2"])
def test_joints_inbound_filters_unknown_when_names_intersect(_n: MagicMock) -> None:
    twin = _make_joint_twin(
        metadata={
            "mqtt": {
                "topics": {JOINT_UPDATE_TOPIC_SLUG: {}},
                "commands": {"supported": []},
            }
        }
    )
    topic = JOINT_UPDATE_TOPIC_SLUG.format(twin_uuid="arm-1")
    _ = twin.joints
    # Payload shares j1 with schema -> "bogus" must be dropped, j1 kept.
    _inject(
        twin.client.mqtt,
        topic,
        {"positions": {"j1": 0.3, "bogus": 9.9}, "source_type": "edge"},
    )
    got = twin.joints.get(timeout=0.0)
    assert got["j1"] == 0.3
    assert "bogus" not in dict(got)


def test_joints_list_is_cached_after_first_resolution() -> None:
    twin = _make_joint_twin(
        metadata={
            "mqtt": {
                "topics": {JOINT_UPDATE_TOPIC_SLUG: {}},
                "commands": {"supported": []},
            }
        }
    )
    with patch.object(
        _joints, "controllable_joint_names", return_value=["j1", "j2"]
    ) as spy:
        handle = twin.joints
        handle.list()
        handle.list()
        handle.list()
    assert spy.call_count == 1


@patch.object(
    _joints, "controllable_joint_names",
    return_value=["j1", "j2"],
)
def test_joints_get_timeout_returns_zeros_from_schema(_mock_names: MagicMock) -> None:
    twin = _make_joint_twin(
        metadata={
            "mqtt": {
                "topics": {JOINT_UPDATE_TOPIC_SLUG: {}},
                "commands": {"supported": []},
            }
        }
    )
    got = twin.joints.get(timeout=0.05)
    assert got == {"j1": 0.0, "j2": 0.0}
    again = twin.get_joints()
    assert again == {"j1": 0.0, "j2": 0.0}


@patch.object(
    _joints, "controllable_joint_names",
    return_value=["j1", "j2"],
)
def test_joints_get_partial_mqtt_fills_missing_joints_with_zero(
    _mock_names: MagicMock,
) -> None:
    twin = _make_joint_twin(
        metadata={
            "mqtt": {
                "topics": {JOINT_UPDATE_TOPIC_SLUG: {}},
                "commands": {"supported": []},
            }
        }
    )
    topic = JOINT_UPDATE_TOPIC_SLUG.format(twin_uuid="arm-1")
    result: list[dict[str, float]] = []
    thread = threading.Thread(
        target=lambda: result.append(twin.joints.get(what_joints=["j1", "j2"]))
    )
    thread.start()
    time.sleep(0.05)
    _inject(twin.client.mqtt, topic, {"j1": 0.42, "source_type": "edge"})
    thread.join(timeout=2.0)
    assert result[0]["j1"] == 0.42
    assert result[0]["j2"] == 0.0


@patch.object(
    _joints, "controllable_joint_names",
    return_value=["j1", "j2"],
)
def test_joints_get_uses_runtime_mode_bucket(_mock_names: MagicMock) -> None:
    twin = _make_joint_twin(
        metadata={
            "mqtt": {
                "topics": {JOINT_UPDATE_TOPIC_SLUG: {}},
                "commands": {"supported": []},
            }
        }
    )
    topic = JOINT_UPDATE_TOPIC_SLUG.format(twin_uuid="arm-1")
    _ = twin.joints
    _inject(twin.client.mqtt, topic, {"j1": 0.1, "source_type": "tele"})
    assert twin.joints.get() == {"j1": 0.1, "j2": 0.0}

    twin.client.config.runtime_mode = "simulation"
    _inject(twin.client.mqtt, topic, {"j1": 0.9, "source_type": "sim_tele"})
    assert twin.joints.get() == {"j1": 0.9, "j2": 0.0}

    twin.client.config.runtime_mode = "live"
    assert twin.joints.get() == {"j1": 0.1, "j2": 0.0}


@patch.object(
    _joints, "controllable_joint_names",
    return_value=["j1"],
)
def test_joints_sim_mqtt_ignored_when_runtime_mode_live(_mock_names: MagicMock) -> None:
    twin = _make_joint_twin(
        metadata={
            "mqtt": {
                "topics": {JOINT_UPDATE_TOPIC_SLUG: {}},
                "commands": {"supported": []},
            }
        }
    )
    topic = JOINT_UPDATE_TOPIC_SLUG.format(twin_uuid="arm-1")
    _ = twin.joints
    _inject(twin.client.mqtt, topic, {"j1": 0.5, "source_type": "sim_tele"})
    assert twin.joints.get(timeout=0.0) == {"j1": 0.0}


def test_pose_get_respects_runtime_mode_bucket() -> None:
    twin = _make_locomote_twin()
    pos_topic = TWIN_POSITION_TOPIC_SLUG.format(twin_uuid=twin.uuid)
    # Pose MQTT listeners attach on first get(), not on property access.
    twin.pose.get(timeout=0.05)
    _inject(
        twin.client.mqtt,
        pos_topic,
        {"position": {"x": 1.0, "y": 0.0, "z": 0.0}, "source_type": "tele"},
    )
    assert twin.pose.get(timeout=0.0).position.x == 1.0

    twin.client.config.runtime_mode = "simulation"
    _inject(
        twin.client.mqtt,
        pos_topic,
        {"position": {"x": 9.0, "y": 0.0, "z": 0.0}, "source_type": "sim_tele"},
    )
    assert twin.pose.get(timeout=0.0).position.x == 9.0

    twin.client.config.runtime_mode = "live"
    assert twin.pose.get(timeout=0.0).position.x == 1.0
