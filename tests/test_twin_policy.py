"""Policy manager and twin.policy handle tests."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from cyberwave.twin import transport as _transport
from cyberwave.twin.capabilities import joints as _joints

import pytest

from cyberwave.exceptions import CyberwaveError
from cyberwave.managers.policies import PolicyManager
from cyberwave.twin import JointTwin, LocomoteTwin, Twin
from cyberwave.twin._helpers import _build_controller_assignment_metadata
from cyberwave.twin.classes import FlyingTwin
from cyberwave.twin.factory import create_twin


def _policy(uuid: str, *, controller_type: str = "teleop", input_device: str = "sdk") -> SimpleNamespace:
    return SimpleNamespace(
        uuid=uuid,
        name=f"policy-{uuid[:8]}",
        controller_type=controller_type,
        metadata={"input_device": input_device},
    )


def test_policy_manager_list_filters_by_twin_asset_and_workspace() -> None:
    api = MagicMock()
    api.src_app_api_controller_policies_list_controller_policies.return_value = []
    client = SimpleNamespace(
        twins=SimpleNamespace(api=api),
        environments=MagicMock(),
    )
    twin = Twin(
        client,
        SimpleNamespace(
            uuid="twin-1",
            asset_uuid="asset-1",
            environment_uuid="env-1",
        ),
    )
    twin._get_workspace_uuid = lambda: "ws-1"  # type: ignore[method-assign]

    PolicyManager(client).list(twin=twin)

    api.src_app_api_controller_policies_list_controller_policies.assert_called_once_with(
        asset_uuid="asset-1",
        workspace_uuid="ws-1",
    )


def test_build_controller_assignment_metadata_clears_on_unassign() -> None:
    twin_data = SimpleNamespace(
        metadata={
            "controller_policy_uuid": "p1",
            "controller_policy_name": "Keyboard",
            "controller_type": "teleop",
            "control_mode": "joint_control",
            "locomotion_animations": {"bindings": {}},
        }
    )
    meta = _build_controller_assignment_metadata(twin_data, None)
    assert meta["controller_policy_uuid"] is None
    assert meta["controller_policy_name"] is None
    assert meta["controller_type"] is None
    assert meta["control_mode"] is None
    assert meta["locomotion_animations"] == {"bindings": {}}


def test_twin_policy_handle_unassign_clears_controller() -> None:
    policy = _policy("p1")
    twin_data = _twin_data_after_assign("twin-1", policy)
    twins_mgr = MagicMock()
    twins_mgr.update.return_value = SimpleNamespace(
        uuid="twin-1",
        controller_policy_uuid=None,
        metadata=_build_controller_assignment_metadata(twin_data, None),
    )
    publish_alert = MagicMock()
    client = SimpleNamespace(
        twins=SimpleNamespace(update=twins_mgr.update),
        config=SimpleNamespace(runtime_mode="live"),
        publish_alert=publish_alert,
    )
    twin = JointTwin(
        client,
        SimpleNamespace(
            uuid="twin-1",
            asset_uuid="a1",
            metadata=twin_data.metadata,
            controller_policy_uuid=twin_data.controller_policy_uuid,
            capabilities={"has_joints": True},
        ),
    )
    twin._controller_ensured = True

    twin.policy.unassign()

    twins_mgr.update.assert_called_once()
    assert twin._controller_ensured is False
    publish_alert.assert_called_once()
    assert publish_alert.call_args[0][1] == "Removing controller"


def test_twin_disconnect_unassigns_in_live_mode() -> None:
    policy = _policy("p1")
    twin_data = _twin_data_after_assign("twin-1", policy)
    twins_mgr = MagicMock()
    twins_mgr.update.return_value = SimpleNamespace(
        uuid="twin-1",
        controller_policy_uuid=None,
        metadata=_build_controller_assignment_metadata(twin_data, None),
    )
    publish_alert = MagicMock()
    client = SimpleNamespace(
        twins=SimpleNamespace(update=twins_mgr.update),
        config=SimpleNamespace(runtime_mode="live"),
        publish_alert=publish_alert,
    )
    twin = Twin(client, twin_data)

    twin.disconnect()

    twins_mgr.update.assert_called_once()
    publish_alert.assert_called_once()
    assert publish_alert.call_args[0][1] == "Disconnecting"
    assert "removing controller" in publish_alert.call_args[1]["description"].lower()


def test_twin_disconnect_skips_unassign_in_simulation() -> None:
    policy = _policy("p1")
    twin_data = _twin_data_after_assign("twin-1", policy)
    twins_mgr = MagicMock()
    client = SimpleNamespace(
        twins=SimpleNamespace(update=twins_mgr.update),
        config=SimpleNamespace(runtime_mode="simulation"),
    )
    twin = Twin(client, twin_data)

    twin.disconnect()

    twins_mgr.update.assert_not_called()


def test_policy_manager_unassign_clears_controller() -> None:
    policy = _policy("p1")
    twin_data = _twin_data_after_assign("twin-1", policy)
    twins_mgr = MagicMock()
    twins_mgr.update.return_value = SimpleNamespace(
        uuid="twin-1",
        controller_policy_uuid=None,
        metadata=_build_controller_assignment_metadata(twin_data, None),
    )
    publish_alert = MagicMock()
    client = SimpleNamespace(
        twins=SimpleNamespace(update=twins_mgr.update),
        publish_alert=publish_alert,
    )
    twin = Twin(client, twin_data)
    twin._controller_ensured = True

    PolicyManager(client).unassign(twin)

    twins_mgr.update.assert_called_once()
    _args, kwargs = twins_mgr.update.call_args
    assert kwargs["controller_policy_uuid"] == ""
    assert kwargs["metadata"]["controller_policy_uuid"] is None
    assert twin._controller_ensured is False
    publish_alert.assert_called_once()
    assert publish_alert.call_args[0][1] == "Removing controller"


def test_twin_apply_controller_policy_publishes_assigning_alert() -> None:
    policy = _policy("p1")
    twins_mgr = MagicMock()
    twins_mgr.update.return_value = _twin_data_after_assign("twin-1", policy)
    publish_alert = MagicMock()
    client = SimpleNamespace(
        twins=SimpleNamespace(update=twins_mgr.update),
        config=SimpleNamespace(runtime_mode="live"),
        publish_alert=publish_alert,
    )
    twin = Twin(
        client,
        SimpleNamespace(uuid="twin-1", name="Go2", metadata={}, controller_policy_uuid=None),
    )

    twin._apply_controller_policy(policy)

    publish_alert.assert_called_once()
    assert publish_alert.call_args[0][1] == "Assigning controller"
    assert publish_alert.call_args[1]["alert_type"] == "controller_state"


def test_prepare_outbound_command_assigns_when_unassigned() -> None:
    policies = [_policy("p1", input_device="sdk")]
    api = MagicMock()
    api.src_app_api_controller_policies_list_controller_policies.return_value = policies
    twins_mgr = MagicMock()
    twins_mgr.update.return_value = _twin_data_after_assign("twin-1", policies[0])
    client = SimpleNamespace(
        twins=SimpleNamespace(api=api, update=twins_mgr.update),
        environments=MagicMock(),
        config=SimpleNamespace(runtime_mode="live"),
        policies=PolicyManager(
            SimpleNamespace(
                twins=SimpleNamespace(api=api, update=twins_mgr.update),
                environments=MagicMock(),
            )
        ),
    )
    twin = JointTwin(
        client,
        SimpleNamespace(
            uuid="twin-1",
            asset_uuid="a1",
            metadata={},
            capabilities={"has_joints": True, "can_locomote": False},
        ),
    )
    twin._get_workspace_uuid = lambda: None  # type: ignore[method-assign]

    with patch.object(twin.policy, "list", return_value=policies):
        with patch.object(twin.policy, "_pick_controller_policy", return_value=policies[0]):
            twin.policy.ensure_attached()

    twins_mgr.update.assert_called()


def _twin_data_after_assign(twin_uuid: str, policy: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        uuid=twin_uuid,
        controller_policy_uuid=str(policy.uuid),
        metadata={
            "controller_policy_uuid": str(policy.uuid),
            "controller_policy_name": policy.name,
            "controller_type": "teleop",
            "control_mode": "joint_control",
        },
    )


def _locomote_twin_with_mqtt(
    *,
    policies: list,
    twin_uuid: str = "go2",
) -> tuple[LocomoteTwin, MagicMock]:
    api = MagicMock()
    api.src_app_api_controller_policies_list_controller_policies.return_value = policies
    twins_mgr = MagicMock()
    if policies:
        twins_mgr.update.return_value = _twin_data_after_assign(twin_uuid, policies[0])
    mqtt = MagicMock()
    mqtt.connected = True
    assets = MagicMock()
    assets.get.return_value = SimpleNamespace(
        metadata={
            "mqtt": {
                "topics": {"cyberwave/twin/{twin_uuid}/command": {}},
                "commands": {"supported": ["move_forward", "stop"]},
            }
        }
    )
    client = SimpleNamespace(
        mqtt=mqtt,
        assets=assets,
        twins=SimpleNamespace(api=api, update=twins_mgr.update),
        environments=MagicMock(),
        config=SimpleNamespace(runtime_mode="live", source_type="tele"),
        policies=PolicyManager(
            SimpleNamespace(
                twins=SimpleNamespace(api=api, update=twins_mgr.update),
                environments=MagicMock(),
            )
        ),
    )
    twin = LocomoteTwin(
        client,
        SimpleNamespace(
            uuid="go2",
            name="Go2",
            asset_uuid="a1",
            metadata={},
            capabilities={"can_locomote": True},
        ),
    )
    twin._get_workspace_uuid = lambda: None  # type: ignore[method-assign]
    return twin, mqtt


def test_locomotion_ensure_attached_before_mqtt_publish() -> None:
    policies = [_policy("p1", input_device="sdk")]
    twin, mqtt = _locomote_twin_with_mqtt(policies=policies)
    with patch.object(twin.policy, "list", return_value=policies):
        with patch.object(twin.policy, "_pick_controller_policy", return_value=policies[0]):
            with patch.object(_transport.time, "sleep"):
                twin.locomotion.move_forward(1.0, duration=0.1, rate_hz=10)
    assert mqtt.publish.call_count >= 1


def test_joints_ensure_attached_before_mqtt_publish() -> None:
    policies = [_policy("p1", input_device="sdk")]
    twin, mqtt = _locomote_twin_with_mqtt(policies=policies)
    twin.client.assets.get.return_value = SimpleNamespace(
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
        twin.client,
        SimpleNamespace(
            uuid="arm-1",
            name="Arm",
            asset_uuid="a1",
            metadata={},
            capabilities={"has_joints": True},
        ),
    )
    twin._get_workspace_uuid = lambda: None  # type: ignore[method-assign]
    with patch.object(_joints, "controllable_joint_names", return_value=["j1"]):
        with patch.object(twin.policy, "list", return_value=policies):
            with patch.object(twin.policy, "_pick_controller_policy", return_value=policies[0]):
                twin.joints.set({"j1": 1.0})
    mqtt.publish.assert_called_once()


def test_flight_ensure_attached_before_mqtt_publish() -> None:
    policies = [_policy("p1", input_device="sdk")]
    api = MagicMock()
    api.src_app_api_controller_policies_list_controller_policies.return_value = policies
    twins_mgr = MagicMock()
    twins_mgr.update.return_value = _twin_data_after_assign("drone", policies[0])
    mqtt = MagicMock()
    mqtt.connected = True
    assets = MagicMock()
    assets.get.return_value = SimpleNamespace(
        metadata={
            "mqtt": {
                "topics": {"cyberwave/twin/{twin_uuid}/command": {}},
                "commands": {"supported": ["takeoff", "land"]},
            }
        }
    )
    client = SimpleNamespace(
        mqtt=mqtt,
        assets=assets,
        twins=SimpleNamespace(
            api=api,
            update=twins_mgr.update,
        ),
        environments=MagicMock(),
        config=SimpleNamespace(runtime_mode="live", source_type="tele"),
        policies=PolicyManager(
            SimpleNamespace(
                twins=SimpleNamespace(api=api, update=twins_mgr.update),
                environments=MagicMock(),
            )
        ),
    )
    twin = FlyingTwin(
        client,
        SimpleNamespace(
            uuid="drone",
            name="Drone",
            asset_uuid="a1",
            metadata={},
            capabilities={"can_fly": True, "can_locomote": True},
        ),
    )
    twin._get_workspace_uuid = lambda: None  # type: ignore[method-assign]
    with patch.object(twin.policy, "list", return_value=policies):
        with patch.object(twin.policy, "_pick_controller_policy", return_value=policies[0]):
            twin.takeoff(altitude=1.0)
    mqtt.publish.assert_called_once()


def test_motion_publish_blocked_without_teleop_policy() -> None:
    twin, mqtt = _locomote_twin_with_mqtt(policies=[])
    with patch.dict("os.environ", {"CYBERWAVE_SDK_AUTO_ATTACH_CONTROLLER": "0"}):
        with pytest.raises(CyberwaveError, match="teleop controller policy"):
            twin.locomotion.move_forward(1.0)
    mqtt.publish.assert_not_called()


def test_playground_actuations_empty_without_attached_policy() -> None:
    twin = JointTwin(
        SimpleNamespace(twins=SimpleNamespace(api=MagicMock())),
        SimpleNamespace(
            uuid="twin-1",
            metadata={},
            controller_policy_uuid=None,
            capabilities={"has_joints": True},
        ),
    )
    assert twin.policy.playground_actuations() == frozenset()


def test_playground_actuations_reads_keyboard_bindings_with_playground_config() -> None:
    api = MagicMock()
    api.src_app_api_controller_policies_get_controller_policy.return_value = SimpleNamespace(
        metadata={
            "keyboard_bindings": [
                {"actuation": "move_forward", "continuous": True, "playground": {"linear_x": 1.0}},
                {"actuation": "turn_left", "continuous": True, "playground": {"angular_z": 1.5}},
                # No `playground` extension declared for this actuation -- excluded.
                {"actuation": "recovery_stand", "continuous": False},
            ]
        }
    )
    twin = LocomoteTwin(
        SimpleNamespace(twins=SimpleNamespace(api=api)),
        SimpleNamespace(
            uuid="twin-1",
            metadata={"controller_policy_uuid": "p1"},
            controller_policy_uuid="p1",
            capabilities={"can_locomote": True},
        ),
    )

    actuations = twin.policy.playground_actuations()

    assert actuations == frozenset({"move_forward", "turn_left"})
    api.src_app_api_controller_policies_get_controller_policy.assert_called_once_with("p1")


def test_playground_actuations_caches_per_policy_uuid() -> None:
    api = MagicMock()
    api.src_app_api_controller_policies_get_controller_policy.return_value = SimpleNamespace(
        metadata={"keyboard_bindings": [{"actuation": "takeoff", "playground": {"delta_z": 2.0}}]}
    )
    twin = FlyingTwin(
        SimpleNamespace(twins=SimpleNamespace(api=api)),
        SimpleNamespace(
            uuid="drone-1",
            metadata={"controller_policy_uuid": "p1"},
            controller_policy_uuid="p1",
            capabilities={"can_fly": True, "can_locomote": True},
        ),
    )

    first = twin.policy.playground_actuations()
    second = twin.policy.playground_actuations()

    assert first == second == frozenset({"takeoff"})
    # Only one network round-trip -- the burst-loop preflight check (up to 20 Hz)
    # must not refetch the policy on every call.
    api.src_app_api_controller_policies_get_controller_policy.assert_called_once_with("p1")


def test_ensure_attached_logs_retry_warning_when_policy_newly_attached(
    caplog: pytest.LogCaptureFixture,
) -> None:
    policies = [_policy("p1", input_device="sdk")]
    twin, _mqtt = _locomote_twin_with_mqtt(policies=policies)
    caplog.set_level(logging.WARNING)
    with patch.object(twin.policy, "list", return_value=policies):
        with patch.object(twin.policy, "_pick_controller_policy", return_value=policies[0]):
            with patch.object(_transport.time, "sleep"):
                twin.locomotion.move_forward(1.0, duration=0.1, rate_hz=10)
    assert any("retry in a few seconds" in record.message for record in caplog.records)
