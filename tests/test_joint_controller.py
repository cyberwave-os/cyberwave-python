"""Tests for JointController.refresh(), .get(), .get_all(), .list()."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cyberwave.constants import SOURCE_TYPE_TELE
from cyberwave.exceptions import CyberwaveError
from cyberwave.rest.models.joint_states_schema import JointStatesSchema
from cyberwave.twin import JointController


def _make_joint_controller(joint_states_response=None, side_effect=None):
    """Build a JointController with a mocked twin and client."""
    twins_manager = MagicMock()
    if side_effect is not None:
        twins_manager.get_joint_states.side_effect = side_effect
    else:
        twins_manager.get_joint_states.return_value = joint_states_response
    client = SimpleNamespace(twins=twins_manager)
    twin = SimpleNamespace(uuid="twin-uuid", client=client)
    controller = JointController(twin)
    return controller, twins_manager


def _make_joint_states_schema(
    names=None, positions=None, velocities=None, efforts=None
):
    """Build a JointStatesSchema matching the ROS JointState format."""
    return JointStatesSchema(
        header={"stamp": "2026-04-02T13:45:01Z", "frame_id": ""},
        name=names if names is not None else ["joint_a", "joint_b", "joint_c"],
        position=positions if positions is not None else [0.0, 1.57, -0.5],
        velocity=velocities if velocities is not None else [0.0, 0.0, 0.0],
        effort=efforts if efforts is not None else [0.0, 0.0, 0.0],
    )


def _make_controller_for_set():
    """Build a JointController with mocks deep enough for set() (MQTT path)."""
    twins_manager = MagicMock()
    client = MagicMock()
    client.twins = twins_manager
    client.mqtt = MagicMock()
    twin = MagicMock()
    twin.uuid = "twin-uuid"
    twin.client = client
    controller = JointController(twin)
    return controller, client


# ---------------------------------------------------------------------------
# refresh()
# ---------------------------------------------------------------------------


class TestJointControllerRefresh:
    def test_populates_joint_states_from_parallel_arrays(self):
        schema = _make_joint_states_schema(
            names=["wheel_left", "wheel_right"],
            positions=[0.0, 1.57],
        )
        controller, mgr = _make_joint_controller(schema)

        controller.refresh()

        assert controller._joint_states == {"wheel_left": 0.0, "wheel_right": 1.57}
        mgr.get_joint_states.assert_called_once_with("twin-uuid")

    def test_empty_response_yields_empty_dict(self):
        schema = _make_joint_states_schema(names=[], positions=[])
        controller, mgr = _make_joint_controller(schema)

        controller.refresh()

        assert controller._joint_states == {}
        mgr.get_joint_states.assert_called_once_with("twin-uuid")

    def test_propagates_api_errors(self):
        controller, _ = _make_joint_controller(
            side_effect=RuntimeError("network error")
        )

        with pytest.raises(CyberwaveError, match="Failed to refresh joint states"):
            controller.refresh()


# ---------------------------------------------------------------------------
# get_all()
# ---------------------------------------------------------------------------


class TestJointControllerGetAll:
    def test_returns_all_joint_positions(self):
        schema = _make_joint_states_schema(
            names=["shoulder", "elbow"],
            positions=[0.785, 1.57],
        )
        controller, _ = _make_joint_controller(schema)

        result = controller.get_all()

        assert result == {"shoulder": 0.785, "elbow": 1.57}

    def test_returns_copy_not_reference(self):
        schema = _make_joint_states_schema(names=["j1"], positions=[1.0])
        controller, _ = _make_joint_controller(schema)

        result1 = controller.get_all()
        result2 = controller.get_all()
        assert result1 is not result2

    def test_empty_when_no_joints(self):
        schema = _make_joint_states_schema(names=[], positions=[])
        controller, _ = _make_joint_controller(schema)

        assert controller.get_all() == {}


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------


class TestJointControllerGet:
    def test_returns_specific_joint_position(self):
        schema = _make_joint_states_schema(
            names=["shoulder", "elbow", "wrist"],
            positions=[0.5, 1.0, -0.3],
        )
        controller, _ = _make_joint_controller(schema)

        assert controller.get("elbow") == 1.0

    def test_raises_for_unknown_joint(self):
        schema = _make_joint_states_schema(names=["shoulder"], positions=[0.0])
        controller, _ = _make_joint_controller(schema)

        with pytest.raises(CyberwaveError, match="Joint 'nonexistent' not found"):
            controller.get("nonexistent")


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


class TestJointControllerList:
    def test_returns_all_joint_names(self):
        schema = _make_joint_states_schema(
            names=["wheel_fl", "wheel_fr", "pt_pan"],
            positions=[0, 0, 0],
        )
        controller, _ = _make_joint_controller(schema)

        assert sorted(controller.list()) == ["pt_pan", "wheel_fl", "wheel_fr"]

    def test_empty_list_when_no_joints(self):
        schema = _make_joint_states_schema(names=[], positions=[])
        controller, _ = _make_joint_controller(schema)

        assert controller.list() == []


# ---------------------------------------------------------------------------
# set()
# ---------------------------------------------------------------------------


class TestJointControllerSet:
    def test_calls_mqtt_update_joint_state(self):
        controller, client = _make_controller_for_set()

        with patch(
            "cyberwave.twin._default_control_source_type",
            return_value=SOURCE_TYPE_TELE,
        ):
            controller.set("elbow", 1.57, degrees=False)

        client.mqtt.update_joint_state.assert_called_once()
        call_args = client.mqtt.update_joint_state.call_args
        # update_joint_state(twin_uuid, joint_name, position=..., ...)
        assert call_args[0][0] == "twin-uuid"
        assert call_args[0][1] == "elbow"
        assert call_args[1]["position"] == 1.57
        assert call_args[1]["source_type"] == SOURCE_TYPE_TELE

    def test_updates_cache_after_set(self):
        controller, client = _make_controller_for_set()

        with patch(
            "cyberwave.twin._default_control_source_type",
            return_value=SOURCE_TYPE_TELE,
        ):
            controller.set("shoulder", 0.785, degrees=False)

        client.mqtt.update_joint_state.assert_called_once()
        assert controller._joint_states == {"shoulder": 0.785}
