"""JointController.refresh parses both legacy and OpenAPI joint-state shapes."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from cyberwave.rest.models.joint_states_schema import JointStatesSchema
from cyberwave.twin import JointController


def test_refresh_parses_parallel_name_position_lists() -> None:
    states = JointStatesSchema(
        header={"t": "1"},
        name=["pan_joint", "tilt_joint"],
        position=[0.25, -0.1],
        velocity=[0.0, 0.0],
        effort=[0.0, 0.0],
    )
    twin = MagicMock()
    twin.uuid = "twin-uuid"
    twin.client.twins.get_joint_states.return_value = states

    jc = JointController(twin)
    jc.refresh()

    assert jc.get_all() == {"pan_joint": 0.25, "tilt_joint": -0.1}
    twin.client.twins.get_joint_states.assert_called_once_with("twin-uuid")


def test_refresh_parses_legacy_joint_states_objects() -> None:
    states = SimpleNamespace(
        joint_states=[
            SimpleNamespace(joint_name="a", position=1.5),
            SimpleNamespace(joint_name="b", position=2.5),
        ]
    )
    twin = MagicMock()
    twin.uuid = "u2"
    twin.client.twins.get_joint_states.return_value = states

    jc = JointController(twin)
    jc.refresh()

    assert jc.get_all() == {"a": 1.5, "b": 2.5}
