"""WorldObstaclesMixin: runtime world scene over twin/command, persisted, atomic swap."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cyberwave.driver.control.world_obstacles import WorldObstaclesMixin
from cyberwave.driver.interface import DriverOperationMode
from cyberwave.driver.interface.registry import DriverInterfaceRegistry
from cyberwave.driver.motion.scene import Pose, Primitive, parse_scene_set


def _box(identifier: str = "desk") -> dict:
    return {
        "id": identifier,
        "primitive": "box",
        "dimensions": [0.8, 0.6, 0.04],
        "pose": {
            "position": {"x": 0.4, "y": 0.0, "z": -0.02},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }


_FLOOR = Primitive(
    "floor", "box", (4.0, 4.0, 0.02), Pose((0.0, 0.0, -0.025), (0.0, 0.0, 0.0, 1.0))
)


class _TerminalBase:
    def define_interface(self, iface):
        pass


class _Controller:
    """The ArmController surface the mixin uses: its config and its solver."""

    def __init__(self, kin) -> None:
        self.config = object()
        self._kin = kin

    @property
    def loaded_kinematics(self):
        return self._kin

    def replace_kinematics(self, candidate) -> None:
        self._kin = candidate


class _Driver(WorldObstaclesMixin, _TerminalBase):
    """Mixin + fakes: an ArmCapabilityMixin-shaped controller and a twins API."""

    def __init__(self, *, old_kin=None, candidate=None, floor: bool = False) -> None:
        self._init_world_obstacles()
        self._arm_ctrl = _Controller(old_kin or object())
        self._build_arm_kinematics = MagicMock(
            return_value=candidate
            or SimpleNamespace(
                collision_model_valid=True, obstacle_collision_available=True
            )
        )
        self._on_arm_command_failed = MagicMock()
        self.replaced = MagicMock()
        self.busy: str | None = None
        self.floor = floor
        self.logger = MagicMock()

    def get_logger(self):
        return self.logger

    @property
    def twin_uuid(self) -> str:  # BaseDriver contract
        return self._twin.uuid

    def bind(self, metadata=None) -> None:
        self._twin = SimpleNamespace(uuid="test-twin", metadata=metadata)
        self._cw = SimpleNamespace(twins=SimpleNamespace(update=MagicMock()))

    def _default_world_obstacles(self):
        return (_FLOOR,) if self.floor else ()

    def _world_obstacles_busy_detail(self):
        return self.busy

    def _on_world_obstacles_replaced(self, obstacles):
        self.replaced(obstacles)


def _saved(driver) -> dict:
    return driver._cw.twins.update.call_args.kwargs["metadata"]["world_obstacles"]


def test_commands_are_registered_in_every_operation_mode() -> None:
    iface = DriverInterfaceRegistry()
    _Driver().define_interface(iface)
    for mode in DriverOperationMode:
        table = iface.command_dispatch_table(mode)
        assert {"add_obstacles", "remove_obstacle"} <= set(table)


def test_commands_can_be_opted_out() -> None:
    class Quiet(_Driver):
        use_world_obstacle_commands = False

    iface = DriverInterfaceRegistry()
    Quiet().define_interface(iface)
    assert "add_obstacles" not in iface.command_dispatch_table(
        DriverOperationMode.NO_OP
    )


def test_a_valid_scene_replaces_the_solver_once_and_persists_under_the_new_key() -> (
    None
):
    driver = _Driver()
    driver.bind()

    driver._on_add_obstacles_command({"data": {"geometries": [_box()]}})

    candidate = driver._build_arm_kinematics.return_value
    assert driver._arm_ctrl._kin is candidate
    obstacles = driver._build_arm_kinematics.call_args.kwargs["obstacles"]
    assert len(obstacles) == 1 and obstacles[0].id == "desk"
    driver._cw.twins.update.assert_called_once_with(
        "test-twin",
        metadata={"world_obstacles": {"frame_id": "base_link", "geometries": [_box()]}},
    )
    driver.replaced.assert_called_once_with(obstacles)
    driver._on_arm_command_failed.assert_not_called()


def test_the_default_floor_is_recorded_readable_and_removable() -> None:
    driver = _Driver(
        candidate=SimpleNamespace(
            collision_model_valid=True, obstacle_collision_available=False
        ),
        floor=True,
    )
    driver.bind(metadata={})
    driver.persist_default_world_obstacles_if_absent()
    floor = _saved(driver)["geometries"][0]
    assert floor["id"] == "floor"
    assert floor["dimensions"] == [4.0, 4.0, 0.02]
    assert floor["pose"]["position"]["z"] == pytest.approx(-0.025)

    driver._on_remove_obstacle_command({"data": {"id": "floor"}})

    assert _saved(driver)["geometries"] == []
    assert driver._build_arm_kinematics.call_args.kwargs["obstacles"] == ()
    assert driver.current_world_obstacles() == ()
    driver._on_arm_command_failed.assert_not_called()


@pytest.mark.parametrize("key", ["world_obstacles", "pinocchio_obstacles"])
@pytest.mark.parametrize("geometries", [[_box("desk")], []])
def test_a_saved_scene_restores_without_writing_back(key, geometries) -> None:
    """The legacy key is read for backward compatibility; nothing is re-written."""
    old = SimpleNamespace(collision_model_valid=True)
    driver = _Driver(
        old_kin=old,
        candidate=SimpleNamespace(
            collision_model_valid=True, obstacle_collision_available=bool(geometries)
        ),
        floor=True,
    )
    driver.bind(metadata={key: {"frame_id": "base_link", "geometries": geometries}})

    assert driver.restore_world_obstacles_from_metadata() is True

    assert driver._arm_ctrl._kin is driver._build_arm_kinematics.return_value
    assert [o.id for o in driver.current_world_obstacles()] == [
        g["id"] for g in geometries
    ]
    driver.persist_default_world_obstacles_if_absent()
    driver._cw.twins.update.assert_not_called()


def test_the_new_key_wins_over_a_legacy_one() -> None:
    driver = _Driver()
    driver.bind(
        metadata={
            "pinocchio_obstacles": {
                "frame_id": "base_link",
                "geometries": [_box("old")],
            },
            "world_obstacles": {"frame_id": "base_link", "geometries": [_box("new")]},
        }
    )
    assert driver.restore_world_obstacles_from_metadata()
    assert [o.id for o in driver.current_world_obstacles()] == ["new"]


def test_a_missing_saved_scene_keeps_the_default() -> None:
    driver = _Driver(floor=True)
    driver.bind(metadata={})
    assert driver.restore_world_obstacles_from_metadata() is False
    assert [o.id for o in driver.current_world_obstacles()] == ["floor"]
    driver._build_arm_kinematics.assert_not_called()
    driver._cw.twins.update.assert_not_called()


@pytest.mark.parametrize(
    "saved",
    [
        {"frame_id": "world", "geometries": [_box()]},
        {"frame_id": "base_link", "geometries": [{**_box(), "dimensions": [-1, 1, 1]}]},
        {"frame_id": "base_link", "geometries": "not a list"},
        {"geometries": [_box()]},  # no frame at all
        "not an object",
    ],
)
def test_an_invalid_saved_scene_raises_without_replacing_anything(saved) -> None:
    """Starting with a silently different scene than the stored one is worse."""
    old = object()
    driver = _Driver(old_kin=old)
    driver.bind(metadata={"world_obstacles": saved})
    with pytest.raises(ValueError):
        driver.restore_world_obstacles_from_metadata()
    assert driver._arm_ctrl._kin is old
    driver._build_arm_kinematics.assert_not_called()
    driver._cw.twins.update.assert_not_called()


def test_an_unbuildable_saved_scene_raises_and_clears_the_update_flag() -> None:
    old = object()
    driver = _Driver(old_kin=old)
    driver._build_arm_kinematics.side_effect = RuntimeError("FCL failed")
    driver.bind(
        metadata={"world_obstacles": {"frame_id": "base_link", "geometries": [_box()]}}
    )
    with pytest.raises(RuntimeError, match="FCL failed"):
        driver.restore_world_obstacles_from_metadata()
    assert driver._arm_ctrl._kin is old
    assert driver.world_obstacles_updating is False


def test_remove_one_preserves_the_others_and_rejects_an_unknown_id() -> None:
    driver = _Driver()
    driver.bind()
    driver._on_add_obstacles_command(
        {"data": {"geometries": [_box("desk"), _box("wall")]}}
    )

    driver._on_remove_obstacle_command({"data": {"id": "desk"}})

    assert [g["id"] for g in _saved(driver)["geometries"]] == ["wall"]
    assert [
        o.id for o in driver._build_arm_kinematics.call_args.kwargs["obstacles"]
    ] == ["wall"]
    calls = driver._build_arm_kinematics.call_count
    driver._on_remove_obstacle_command({"data": {"id": "missing"}})
    driver._on_remove_obstacle_command({"data": {"id": ""}})
    assert driver._build_arm_kinematics.call_count == calls
    assert [o.id for o in driver.current_world_obstacles()] == ["wall"]
    assert "unknown" in driver._on_arm_command_failed.call_args_list[0].args[1]


def test_a_failed_build_keeps_the_installed_scene_and_solver() -> None:
    old = object()
    driver = _Driver(old_kin=old)
    driver._build_arm_kinematics.side_effect = RuntimeError("FCL failed")
    driver._world_obstacles = parse_scene_set(
        {"scene_revision": "t", "objects": [_box("desk"), _box("wall")]}
    ).objects

    driver._on_remove_obstacle_command({"data": {"id": "desk"}})

    assert driver._arm_ctrl._kin is old
    assert [o.id for o in driver.current_world_obstacles()] == ["desk", "wall"]
    assert "FCL failed" in driver._on_arm_command_failed.call_args.args[1]


def test_a_metadata_failure_keeps_the_previous_solver_and_scene() -> None:
    old = object()
    driver = _Driver(old_kin=old)
    driver.bind()
    driver._cw.twins.update.side_effect = RuntimeError("API offline")

    driver._on_add_obstacles_command({"data": {"geometries": [_box()]}})

    assert driver._arm_ctrl._kin is old
    assert driver.current_world_obstacles() == ()
    assert driver.world_obstacles_updating is False
    assert "API offline" in driver._on_arm_command_failed.call_args.args[1]


@pytest.mark.parametrize(
    "candidate,detail",
    [
        (
            SimpleNamespace(
                collision_model_valid=False, obstacle_collision_available=True
            ),
            "self-collision",
        ),
        (
            SimpleNamespace(
                collision_model_valid=True, obstacle_collision_available=False
            ),
            "not paired",
        ),
    ],
)
def test_an_invalid_candidate_is_refused(candidate, detail) -> None:
    old = SimpleNamespace(collision_model_valid=True)
    driver = _Driver(old_kin=old, candidate=candidate)
    driver.bind()
    driver._on_add_obstacles_command({"data": {"geometries": [_box()]}})
    assert driver._arm_ctrl._kin is old
    driver._cw.twins.update.assert_not_called()
    assert detail in driver._on_arm_command_failed.call_args.args[1]


def test_invalid_payloads_never_reach_the_builder() -> None:
    driver = _Driver()
    for payload in (
        {"data": {"geometries": [_box(), _box()]}},  # duplicate ids
        {"data": "nope"},
        {"data": {"geometries": [_box()], "frame_id": "world"}},
    ):
        driver._on_add_obstacles_command(payload)
    driver._build_arm_kinematics.assert_not_called()
    assert driver._on_arm_command_failed.call_count == 3


def test_an_update_is_refused_while_motion_is_in_flight() -> None:
    driver = _Driver()
    driver.busy = "trajectory executing"
    driver._on_add_obstacles_command({"data": {"geometries": [_box()]}})
    driver._build_arm_kinematics.assert_not_called()
    reason = driver._on_arm_command_failed.call_args.args[1]
    assert "motion" in reason and "trajectory executing" in reason


def test_an_unconfigured_solver_is_refused() -> None:
    driver = _Driver()
    driver._arm_ctrl = None
    driver._on_add_obstacles_command({"data": {"geometries": [_box()]}})
    assert "not configured" in driver._on_arm_command_failed.call_args.args[1]
    assert driver.world_obstacles_updating is False


def test_default_recording_errors_are_logged_not_raised() -> None:
    driver = _Driver(floor=True)
    driver.bind(metadata={})
    driver._cw.twins.update.side_effect = RuntimeError("API offline")
    driver.persist_default_world_obstacles_if_absent()
    driver.logger.error.assert_called_once()


def test_nothing_is_recorded_before_the_twin_is_bound() -> None:
    driver = _Driver(floor=True)
    driver.persist_default_world_obstacles_if_absent()  # no _cw/_twin: no-op
    assert driver.restore_world_obstacles_from_metadata() is False
