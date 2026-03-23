"""Tests for twin calibration methods (get_calibration, update_calibration, delete_calibration)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from cyberwave.twin import Twin


def _make_twin(client=None):
    source_twin_data = SimpleNamespace(
        uuid="twin-123",
        name="Robot",
        description="Test twin",
        asset_uuid="asset-uuid",
        environment_uuid="env-uuid",
        position_x=0.0,
        position_y=0.0,
        position_z=0.0,
        rotation_w=1.0,
        rotation_x=0.0,
        rotation_y=0.0,
        rotation_z=0.0,
        scale_x=1.0,
        scale_y=1.0,
        scale_z=1.0,
        metadata={},
        fixed_base=False,
    )
    if client is None:
        client = SimpleNamespace(
            twins=MagicMock(),
            _api_client=MagicMock(),
        )
    return Twin(client, source_twin_data)


def test_delete_calibration_clears_both():
    """delete_calibration() with no robot_type calls API to clear both."""
    twins_manager = MagicMock()
    client = SimpleNamespace(twins=twins_manager, _api_client=MagicMock())
    twin = _make_twin(client=client)

    twin.delete_calibration()

    twins_manager.delete_calibration.assert_called_once_with("twin-123", robot_type=None)


def test_delete_calibration_clears_leader_only():
    """delete_calibration(robot_type='leader') passes robot_type to API."""
    twins_manager = MagicMock()
    client = SimpleNamespace(twins=twins_manager, _api_client=MagicMock())
    twin = _make_twin(client=client)

    twin.delete_calibration(robot_type="leader")

    twins_manager.delete_calibration.assert_called_once_with(
        "twin-123", robot_type="leader"
    )


def test_delete_calibration_clears_follower_only():
    """delete_calibration(robot_type='follower') passes robot_type to API."""
    twins_manager = MagicMock()
    client = SimpleNamespace(twins=twins_manager, _api_client=MagicMock())
    twin = _make_twin(client=client)

    twin.delete_calibration(robot_type="follower")

    twins_manager.delete_calibration.assert_called_once_with(
        "twin-123", robot_type="follower"
    )
