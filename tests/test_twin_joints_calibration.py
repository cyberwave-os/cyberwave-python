"""Nested joints.calibration REST tests."""

import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cyberwave.twin.classes import JointTwin


def _joint_twin() -> JointTwin:
    twins = MagicMock()
    client = SimpleNamespace(twins=twins)
    return JointTwin(client, SimpleNamespace(uuid="twin-123", name="Arm"))


def test_joints_calibration_get_delegates_to_twins_api() -> None:
    twin = _joint_twin()
    twin.client.twins.get_calibration.return_value = {"ok": True}
    result = twin.joints.calibration.get(robot_type="leader")
    twin.client.twins.get_calibration.assert_called_once_with("twin-123", robot_type="leader")
    assert result == {"ok": True}


def test_joints_calibration_set_does_not_call_prepare_outbound() -> None:
    from unittest.mock import patch

    twin = _joint_twin()
    with patch.object(twin, "_prepare_outbound_command") as gate:
        twin.joints.calibration.set({"j": {}}, robot_type="leader")
    gate.assert_not_called()
    twin.client.twins.update_calibration.assert_called_once()


def test_get_calibration_emits_deprecation_warning() -> None:
    twin = _joint_twin()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        twin.get_calibration()
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
