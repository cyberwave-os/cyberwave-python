from unittest.mock import MagicMock, patch

from cyberwave.alerts import TwinAlertManager


def _make_twin():
    twin = MagicMock()
    twin.uuid = "twin-uuid"
    twin.environment_id = None
    twin.client = MagicMock()
    twin.client.config.workspace_id = "workspace-uuid"
    return twin


def test_twin_alert_manager_create_omits_force_by_default():
    twin = _make_twin()
    manager = TwinAlertManager(twin)

    with patch("cyberwave.alerts._create_alert", return_value={"uuid": "alert-uuid"}) as mock_create:
        manager.create(name="Calibration needed")

    _, payload = mock_create.call_args.args
    assert "force" not in payload


def test_twin_alert_manager_create_sets_force_when_requested():
    twin = _make_twin()
    manager = TwinAlertManager(twin)

    with patch("cyberwave.alerts._create_alert", return_value={"uuid": "alert-uuid"}) as mock_create:
        manager.create(name="Calibration needed", force=True)

    _, payload = mock_create.call_args.args
    assert payload["force"] is True
