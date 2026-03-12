from unittest.mock import MagicMock, patch

import pytest

from cyberwave.alerts import Alert, TwinAlertManager
from cyberwave.exceptions import CyberwaveError


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


def test_twin_alert_manager_create_includes_media_when_provided():
    twin = _make_twin()
    manager = TwinAlertManager(twin)

    with patch(
        "cyberwave.alerts._create_alert", return_value={"uuid": "alert-uuid"}
    ) as mock_create:
        manager.create(
            name="Calibration needed",
            media="https://cdn.example.com/alerts/calibration.gif",
        )

    _, payload = mock_create.call_args.args
    assert payload["media"] == "https://cdn.example.com/alerts/calibration.gif"


def test_alert_media_property_reads_payload_value():
    client = MagicMock()
    alert = Alert(
        client,
        {"uuid": "alert-uuid", "media": "https://cdn.example.com/alerts/help.mp4"},
    )

    assert alert.media == "https://cdn.example.com/alerts/help.mp4"


def test_alert_press_button_delegates_to_backend_action():
    client = MagicMock()
    alert = Alert(client, {"uuid": "alert-uuid"})

    with patch(
        "cyberwave.alerts._post_alert_button", return_value={"uuid": "alert-uuid"}
    ) as mock_press:
        alert.press_button(2)

    mock_press.assert_called_once_with(client, "alert-uuid", 2)


def test_alert_press_button_rejects_negative_index():
    client = MagicMock()
    alert = Alert(client, {"uuid": "alert-uuid"})

    with patch("cyberwave.alerts._post_alert_button") as mock_press:
        with pytest.raises(CyberwaveError):
            alert.press_button(-1)
        mock_press.assert_not_called()
