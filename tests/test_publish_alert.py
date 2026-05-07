"""Regression tests for ``Cyberwave.publish_alert()``."""

from unittest.mock import patch

import pytest


@pytest.fixture
def cw_client():
    """Create a Cyberwave client with mocked REST dependencies."""
    with (
        patch("cyberwave.rest.ApiClient"),
        patch("cyberwave.rest.DefaultApi"),
        patch("cyberwave.rest.Configuration"),
    ):
        from cyberwave.client import Cyberwave

        client = Cyberwave(api_key="test-key", base_url="http://localhost:8000")
        client.config.workspace_id = "workspace-uuid"
        return client


def test_publish_alert_builds_expected_payload(cw_client):
    with patch(
        "cyberwave.alerts._create_alert", return_value={"uuid": "alert-uuid"}
    ) as mock_create:
        cw_client.publish_alert(
            "twin-uuid",
            "Person detected",
            description="A person entered the workspace",
            alert_type="person_detected",
            severity="WARNING",
            metadata={"confidence": 0.97},
        )

    mock_create.assert_called_once()
    called_client, payload = mock_create.call_args.args
    assert called_client is cw_client
    assert payload == {
        "name": "Person detected",
        "description": "A person entered the workspace",
        "alert_type": "person_detected",
        "severity": "warning",
        "source_type": "edge",
        "category": "business",
        "twin_uuid": "twin-uuid",
        "workspace_uuid": "workspace-uuid",
        "metadata": {"confidence": 0.97},
    }


def test_publish_alert_omits_optional_workspace_and_metadata(cw_client):
    cw_client.config.workspace_id = None

    with patch(
        "cyberwave.alerts._create_alert", return_value={"uuid": "alert-uuid"}
    ) as mock_create:
        cw_client.publish_alert("twin-uuid", "Heartbeat")

    _, payload = mock_create.call_args.args
    assert payload["name"] == "Heartbeat"
    assert payload["description"] == ""
    assert payload["alert_type"] == ""
    assert payload["severity"] == "info"
    assert payload["source_type"] == "edge"
    assert payload["category"] == "business"
    assert payload["twin_uuid"] == "twin-uuid"
    assert "workspace_uuid" not in payload
    assert "metadata" not in payload


def test_publish_alert_sets_force_when_requested(cw_client):
    with patch(
        "cyberwave.alerts._create_alert", return_value={"uuid": "alert-uuid"}
    ) as mock_create:
        cw_client.publish_alert("twin-uuid", "Person detected", force=True)

    _, payload = mock_create.call_args.args
    assert payload["force"] is True


def test_publish_alert_swallows_backend_errors(cw_client):
    with (
        patch("cyberwave.alerts._create_alert", side_effect=RuntimeError("boom")),
        patch("cyberwave.client.logger.exception") as mock_exception,
    ):
        cw_client.publish_alert(
            "twin-uuid",
            "Person detected",
            alert_type="person_detected",
        )

    mock_exception.assert_called_once()
