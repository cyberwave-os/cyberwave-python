"""
Unit tests for ``WorkflowExecutionReporter`` and ``WorkflowExecutionManager``.

Reporters speak MQTT only. We mock the MQTT publish hook so no network
access is required. Tests focus on:

- ``start()`` sends the initial ``started`` event on the right topic
  and returns a reporter pointing at the right execution.
- Node / finish events route to the correct topics and honor the
  broker's ``topic_prefix``.
- Idempotency: ``finished()`` twice is a no-op.
- Error handling: publish failures are swallowed unless ``strict=True``,
  except the initial ``started`` which is always strict.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from cyberwave.exceptions import CyberwaveError
from cyberwave.workflow_executions import WorkflowExecutionManager

WORKFLOW_UUID = "11111111-1111-1111-1111-111111111111"
NODE_UUID = "22222222-2222-2222-2222-222222222222"


def _make_client(*, mqtt_connected: bool = True) -> MagicMock:
    """Return a mock ``Cyberwave`` client with a stub MQTT transport."""
    client = MagicMock()
    mqtt = MagicMock()
    mqtt.connected = mqtt_connected
    mqtt.topic_prefix = ""
    client.mqtt = mqtt
    return client


def _make_client_with_api(response_payload: list[dict]) -> MagicMock:
    client = _make_client()
    api_client = MagicMock()
    api_client.param_serialize.return_value = ()
    response = MagicMock()
    response.data = json.dumps(response_payload).encode("utf-8")
    api_client.call_api.return_value = response
    client.api.api_client = api_client
    return client


def _topics(client: MagicMock) -> list[str]:
    return [call.args[0] for call in client.mqtt.publish.call_args_list]


def _payloads(client: MagicMock) -> list[dict]:
    return [call.args[1] for call in client.mqtt.publish.call_args_list]


# ======================================================================
# Happy path
# ======================================================================


class TestStart:

    def test_publishes_started_event_on_workflow_topic(self):
        client = _make_client()
        manager = WorkflowExecutionManager(client)

        reporter = manager.start(
            workflow_uuid=WORKFLOW_UUID,
            trigger_data={"source": "unit-test"},
        )

        assert _topics(client) == [
            f"cyberwave/workflow/{WORKFLOW_UUID}/execution/started"
        ]
        payload = _payloads(client)[0]
        assert payload["execution_uuid"] == reporter.execution_uuid
        assert payload["trigger_data"] == {"source": "unit-test"}
        assert payload["source_type"] == "edge"

    def test_uses_caller_provided_execution_uuid(self):
        client = _make_client()
        manager = WorkflowExecutionManager(client)

        reporter = manager.start(
            workflow_uuid=WORKFLOW_UUID,
            execution_uuid="deadbeef-dead-beef-dead-beefdeadbeef",
        )
        assert reporter.execution_uuid == "deadbeef-dead-beef-dead-beefdeadbeef"


class TestListCaptures:

    def test_filters_execution_captures_by_workflow_node_uuid(self):
        client = _make_client_with_api(
            [
                {
                    "uuid": "attachment-1",
                    "file_url": "https://example.com/one.jpg",
                    "metadata": {"workflow_node_uuid": NODE_UUID},
                },
                {
                    "uuid": "attachment-2",
                    "file_url": "https://example.com/two.jpg",
                    "metadata": {"workflow_node_uuid": "other-node"},
                },
            ]
        )
        manager = WorkflowExecutionManager(client)

        captures = manager.list_captures("exec-123", workflow_node_uuid=NODE_UUID)

        assert captures == [
            {
                "attachment_uuid": "attachment-1",
                "uuid": "attachment-1",
                "file_url": "https://example.com/one.jpg",
                "metadata": {"workflow_node_uuid": NODE_UUID},
            }
        ]
        client.api.api_client.param_serialize.assert_called_once_with(
            method="GET",
            resource_path="/api/v1/workflows/executions/exec-123/captures",
            auth_settings=["CustomTokenAuthentication"],
        )


class TestNodeAndFinishEvents:

    def test_node_events_land_on_per_node_topic(self):
        client = _make_client()
        reporter = WorkflowExecutionManager(client).start(
            workflow_uuid=WORKFLOW_UUID
        )
        client.mqtt.publish.reset_mock()

        reporter.node_started(NODE_UUID, input_data=[{"waypoint": "dock-a"}])
        reporter.node_finished(NODE_UUID, output_data=[{"arrived": True}])

        topics = _topics(client)
        payloads = _payloads(client)
        expected_topic = (
            f"cyberwave/workflow/{WORKFLOW_UUID}/execution/"
            f"{reporter.execution_uuid}/node/{NODE_UUID}"
        )
        assert topics == [expected_topic, expected_topic]
        assert payloads[0]["status"] == "running"
        assert payloads[0]["input_data"] == [{"waypoint": "dock-a"}]
        assert payloads[1]["status"] == "success"
        assert payloads[1]["output_data"] == [{"arrived": True}]

    def test_node_error_carries_error_message(self):
        client = _make_client()
        reporter = WorkflowExecutionManager(client).start(
            workflow_uuid=WORKFLOW_UUID
        )
        client.mqtt.publish.reset_mock()

        reporter.node_error(NODE_UUID, error="boom")

        payload = _payloads(client)[0]
        assert payload["status"] == "error"
        assert payload["error_message"] == "boom"

    def test_finished_publishes_terminal_status(self):
        client = _make_client()
        reporter = WorkflowExecutionManager(client).start(
            workflow_uuid=WORKFLOW_UUID
        )
        client.mqtt.publish.reset_mock()

        reporter.finished(status="success")

        assert _topics(client) == [
            f"cyberwave/workflow/{WORKFLOW_UUID}/execution/"
            f"{reporter.execution_uuid}/finished"
        ]
        assert _payloads(client)[0]["status"] == "success"

    def test_finished_rejects_non_terminal_status(self):
        client = _make_client()
        reporter = WorkflowExecutionManager(client).start(
            workflow_uuid=WORKFLOW_UUID
        )

        with pytest.raises(ValueError, match="Invalid execution status"):
            reporter.finished(status="running")

    def test_finished_is_idempotent(self):
        client = _make_client()
        reporter = WorkflowExecutionManager(client).start(
            workflow_uuid=WORKFLOW_UUID
        )
        client.mqtt.publish.reset_mock()

        reporter.finished(status="success")
        reporter.finished(status="success")

        assert client.mqtt.publish.call_count == 1


# ======================================================================
# Topic prefix
# ======================================================================


class TestTopicPrefix:

    def test_all_topics_carry_broker_prefix(self):
        client = _make_client()
        client.mqtt.topic_prefix = "dev/"

        reporter = WorkflowExecutionManager(client).start(
            workflow_uuid=WORKFLOW_UUID
        )
        reporter.node_started(NODE_UUID)
        reporter.finished(status="success")

        topics = _topics(client)
        assert all(t.startswith("dev/") for t in topics)
        assert topics[0] == (
            f"dev/cyberwave/workflow/{WORKFLOW_UUID}/execution/started"
        )
        assert topics[1] == (
            f"dev/cyberwave/workflow/{WORKFLOW_UUID}/execution/"
            f"{reporter.execution_uuid}/node/{NODE_UUID}"
        )
        assert topics[2] == (
            f"dev/cyberwave/workflow/{WORKFLOW_UUID}/execution/"
            f"{reporter.execution_uuid}/finished"
        )


# ======================================================================
# Context manager sugar
# ======================================================================


class TestContextManager:

    def test_clean_exit_marks_success(self):
        client = _make_client()
        manager = WorkflowExecutionManager(client)

        with manager.start(workflow_uuid=WORKFLOW_UUID) as reporter:
            pass

        payloads = _payloads(client)
        assert payloads[0].get("execution_uuid") == reporter.execution_uuid
        assert payloads[-1]["status"] == "success"
        assert reporter._finished is True

    def test_exception_marks_error(self):
        client = _make_client()
        manager = WorkflowExecutionManager(client)

        with pytest.raises(RuntimeError):
            with manager.start(workflow_uuid=WORKFLOW_UUID):
                raise RuntimeError("kaboom")

        finish_payload = _payloads(client)[-1]
        assert finish_payload["status"] == "error"
        assert "kaboom" in finish_payload["error_message"]


# ======================================================================
# Error handling
# ======================================================================


class TestErrorHandling:

    def test_started_always_raises_on_publish_failure(self):
        """The initial ``started`` must be strict even when the
        reporter is not — subsequent node events would 404 otherwise."""
        client = _make_client()
        client.mqtt.publish.side_effect = RuntimeError("broker offline")

        with pytest.raises(CyberwaveError, match="broker offline"):
            WorkflowExecutionManager(client).start(workflow_uuid=WORKFLOW_UUID)

    def test_started_always_raises_when_mqtt_not_connected(self):
        client = _make_client(mqtt_connected=False)

        with pytest.raises(CyberwaveError, match="not connected"):
            WorkflowExecutionManager(client).start(workflow_uuid=WORKFLOW_UUID)

    def test_node_failure_is_swallowed_by_default(self):
        client = _make_client()
        reporter = WorkflowExecutionManager(client).start(
            workflow_uuid=WORKFLOW_UUID
        )
        client.mqtt.publish.side_effect = RuntimeError("transient blip")

        # Should log but not raise.
        reporter.node_started(NODE_UUID)

    def test_node_failure_raises_when_strict(self):
        client = _make_client()
        reporter = WorkflowExecutionManager(client).start(
            workflow_uuid=WORKFLOW_UUID,
            strict=True,
        )
        client.mqtt.publish.side_effect = RuntimeError("transient blip")

        with pytest.raises(CyberwaveError, match="transient blip"):
            reporter.node_started(NODE_UUID)
