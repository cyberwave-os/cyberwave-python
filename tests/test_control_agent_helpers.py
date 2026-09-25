import json

import pytest

from cyberwave.actions import ActionsClient
from cyberwave.agents import AgentManager
from cyberwave.exceptions import CyberwaveAPIError


class FakeResponse:
    def __init__(self, payload):
        self.data = json.dumps(payload).encode("utf-8")

    def read(self):
        return None


class FakeApiClient:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.serialized = []
        self.calls = []

    def param_serialize(self, **kwargs):
        self.serialized.append(kwargs)
        return (kwargs,)

    def call_api(self, *args):
        self.calls.append(args)
        payload = self.payloads.pop(0)
        return FakeResponse(payload)


def test_control_agent_plan_posts_to_plan_endpoint():
    api_client = FakeApiClient(
        {
            "summary": "Move Go2 to Waypoint A.",
            "mode": "simulation",
            "source_type": "sim",
            "actions": [],
            "readiness": "ready",
        }
    )
    agents = AgentManager(api_client)

    result = agents.control.plan(
        "env-uuid",
        "Move the Go2 to Waypoint A",
        twin_uuid="twin-uuid",
        mode="simulation",
        simulation_backend="playground",
    )

    assert result["summary"] == "Move Go2 to Waypoint A."
    assert api_client.serialized[0]["method"] == "POST"
    assert (
        api_client.serialized[0]["resource_path"]
        == "/api/v1/agents/environments/{environment_uuid}/control/plan"
    )
    assert api_client.serialized[0]["path_params"] == {"environment_uuid": "env-uuid"}
    assert api_client.serialized[0]["body"] == {
        "prompt": "Move the Go2 to Waypoint A",
        "mode": "simulation",
        "twin_uuid": "twin-uuid",
        "simulation_backend": "playground",
    }


def test_control_agent_lists_surfaces_from_read_endpoint():
    api_client = FakeApiClient(
        [
            {
                "twin_uuid": "twin-uuid",
                "capabilities": ["locomotion"],
                "controls": [{"kind": "navigation_command", "available": True}],
            }
        ]
    )
    agents = AgentManager(api_client)

    result = agents.control.list_surfaces("env-uuid")

    assert result[0]["twin_uuid"] == "twin-uuid"
    assert api_client.serialized[0]["method"] == "GET"
    assert (
        api_client.serialized[0]["resource_path"]
        == "/api/v1/agents/environments/{environment_uuid}/control/surfaces"
    )
    assert api_client.serialized[0]["path_params"] == {"environment_uuid": "env-uuid"}


def test_control_agent_options_gets_backend_contract():
    api_client = FakeApiClient({"action_specs": [{"kind": "navigation_command"}]})
    agents = AgentManager(api_client)

    result = agents.control.options("env-uuid")

    assert result["action_specs"][0]["kind"] == "navigation_command"
    assert api_client.serialized[0]["method"] == "GET"
    assert (
        api_client.serialized[0]["resource_path"]
        == "/api/v1/agents/environments/{environment_uuid}/control/options"
    )


def test_control_agent_resolve_route_posts_backend_payload():
    api_client = FakeApiClient({"summary": "Resolved.", "dispatchable_actions": []})
    agents = AgentManager(api_client)

    result = agents.control.resolve_route(
        "env-uuid",
        "navigation",
        "twin-uuid",
        {"position": [1, 2, 0]},
        simulation_backend="playground",
    )

    assert result["summary"] == "Resolved."
    assert api_client.serialized[0]["method"] == "POST"
    assert (
        api_client.serialized[0]["resource_path"]
        == "/api/v1/agents/environments/{environment_uuid}/control/routes/resolve"
    )
    assert api_client.serialized[0]["body"] == {
        "route_id": "navigation",
        "target_twin_uuid": "twin-uuid",
        "inputs": {"position": [1, 2, 0]},
        "mode": "simulation",
        "simulation_backend": "playground",
    }


def test_control_agent_dispatch_posts_explicit_action():
    api_client = FakeApiClient({"action_id": "action-1", "status": "queued"})
    agents = AgentManager(api_client)
    action = {
        "kind": "navigation_command",
        "target_twin_uuid": "twin-uuid",
        "payload": {"position": [1, 2, 0]},
    }

    result = agents.control.dispatch("env-uuid", action, confirmed=True)

    assert result["action_id"] == "action-1"
    assert api_client.serialized[0]["method"] == "POST"
    assert (
        api_client.serialized[0]["resource_path"]
        == "/api/v1/agents/environments/{environment_uuid}/control/actions/dispatch"
    )
    assert api_client.serialized[0]["body"] == {
        "action": action,
        "mode": "simulation",
        "confirmed": True,
    }


def test_environment_agent_uses_canonical_routes():
    api_client = FakeApiClient(
        {"models": []},
        {"environment": {"uuid": "env-new"}},
        {"answer": "done"},
    )
    agents = AgentManager(api_client)

    assert agents.environment.models()["models"] == []
    created = agents.environment.create("create a warehouse", workspace_uuid="ws-1")
    result = agents.environment.message("env-uuid", "add a safety zone")

    assert created["environment"]["uuid"] == "env-new"
    assert result["answer"] == "done"
    assert [call["resource_path"] for call in api_client.serialized] == [
        "/api/v1/agents/environments/models",
        "/api/v1/agents/environments",
        "/api/v1/agents/environments/{environment_uuid}/messages",
    ]
    assert api_client.serialized[1]["body"] == {
        "prompt": "create a warehouse",
        "cyberwave_api_key": "",
        "workspace_uuid": "ws-1",
    }
    assert api_client.serialized[2]["body"] == {
        "message": "add a safety zone",
        "cyberwave_api_key": "",
        "history": [],
        "current_twins": [],
    }


def test_workflow_agent_uses_plan_preview_draft_and_apply_routes():
    api_client = FakeApiClient(
        {"agent_plan": {}},
        {"preview": {}},
        {"workflow": {"uuid": "wf-new"}},
        {"workflow": {"uuid": "wf-1"}},
    )
    agents = AgentManager(api_client)

    agents.workflow.plan("env-uuid", "plan inspection", llm_model_name="gemini")
    agents.workflow.preview("env-uuid", "preview inspection", mode="preview")
    agents.workflow.setup_and_draft(
        "env-uuid",
        "draft inspection",
        confirmed_actions=[{"action": "select_existing_twin"}],
        setup_mode="auto",
    )
    agents.workflow.apply_plan(
        "env-uuid",
        "wf-1",
        "edit inspection",
        workflow_edit={"name": "Updated"},
    )

    assert [call["resource_path"] for call in api_client.serialized] == [
        "/api/v1/agents/environments/{environment_uuid}/workflows/template-plan",
        "/api/v1/agents/environments/{environment_uuid}/workflows/preview",
        "/api/v1/agents/environments/{environment_uuid}/workflows/setup-and-draft",
        "/api/v1/agents/environments/{environment_uuid}/workflows/{workflow_uuid}/apply-plan",
    ]
    assert api_client.serialized[0]["body"] == {
        "prompt": "plan inspection",
        "llm_model_name": "gemini",
        "mode": "preview",
    }
    assert api_client.serialized[2]["body"] == {
        "prompt": "draft inspection",
        "confirmed_actions": [{"action": "select_existing_twin"}],
        "mode": "simulation",
        "visibility": "private",
        "setup_mode": "auto",
    }
    assert api_client.serialized[3]["path_params"] == {
        "environment_uuid": "env-uuid",
        "workflow_uuid": "wf-1",
    }


def test_embodiment_agent_returns_context_from_preview_surface():
    api_client = FakeApiClient(
        {"embodiment_context": {"environment_uuid": "env-uuid", "twin_uuid": "twin-1"}}
    )
    agents = AgentManager(api_client)

    result = agents.embodiment.context("env-uuid", twin_uuid="twin-1")

    assert result == {"environment_uuid": "env-uuid", "twin_uuid": "twin-1"}
    assert (
        api_client.serialized[0]["resource_path"]
        == "/api/v1/agents/environments/{environment_uuid}/workflows/preview"
    )
    assert api_client.serialized[0]["body"] == {
        "prompt": "Summarize embodiment context.",
        "twin_uuid": "twin-1",
        "mode": "preview",
    }


def test_actions_get_status_uses_existing_twin_scoped_route():
    api_client = FakeApiClient({"action_id": "action-1", "status": "running"})
    actions = ActionsClient(api_client)

    result = actions.get_status("action-1", twin_uuid="twin-uuid")

    assert result["status"] == "running"
    assert api_client.serialized[0]["method"] == "GET"
    assert (
        api_client.serialized[0]["resource_path"]
        == "/api/v1/twins/{uuid}/actions/{action_id}"
    )
    assert api_client.serialized[0]["path_params"] == {
        "uuid": "twin-uuid",
        "action_id": "action-1",
    }


def test_actions_wait_polls_until_terminal(monkeypatch):
    api_client = FakeApiClient(
        {"action_id": "action-1", "status": "queued"},
        {"action_id": "action-1", "status": "completed"},
    )
    actions = ActionsClient(api_client)
    monkeypatch.setattr("cyberwave.actions.time.sleep", lambda _seconds: None)

    result = actions.wait(
        "action-1",
        twin_uuid="twin-uuid",
        timeout=5,
        poll_interval=0.1,
    )

    assert result["status"] == "completed"
    assert len(api_client.calls) == 2


def test_actions_get_status_requires_twin_uuid():
    actions = ActionsClient(FakeApiClient())

    with pytest.raises(ValueError, match="twin_uuid"):
        actions.get_status("action-1", twin_uuid="")


class StatusResponse(FakeResponse):
    """A response that carries an HTTP status, as the real REST client does."""

    def __init__(self, payload, status):
        super().__init__(payload)
        self.status = status


class StatusApiClient(FakeApiClient):
    def __init__(self, *responses):
        super().__init__()
        self.responses = list(responses)

    def call_api(self, *args):
        self.calls.append(args)
        # The last response repeats, so a spin-fast timeout test does not have
        # to guess how many polls fit inside the deadline.
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


def test_actions_get_status_raises_on_a_non_2xx_response():
    # call_api does not check the HTTP status (the generated client does that
    # in response_deserialize, which this path skips), so without an explicit
    # check an error body reads as an action record with no status.
    api_client = StatusApiClient(StatusResponse({"detail": "Action not found"}, 404))
    actions = ActionsClient(api_client)

    with pytest.raises(CyberwaveAPIError, match="HTTP 404: Action not found"):
        actions.get_status("action-1", twin_uuid="twin-uuid")


def test_actions_wait_survives_a_transient_poll_failure(monkeypatch):
    # One bad poll must not end a wait that is holding a robot in motion.
    api_client = StatusApiClient(
        StatusResponse({"detail": "Service Unavailable"}, 503),
        StatusResponse({"action_id": "action-1", "status": "completed"}, 200),
    )
    actions = ActionsClient(api_client)
    monkeypatch.setattr("cyberwave.actions.time.sleep", lambda _seconds: None)

    result = actions.wait(
        "action-1", twin_uuid="twin-uuid", timeout=5, poll_interval=0.1
    )

    assert result["status"] == "completed"
    assert len(api_client.calls) == 2


def test_actions_wait_names_the_poll_failure_in_the_timeout(monkeypatch):
    # A permanently failing poll still raises TimeoutError — callers' recovery
    # paths are written against it — but the message has to say why.
    api_client = StatusApiClient(StatusResponse({"detail": "Action not found"}, 404))
    actions = ActionsClient(api_client)
    monkeypatch.setattr("cyberwave.actions.time.sleep", lambda _seconds: None)

    with pytest.raises(TimeoutError, match="last status poll failed.*HTTP 404"):
        actions.wait("action-1", twin_uuid="twin-uuid", timeout=0.2, poll_interval=0.1)
