"""Agent API helpers for Cyberwave."""

from __future__ import annotations

import json
from typing import Any, Literal

from cyberwave.exceptions import CyberwaveAPIError

ControlMode = Literal["simulation", "live", "preview"]


def _decode_json_response(response_data: Any) -> Any:
    response_data.read()
    payload = getattr(response_data, "data", None)
    if isinstance(payload, bytes):
        return json.loads(payload.decode("utf-8"))
    if isinstance(payload, str):
        return json.loads(payload)
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return payload
    return {}


class AgentClientBase:
    """Small REST helper shared by typed agent clients."""

    def __init__(self, api_client: Any):
        self._api_client = api_client

    def _call(
        self,
        method: str,
        path: str,
        *,
        path_params: dict[str, str] | None = None,
        body: Any = None,
    ) -> Any:
        _param = self._api_client.param_serialize(
            method=method,
            resource_path=path,
            path_params=path_params or {},
            body=body,
            header_params={"Content-Type": "application/json"}
            if body is not None
            else None,
            auth_settings=["CustomTokenAuthentication"],
        )
        return _decode_json_response(self._api_client.call_api(*_param))

    def _environment_call(
        self,
        method: str,
        path: str,
        environment_uuid: str,
        body: Any = None,
    ) -> Any:
        return self._call(
            method,
            path,
            path_params={"environment_uuid": environment_uuid},
            body=body,
        )


class ControlAgentClient(AgentClientBase):
    """Control Agent helpers for route/action planning and dispatch."""

    def plan(
        self,
        environment_uuid: str,
        message: str,
        *,
        twin_uuid: str | None = None,
        mode: ControlMode = "simulation",
        simulation_backend: str | None = None,
        llm_model_uuid: str | None = None,
        llm_model_name: str | None = None,
        mlmodel_uuid: str | None = None,
        controller_policy_uuid: str | None = None,
    ) -> dict[str, Any]:
        """Plan control actions for an environment without executing them."""
        if not environment_uuid:
            raise ValueError("environment_uuid is required")
        prompt = (message or "").strip()
        if not prompt:
            raise ValueError("message is required")

        payload: dict[str, Any] = {
            "prompt": prompt,
            "mode": mode,
        }
        optional_fields = {
            "twin_uuid": twin_uuid,
            "simulation_backend": simulation_backend,
            "llm_model_uuid": llm_model_uuid,
            "llm_model_name": llm_model_name,
            "mlmodel_uuid": mlmodel_uuid,
            "controller_policy_uuid": controller_policy_uuid,
        }
        payload.update(
            {key: value for key, value in optional_fields.items() if value is not None}
        )

        try:
            payload = self._environment_call(
                "POST",
                "/api/v1/agents/environments/{environment_uuid}/control/plan",
                environment_uuid,
                payload,
            )
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            raise CyberwaveAPIError(f"Failed to plan control action: {exc}") from exc

    def surfaces(self, environment_uuid: str) -> list[dict[str, Any]]:
        return self.list_surfaces(environment_uuid)

    def list_surfaces(self, environment_uuid: str) -> list[dict[str, Any]]:
        """List metadata-derived twin control surfaces for an environment."""
        if not environment_uuid:
            raise ValueError("environment_uuid is required")

        try:
            payload = self._environment_call(
                "GET",
                "/api/v1/agents/environments/{environment_uuid}/control/surfaces",
                environment_uuid,
            )
            return payload if isinstance(payload, list) else []
        except Exception as exc:
            raise CyberwaveAPIError(f"Failed to list control surfaces: {exc}") from exc

    def options(self, environment_uuid: str) -> dict[str, Any]:
        """Return control routes, action specs, and helper options."""
        if not environment_uuid:
            raise ValueError("environment_uuid is required")
        try:
            payload = self._environment_call(
                "GET",
                "/api/v1/agents/environments/{environment_uuid}/control/options",
                environment_uuid,
            )
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            raise CyberwaveAPIError(f"Failed to get control options: {exc}") from exc

    def resolve_route(
        self,
        environment_uuid: str,
        route_id: str,
        twin_uuid: str,
        inputs: dict[str, Any] | None = None,
        *,
        mode: ControlMode = "simulation",
        simulation_backend: str | None = None,
    ) -> dict[str, Any]:
        """Resolve one backend route into a plan without executing it."""
        if not environment_uuid or not route_id or not twin_uuid:
            raise ValueError("environment_uuid, route_id, and twin_uuid are required")
        body = {
            "route_id": route_id,
            "target_twin_uuid": twin_uuid,
            "inputs": inputs or {},
            "mode": mode,
        }
        if simulation_backend is not None:
            body["simulation_backend"] = simulation_backend
        try:
            payload = self._environment_call(
                "POST",
                "/api/v1/agents/environments/{environment_uuid}/control/routes/resolve",
                environment_uuid,
                body,
            )
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            raise CyberwaveAPIError(f"Failed to resolve control route: {exc}") from exc

    def dispatch(
        self,
        environment_uuid: str,
        action: dict[str, Any],
        *,
        confirmed: bool = False,
        mode: ControlMode = "simulation",
        simulation_backend: str | None = None,
        source_type: str | None = None,
    ) -> dict[str, Any]:
        """Dispatch one explicit action returned by plan() or resolve_route()."""
        if not environment_uuid or not action:
            raise ValueError("environment_uuid and action are required")
        body = {"action": action, "mode": mode, "confirmed": confirmed}
        body.update(
            {
                key: value
                for key, value in {
                    "simulation_backend": simulation_backend,
                    "source_type": source_type,
                }.items()
                if value is not None
            }
        )
        try:
            payload = self._environment_call(
                "POST",
                "/api/v1/agents/environments/{environment_uuid}/control/actions/dispatch",
                environment_uuid,
                body,
            )
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            raise CyberwaveAPIError(
                f"Failed to dispatch control action: {exc}"
            ) from exc


class EnvironmentAgentClient(AgentClientBase):
    """Environment editor agent helpers."""

    def models(self) -> dict[str, Any]:
        """List models that can run the environment editor agent."""
        try:
            payload = self._call("GET", "/api/v1/agents/environments/models")
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            raise CyberwaveAPIError(
                f"Failed to list environment agent models: {exc}"
            ) from exc

    def create(
        self,
        prompt: str,
        *,
        workspace_uuid: str | None = None,
        project_uuid: str | None = None,
        mlmodel_uuid: str | None = None,
        cyberwave_api_key: str = "",
    ) -> dict[str, Any]:
        """Create an environment from an agent prompt."""
        if not (prompt or "").strip():
            raise ValueError("prompt is required")
        body = {"prompt": prompt.strip(), "cyberwave_api_key": cyberwave_api_key}
        body.update(
            {
                key: value
                for key, value in {
                    "workspace_uuid": workspace_uuid,
                    "project_uuid": project_uuid,
                    "mlmodel_uuid": mlmodel_uuid,
                }.items()
                if value is not None
            }
        )
        try:
            payload = self._call("POST", "/api/v1/agents/environments", body=body)
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            raise CyberwaveAPIError(
                f"Failed to create agent environment: {exc}"
            ) from exc

    def message(
        self,
        environment_uuid: str,
        message: str,
        *,
        history: list[dict[str, str]] | None = None,
        pending_confirmation: dict[str, Any] | None = None,
        current_twins: list[dict[str, Any]] | None = None,
        mlmodel_uuid: str | None = None,
        cyberwave_api_key: str = "",
        image_base64: str | None = None,
        image_mime_type: str | None = None,
        image_name: str | None = None,
    ) -> dict[str, Any]:
        """Send a message to the environment editor agent."""
        if not environment_uuid:
            raise ValueError("environment_uuid is required")
        if not (message or "").strip():
            raise ValueError("message is required")
        body: dict[str, Any] = {
            "message": message.strip(),
            "cyberwave_api_key": cyberwave_api_key,
            "history": history or [],
            "current_twins": current_twins or [],
        }
        body.update(
            {
                key: value
                for key, value in {
                    "pending_confirmation": pending_confirmation,
                    "mlmodel_uuid": mlmodel_uuid,
                    "image_base64": image_base64,
                    "image_mime_type": image_mime_type,
                    "image_name": image_name,
                }.items()
                if value is not None
            }
        )
        try:
            payload = self._environment_call(
                "POST",
                "/api/v1/agents/environments/{environment_uuid}/messages",
                environment_uuid,
                body,
            )
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            raise CyberwaveAPIError(f"Failed to run environment agent: {exc}") from exc

    run = message


class WorkflowAgentClient(AgentClientBase):
    """Workflow Agent helpers for planning, previewing, drafting, and applying."""

    def plan(
        self,
        environment_uuid: str,
        prompt: str,
        *,
        mlmodel_uuid: str | None = None,
        llm_model_uuid: str | None = None,
        llm_model_name: str | None = None,
        twin_uuid: str | None = None,
        controller_policy_uuid: str | None = None,
        workflow_uuid: str | None = None,
        mode: ControlMode = "preview",
        simulation_backend: str | None = None,
    ) -> dict[str, Any]:
        """Run the read-only LLM workflow planning pass."""
        return self._prompt_post(
            environment_uuid,
            "/api/v1/agents/environments/{environment_uuid}/workflows/template-plan",
            prompt,
            {
                "mlmodel_uuid": mlmodel_uuid,
                "llm_model_uuid": llm_model_uuid,
                "llm_model_name": llm_model_name,
                "twin_uuid": twin_uuid,
                "controller_policy_uuid": controller_policy_uuid,
                "workflow_uuid": workflow_uuid,
                "mode": mode,
                "simulation_backend": simulation_backend,
            },
            "plan workflow",
        )

    def preview(
        self,
        environment_uuid: str,
        prompt: str,
        *,
        twin_uuid: str | None = None,
        controller_policy_uuid: str | None = None,
        mode: ControlMode = "preview",
        simulation_backend: str | None = None,
    ) -> dict[str, Any]:
        """Preview deterministic workflow strategy selection without mutation."""
        return self._prompt_post(
            environment_uuid,
            "/api/v1/agents/environments/{environment_uuid}/workflows/preview",
            prompt,
            {
                "twin_uuid": twin_uuid,
                "controller_policy_uuid": controller_policy_uuid,
                "mode": mode,
                "simulation_backend": simulation_backend,
            },
            "preview workflow",
        )

    def setup_and_draft(
        self,
        environment_uuid: str,
        prompt: str,
        *,
        confirmed_actions: list[dict[str, Any]] | None = None,
        agent_plan: dict[str, Any] | None = None,
        mlmodel_uuid: str | None = None,
        twin_uuid: str | None = None,
        controller_policy_uuid: str | None = None,
        mode: ControlMode = "simulation",
        simulation_backend: str | None = None,
        visibility: str = "private",
    ) -> dict[str, Any]:
        """Apply confirmed setup actions and draft a workflow."""
        return self._prompt_post(
            environment_uuid,
            "/api/v1/agents/environments/{environment_uuid}/workflows/setup-and-draft",
            prompt,
            {
                "confirmed_actions": confirmed_actions or [],
                "agent_plan": agent_plan,
                "mlmodel_uuid": mlmodel_uuid,
                "twin_uuid": twin_uuid,
                "controller_policy_uuid": controller_policy_uuid,
                "mode": mode,
                "simulation_backend": simulation_backend,
                "visibility": visibility,
            },
            "setup and draft workflow",
        )

    def apply_plan(
        self,
        environment_uuid: str,
        workflow_uuid: str,
        prompt: str,
        *,
        agent_plan: dict[str, Any] | None = None,
        workflow_edit: dict[str, Any] | None = None,
        twin_uuid: str | None = None,
        controller_policy_uuid: str | None = None,
        mode: ControlMode = "preview",
        simulation_backend: str | None = None,
    ) -> dict[str, Any]:
        """Apply a constrained Workflow Agent edit to an existing workflow."""
        if not workflow_uuid:
            raise ValueError("workflow_uuid is required")
        body = self._prompt_body(
            prompt,
            {
                "agent_plan": agent_plan,
                "workflow_edit": workflow_edit,
                "twin_uuid": twin_uuid,
                "controller_policy_uuid": controller_policy_uuid,
                "mode": mode,
                "simulation_backend": simulation_backend,
            },
        )
        try:
            payload = self._call(
                "POST",
                "/api/v1/agents/environments/{environment_uuid}/workflows/{workflow_uuid}/apply-plan",
                path_params={
                    "environment_uuid": environment_uuid,
                    "workflow_uuid": workflow_uuid,
                },
                body=body,
            )
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            raise CyberwaveAPIError(
                f"Failed to apply workflow agent plan: {exc}"
            ) from exc

    def _prompt_post(
        self,
        environment_uuid: str,
        path: str,
        prompt: str,
        options: dict[str, Any],
        action: str,
    ) -> dict[str, Any]:
        if not environment_uuid:
            raise ValueError("environment_uuid is required")
        body = self._prompt_body(prompt, options)
        try:
            payload = self._environment_call("POST", path, environment_uuid, body)
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            raise CyberwaveAPIError(f"Failed to {action}: {exc}") from exc

    @staticmethod
    def _prompt_body(prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        if not (prompt or "").strip():
            raise ValueError("prompt is required")
        body = {"prompt": prompt.strip()}
        body.update({key: value for key, value in options.items() if value is not None})
        return body


class EmbodimentAgentClient(AgentClientBase):
    """Read embodiment context from existing agent planning surfaces."""

    def context(
        self,
        environment_uuid: str,
        *,
        prompt: str = "Summarize embodiment context.",
        twin_uuid: str | None = None,
        controller_policy_uuid: str | None = None,
        mode: ControlMode = "preview",
        simulation_backend: str | None = None,
    ) -> dict[str, Any]:
        """Return server-built embodiment context for an environment/twin."""
        preview = WorkflowAgentClient(self._api_client).preview(
            environment_uuid,
            prompt,
            twin_uuid=twin_uuid,
            controller_policy_uuid=controller_policy_uuid,
            mode=mode,
            simulation_backend=simulation_backend,
        )
        context = preview.get("embodiment_context")
        return context if isinstance(context, dict) else {}

    profile = context


class AgentManager:
    """Container for agent API surfaces."""

    def __init__(self, api_client: Any):
        self.control = ControlAgentClient(api_client)
        self.environment = EnvironmentAgentClient(api_client)
        self.workflow = WorkflowAgentClient(api_client)
        self.embodiment = EmbodimentAgentClient(api_client)
