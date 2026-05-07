"""
Workflow and WorkflowRun abstractions for the Cyberwave SDK.

Provides:
- ``Workflow`` object with convenience ``trigger()``, ``runs()``, and
  ``is_running()`` methods.
- ``WorkflowRun`` object with ``refresh()``, ``wait()``, ``cancel()``,
  and ``on_status_change()`` for MQTT-based real-time updates.
- ``WorkflowManager`` (``client.workflows``) for list / get / trigger.
- ``WorkflowRunManager`` (``client.workflow_runs``) for list / get / cancel.

Example::

    client = Cyberwave(api_key="...")

    # List available workflows
    workflows = client.workflows.list()
    for wf in workflows:
        print(f"{wf.name} ({wf.uuid}) — status: {wf.status}")

    # Trigger a workflow
    run = client.workflows.trigger(
        workflow_id="wf-uuid",
        inputs={"target_position": [1.0, 2.0, 0.0], "speed": 0.5},
    )
    print(f"Run started: {run.uuid}")

    # Poll until done
    run.wait(timeout=60)
    print(f"Final status: {run.status}")
    print(f"Output: {run.result}")

    # Check if a workflow is currently running
    wf = client.workflows.get("wf-uuid")
    if wf.is_running():
        print("Workflow is currently executing")
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from .exceptions import CyberwaveError, CyberwaveTimeoutError

if TYPE_CHECKING:
    from .client import Cyberwave

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = frozenset({"success", "error", "canceled"})
CANCELABLE_STATUSES = frozenset({"running", "waiting", "requested"})

_AUTH = ["CustomTokenAuthentication"]


def _attr(data: Any, key: str, default: Any = None) -> Any:
    """Read *key* from an object or dict, returning *default* if missing."""
    if hasattr(data, key):
        return getattr(data, key)
    if isinstance(data, dict):
        return data.get(key, default)
    return default


# ======================================================================
# Workflow object
# ======================================================================


class Workflow:
    """
    A workflow with convenience methods.

    Typically obtained from :meth:`WorkflowManager.list` or
    :meth:`WorkflowManager.get`.
    """

    def __init__(self, client: "Cyberwave", data: Any) -> None:
        self._client = client
        self._data = data

    @property
    def uuid(self) -> str:
        return str(_attr(self._data, "uuid", ""))

    @property
    def name(self) -> str:
        return str(_attr(self._data, "name", ""))

    @property
    def slug(self) -> str:
        return str(_attr(self._data, "slug", ""))

    @property
    def description(self) -> str:
        return str(_attr(self._data, "description", ""))

    @property
    def is_active(self) -> bool:
        return bool(_attr(self._data, "is_active", False))

    @property
    def status(self) -> str:
        """Human-friendly status derived from ``is_active``."""
        return "active" if self.is_active else "inactive"

    @property
    def workspace_uuid(self) -> str:
        return str(_attr(self._data, "workspace_uuid", ""))

    @property
    def visibility(self) -> str:
        return str(_attr(self._data, "visibility", ""))

    @property
    def created_at(self) -> Optional[str]:
        val = _attr(self._data, "created_at")
        return str(val) if val else None

    @property
    def updated_at(self) -> Optional[str]:
        val = _attr(self._data, "updated_at")
        return str(val) if val else None

    @property
    def metadata(self) -> Dict[str, Any]:
        val = _attr(self._data, "metadata")
        return val if isinstance(val, dict) else {}

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def trigger(self, inputs: Optional[Dict[str, Any]] = None) -> "WorkflowRun":
        """Trigger a new run of this workflow.

        Args:
            inputs: Payload passed to the workflow trigger.

        Returns:
            A :class:`WorkflowRun` representing the started execution.
        """
        return self._client.workflows.trigger(self.uuid, inputs=inputs)

    def runs(
        self, *, status: Optional[str] = None
    ) -> List["WorkflowRun"]:
        """List past runs of this workflow.

        Args:
            status: Optional status filter (e.g. ``"running"``).

        Returns:
            List of :class:`WorkflowRun` instances.
        """
        return self._client.workflow_runs.list(
            workflow_id=self.uuid, status=status
        )

    def is_running(self) -> bool:
        """Check whether this workflow has any currently active run.

        A run is considered active when its status is ``"running"``,
        ``"waiting"``, or ``"requested"``.

        Returns:
            ``True`` if at least one active run exists, ``False`` otherwise.
        """
        active_runs = self._client.workflow_runs.list(
            workflow_id=self.uuid, status="running"
        )
        if active_runs:
            return True
        waiting_runs = self._client.workflow_runs.list(
            workflow_id=self.uuid, status="waiting"
        )
        if waiting_runs:
            return True
        requested_runs = self._client.workflow_runs.list(
            workflow_id=self.uuid, status="requested"
        )
        return bool(requested_runs)

    def __repr__(self) -> str:
        return (
            f"Workflow(uuid='{self.uuid}', name='{self.name}', "
            f"status='{self.status}')"
        )


# ======================================================================
# WorkflowRun object
# ======================================================================


class WorkflowRun:
    """
    A single workflow run (execution) with polling and MQTT helpers.

    Typically obtained from :meth:`WorkflowManager.trigger`,
    :meth:`WorkflowRunManager.get`, or :meth:`Workflow.trigger`.
    """

    def __init__(self, client: "Cyberwave", data: Any) -> None:
        self._client = client
        self._data = data

    @property
    def uuid(self) -> str:
        return str(_attr(self._data, "uuid", ""))

    @property
    def workflow_id(self) -> str:
        return str(_attr(self._data, "workflow_id", ""))

    @property
    def status(self) -> str:
        return str(_attr(self._data, "status", ""))

    @property
    def inputs(self) -> Dict[str, Any]:
        val = _attr(self._data, "inputs")
        return val if isinstance(val, dict) else {}

    @property
    def result(self) -> Optional[Dict[str, Any]]:
        val = _attr(self._data, "result")
        return val if isinstance(val, dict) else None

    @property
    def error(self) -> Optional[str]:
        val = _attr(self._data, "error")
        return str(val) if val else None

    @property
    def started_at(self) -> Optional[str]:
        val = _attr(self._data, "started_at")
        return str(val) if val else None

    @property
    def finished_at(self) -> Optional[str]:
        val = _attr(self._data, "finished_at")
        return str(val) if val else None

    @property
    def completed_at(self) -> Optional[str]:
        """Alias for :attr:`finished_at`."""
        return self.finished_at

    @property
    def duration(self) -> Optional[float]:
        """Elapsed seconds between start and finish, or ``None``."""
        from datetime import datetime

        start = _attr(self._data, "started_at")
        end = _attr(self._data, "finished_at")
        if not start or not end:
            return None
        try:
            if isinstance(start, datetime) and isinstance(end, datetime):
                return (end - start).total_seconds()
            start_dt = datetime.fromisoformat(str(start))
            end_dt = datetime.fromisoformat(str(end))
            return (end_dt - start_dt).total_seconds()
        except (ValueError, TypeError):
            return None

    @property
    def is_terminal(self) -> bool:
        """Whether the run has reached a terminal state."""
        return self.status in TERMINAL_STATUSES

    # ------------------------------------------------------------------
    # Lifecycle methods
    # ------------------------------------------------------------------

    def refresh(self) -> "WorkflowRun":
        """Re-fetch this run from the server.

        Returns:
            Self (updated in-place).
        """
        data = _get_workflow_run(self._client, self.uuid)
        self._data = data
        return self

    def wait(
        self,
        timeout: float = 60,
        poll_interval: float = 2.0,
    ) -> "WorkflowRun":
        """Block until the run reaches a terminal state or *timeout* expires.

        Args:
            timeout: Maximum seconds to wait.
            poll_interval: Seconds between polling requests.

        Returns:
            Self (updated in-place).

        Raises:
            CyberwaveTimeoutError: If *timeout* expires before completion.
        """
        deadline = time.monotonic() + timeout
        while not self.is_terminal:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CyberwaveTimeoutError(
                    f"Workflow run {self.uuid} did not complete within "
                    f"{timeout}s (last status: {self.status})"
                )
            time.sleep(min(poll_interval, remaining))
            self.refresh()
        return self

    def cancel(self) -> "WorkflowRun":
        """Cancel this run.

        Returns:
            Self (updated in-place).
        """
        data = _cancel_workflow_run(self._client, self.uuid)
        self._data = data
        return self

    # ------------------------------------------------------------------
    # MQTT real-time subscription
    # ------------------------------------------------------------------

    def on_status_change(
        self, callback: Callable[[str, "WorkflowRun"], None]
    ) -> None:
        """Subscribe via MQTT for real-time status updates.

        The *callback* receives ``(new_status, run)`` each time the
        backend publishes a status change for this run.

        Requires the client's MQTT connection to be active.

        Args:
            callback: ``(status: str, run: WorkflowRun) -> None``
        """
        mqtt = self._client.mqtt
        if not mqtt.connected:
            mqtt.connect()

        prefix = mqtt.topic_prefix
        topic = f"{prefix}cyberwave/workflow-run/{self.uuid}/status"

        def _handler(payload: Dict[str, Any]) -> None:
            new_status = payload.get("status") if isinstance(payload, dict) else None
            if new_status:
                # Update local data with the received status
                if hasattr(self._data, "status"):
                    self._data.status = new_status
                elif isinstance(self._data, dict):
                    self._data["status"] = new_status
            callback(str(new_status or ""), self)

        mqtt.subscribe(topic, _handler)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"WorkflowRun(uuid='{self.uuid}', status='{self.status}', "
            f"workflow_id='{self.workflow_id}')"
        )


# ======================================================================
# Manager classes
# ======================================================================


class WorkflowManager:
    """
    Manager for workflow operations.

    Accessed via ``client.workflows``.
    """

    def __init__(self, client: "Cyberwave") -> None:
        self._client = client

    def list(self) -> List[Workflow]:
        """List all workflows visible to the authenticated user.

        Returns:
            List of :class:`Workflow` instances.
        """
        items = _list_workflows(self._client)
        return [Workflow(self._client, item) for item in items]

    def get(self, workflow_id: str) -> Workflow:
        """Get a single workflow by UUID or unified slug.

        When *workflow_id* looks like a UUID it is used directly. Otherwise
        the method tries the ``/by-slug`` endpoint (full unified slug such as
        ``acme/workflows/pick-and-place``), then falls back to the legacy
        ``/{uuid}`` path.

        Args:
            workflow_id: UUID or full unified slug.

        Returns:
            A :class:`Workflow` instance.
        """
        data = _get_workflow(self._client, workflow_id)
        return Workflow(self._client, data)

    def get_by_slug(self, slug: str, workspace_id: Optional[str] = None) -> Optional[Workflow]:
        """Get a workflow by its unified slug.

        Supports two calling conventions:

        1. **Full unified slug** (recommended):
           ``cw.workflows.get_by_slug("acme/workflows/pick-and-place")``
        2. **Legacy workspace-scoped slug** (backward-compatible):
           ``cw.workflows.get_by_slug("pick-and-place", workspace_id="ws-uuid")``

        When a *workspace_id* is provided the method uses the list endpoint
        with a ``slug`` query filter (legacy behaviour).  Otherwise it hits
        the dedicated ``GET /workflows/by-slug`` endpoint.

        Args:
            slug: Full unified slug or legacy workspace-scoped slug.
            workspace_id: Optional workspace UUID for the legacy lookup.

        Returns:
            A :class:`Workflow` instance if found, otherwise ``None``.
        """
        if workspace_id:
            items = _list_workflows(
                self._client, workspace_id=workspace_id, slug=slug
            )
            if not items:
                return None
            return Workflow(self._client, items[0])
        data = _get_workflow_by_slug(self._client, slug)
        if data is None:
            return None
        return Workflow(self._client, data)

    def trigger(
        self,
        workflow_id: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> WorkflowRun:
        """Trigger a workflow run.

        Args:
            workflow_id: UUID or unified slug of the workflow to trigger.
                When a slug is provided (e.g.
                ``"acme/workflows/pick-and-place"``), the method first
                resolves the workflow UUID via the ``/by-slug`` endpoint.
            inputs: Payload passed to the workflow.

        Returns:
            A :class:`WorkflowRun` representing the started execution.
        """
        resolved_id = workflow_id
        if not _is_uuid(workflow_id) and "/" in workflow_id:
            wf = self.get(workflow_id)
            resolved_id = wf.uuid
        data = _trigger_workflow(self._client, resolved_id, inputs)
        return WorkflowRun(self._client, data)

    def is_running(self, workflow_id: str) -> bool:
        """Check whether a workflow has any currently active run.

        A run is considered active when its status is ``"running"``,
        ``"waiting"``, or ``"requested"``.

        Args:
            workflow_id: UUID of the workflow to check.

        Returns:
            ``True`` if at least one active run exists, ``False`` otherwise.
        """
        wf = self.get(workflow_id)
        return wf.is_running()


class WorkflowRunManager:
    """
    Manager for workflow run (execution) operations.

    Accessed via ``client.workflow_runs``.
    """

    def __init__(self, client: "Cyberwave") -> None:
        self._client = client

    def list(
        self,
        *,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[WorkflowRun]:
        """List workflow runs, with optional filtering.

        Args:
            workflow_id: Only return runs for this workflow.
            status: Only return runs with this status
                (e.g. ``"running"``, ``"success"``, ``"error"``).

        Returns:
            List of :class:`WorkflowRun` instances.
        """
        items = _list_workflow_runs(self._client, workflow_id, status)
        return [WorkflowRun(self._client, item) for item in items]

    def get(self, run_id: str) -> WorkflowRun:
        """Get a single workflow run by UUID.

        Args:
            run_id: The run UUID.

        Returns:
            A :class:`WorkflowRun` instance.
        """
        data = _get_workflow_run(self._client, run_id)
        return WorkflowRun(self._client, data)

    def cancel(self, run_id: str) -> WorkflowRun:
        """Cancel a running workflow run.

        Args:
            run_id: The run UUID.

        Returns:
            The updated :class:`WorkflowRun`.
        """
        data = _cancel_workflow_run(self._client, run_id)
        return WorkflowRun(self._client, data)


# ======================================================================
# Private HTTP helpers (param_serialize pattern, same as alerts/edges)
# ======================================================================


def _api(client: "Cyberwave") -> Any:
    """Return the low-level api_client from the Cyberwave instance."""
    return client.api.api_client


def _list_workflows(
    client: "Cyberwave",
    workspace_id: Optional[str] = None,
    slug: Optional[str] = None,
) -> list:
    try:
        query_params: list[tuple[str, str]] = []
        if workspace_id:
            query_params.append(("workspace_uuid", workspace_id))
        if slug:
            query_params.append(("slug", slug))
        _param = _api(client).param_serialize(
            method="GET",
            resource_path="/api/v1/workflows",
            query_params=query_params,
            auth_settings=_AUTH,
        )
        response = _api(client).call_api(*_param)
        response.read()
        return _api(client).response_deserialize(
            response_data=response,
            response_types_map={"200": "List[WorkflowSchema]"},
        ).data
    except Exception as e:
        raise CyberwaveError(f"Failed to list workflows: {e}") from e


def _is_uuid(value: str) -> bool:
    """Check whether *value* looks like a UUID."""
    try:
        import uuid as _uuid_mod
        _uuid_mod.UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


def _get_workflow(client: "Cyberwave", workflow_id: str) -> Any:
    try:
        _param = _api(client).param_serialize(
            method="GET",
            resource_path="/api/v1/workflows/{uuid}",
            path_params={"uuid": workflow_id},
            auth_settings=_AUTH,
        )
        response = _api(client).call_api(*_param)
        response.read()
        return _api(client).response_deserialize(
            response_data=response,
            response_types_map={"200": "WorkflowSchema"},
        ).data
    except Exception as e:
        if not _is_uuid(workflow_id) and "/" in workflow_id:
            result = _get_workflow_by_slug(client, workflow_id)
            if result is not None:
                return result
        raise CyberwaveError(f"Failed to get workflow {workflow_id}: {e}") from e


def _get_workflow_by_slug(client: "Cyberwave", slug: str) -> Any:
    """Fetch a workflow by its full unified slug via ``GET /workflows/by-slug``."""
    try:
        _param = _api(client).param_serialize(
            method="GET",
            resource_path="/api/v1/workflows/by-slug",
            query_params=[("slug", slug)],
            auth_settings=_AUTH,
        )
        response = _api(client).call_api(*_param)
        response.read()
        return _api(client).response_deserialize(
            response_data=response,
            response_types_map={"200": "WorkflowSchema"},
        ).data
    except Exception:
        return None


def _trigger_workflow(
    client: "Cyberwave",
    workflow_id: str,
    inputs: Optional[Dict[str, Any]],
) -> Any:
    try:
        body: Dict[str, Any] = {}
        if inputs is not None:
            body["inputs"] = inputs
        _param = _api(client).param_serialize(
            method="POST",
            resource_path="/api/v1/workflows/{uuid}/trigger",
            path_params={"uuid": workflow_id},
            body=body,
            auth_settings=_AUTH,
        )
        response = _api(client).call_api(*_param)
        response.read()
        return _api(client).response_deserialize(
            response_data=response,
            response_types_map={"200": "WorkflowRunSchema"},
        ).data
    except Exception as e:
        raise CyberwaveError(f"Failed to trigger workflow {workflow_id}: {e}") from e


def _list_workflow_runs(
    client: "Cyberwave",
    workflow_id: Optional[str],
    status: Optional[str],
) -> list:
    try:
        query_params: list = []
        if workflow_id:
            query_params.append(("workflow_id", workflow_id))
        if status:
            query_params.append(("status", status))
        _param = _api(client).param_serialize(
            method="GET",
            resource_path="/api/v1/workflow-runs",
            query_params=query_params,
            auth_settings=_AUTH,
        )
        response = _api(client).call_api(*_param)
        response.read()
        return _api(client).response_deserialize(
            response_data=response,
            response_types_map={"200": "List[WorkflowRunSchema]"},
        ).data
    except Exception as e:
        raise CyberwaveError(f"Failed to list workflow runs: {e}") from e


def _get_workflow_run(client: "Cyberwave", run_id: str) -> Any:
    try:
        _param = _api(client).param_serialize(
            method="GET",
            resource_path="/api/v1/workflow-runs/{uuid}",
            path_params={"uuid": run_id},
            auth_settings=_AUTH,
        )
        response = _api(client).call_api(*_param)
        response.read()
        return _api(client).response_deserialize(
            response_data=response,
            response_types_map={"200": "WorkflowRunSchema"},
        ).data
    except Exception as e:
        raise CyberwaveError(f"Failed to get workflow run {run_id}: {e}") from e


def _cancel_workflow_run(client: "Cyberwave", run_id: str) -> Any:
    try:
        _param = _api(client).param_serialize(
            method="POST",
            resource_path="/api/v1/workflow-runs/{uuid}/cancel",
            path_params={"uuid": run_id},
            auth_settings=_AUTH,
        )
        response = _api(client).call_api(*_param)
        response.read()
        return _api(client).response_deserialize(
            response_data=response,
            response_types_map={"200": "WorkflowRunSchema"},
        ).data
    except Exception as e:
        raise CyberwaveError(f"Failed to cancel workflow run {run_id}: {e}") from e
