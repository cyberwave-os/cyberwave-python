"""
High-level Alert abstractions for the Cyberwave SDK.

Provides an ``Alert`` object with lifecycle helpers (acknowledge, resolve,
silence) and a ``TwinAlertManager`` that is accessed via ``twin.alerts``.

Example::

    twin = client.twin(twin_id="...")

    # Create an alert for this twin
    alert = twin.alerts.create(
        name="Calibration needed",
        description="Joint 3 is drifting",
        severity="warning",
        alert_type="calibration_needed",
    )

    # List active alerts
    for a in twin.alerts.list():
        print(a.name, a.severity, a.status)

    # Lifecycle actions
    alert.acknowledge()
    alert.resolve()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .exceptions import CyberwaveError

if TYPE_CHECKING:
    from .client import Cyberwave
    from .twin import Twin


def _attr(data: Any, key: str, default: Any = None) -> Any:
    """Read *key* from an object or dict, returning *default* if missing."""
    if hasattr(data, key):
        return getattr(data, key)
    if isinstance(data, dict):
        return data.get(key, default)
    return default


class Alert:
    """
    A single alert with convenience lifecycle methods.

    You normally obtain ``Alert`` instances from :pymethod:`TwinAlertManager.list`
    or :pymethod:`TwinAlertManager.create` rather than constructing them directly.
    """

    def __init__(self, client: "Cyberwave", data: Any) -> None:
        self._client = client
        self._data = data

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def uuid(self) -> str:
        return str(_attr(self._data, "uuid", ""))

    @property
    def name(self) -> str:
        return str(_attr(self._data, "name", ""))

    @property
    def description(self) -> str:
        return str(_attr(self._data, "description", ""))

    @property
    def alert_type(self) -> str:
        return str(_attr(self._data, "alert_type", ""))

    @property
    def severity(self) -> str:
        return str(_attr(self._data, "severity", ""))

    @property
    def status(self) -> str:
        return str(_attr(self._data, "status", ""))

    @property
    def source_type(self) -> str:
        return str(_attr(self._data, "source_type", ""))

    @property
    def twin_uuid(self) -> Optional[str]:
        val = _attr(self._data, "twin_uuid")
        return str(val) if val else None

    @property
    def environment_uuid(self) -> Optional[str]:
        val = _attr(self._data, "environment_uuid")
        return str(val) if val else None

    @property
    def workflow_uuid(self) -> Optional[str]:
        val = _attr(self._data, "workflow_uuid")
        return str(val) if val else None

    @property
    def workspace_uuid(self) -> str:
        return str(_attr(self._data, "workspace_uuid", ""))

    @property
    def created_at(self) -> Optional[str]:
        val = _attr(self._data, "created_at")
        return str(val) if val else None

    @property
    def updated_at(self) -> Optional[str]:
        val = _attr(self._data, "updated_at")
        return str(val) if val else None

    @property
    def resolved_at(self) -> Optional[str]:
        val = _attr(self._data, "resolved_at")
        return str(val) if val else None

    # ------------------------------------------------------------------
    # Lifecycle actions
    # ------------------------------------------------------------------

    def acknowledge(self) -> "Alert":
        """Mark this alert as acknowledged.

        Returns:
            Updated Alert instance.
        """
        data = _post_alert_action(self._client, self.uuid, "acknowledge")
        self._data = data
        return self

    def resolve(self) -> "Alert":
        """Mark this alert as resolved (sets ``resolved_at``).

        Returns:
            Updated Alert instance.
        """
        data = _post_alert_action(self._client, self.uuid, "resolve")
        self._data = data
        return self

    def silence(self) -> "Alert":
        """Silence this alert workspace-wide (sets ``resolved_at``).

        Returns:
            Updated Alert instance.
        """
        data = _post_alert_action(self._client, self.uuid, "silence")
        self._data = data
        return self

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def update(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
    ) -> "Alert":
        """Update mutable fields on this alert.

        Only the fields you pass will be changed; the rest stay as-is.

        Returns:
            Updated Alert instance.
        """
        payload: Dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if alert_type is not None:
            payload["alert_type"] = alert_type
        if severity is not None:
            payload["severity"] = severity
        if status is not None:
            payload["status"] = status

        data = _put_alert(self._client, self.uuid, payload)
        self._data = data
        return self

    def delete(self) -> None:
        """Delete this alert."""
        _delete_alert(self._client, self.uuid)

    def refresh(self) -> "Alert":
        """Re-fetch this alert from the server.

        Returns:
            Updated Alert instance.
        """
        data = _get_alert(self._client, self.uuid)
        self._data = data
        return self

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Alert(uuid='{self.uuid}', name='{self.name}', "
            f"severity='{self.severity}', status='{self.status}')"
        )


class TwinAlertManager:
    """
    Alert manager scoped to a specific :class:`Twin`.

    Accessed via ``twin.alerts``.
    """

    def __init__(self, twin: "Twin") -> None:
        self._twin = twin

    def create(
        self,
        name: str,
        *,
        description: str = "",
        alert_type: str = "",
        severity: str = "warning",
        source_type: str = "edge",
        environment_uuid: Optional[str] = None,
        workflow_uuid: Optional[str] = None,
        workspace_uuid: Optional[str] = None,
    ) -> Alert:
        """Create a new **active** alert for this twin.

        Args:
            name: Human-readable title.
            description: Optional details.
            alert_type: Machine-readable code (e.g. ``calibration_needed``).
            severity: One of ``info``, ``warning``, ``error``, ``critical``.
            source_type: One of ``edge``, ``cloud``, ``workflow``.
            environment_uuid: Optionally attach to an environment.
            workflow_uuid: Optionally attach to a workflow.
            workspace_uuid: Workspace to associate the alert with.
                Defaults to the workspace configured on the client.

        Returns:
            The newly created :class:`Alert`.
        """
        ws_uuid = workspace_uuid or self._twin.client.config.workspace_id
        env_uuid = environment_uuid or self._twin.environment_id or None
        payload: Dict[str, Any] = {
            "name": name,
            "description": description,
            "alert_type": alert_type,
            "severity": severity,
            "source_type": source_type,
            "twin_uuid": self._twin.uuid,
        }
        if ws_uuid is not None:
            payload["workspace_uuid"] = ws_uuid
        if env_uuid is not None:
            payload["environment_uuid"] = env_uuid
        if workflow_uuid is not None:
            payload["workflow_uuid"] = workflow_uuid

        data = _create_alert(self._twin.client, payload)
        return Alert(self._twin.client, data)

    def get(self, uuid: str) -> Alert:
        """Fetch a single alert by UUID.

        Args:
            uuid: The alert UUID.

        Returns:
            The :class:`Alert` instance.
        """
        data = _get_alert(self._twin.client, uuid)
        return Alert(self._twin.client, data)

    def list(
        self,
        *,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """List alerts for this twin.

        Args:
            status: Comma-separated statuses to filter
                    (e.g. ``"active"`` or ``"active,acknowledged"``).
                    Defaults to all statuses.
            severity: Comma-separated severities to filter
                      (e.g. ``"error,critical"``).
            limit: Maximum number of results (default 100).

        Returns:
            List of :class:`Alert` instances.
        """
        params: Dict[str, Any] = {
            "twin_uuid": self._twin.uuid,
            "limit": limit,
        }
        if status is not None:
            params["status"] = status
        if severity is not None:
            params["severity"] = severity

        items = _list_alerts(self._twin.client, params)
        return [Alert(self._twin.client, item) for item in items]


# ======================================================================
# Private HTTP helpers (use the same param_serialize pattern as EdgeManager)
# ======================================================================

_AUTH = ["CustomTokenAuthentication"]


def _api(client: "Cyberwave") -> Any:
    """Return the low-level api_client from the Cyberwave instance."""
    return client.api.api_client


def _list_alerts(client: "Cyberwave", query: Dict[str, Any]) -> list:
    query_params = [(k, v) for k, v in query.items()]
    try:
        _param = _api(client).param_serialize(
            method="GET",
            resource_path="/api/v1/alerts",
            query_params=query_params,
            auth_settings=_AUTH,
        )
        response = _api(client).call_api(*_param)
        response.read()
        return _api(client).response_deserialize(
            response_data=response,
            response_types_map={"200": "List[AlertSchema]"},
        ).data
    except Exception as e:
        raise CyberwaveError(f"Failed to list alerts: {e}") from e


def _get_alert(client: "Cyberwave", uuid: str) -> Any:
    try:
        _param = _api(client).param_serialize(
            method="GET",
            resource_path="/api/v1/alerts/{uuid}",
            path_params={"uuid": uuid},
            auth_settings=_AUTH,
        )
        response = _api(client).call_api(*_param)
        response.read()
        return _api(client).response_deserialize(
            response_data=response,
            response_types_map={"200": "AlertSchema"},
        ).data
    except Exception as e:
        raise CyberwaveError(f"Failed to get alert {uuid}: {e}") from e


def _create_alert(client: "Cyberwave", payload: Dict[str, Any]) -> Any:
    try:
        _param = _api(client).param_serialize(
            method="POST",
            resource_path="/api/v1/alerts",
            body=payload,
            auth_settings=_AUTH,
        )
        response = _api(client).call_api(*_param)
        response.read()
        return _api(client).response_deserialize(
            response_data=response,
            response_types_map={"200": "AlertSchema"},
        ).data
    except Exception as e:
        raise CyberwaveError(f"Failed to create alert: {e}") from e


def _put_alert(client: "Cyberwave", uuid: str, payload: Dict[str, Any]) -> Any:
    try:
        _param = _api(client).param_serialize(
            method="PUT",
            resource_path="/api/v1/alerts/{uuid}",
            path_params={"uuid": uuid},
            body=payload,
            auth_settings=_AUTH,
        )
        response = _api(client).call_api(*_param)
        response.read()
        return _api(client).response_deserialize(
            response_data=response,
            response_types_map={"200": "AlertSchema"},
        ).data
    except Exception as e:
        raise CyberwaveError(f"Failed to update alert {uuid}: {e}") from e


def _delete_alert(client: "Cyberwave", uuid: str) -> None:
    try:
        _param = _api(client).param_serialize(
            method="DELETE",
            resource_path="/api/v1/alerts/{uuid}",
            path_params={"uuid": uuid},
            auth_settings=_AUTH,
        )
        response = _api(client).call_api(*_param)
        response.read()
    except Exception as e:
        raise CyberwaveError(f"Failed to delete alert {uuid}: {e}") from e


def _post_alert_action(client: "Cyberwave", uuid: str, action: str) -> Any:
    """POST /api/v1/alerts/{uuid}/{action} (acknowledge, resolve, silence)."""
    try:
        _param = _api(client).param_serialize(
            method="POST",
            resource_path=f"/api/v1/alerts/{{uuid}}/{action}",
            path_params={"uuid": uuid},
            auth_settings=_AUTH,
        )
        response = _api(client).call_api(*_param)
        response.read()
        return _api(client).response_deserialize(
            response_data=response,
            response_types_map={"200": "AlertSchema"},
        ).data
    except Exception as e:
        raise CyberwaveError(f"Failed to {action} alert {uuid}: {e}") from e
