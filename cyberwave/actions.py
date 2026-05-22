"""Thin action-status helpers for Cyberwave API actions."""

from __future__ import annotations

import json
import time
from typing import Any

from cyberwave.exceptions import CyberwaveAPIError

TERMINAL_STATUSES = {"completed", "failed", "cancelled", "blocked"}


def _decode_json_response(response_data: Any) -> dict[str, Any]:
    response_data.read()
    payload = getattr(response_data, "data", None)
    if isinstance(payload, bytes):
        return json.loads(payload.decode("utf-8"))
    if isinstance(payload, str):
        return json.loads(payload)
    if isinstance(payload, dict):
        return payload
    return {}


class ActionsClient:
    """Helpers for polling action status via existing twin action APIs."""

    def __init__(self, api_client: Any):
        self._api_client = api_client

    def get_status(
        self, action_id: str, *, twin_uuid: str | None = None
    ) -> dict[str, Any]:
        """Return the current status for an action.

        The backend action-status route is scoped by twin for authorization, so
        callers must provide the twin UUID associated with the action.
        """
        if not action_id:
            raise ValueError("action_id is required")
        if not twin_uuid:
            raise ValueError("twin_uuid is required by the action status endpoint")

        try:
            _param = self._api_client.param_serialize(
                method="GET",
                resource_path="/api/v1/twins/{uuid}/actions/{action_id}",
                path_params={"uuid": twin_uuid, "action_id": action_id},
                auth_settings=["CustomTokenAuthentication"],
            )
            response_data = self._api_client.call_api(*_param)
            return _decode_json_response(response_data)
        except Exception as exc:
            raise CyberwaveAPIError(f"Failed to get action status: {exc}") from exc

    def wait(
        self,
        action_id: str,
        *,
        twin_uuid: str | None = None,
        timeout: float = 120.0,
        poll_interval: float = 1.0,
        raise_on_failure: bool = True,
    ) -> dict[str, Any]:
        """Poll until an action reaches a terminal status."""
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be greater than zero")

        deadline = time.monotonic() + timeout
        last_status: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            last_status = self.get_status(action_id, twin_uuid=twin_uuid)
            status = str(last_status.get("status") or "").lower()
            if status in TERMINAL_STATUSES:
                if raise_on_failure and status != "completed":
                    raise RuntimeError(
                        f"action {action_id} finished with status {status!r}: "
                        f"{last_status.get('message')!r}"
                    )
                return last_status
            time.sleep(poll_interval)

        raise TimeoutError(
            f"action {action_id} did not reach a terminal status within {timeout}s"
        )
