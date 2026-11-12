"""Thin action-status helpers for Cyberwave API actions."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from cyberwave._rest import request
from cyberwave.exceptions import CyberwaveAPIError

TERMINAL_STATUSES = {"completed", "failed", "cancelled", "blocked"}


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
            # Non-2xx raises: an error body would otherwise read as an action
            # record with no ``status``, which every caller below reads as
            # "not finished yet".
            _, payload = request(
                self._api_client,
                "GET",
                "/api/v1/twins/{uuid}/actions/{action_id}",
                path_params={"uuid": twin_uuid, "action_id": action_id},
            )
        except CyberwaveAPIError:
            raise
        except Exception as exc:
            raise CyberwaveAPIError(f"Failed to get action status: {exc}") from exc
        return payload if isinstance(payload, dict) else {}

    def wait(
        self,
        action_id: str,
        *,
        twin_uuid: str | None = None,
        timeout: float = 120.0,
        poll_interval: float = 1.0,
        raise_on_failure: bool = True,
        on_poll: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Poll until an action reaches a terminal status.

        ``on_poll`` runs before every status poll. Whatever it raises ends the
        wait and propagates, which is how a caller that owns the wait (a
        heartbeat, a cancel check) can stop it between polls.
        """
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be greater than zero")

        deadline = time.monotonic() + timeout
        last_status: dict[str, Any] | None = None
        last_error: CyberwaveAPIError | None = None
        while time.monotonic() < deadline:
            if on_poll is not None:
                on_poll()
            # A single failed poll must not end the wait: callers hold a robot
            # in motion for the whole timeout, and their recovery is written
            # against TimeoutError. Keep polling and report the last failure
            # in the timeout message so the cause is not lost.
            try:
                last_status = self.get_status(action_id, twin_uuid=twin_uuid)
            except CyberwaveAPIError as exc:
                last_error = exc
                time.sleep(poll_interval)
                continue
            last_error = None
            status = str(last_status.get("status") or "").lower()
            if status in TERMINAL_STATUSES:
                if raise_on_failure and status != "completed":
                    raise RuntimeError(
                        f"action {action_id} finished with status {status!r}: "
                        f"{last_status.get('message')!r}"
                    )
                return last_status
            time.sleep(poll_interval)

        message = (
            f"action {action_id} did not reach a terminal status within {timeout}s"
        )
        if last_error is not None:
            message += f" (last status poll failed: {last_error})"
        raise TimeoutError(message)
