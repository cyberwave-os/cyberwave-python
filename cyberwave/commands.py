"""Governed twin commands: the platform command gateway, audit, policy and grants.

``POST /api/v1/twins/{uuid}/commands`` is the one server-side ingress for twin
commands. The backend decides each command (token scope, role, the twin's
control policy, operator grants, the catalog's typed arguments), writes an
append-only audit record for the decision whatever it is, and only then
publishes the command to the driver. With ``timeout_s > 0`` it also waits for
the driver's MQTT v5 ``twin.command_response.v1`` reply.

On a *governed* twin (control policy ``governed=true``) the broker refuses
direct publishes on the twin's command topic, so the gateway is the only way
to command it. On other twins it is still the way to get an audited,
policy-checked command and its outcome::

    result = twin.commands.send("takeoff", {"altitude_m": 30}, timeout_s=10)
    if result.status != "accepted":
        print(result.status, result.reason)

    for record in twin.commands.audit(limit=20)["items"]:
        print(record["created_at"], record["command"], record["decision"])

    twin.commands.control_policy.put(level="limited", governed=True)
    grant = twin.commands.grants.create(operator_uuid, duration_s=3600)
    twin.commands.grants.revoke(grant["uuid"])

The same calls exist on the client, keyed by twin UUID:
``client.commands.send(twin_uuid, "takeoff", ...)``,
``client.commands.for_twin(twin_uuid).audit()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from ._rest import request
from .exceptions import CyberwaveError

if TYPE_CHECKING:
    from .client import Cyberwave

#: Gateway statuses (``CommandResult.status``). ``denied``: refused by policy,
#: nothing was published. ``sent``: published, no reply awaited.
#: ``publish_failed``: the broker did not take it. ``no_response``: no reply
#: within ``timeout_s``. ``invalid_response``: the reply was not a
#: ``twin.command_response.v1``. Otherwise the driver's reply status:
#: ``accepted`` / ``rejected`` / ``denied`` / ``timeout``.
GATEWAY_STATUSES = frozenset(
    {
        "denied",
        "sent",
        "publish_failed",
        "no_response",
        "invalid_response",
        "accepted",
        "rejected",
        "timeout",
    }
)
# HTTP statuses on which the gateway still answers with a TwinCommandResult.
_RESULT_HTTP_STATUSES = frozenset({200, 400, 403, 503})

CONTROL_LEVELS = ("observe", "plan", "limited", "full")
GRANT_LEVELS = ("plan", "limited", "full")


@dataclass(frozen=True)
class CommandResult:
    """The gateway's decision and, when awaited, the driver's reply.

    Attributes:
        audit_id: The command audit record written for this decision.
        status: See :data:`GATEWAY_STATUSES`.
        reason: Why, for anything other than an accepted/sent command.
        response: The driver's ``twin.command_response.v1`` reply, if any.
        http_status: The gateway's HTTP status (200, 400, 403 or 503).
    """

    audit_id: str
    status: str
    reason: Optional[str] = None
    response: Optional[Dict[str, Any]] = None
    http_status: int = 200
    raw: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def accepted(self) -> bool:
        """The driver replied ``accepted``."""
        return self.status == "accepted"

    @property
    def delivered(self) -> bool:
        """Published to the driver (``sent``) or accepted by it."""
        return self.status in {"sent", "accepted"}

    @property
    def denied_by_policy(self) -> bool:
        """Refused by the gateway itself; nothing reached the driver."""
        return self.status == "denied" and self.response is None

    @classmethod
    def from_payload(
        cls, payload: Dict[str, Any], *, http_status: int = 200
    ) -> "CommandResult":
        response = payload.get("response")
        return cls(
            audit_id=str(payload.get("audit_id") or ""),
            status=str(payload.get("status") or ""),
            reason=payload.get("reason"),
            response=response if isinstance(response, dict) else None,
            http_status=http_status,
            raw=dict(payload),
        )


# --------------------------------------------------------------------------- #
# Twin-scoped managers
# --------------------------------------------------------------------------- #


class TwinControlPolicyManager:
    """A twin's control policy (``/twins/{uuid}/control-policy``).

    Calling the manager is a shorthand for :meth:`get`.
    """

    def __init__(self, client: "Cyberwave", twin_uuid: str) -> None:
        self._client = client
        self._twin_uuid = twin_uuid

    def __call__(self) -> Optional[Dict[str, Any]]:
        return self.get()

    def get(self) -> Optional[Dict[str, Any]]:
        """The policy, or ``None`` when the twin has none (it is ungoverned)."""
        status, data = request(
            self._client.api.api_client,
            "GET",
            "/api/v1/twins/{uuid}/control-policy",
            path_params={"uuid": self._twin_uuid},
            allow_statuses={404},
        )
        return None if status == 404 else data

    def put(
        self,
        level: str,
        *,
        allowed_commands: Iterable[str] = (),
        forbidden_commands: Iterable[str] = (),
        plan_commands: Optional[Iterable[str]] = None,
        minimum_role: str = "developer",
        require_operator_grant_for: Iterable[str] = (),
        governed: bool = False,
    ) -> Dict[str, Any]:
        """Create or replace the policy (twin admins).

        Args:
            level: ``observe``, ``plan``, ``limited`` or ``full``.
            allowed_commands: Commands allowed at ``limited``.
            forbidden_commands: Commands never allowed.
            plan_commands: Non-actuating commands; ``None`` keeps the current
                list (or the backend default).
            minimum_role: ``developer``, ``writer``, ``admin`` or ``owner``.
            require_operator_grant_for: Commands that also need an active grant.
            governed: Make the twin's command topic backend-only at the broker.
        """
        if level not in CONTROL_LEVELS:
            raise CyberwaveError(
                f"level must be one of {CONTROL_LEVELS}, got {level!r}"
            )
        body: Dict[str, Any] = {
            "level": level,
            "allowed_commands": list(allowed_commands),
            "forbidden_commands": list(forbidden_commands),
            "minimum_role": minimum_role,
            "require_operator_grant_for": list(require_operator_grant_for),
            "governed": bool(governed),
        }
        if plan_commands is not None:
            body["plan_commands"] = list(plan_commands)
        _, data = request(
            self._client.api.api_client,
            "PUT",
            "/api/v1/twins/{uuid}/control-policy",
            path_params={"uuid": self._twin_uuid},
            body=body,
        )
        return data

    def delete(self) -> bool:
        """Remove the policy. ``False`` when there was none."""
        status, _ = request(
            self._client.api.api_client,
            "DELETE",
            "/api/v1/twins/{uuid}/control-policy",
            path_params={"uuid": self._twin_uuid},
            allow_statuses={404},
        )
        return status != 404

    def __repr__(self) -> str:
        return f"TwinControlPolicyManager(twin={self._twin_uuid!r})"


class TwinControlGrantManager:
    """Operator grants on a twin (``/twins/{uuid}/control-grants``)."""

    def __init__(self, client: "Cyberwave", twin_uuid: str) -> None:
        self._client = client
        self._twin_uuid = twin_uuid

    def list(
        self,
        *,
        active: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Grants, newest first; ``active`` filters live/expired-or-revoked."""
        query: Dict[str, Any] = {"limit": limit, "offset": offset}
        if active is not None:
            query["active"] = "true" if active else "false"
        _, data = request(
            self._client.api.api_client,
            "GET",
            "/api/v1/twins/{uuid}/control-grants",
            path_params={"uuid": self._twin_uuid},
            query=query,
        )
        return list(data or [])

    def create(
        self,
        grantee_uuid: str,
        *,
        duration_s: int,
        level: str = "limited",
        mission_id: str = "",
    ) -> Dict[str, Any]:
        """Grant a user time-boxed command authority (twin admins)."""
        if level not in GRANT_LEVELS:
            raise CyberwaveError(f"level must be one of {GRANT_LEVELS}, got {level!r}")
        _, data = request(
            self._client.api.api_client,
            "POST",
            "/api/v1/twins/{uuid}/control-grants",
            path_params={"uuid": self._twin_uuid},
            body={
                "grantee_uuid": str(grantee_uuid),
                "level": level,
                "mission_id": mission_id,
                "duration_s": int(duration_s),
            },
        )
        return data

    def revoke(self, grant_uuid: str) -> Dict[str, Any]:
        """Revoke a grant now (twin admins, or the grantee handing it back)."""
        _, data = request(
            self._client.api.api_client,
            "POST",
            "/api/v1/twins/{uuid}/control-grants/{grant_uuid}/revoke",
            path_params={"uuid": self._twin_uuid, "grant_uuid": str(grant_uuid)},
        )
        return data

    def __repr__(self) -> str:
        return f"TwinControlGrantManager(twin={self._twin_uuid!r})"


class TwinCommandGateway:
    """The command gateway bound to one twin (``client.commands.for_twin``)."""

    def __init__(self, client: "Cyberwave", twin_uuid: str) -> None:
        self._client = client
        self.twin_uuid = str(twin_uuid)
        self.control_policy = TwinControlPolicyManager(client, self.twin_uuid)
        self.grants = TwinControlGrantManager(client, self.twin_uuid)

    def send(
        self,
        command: str,
        data: Optional[Dict[str, Any]] = None,
        *,
        timeout_s: float = 0,
        mode: Optional[str] = None,
    ) -> CommandResult:
        """Issue one command through the gateway.

        Args:
            command: Command name (``[A-Za-z0-9_.:-]{1,128}``).
            data: Command arguments, checked against the catalog's typed args.
            timeout_s: Wait up to this long (max 15 s) for the driver's reply.
                ``0`` publishes and returns ``sent``.
            mode: ``live`` or ``simulation``. Default: ``simulation`` when the
                client runs in simulation mode, else the backend's ``live``.

        Returns:
            A :class:`CommandResult`. Policy denials, publish failures and
            driver refusals are results, not exceptions.

        Raises:
            CyberwaveAPIError: the request itself failed (auth, unknown twin,
                malformed arguments, server error).
        """
        if timeout_s < 0:
            raise CyberwaveError("timeout_s must be >= 0")
        body: Dict[str, Any] = {"command": command, "data": dict(data or {})}
        if timeout_s:
            body["timeout_s"] = float(timeout_s)
        resolved_mode = mode or _default_mode(self._client)
        if resolved_mode is not None:
            if resolved_mode not in {"live", "simulation"}:
                raise CyberwaveError("mode must be 'live' or 'simulation'")
            body["mode"] = resolved_mode
        status, payload = request(
            self._client.api.api_client,
            "POST",
            "/api/v1/twins/{uuid}/commands",
            path_params={"uuid": self.twin_uuid},
            body=body,
            allow_statuses=_RESULT_HTTP_STATUSES,
            result_shape=("audit_id", "status"),
        )
        return CommandResult.from_payload(payload or {}, http_status=status)

    def audit(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        decision: Optional[str] = None,
        command: Optional[str] = None,
    ) -> Dict[str, Any]:
        """The twin's command audit, newest first.

        Returns the page: ``{"items": [...], "total", "limit", "offset",
        "has_more"}``. ``decision`` is ``allowed`` or ``denied``.
        """
        query: Dict[str, Any] = {"limit": limit, "offset": offset}
        if decision is not None:
            if decision not in {"allowed", "denied"}:
                raise CyberwaveError("decision must be 'allowed' or 'denied'")
            query["decision"] = decision
        if command is not None:
            query["command"] = command
        _, data = request(
            self._client.api.api_client,
            "GET",
            "/api/v1/twins/{uuid}/command-audit",
            path_params={"uuid": self.twin_uuid},
            query=query,
        )
        return data or {
            "items": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "has_more": False,
        }

    def __repr__(self) -> str:
        return f"TwinCommandGateway(twin={self.twin_uuid!r})"


class CommandGateway:
    """Workspace-level access to the command gateway (``client.commands``)."""

    def __init__(self, client: "Cyberwave") -> None:
        self._client = client

    def for_twin(self, twin_uuid: str) -> TwinCommandGateway:
        return TwinCommandGateway(self._client, twin_uuid)

    def send(
        self,
        twin_uuid: str,
        command: str,
        data: Optional[Dict[str, Any]] = None,
        *,
        timeout_s: float = 0,
        mode: Optional[str] = None,
    ) -> CommandResult:
        """See :meth:`TwinCommandGateway.send`."""
        return self.for_twin(twin_uuid).send(
            command, data, timeout_s=timeout_s, mode=mode
        )

    def audit(self, twin_uuid: str, **kwargs: Any) -> Dict[str, Any]:
        """See :meth:`TwinCommandGateway.audit`."""
        return self.for_twin(twin_uuid).audit(**kwargs)

    def control_policy(self, twin_uuid: str) -> TwinControlPolicyManager:
        return TwinControlPolicyManager(self._client, twin_uuid)

    def grants(self, twin_uuid: str) -> TwinControlGrantManager:
        return TwinControlGrantManager(self._client, twin_uuid)

    def __repr__(self) -> str:
        return "CommandGateway()"


def _default_mode(client: Any) -> Optional[str]:
    runtime_mode = getattr(getattr(client, "config", None), "runtime_mode", None)
    return "simulation" if runtime_mode == "simulation" else None


__all__ = [
    "CONTROL_LEVELS",
    "GATEWAY_STATUSES",
    "GRANT_LEVELS",
    "CommandGateway",
    "CommandResult",
    "TwinCommandGateway",
    "TwinControlGrantManager",
    "TwinControlPolicyManager",
]
