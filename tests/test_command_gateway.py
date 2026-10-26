"""Command gateway client: ``client.commands`` / ``twin.commands`` (CYB-4275 G1-G4)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from cyberwave.commands import CommandGateway, CommandResult, TwinCommandGateway
from cyberwave.exceptions import CyberwaveAPIError, CyberwaveError
from cyberwave.twin.commands import TwinCommandsHandle

TWIN = "5f2c7d0e-8a51-4d0f-9e0b-2a9c1f4d6b11"


class FakeApi:
    """Records param_serialize calls and answers call_api from a queue."""

    def __init__(self, *responses: tuple[int, Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def param_serialize(self, **kwargs: Any) -> tuple:
        self.calls.append(kwargs)
        return (kwargs,)

    def call_api(self, *_params: Any) -> Any:
        status, body = self.responses.pop(0)
        data = b"" if body is None else json.dumps(body).encode()
        return SimpleNamespace(status=status, data=data, read=lambda: data)


def _client(*responses: tuple[int, Any], runtime_mode: str = "live") -> Any:
    api = FakeApi(*responses)
    return SimpleNamespace(
        api=SimpleNamespace(api_client=api),
        config=SimpleNamespace(runtime_mode=runtime_mode),
        mqtt=MagicMock(),
    )


def _call(client: Any, index: int = -1) -> dict[str, Any]:
    return client.api.api_client.calls[index]


def test_send_posts_command_and_returns_the_reply() -> None:
    reply = {"contract": "twin.command_response.v1", "status": "accepted"}
    client = _client(
        (
            200,
            {"audit_id": "a1", "status": "accepted", "reason": None, "response": reply},
        )
    )
    result = TwinCommandGateway(client, TWIN).send("takeoff", {"alt": 30}, timeout_s=10)
    assert result == CommandResult(audit_id="a1", status="accepted", response=reply)
    assert result.accepted and result.delivered and not result.denied_by_policy
    call = _call(client)
    assert call["method"] == "POST"
    assert call["resource_path"] == "/api/v1/twins/{uuid}/commands"
    assert call["path_params"] == {"uuid": TWIN}
    assert call["body"] == {
        "command": "takeoff",
        "data": {"alt": 30},
        "timeout_s": 10.0,
    }
    assert call["auth_settings"] == ["CustomTokenAuthentication"]


def test_send_fire_and_forget_omits_timeout_and_uses_simulation_mode() -> None:
    client = _client(
        (200, {"audit_id": "a2", "status": "sent"}), runtime_mode="simulation"
    )
    result = CommandGateway(client).send(TWIN, "land")
    assert result.status == "sent" and result.delivered
    assert _call(client)["body"] == {
        "command": "land",
        "data": {},
        "mode": "simulation",
    }


@pytest.mark.parametrize(
    ("http_status", "status", "reason"),
    [
        (403, "denied", "policy level observe"),
        (400, "rejected", "bad args"),
        (503, "publish_failed", None),
    ],
)
def test_gateway_error_statuses_are_results(
    http_status: int, status: str, reason: str | None
) -> None:
    client = _client(
        (http_status, {"audit_id": "a3", "status": status, "reason": reason})
    )
    result = TwinCommandGateway(client, TWIN).send("arm", mode="live")
    assert result.http_status == http_status
    assert result.status == status
    assert result.reason == reason
    assert _call(client)["body"]["mode"] == "live"
    if status == "denied":
        assert result.denied_by_policy


def test_framework_errors_raise() -> None:
    client = _client((403, {"detail": "Forbidden"}))
    with pytest.raises(CyberwaveAPIError) as exc:
        TwinCommandGateway(client, TWIN).send("arm")
    assert exc.value.status_code == 403
    assert "Forbidden" in str(exc.value)
    client = _client((422, {"detail": [{"msg": "bad"}]}))
    with pytest.raises(CyberwaveAPIError):
        TwinCommandGateway(client, TWIN).send("arm")
    with pytest.raises(CyberwaveError):
        TwinCommandGateway(_client(), TWIN).send("arm", timeout_s=-1)
    with pytest.raises(CyberwaveError):
        TwinCommandGateway(_client(), TWIN).send("arm", mode="replay")


def test_audit_query() -> None:
    page = {
        "items": [{"command": "arm"}],
        "total": 1,
        "limit": 20,
        "offset": 0,
        "has_more": False,
    }
    client = _client((200, page))
    assert (
        TwinCommandGateway(client, TWIN).audit(
            limit=20, decision="denied", command="arm"
        )
        == page
    )
    call = _call(client)
    assert call["method"] == "GET"
    assert call["resource_path"] == "/api/v1/twins/{uuid}/command-audit"
    assert dict(call["query_params"]) == {
        "limit": 20,
        "offset": 0,
        "decision": "denied",
        "command": "arm",
    }
    with pytest.raises(CyberwaveError):
        TwinCommandGateway(client, TWIN).audit(decision="maybe")


def test_control_policy_get_put_delete() -> None:
    policy = {"level": "limited", "governed": True}
    client = _client(
        (404, {"detail": "none"}),
        (200, policy),
        (200, policy),
        (204, None),
        (404, None),
    )
    manager = TwinCommandGateway(client, TWIN).control_policy
    assert manager() is None
    assert manager.get() == policy
    assert manager.put("limited", allowed_commands=["arm"], governed=True) == policy
    assert _call(client)["method"] == "PUT"
    assert _call(client)["body"] == {
        "level": "limited",
        "allowed_commands": ["arm"],
        "forbidden_commands": [],
        "minimum_role": "developer",
        "require_operator_grant_for": [],
        "governed": True,
    }
    assert manager.delete() is True
    assert manager.delete() is False
    with pytest.raises(CyberwaveError):
        manager.put("everything")


def test_grants_list_create_revoke() -> None:
    grant = {"uuid": "g1", "active": True}
    client = _client((200, [grant]), (201, grant), (200, {**grant, "active": False}))
    grants = CommandGateway(client).grants(TWIN)
    assert grants.list(active=True) == [grant]
    assert dict(_call(client)["query_params"])["active"] == "true"
    assert grants.create("user-1", duration_s=600, mission_id="m1") == grant
    assert _call(client)["body"] == {
        "grantee_uuid": "user-1",
        "level": "limited",
        "mission_id": "m1",
        "duration_s": 600,
    }
    assert grants.revoke("g1")["active"] is False
    assert _call(client)["path_params"] == {"uuid": TWIN, "grant_uuid": "g1"}
    with pytest.raises(CyberwaveError):
        grants.create("user-1", duration_s=1, level="observe")


def test_twin_commands_handle_exposes_gateway_and_v5_request() -> None:
    client = _client((200, {"audit_id": "a9", "status": "sent"}), (200, []))
    twin = SimpleNamespace(
        uuid=TWIN,
        client=client,
        driver=SimpleNamespace(
            get_mqtt_schema=lambda **_: {
                "commands": {"supported": ["send", "audit", "takeoff"]}
            },
            get_supported_commands=lambda: ["send", "audit", "takeoff"],
        ),
        _request_command=MagicMock(return_value={"status": "accepted"}),
    )
    handle = TwinCommandsHandle(twin)  # type: ignore[arg-type]
    # Gateway names are never shadowed by catalog commands of the same name.
    assert "send" not in handle._bound_catalog_commands
    assert "takeoff" in handle._bound_catalog_commands
    assert handle.send("takeoff").status == "sent"
    assert handle.grants.list() == []
    assert handle.control_policy is handle.gateway.control_policy
    assert handle.request("arm", {"x": 1}, timeout_s=3) == {"status": "accepted"}
    twin._request_command.assert_called_once_with(
        "arm", {"x": 1}, timeout_s=3, source_type=None
    )
    assert "send" in handle.describe_section()["gateway_methods"]
