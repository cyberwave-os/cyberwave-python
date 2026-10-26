"""Twin command request/response over MQTT v5 (``twin.command_response.v1``).

A twin command (``{prefix}cyberwave/twin/{twin_uuid}/command``) is
fire-and-forget over MQTT 3.1.1. Over MQTT v5 the requester can ask for the
outcome instead:

1. it subscribes to its own reply inbox under the twin,
   ``{prefix}cyberwave/twin/{twin_uuid}/command/{client_id}/response`` (the
   broker lets only that client id read it);
2. it publishes the command with that inbox as the **Response Topic** and an
   opaque, per-request **Correlation Data**;
3. the twin's driver publishes exactly one ``twin.command_response.v1`` payload
   to the Response Topic, QoS 1, echoing the Correlation Data.

A request without a Response Topic gets no reply, so v3.1.1 publishers are
unaffected, and requesters must bound their wait: a handler that predates the
contract never replies.

This module holds the contract-level pieces both sides share. It imports only
the standard library at module load (paho is imported where properties are
built), so it is cheap to import from the MQTT client and from drivers.
The schema lives in ``cyberwave-contracts`` as
``twin.command_response.v1.schema.json``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from .request_response import (
    message_property,
    reply_properties,
    reply_to,
    request_properties,
)

CONTRACT_ID = "twin.command_response.v1"

STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"
STATUS_DENIED = "denied"
STATUS_TIMEOUT = "timeout"
#: ``accepted``: the device took the command. ``rejected``: the handler refused
#: it before it reached the device (stale, wrong source type, unsupported or
#: malformed). ``denied``: the device refused it. ``timeout``: the device did
#: not answer in time.
STATUSES = frozenset({STATUS_ACCEPTED, STATUS_REJECTED, STATUS_DENIED, STATUS_TIMEOUT})

#: ``source_type`` values a reply may carry: the runtime that produced it.
REPLY_SOURCE_EDGE = "edge"
REPLY_SOURCE_SIM = "sim"

_COMMAND_TOPIC = "cyberwave/twin/{twin_uuid}/command"
_REPLY_TOPIC = "cyberwave/twin/{twin_uuid}/command/{client_id}/response"
_REPLY_SUFFIX = "/response"


def command_topic(twin_uuid: str, *, topic_prefix: str = "") -> str:
    """The twin's command topic, ``{prefix}cyberwave/twin/{uuid}/command``."""
    return f"{topic_prefix}{_COMMAND_TOPIC.format(twin_uuid=twin_uuid)}"


def reply_inbox_topic(twin_uuid: str, client_id: str, *, topic_prefix: str = "") -> str:
    """A requester's reply inbox under ``twin_uuid`` (its v5 Response Topic)."""
    if not _valid_segment(client_id):
        raise ValueError(f"client id {client_id!r} cannot be a topic segment")
    return (
        f"{topic_prefix}{_REPLY_TOPIC.format(twin_uuid=twin_uuid, client_id=client_id)}"
    )


def _valid_segment(value: str) -> bool:
    return bool(value) and not any(char in value for char in "/+#")


def validated_reply_topic(
    response_topic: Optional[str], *, twin_uuid: str, topic_prefix: str = ""
) -> Optional[str]:
    """``response_topic`` if it is this twin's reply inbox, else ``None``.

    A handler answers only on ``{prefix}cyberwave/twin/{twin_uuid}/command/
    {requester}/response``, so a request cannot make one twin's driver publish
    somewhere else.
    """
    if not response_topic:
        return None
    topic = str(response_topic)
    head = f"{topic_prefix}cyberwave/twin/{twin_uuid}/command/"
    if not (topic.startswith(head) and topic.endswith(_REPLY_SUFFIX)):
        return None
    requester = topic[len(head) : -len(_REPLY_SUFFIX)]
    return topic if _valid_segment(requester) else None


@dataclass(frozen=True)
class CommandOutcome:
    """What happened to one command, as the requester needs to know it."""

    status: str
    reason: Optional[str] = None
    device_result: Optional[str] = None
    messages: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(
                f"unknown command status {self.status!r}; expected one of "
                f"{sorted(STATUSES)}"
            )
        # Accept any iterable of strings, store an immutable tuple.
        object.__setattr__(self, "messages", tuple(str(m) for m in self.messages))

    @property
    def accepted(self) -> bool:
        return self.status == STATUS_ACCEPTED

    @classmethod
    def accept(cls) -> "CommandOutcome":
        return cls(STATUS_ACCEPTED)

    @classmethod
    def rejected(cls, reason: str) -> "CommandOutcome":
        return cls(STATUS_REJECTED, reason=reason)

    @classmethod
    def denied(
        cls,
        reason: str,
        *,
        device_result: Optional[str] = None,
        messages: Iterable[str] = (),
    ) -> "CommandOutcome":
        return cls(
            STATUS_DENIED,
            reason=reason,
            device_result=device_result,
            messages=tuple(messages),
        )

    @classmethod
    def timed_out(
        cls, reason: str = "the device did not answer in time"
    ) -> "CommandOutcome":
        return cls(STATUS_TIMEOUT, reason=reason)


def build_command_response(
    outcome: CommandOutcome,
    *,
    twin_uuid: str,
    command: str,
    source_type: str = REPLY_SOURCE_EDGE,
    timestamp: Optional[float] = None,
) -> dict[str, Any]:
    """The ``twin.command_response.v1`` payload for ``outcome``.

    Optional fields are omitted rather than sent empty, as the contract (and
    the shared aerial kit) does.
    """
    payload: dict[str, Any] = {
        "contract": CONTRACT_ID,
        "twin_uuid": twin_uuid,
        "command": command,
        "status": outcome.status,
        "source_type": source_type,
        "timestamp": time.time() if timestamp is None else float(timestamp),
    }
    if outcome.reason:
        payload["reason"] = outcome.reason
    if outcome.device_result:
        payload["device_result"] = outcome.device_result
    if outcome.messages:
        payload["messages"] = list(outcome.messages)
    return payload


def reply_source_type(source_type: Optional[str]) -> str:
    """Map an SDK ``source_type`` to the reply's runtime (``edge`` / ``sim``)."""
    return REPLY_SOURCE_SIM if source_type in {"sim", "sim_tele"} else REPLY_SOURCE_EDGE


def is_command_response(payload: Any) -> bool:
    """Whether ``payload`` looks like a ``twin.command_response.v1`` reply."""
    return (
        isinstance(payload, Mapping)
        and payload.get("contract") == CONTRACT_ID
        and payload.get("status") in STATUSES
    )


def publish_command_response(
    paho_client: Any,
    message: Any,
    outcome: CommandOutcome,
    *,
    twin_uuid: str,
    command: str,
    topic_prefix: str = "",
    source_type: str = REPLY_SOURCE_EDGE,
) -> bool:
    """Answer one MQTT v5 request on its Response Topic (raw paho client).

    ``message`` is the paho message that carried the command. Returns ``False``
    (and publishes nothing) when it has no Response Topic, or one that is not
    this twin's reply inbox.
    """
    topic = validated_reply_topic(
        message_property(message, "ResponseTopic"),
        twin_uuid=twin_uuid,
        topic_prefix=topic_prefix,
    )
    if topic is None:
        return False
    return reply_to(
        paho_client,
        message,
        build_command_response(
            outcome, twin_uuid=twin_uuid, command=command, source_type=source_type
        ),
        topic=topic,
    )


__all__ = [
    "CONTRACT_ID",
    "STATUSES",
    "STATUS_ACCEPTED",
    "STATUS_REJECTED",
    "STATUS_DENIED",
    "STATUS_TIMEOUT",
    "REPLY_SOURCE_EDGE",
    "REPLY_SOURCE_SIM",
    "CommandOutcome",
    "build_command_response",
    "command_topic",
    "is_command_response",
    "message_property",
    "publish_command_response",
    "reply_inbox_topic",
    "reply_properties",
    "reply_source_type",
    "request_properties",
    "validated_reply_topic",
]
