"""Driver-side replies to MQTT v5 twin commands (``twin.command_response.v1``).

A command published with a v5 Response Topic and Correlation Data (by the
command gateway, ``client.mqtt.request_command`` or any v5 requester) expects
exactly one reply. :class:`~cyberwave.driver.BaseDriver` builds a
:class:`CommandRequest` (the received command, with ``reply``) for every
command it dispatches; a listener opts in by taking it as a second positional
argument, or by returning a :class:`CommandOutcome`::

    class MyDriver(BaseDriver):
        def define_interface(self, iface):
            command = TopicSpec(namespace="twin", leaf="command")
            iface.add_listener(
                command,
                CallbackGroup(callback=self._on_arm),
                command=CommandArgs(name="arm"),
            )

        async def _on_arm(self, envelope, ctx: CommandRequest):
            if not self.link_up:
                ctx.reject("vehicle link is down")
                return
            result = await self.vehicle.arm()
            ctx.reply("accepted" if result.ok else "denied", result.text)

        # ...or, without the request argument:
        async def _on_disarm(self, envelope):
            return CommandOutcome.accept()

Listeners taking only ``envelope`` are called exactly as before, and a command
sent without a Response Topic (MQTT 3.1.1, or v5 fire-and-forget) gets no
reply: ``ctx.expects_reply`` is ``False`` and ``ctx.reply`` returns ``False``.
:func:`current_command` returns the :class:`CommandRequest` being handled, for
code deeper in the call stack.

The payload builder and topic rules live in :mod:`cyberwave.mqtt.command_response`
and are re-exported here.
"""

from __future__ import annotations

import contextvars
import inspect
import json
import logging
import re
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Union

from ..mqtt.command_response import (
    CONTRACT_ID,
    REPLY_SOURCE_EDGE,
    REPLY_SOURCE_SIM,
    STATUS_ACCEPTED,
    STATUS_DENIED,
    STATUS_REJECTED,
    STATUS_TIMEOUT,
    STATUSES,
    CommandOutcome,
    build_command_response,
    message_property,
    reply_properties,
    reply_source_type,
    validated_reply_topic,
)

logger = logging.getLogger(__name__)

_CURRENT_COMMAND: contextvars.ContextVar[Optional["CommandRequest"]] = (
    contextvars.ContextVar("cyberwave_current_command", default=None)
)

PublishFn = Callable[..., Any]


class CommandRequest:
    """One received twin command and the means to answer it once.

    Attributes:
        command: The command name from the envelope.
        envelope: The decoded command payload (``command``, ``data``, ...).
        twin_uuid: The twin whose command topic carried it.
        message: The paho message (``None`` when the transport has none, e.g.
            a Zenoh command).
        source_type: ``edge`` or ``sim``, stamped on the reply.
        received_at: ``time.monotonic()`` stamp at reception (for latency
            measurement), as on the Zenoh transport's
            :class:`~cyberwave.driver.transports.zenoh_subscriber.CommandContext`.

    The MQTT counterpart of ``CommandContext``: that one carries a Zenoh
    channel command's metadata to its handler, this one also carries the
    MQTT v5 request properties and the one reply.
    """

    def __init__(
        self,
        *,
        command: str,
        envelope: Mapping[str, Any],
        twin_uuid: str,
        message: Any = None,
        publish: Optional[PublishFn] = None,
        topic_prefix: str = "",
        source_type: str = REPLY_SOURCE_EDGE,
    ) -> None:
        self.command = command
        self.envelope = envelope
        self.twin_uuid = twin_uuid
        self.message = message
        self.source_type = source_type
        self.received_at = time.monotonic()
        self._publish = publish
        self._lock = threading.Lock()
        self._reply: Optional[dict[str, Any]] = None
        self.response_topic = validated_reply_topic(
            message_property(message, "ResponseTopic"),
            twin_uuid=twin_uuid,
            topic_prefix=topic_prefix,
        )
        if self.response_topic is None and message_property(message, "ResponseTopic"):
            logger.warning(
                "Ignoring Response Topic %r for command %r: not twin %s's reply inbox",
                message_property(message, "ResponseTopic"),
                command,
                twin_uuid,
            )

    @classmethod
    def from_message(
        cls,
        mqtt_client: Any,
        message: Any,
        envelope: Mapping[str, Any],
        *,
        twin_uuid: str,
        source_type: Optional[str] = None,
    ) -> "CommandRequest":
        """The request for a command received on ``mqtt_client`` (SDK MQTT client).

        Use this from a raw ``@pass_message`` subscriber; ``BaseDriver`` does it
        for registry commands. ``source_type`` is the SDK source type of this
        runtime (``sim``/``sim_tele`` reply as ``sim``, anything else as ``edge``).
        """
        return cls(
            command=str(envelope.get("command") or ""),
            envelope=envelope,
            twin_uuid=twin_uuid,
            message=message,
            publish=getattr(mqtt_client, "publish", None),
            topic_prefix=str(getattr(mqtt_client, "topic_prefix", "") or ""),
            source_type=reply_source_type(source_type),
        )

    @classmethod
    def detached(
        cls, envelope: Mapping[str, Any], *, twin_uuid: str = ""
    ) -> "CommandRequest":
        """A request with nowhere to reply (non-MQTT transports, tests)."""
        return cls(
            command=str(envelope.get("command") or ""),
            envelope=envelope,
            twin_uuid=twin_uuid,
        )

    # -- request properties --------------------------------------------------

    @property
    def data(self) -> Any:
        """The envelope's ``data`` (``{}`` when absent)."""
        value = self.envelope.get("data")
        return {} if value is None else value

    @property
    def properties(self) -> Any:
        """The paho MQTT v5 properties of the request, or ``None``."""
        return getattr(self.message, "properties", None)

    @property
    def correlation_data(self) -> Optional[bytes]:
        value = message_property(self.message, "CorrelationData")
        return None if value is None else bytes(value)

    @property
    def user_properties(self) -> dict[str, str]:
        """MQTT v5 user properties (e.g. the gateway's ``issued_by``)."""
        pairs = message_property(self.message, "UserProperty") or []
        return {str(key): str(value) for key, value in pairs}

    @property
    def issued_by(self) -> Optional[dict[str, Any]]:
        """The gateway's ``issued_by`` (envelope field, else user property)."""
        value = self.envelope.get("issued_by")
        if isinstance(value, dict):
            return value
        raw = self.user_properties.get("issued_by")
        if raw:
            try:
                parsed = json.loads(raw)
            except ValueError:
                return None
            return parsed if isinstance(parsed, dict) else None
        return None

    @property
    def expects_reply(self) -> bool:
        """Whether the requester asked for a reply (and it can be sent)."""
        return self.response_topic is not None and self._publish is not None

    @property
    def replied(self) -> bool:
        return self._reply is not None

    @property
    def reply_payload(self) -> Optional[dict[str, Any]]:
        """The reply that was sent, if any (handy in tests)."""
        return self._reply

    # -- replying ------------------------------------------------------------

    def reply(
        self,
        outcome: Union[CommandOutcome, str],
        reason: Optional[str] = None,
        *,
        device_result: Optional[str] = None,
        messages: Iterable[str] = (),
    ) -> bool:
        """Publish the one ``twin.command_response.v1`` reply.

        ``outcome`` is a :class:`CommandOutcome` or a status string
        (``accepted``, ``rejected``, ``denied``, ``timeout``). Returns ``True``
        when a reply was published; ``False`` when the request did not ask for
        one, or a reply was already sent (only the first counts).
        """
        if isinstance(outcome, str):
            outcome = CommandOutcome(
                outcome,
                reason=reason,
                device_result=device_result,
                messages=tuple(messages),
            )
        if not self.expects_reply:
            return False
        payload = build_command_response(
            outcome,
            twin_uuid=self.twin_uuid,
            command=self.command,
            source_type=self.source_type,
        )
        with self._lock:
            if self._reply is not None:
                logger.warning(
                    "Command %r already answered (%s); dropping second reply %s",
                    self.command,
                    self._reply.get("status"),
                    outcome.status,
                )
                return False
            self._reply = payload
        assert self._publish is not None and self.response_topic is not None
        published = self._publish(
            self.response_topic,
            json.dumps(payload),
            1,
            properties=reply_properties(self.correlation_data),
        )
        return published is not False

    def accept(self) -> bool:
        return self.reply(CommandOutcome.accept())

    def reject(self, reason: str) -> bool:
        return self.reply(CommandOutcome.rejected(reason))

    def deny(
        self,
        reason: str,
        *,
        device_result: Optional[str] = None,
        messages: Iterable[str] = (),
    ) -> bool:
        return self.reply(
            CommandOutcome.denied(
                reason, device_result=device_result, messages=messages
            )
        )

    def timeout(self, reason: str = "the device did not answer in time") -> bool:
        return self.reply(CommandOutcome.timed_out(reason))

    def __repr__(self) -> str:
        return (
            f"CommandRequest(command={self.command!r}, twin={self.twin_uuid!r}, "
            f"expects_reply={self.expects_reply}, replied={self.replied})"
        )


def current_command() -> Optional[CommandRequest]:
    """The :class:`CommandRequest` of the command being handled, if any."""
    return _CURRENT_COMMAND.get()


def accepts_command_request(callback: Any) -> bool:
    """Whether a command listener takes a :class:`CommandRequest` argument.

    True when it has two required positional parameters (``(envelope, ctx)``),
    or an optional second one named ``ctx``/``context``/``request`` or annotated
    ``CommandRequest``. ``(envelope)`` listeners, ``*args`` catch-alls and
    unintrospectable callables keep the one-argument call.
    """
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return False
    params = list(signature.parameters.values())
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return False
    positional = [
        p
        for p in params
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) < 2:
        return False
    second = positional[1]
    if second.default is inspect.Parameter.empty:
        return True
    if second.name in {"ctx", "context", "request"}:
        return True
    annotation = second.annotation
    if isinstance(annotation, str):  # postponed evaluation (PEP 563)
        return re.search(r"\bCommandRequest\b", annotation) is not None
    return annotation is CommandRequest or CommandRequest in getattr(
        annotation, "__args__", ()
    )


async def invoke_command_listener(
    callback: Callable[..., Any],
    envelope: Mapping[str, Any],
    ctx: Optional[CommandRequest] = None,
) -> Any:
    """Call a command listener, passing ``ctx`` when it takes one.

    The request is also visible through :func:`current_command` while the
    listener runs. A listener that returns a :class:`CommandOutcome` without
    having replied gets that outcome sent as its reply.
    """
    if ctx is None:
        ctx = CommandRequest.detached(envelope)
    token = _CURRENT_COMMAND.set(ctx)
    try:
        if accepts_command_request(callback):
            result = callback(envelope, ctx)
        else:
            result = callback(envelope)
        if inspect.isawaitable(result):
            result = await result
    finally:
        _CURRENT_COMMAND.reset(token)
    if isinstance(result, CommandOutcome) and not ctx.replied:
        ctx.reply(result)
    return result


__all__ = [
    "CONTRACT_ID",
    "STATUSES",
    "STATUS_ACCEPTED",
    "STATUS_REJECTED",
    "STATUS_DENIED",
    "STATUS_TIMEOUT",
    "REPLY_SOURCE_EDGE",
    "REPLY_SOURCE_SIM",
    "CommandRequest",
    "CommandOutcome",
    "accepts_command_request",
    "build_command_response",
    "current_command",
    "invoke_command_listener",
    "reply_source_type",
    "validated_reply_topic",
]
