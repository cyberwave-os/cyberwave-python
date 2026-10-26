"""MQTT v5 twin command request/response (``twin.command_response.v1``).

Covers the payload builder against the embedded contract schema, the reply-topic
rule, the driver-side :class:`CommandRequest`, ``BaseDriver`` dispatch of a v5
command to a context-taking listener, and ``request_command`` on the SDK MQTT
client.
"""

from __future__ import annotations

import asyncio
import json
import time
from importlib import resources
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import paho.mqtt.client as paho
import pytest

from _mqtt_fakes import mqtt_client as _mqtt_client
from _mqtt_fakes import v5_message as _v5_message
from cyberwave.driver import (
    BaseDriver,
    CallbackGroup,
    CommandArgs,
    CommandOutcome,
    CommandRequest,
    TopicSpec,
    current_command,
)
from cyberwave.driver.command_response import (
    accepts_command_request,
    invoke_command_listener,
)
from cyberwave.exceptions import CommandResponseTimeoutError
from cyberwave.mqtt import CyberwaveMQTTClient, pass_message, wants_message
from cyberwave.mqtt.command_response import (
    CONTRACT_ID,
    build_command_response,
    publish_command_response,
    reply_inbox_topic,
    validated_reply_topic,
)

TWIN = "5f2c7d0e-8a51-4d0f-9e0b-2a9c1f4d6b11"
OTHER_TWIN = "0b9d3c1a-1111-4222-8333-444455556666"


# -- helpers -------------------------------------------------------------------------


def _schema() -> dict[str, Any]:
    text = (
        resources.files("cyberwave.contracts")
        .joinpath("twin.command_response.v1.schema.json")
        .read_text()
    )
    return json.loads(text)


def _assert_valid(payload: dict[str, Any]) -> None:
    """Check ``payload`` against the contract (no jsonschema dependency)."""
    schema = _schema()
    for key in schema["required"]:
        assert key in payload, key
    props = schema["properties"]
    assert payload["contract"] == props["contract"]["const"]
    assert payload["status"] in props["status"]["enum"]
    for key, value in payload.items():
        kind = props.get(key, {}).get("type")
        if kind == "string":
            assert isinstance(value, str), key
        elif kind == "number":
            assert isinstance(value, (int, float)), key
        elif kind == "array":
            assert isinstance(value, list) and all(isinstance(v, str) for v in value)


# -- payload + topics ------------------------------------------------------------------


def test_build_command_response_matches_the_contract() -> None:
    minimal = build_command_response(
        CommandOutcome.accept(), twin_uuid=TWIN, command="arm", timestamp=10.0
    )
    assert minimal == {
        "contract": CONTRACT_ID,
        "twin_uuid": TWIN,
        "command": "arm",
        "status": "accepted",
        "source_type": "edge",
        "timestamp": 10.0,
    }
    _assert_valid(minimal)
    full = build_command_response(
        CommandOutcome.denied(
            "Arm: preflight failed",
            device_result="MAV_RESULT_DENIED",
            messages=["PreArm: GPS"],
        ),
        twin_uuid=TWIN,
        command="arm",
        source_type="sim",
    )
    _assert_valid(full)
    assert full["messages"] == ["PreArm: GPS"]
    assert full["device_result"] == "MAV_RESULT_DENIED"
    assert full["source_type"] == "sim"


def test_command_outcome_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="unknown command status"):
        CommandOutcome("ok")
    assert CommandOutcome.rejected("stale").reason == "stale"
    assert CommandOutcome.timed_out().status == "timeout"


def test_reply_topic_rule() -> None:
    inbox = reply_inbox_topic(TWIN, "sdk_abc", topic_prefix="dev")
    assert inbox == f"devcyberwave/twin/{TWIN}/command/sdk_abc/response"
    assert validated_reply_topic(inbox, twin_uuid=TWIN, topic_prefix="dev") == inbox
    for bad in (
        None,
        "",
        f"cyberwave/twin/{TWIN}/command/sdk_abc/response",  # wrong prefix
        f"devcyberwave/twin/{OTHER_TWIN}/command/sdk_abc/response",  # other twin
        f"devcyberwave/twin/{TWIN}/command//response",  # empty requester
        f"devcyberwave/twin/{TWIN}/command/a/b/response",  # nested
        f"devcyberwave/twin/{TWIN}/command/+/response",  # wildcard
        f"devcyberwave/twin/{TWIN}/telemetry",
    ):
        assert validated_reply_topic(bad, twin_uuid=TWIN, topic_prefix="dev") is None
    with pytest.raises(ValueError):
        reply_inbox_topic(TWIN, "a/b")


def test_publish_command_response_with_a_raw_paho_client() -> None:
    paho_client = MagicMock()
    inbox = reply_inbox_topic(TWIN, "gcs1")
    message = _v5_message(
        f"cyberwave/twin/{TWIN}/command",
        {"command": "arm"},
        response_topic=inbox,
        correlation=b"c-1",
    )
    assert publish_command_response(
        paho_client, message, CommandOutcome.accept(), twin_uuid=TWIN, command="arm"
    )
    topic, body = paho_client.publish.call_args.args
    assert topic == inbox
    assert json.loads(body)["status"] == "accepted"
    assert paho_client.publish.call_args.kwargs["qos"] == 1
    assert paho_client.publish.call_args.kwargs["properties"].CorrelationData == b"c-1"
    no_reply = _v5_message(f"cyberwave/twin/{TWIN}/command", {"command": "arm"})
    assert not publish_command_response(
        paho_client, no_reply, CommandOutcome.accept(), twin_uuid=TWIN, command="arm"
    )


# -- CommandRequest ---------------------------------------------------------------------


def test_command_request_replies_once_with_correlation() -> None:
    mqtt = _mqtt_client()
    inbox = reply_inbox_topic(TWIN, "requester", topic_prefix="dev")
    message = _v5_message(
        f"devcyberwave/twin/{TWIN}/command",
        {"command": "takeoff", "data": {"alt": 30}},
        response_topic=inbox,
        correlation=b"\x01\x02",
        user_properties=[("issued_by", json.dumps({"via": "command_gateway"}))],
    )
    envelope = json.loads(message.payload)
    ctx = CommandRequest.from_message(
        mqtt, message, envelope, twin_uuid=TWIN, source_type="sim_tele"
    )
    assert ctx.expects_reply and not ctx.replied
    assert ctx.data == {"alt": 30}
    assert ctx.correlation_data == b"\x01\x02"
    assert ctx.issued_by == {"via": "command_gateway"}

    assert ctx.deny("not armable", messages=["PreArm: compass"])
    assert not ctx.accept()  # only the first reply counts
    assert mqtt.client.publish.call_count == 1
    topic, body = mqtt.client.publish.call_args.args
    assert topic == inbox
    reply = json.loads(body)
    _assert_valid(reply)
    assert reply["status"] == "denied"
    assert reply["source_type"] == "sim"
    assert "session_id" not in reply
    assert (
        mqtt.client.publish.call_args.kwargs["properties"].CorrelationData
        == b"\x01\x02"
    )
    assert ctx.reply_payload == reply


def test_command_request_without_or_with_foreign_response_topic_does_not_reply() -> (
    None
):
    mqtt = _mqtt_client()
    plain = _v5_message(f"devcyberwave/twin/{TWIN}/command", {"command": "arm"})
    ctx = CommandRequest.from_message(mqtt, plain, {"command": "arm"}, twin_uuid=TWIN)
    assert not ctx.expects_reply
    assert ctx.reply("accepted") is False
    foreign = _v5_message(
        f"devcyberwave/twin/{TWIN}/command",
        {"command": "arm"},
        response_topic=reply_inbox_topic(OTHER_TWIN, "x", topic_prefix="dev"),
        correlation=b"1",
    )
    ctx = CommandRequest.from_message(mqtt, foreign, {"command": "arm"}, twin_uuid=TWIN)
    assert not ctx.expects_reply
    assert ctx.accept() is False
    mqtt.client.publish.assert_not_called()
    assert CommandRequest.detached({"command": "x"}).reply("accepted") is False


def test_command_request_stamps_reception_like_the_zenoh_context() -> None:
    from cyberwave.driver.transports.zenoh_subscriber import CommandContext

    before = time.monotonic()
    request = CommandRequest.detached({"command": "arm"}, twin_uuid=TWIN)
    zenoh_ctx = CommandContext(channel="commands/arm", twin_uuid=TWIN)
    after = time.monotonic()
    # Same clock and meaning on both transports: time.monotonic() at receipt.
    assert before <= request.received_at <= zenoh_ctx.received_at <= after


def test_accepts_command_request_signatures() -> None:
    def one(envelope):  # noqa: ANN001
        return None

    def two(envelope, ctx):  # noqa: ANN001
        return None

    def optional_named(envelope, ctx=None):  # noqa: ANN001
        return None

    def optional_annotated(envelope, x: CommandRequest | None = None):  # noqa: ANN001
        return None

    def optional_other(envelope, flag=False):  # noqa: ANN001
        return None

    def star(*args):  # noqa: ANN002
        return None

    class Driver:
        def on_cmd(self, envelope):  # noqa: ANN001
            return None

        def on_cmd_ctx(self, envelope, request):  # noqa: ANN001
            return None

    assert not accepts_command_request(one)
    assert accepts_command_request(two)
    assert accepts_command_request(optional_named)
    assert accepts_command_request(optional_annotated)
    assert not accepts_command_request(optional_other)
    assert not accepts_command_request(star)
    assert not accepts_command_request(Driver().on_cmd)
    assert accepts_command_request(Driver().on_cmd_ctx)


def test_invoke_command_listener_passes_request_and_sends_returned_outcome() -> None:
    mqtt = _mqtt_client()
    message = _v5_message(
        f"devcyberwave/twin/{TWIN}/command",
        {"command": "land"},
        response_topic=reply_inbox_topic(TWIN, "r", topic_prefix="dev"),
        correlation=b"x",
    )
    seen: dict[str, Any] = {}

    async def legacy(envelope):  # noqa: ANN001
        seen["legacy"] = envelope
        seen["current"] = current_command()
        return CommandOutcome.rejected("not flying")

    ctx = CommandRequest.from_message(
        mqtt, message, {"command": "land"}, twin_uuid=TWIN
    )
    asyncio.run(invoke_command_listener(legacy, {"command": "land"}, ctx))
    assert seen["legacy"] == {"command": "land"}
    assert seen["current"] is ctx
    assert current_command() is None
    assert json.loads(mqtt.client.publish.call_args.args[1])["status"] == "rejected"

    received: list[Any] = []

    def with_ctx(envelope, ctx):  # noqa: ANN001
        received.append(ctx)

    asyncio.run(invoke_command_listener(with_ctx, {"command": "land"}))
    assert isinstance(received[0], CommandRequest)
    assert not received[0].expects_reply


# -- MQTT handler message passing -------------------------------------------------------


def test_pass_message_handlers_get_the_paho_message() -> None:
    mqtt = _mqtt_client()
    got: list[Any] = []
    plain: list[Any] = []

    @pass_message
    def with_message(data, message):  # noqa: ANN001
        got.append((data, message.topic))

    mqtt.subscribe("devcyberwave/twin/+/command", with_message)
    mqtt.subscribe(f"devcyberwave/twin/{TWIN}/command", plain.append)
    msg = _v5_message(f"devcyberwave/twin/{TWIN}/command", {"command": "x"})
    mqtt._on_message(None, None, msg)
    assert got == [({"command": "x"}, f"devcyberwave/twin/{TWIN}/command")]
    assert plain == [{"command": "x"}]
    assert wants_message(with_message)
    assert wants_message(pass_message(mqtt._on_reply))


# -- BaseDriver dispatch ----------------------------------------------------------------


class _ReplyDriver(BaseDriver):
    REGISTRY_ID = "acme/reply-test"
    driver_family = "python"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.legacy_calls: list[dict[str, Any]] = []

    def define_interface(self, iface: Any) -> None:
        cmd = TopicSpec(
            namespace="twin", leaf="command", payload_schema_ref="TwinCommandPayload"
        )
        iface.add_listener(
            cmd, CallbackGroup(callback=self._on_arm), command=CommandArgs(name="arm")
        )
        iface.add_listener(
            cmd, CallbackGroup(callback=self._on_ping), command=CommandArgs(name="ping")
        )

    async def _on_arm(self, envelope: dict[str, Any], ctx: CommandRequest) -> None:
        await asyncio.sleep(0)
        ctx.deny("PreArm: battery low", device_result="MAV_RESULT_DENIED")

    def _on_ping(self, envelope: dict[str, Any]) -> None:
        self.legacy_calls.append(envelope)

    async def on_configure(self) -> None:
        pass

    async def on_connect_to_device(self) -> None:
        pass

    async def on_register_callbacks(self) -> None:
        pass

    async def on_activate(self) -> None:
        pass

    async def on_shutdown(self) -> None:
        pass

    @classmethod
    def create(cls) -> "_ReplyDriver":
        raise NotImplementedError


def test_base_driver_listener_answers_a_v5_command() -> None:
    mqtt = _mqtt_client()
    twin = SimpleNamespace(uuid=TWIN, environment_id=None, name="t", client=None)
    driver = _ReplyDriver(twin=twin)
    driver._cw = SimpleNamespace(mqtt=mqtt, config=SimpleNamespace(source_type="edge"))
    inbox = reply_inbox_topic(TWIN, "gateway-1", topic_prefix="dev")

    async def scenario() -> None:
        await driver._wire_interface_from_registry()
        mqtt._on_message(
            None,
            None,
            _v5_message(
                f"devcyberwave/twin/{TWIN}/command",
                {"command": "arm", "data": {}, "source_type": "tele"},
                response_topic=inbox,
                correlation=b"corr-9",
            ),
        )
        # A v3.1.1-style command (no properties) to a one-argument listener.
        legacy = paho.MQTTMessage(topic=f"devcyberwave/twin/{TWIN}/command".encode())
        legacy.payload = json.dumps({"command": "ping"}).encode()
        mqtt._on_message(None, None, legacy)
        for _ in range(50):
            if mqtt.client.publish.called and driver.legacy_calls:
                break
            await asyncio.sleep(0.01)

    asyncio.run(scenario())
    assert driver.legacy_calls == [{"command": "ping"}]
    replies = [
        call for call in mqtt.client.publish.call_args_list if call.args[0] == inbox
    ]
    assert len(replies) == 1
    reply = json.loads(replies[0].args[1])
    _assert_valid(reply)
    assert reply["status"] == "denied"
    assert reply["device_result"] == "MAV_RESULT_DENIED"
    assert reply["twin_uuid"] == TWIN
    assert replies[0].kwargs["properties"].CorrelationData == b"corr-9"


# -- request_command --------------------------------------------------------------------


def _answering(mqtt: CyberwaveMQTTClient, status: str = "accepted") -> None:
    """Make the mocked paho client answer every command like a v5 driver."""

    def publish(topic: str, payload: Any, qos: int = 0, properties: Any = None) -> Any:
        if topic.endswith("/command") and properties is not None:
            envelope = json.loads(payload)
            reply = build_command_response(
                CommandOutcome(status), twin_uuid=TWIN, command=envelope["command"]
            )
            mqtt._on_message(
                None,
                None,
                _v5_message(
                    properties.ResponseTopic,
                    reply,
                    correlation=properties.CorrelationData,
                ),
            )
        return SimpleNamespace(rc=paho.MQTT_ERR_SUCCESS)

    mqtt.client.publish.side_effect = publish


def test_request_command_on_a_v5_client() -> None:
    mqtt = _mqtt_client()
    _answering(mqtt, "accepted")
    started = time.monotonic()
    reply = mqtt.request_command(TWIN, "arm", {"force": False}, timeout_s=10)
    # The SUBACK is awaited, not the timeout. Generous, so a loaded parallel run
    # does not fail it.
    assert time.monotonic() - started < 8.0
    assert reply["status"] == "accepted"
    assert reply["command"] == "arm"
    inbox = reply_inbox_topic(TWIN, mqtt.client_id, topic_prefix="dev")
    mqtt.client.subscribe.assert_called_once()
    assert mqtt.client.subscribe.call_args.args[0] == inbox
    command_call = mqtt.client.publish.call_args_list[0]
    assert command_call.args[0] == f"devcyberwave/twin/{TWIN}/command"
    envelope = json.loads(command_call.args[1])
    assert envelope["command"] == "arm"
    assert envelope["data"] == {"force": False}
    assert envelope["source_type"] == "tele"
    assert command_call.kwargs["qos"] == 1
    assert command_call.kwargs["properties"].ResponseTopic == inbox
    # A second request reuses the inbox subscription.
    assert mqtt.request_command(TWIN, "disarm", timeout_s=2)["status"] == "accepted"
    mqtt.client.subscribe.assert_called_once()
    assert mqtt._pending_requests == {}


def test_request_command_times_out_and_ignores_late_or_foreign_replies() -> None:
    mqtt = _mqtt_client(source_type="sim")
    with pytest.raises(CommandResponseTimeoutError, match="command gateway"):
        mqtt.request_command(TWIN, "arm", timeout_s=0.05)
    assert mqtt._pending_requests == {}
    envelope = json.loads(mqtt.client.publish.call_args.args[1])
    assert envelope["source_type"] == "sim_tele"
    inbox = reply_inbox_topic(TWIN, mqtt.client_id, topic_prefix="dev")
    # A late reply (its request is gone) is dropped without error.
    mqtt._on_message(
        None,
        None,
        _v5_message(inbox, {"status": "accepted"}, correlation=b"stale"),
    )
    with pytest.raises(ValueError):
        mqtt.request_command(TWIN, "arm", timeout_s=0)
    assert isinstance(CommandResponseTimeoutError("x"), TimeoutError)


def test_publish_properties_only_on_v5() -> None:
    v311 = _mqtt_client(protocol=paho.MQTTv311)
    assert v311.publish("t", {"a": 1}, properties=object())
    assert "properties" not in v311.client.publish.call_args.kwargs
    v5 = _mqtt_client()
    marker = object()
    assert v5.publish("t", {"a": 1}, qos=1, properties=marker)
    assert v5.client.publish.call_args.kwargs["properties"] is marker
