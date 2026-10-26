"""Generic MQTT v5 request/response: Response Topic + Correlation Data.

The mechanism behind ``twin.command_response.v1``, with nothing twin-specific in
it. A requester

1. subscribes to a reply topic only it may read,
2. publishes the request with that topic as the **Response Topic** and an
   opaque, per-request **Correlation Data**,
3. waits (bounded) for the reply that echoes the same Correlation Data.

The responder publishes to the Response Topic it was handed, echoing the
Correlation Data (:func:`reply_to`). A request without a Response Topic is
answered by nobody, so v3.1.1 publishers are unaffected.

Which topics may carry a reply is a broker ACL decision, not this module's. The
platform broker treats ``cyberwave/{twin|joint|environment}/{uuid}/.../{requester}/response``
as a reply inbox: only the client whose id is ``{requester}`` may read it, and
whoever may write to the resource may publish to it. Name the requester in the
reply topic (``response_topic`` may be a function of the client id), and reply on
a topic under the resource the request concerns. Replies elsewhere need an ACL
rule of their own.

Standard library only at module load (paho is imported where properties are
built), so the MQTT client and drivers can import it cheaply.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional, Union

from .. import _payload_limit

# Oversized replies, reported at most once per interval for each reply topic.
_oversize_reports = _payload_limit.ReportThrottle()


def new_correlation_id() -> bytes:
    """A fresh, opaque Correlation Data value for one request."""
    return uuid.uuid4().hex.encode("ascii")


def message_property(message: Any, name: str) -> Any:
    """A v5 property of a paho message (``None`` on v3.1.1 or when absent)."""
    properties = getattr(message, "properties", None)
    if properties is None:
        return None
    return getattr(properties, name, None)


def request_properties(response_topic: str, correlation_data: bytes) -> Any:
    """PUBLISH properties for a request: Response Topic + Correlation Data."""
    from paho.mqtt.packettypes import PacketTypes
    from paho.mqtt.properties import Properties

    properties = Properties(PacketTypes.PUBLISH)  # type: ignore[no-untyped-call]
    properties.ContentType = "application/json"
    properties.ResponseTopic = response_topic
    properties.CorrelationData = correlation_data
    return properties


def reply_properties(correlation_data: Optional[bytes]) -> Any:
    """PUBLISH properties for a reply: JSON content type, echoed correlation."""
    from paho.mqtt.packettypes import PacketTypes
    from paho.mqtt.properties import Properties

    properties = Properties(PacketTypes.PUBLISH)  # type: ignore[no-untyped-call]
    properties.ContentType = "application/json"
    if correlation_data is not None:
        properties.CorrelationData = bytes(correlation_data)
    return properties


def reply_to(
    paho_client: Any,
    request: Any,
    payload: Union[dict, list, str, bytes],
    *,
    topic: Optional[str] = None,
    qos: int = 1,
    max_payload_bytes: Optional[int] = None,
    broker_max_packet_size: int = 0,
) -> bool:
    """Answer one MQTT v5 request on its Response Topic (raw paho client).

    ``request`` is the paho message that carried the request. The reply goes to
    ``topic`` when given, else to the request's own Response Topic, echoing its
    Correlation Data. Returns ``False`` (and publishes nothing) when there is no
    topic to answer on. A responder that must not be steered by a requester
    validates the Response Topic itself and passes the result as ``topic``.

    A reply over the broker's payload limit is not sent either, and ``False``
    returned: the broker would drop it without telling the requester, which
    would wait out its timeout instead of hearing why. The limit is
    ``max_payload_bytes`` (0 turns the check off), else
    ``CYBERWAVE_MQTT_MAX_PAYLOAD_BYTES``, 32 MiB by default. A raw paho client
    does not say what limit its owner configured, nor the Maximum Packet Size
    the broker sent in CONNACK: pass them when they differ from the defaults
    (``CyberwaveMQTTClient.max_payload_bytes``, for one). Without
    ``broker_max_packet_size`` only the payload is checked, not the packet.
    """
    destination = topic or message_property(request, "ResponseTopic")
    if not destination:
        return False
    body = payload if isinstance(payload, (str, bytes)) else json.dumps(payload)
    properties = reply_properties(message_property(request, "CorrelationData"))
    oversize = _payload_limit.oversize_reason(
        str(destination),
        body,
        max_payload_bytes=_payload_limit.resolve_max_payload_bytes(max_payload_bytes),
        broker_max_packet_size=broker_max_packet_size,
        properties=properties,
    )
    if oversize is not None:
        _oversize_reports.report(
            f"topic:{destination}", f"Not replying on {destination}: {oversize}"
        )
        return False
    paho_client.publish(str(destination), body, qos=qos, properties=properties)
    return True


__all__ = [
    "message_property",
    "new_correlation_id",
    "reply_properties",
    "reply_to",
    "request_properties",
]
