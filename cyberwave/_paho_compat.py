"""Keep a broker's "Packet too large" answer from killing paho's network thread.

A broker that refuses an oversized QoS 1 or 2 publish says so in the PUBACK or
PUBREC, with MQTT 5 reason code 0x95 (149). mosquitto does exactly this when a
payload is over its ``message_size_limit``. paho-mqtt 2.1 and 2.2 only know 149
as a CONNACK and DISCONNECT code, so parsing the ack raises ``ValueError``; nothing
between the parser and ``loop_start``'s thread catches it, and the thread
exits. The client then neither sends, receives nor reconnects, and
``on_disconnect`` never runs to say so -- it stops mid-session without a word.

The fix is to name 149 for those two packets as well, so the ack parses and
``on_publish`` receives the refusal like any other reason code. It is additive
and idempotent: an existing name or packet is never changed, and a paho that
already lists these packets, or has reorganised the table, is left as it is.

Imported for its side effect by ``cyberwave.mqtt``, which builds the SDK's
paho client. The zenoh bridge builds its own, but over MQTT 3.1.1, whose acks
carry no reason code, so it cannot hit this.

A QoS 2 refusal has a second problem, which ``report_refused_pubrecs`` covers:
paho parses the PUBREC code and then ignores it, so the refusal never reaches
``on_publish`` at all.
"""

from __future__ import annotations

import functools
import struct
from typing import Any, Callable

from paho.mqtt.packettypes import PacketTypes  # type: ignore
from paho.mqtt.reasoncodes import ReasonCode  # type: ignore

#: MQTT 5 reason code 0x95.
PACKET_TOO_LARGE = 149
_PACKET_TOO_LARGE_NAME = "Packet too large"
_PUBLISH_ACKS = (PacketTypes.PUBACK, PacketTypes.PUBREC)
_INSTALLED = "_cyberwave_names_packet_too_large_on_publish_acks"


def _naming_packet_too_large_on_publish_acks(
    method: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrap a name lookup so the table it reads covers the publish acks.

    The table is built per instance in ``ReasonCode.__init__``, so it cannot be
    patched once at import. Every path from a number or name to a reason code
    -- ``unpack``, ``getName`` and ``set`` -- goes through one of the two
    lookups this wraps, including the validation ``__init__`` itself runs.
    """

    @functools.wraps(method)
    def lookup(self: Any, *args: Any, **kwargs: Any) -> Any:
        names = getattr(self, "names", None)
        entry = names.get(PACKET_TOO_LARGE) if isinstance(names, dict) else None
        packets = entry.get(_PACKET_TOO_LARGE_NAME) if isinstance(entry, dict) else None
        if isinstance(packets, list):
            for packet in _PUBLISH_ACKS:
                if packet not in packets:
                    packets.append(packet)
        return method(self, *args, **kwargs)

    return lookup


def install() -> None:
    """Apply the fix to paho's ``ReasonCode``. Safe to call more than once."""
    if getattr(ReasonCode, _INSTALLED, False):
        return
    ReasonCode.__getName__ = _naming_packet_too_large_on_publish_acks(  # type: ignore[method-assign]
        ReasonCode.__getName__
    )
    ReasonCode.getId = _naming_packet_too_large_on_publish_acks(ReasonCode.getId)  # type: ignore[method-assign]
    setattr(ReasonCode, _INSTALLED, True)


def report_refused_pubrecs(client: Any, report: Callable[[int, Any], None]) -> None:
    """Call ``report(mid, reason_code)`` when the broker refuses a QoS 2 publish.

    paho 2.1 and 2.2 parse the PUBREC reason code and then drop it: it sends PUBREL
    anyway, the broker completes the flow with a successful PUBCOMP, and
    ``on_publish`` sees only that. A refused QoS 2 publish -- mosquitto refuses
    an oversized one this way -- therefore looks delivered. This wraps the
    one client's PUBREC handler to read the code first; the handler itself runs
    unchanged. A paho without that handler or packet buffer is left as it is.
    """
    handle = getattr(client, "_handle_pubrec", None)
    if not callable(handle) or getattr(handle, _INSTALLED, False):
        return

    @functools.wraps(handle)
    def handle_pubrec(*args: Any, **kwargs: Any) -> Any:
        refusal = None
        try:
            packet = client._in_packet
            # A reason code follows the packet identifier only in MQTT 5.
            if packet["remaining_length"] > 2:
                mid, value = struct.unpack("!HB", packet["packet"][:3])
                if value >= 128:
                    refusal = (mid, value)
        except (AttributeError, KeyError, TypeError, struct.error):
            pass
        result = handle(*args, **kwargs)
        if refusal is not None:
            mid, value = refusal
            try:
                reason_code: Any = ReasonCode(PacketTypes.PUBREC, identifier=value)
            except ValueError:
                reason_code = value
            report(mid, reason_code)
        return result

    setattr(handle_pubrec, _INSTALLED, True)
    client._handle_pubrec = handle_pubrec


install()
