"""The broker's payload limit, as every MQTT publish path in the SDK applies it.

:class:`cyberwave.mqtt.BaseMQTTClient` refuses an oversized publish before
paho sends it; the Zenoh bridge and :func:`cyberwave.mqtt.request_response.reply_to`
drive a raw paho client and use the same rules from here, so no path sends
what the broker would silently drop.
"""

from __future__ import annotations

import collections
import logging
import os
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: The payload limit Cyberwave's brokers are configured with from CYB-4375 on
#: (mosquitto ``message_size_limit``, nanomq ``max_packet_size``); before it they
#: accept up to the protocol's 256 MiB, so this refuses payloads they would
#: still deliver. A broker drops anything
#: larger, and over MQTT 3.1.1 it cannot say so: a QoS 0 message just vanishes
#: and a QoS 1 PUBACK reports success. Refusing it here is the only place the
#: driver can hear about it. A broker configured to accept more needs
#: ``CYBERWAVE_MQTT_MAX_PAYLOAD_BYTES`` raised to match, or 0.
DEFAULT_MAX_PAYLOAD_BYTES = 32 * 1024 * 1024
MAX_PAYLOAD_BYTES_ENV = "CYBERWAVE_MQTT_MAX_PAYLOAD_BYTES"

#: The largest point cloud Cyberwave's telemetry pipeline records: Vector's
#: ``size_guard_colored_pointcloud`` drops a larger ``.../pointcloud`` payload
#: before parsing it, because parsing is what bounds Vector's memory (see
#: ``cyberwave-backend/docs/COLORED_POINTCLOUD_PIPELINE.md``). The broker still
#: delivers such a cloud to live subscribers, so it is published, with a
#: warning: without one the driver could not tell the cloud was not stored.
POINTCLOUD_INGEST_MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
POINTCLOUD_TOPIC_SUFFIX = "/pointcloud"

#: A PUBLISH packet's bytes other than its topic, payload and MQTT 5
#: properties, rounded up: the fixed header (at most 5), the topic length (2),
#: the packet id (2) and an MQTT 5 property block holding a topic alias (at
#: most 7). Properties the caller passes are measured on top.
PUBLISH_PACKET_OVERHEAD_BYTES = 16

#: The largest value the C++ SDK's ``std::size_t`` parse accepts.
_SIZE_MAX = 2**64 - 1
_SIZE_MAX_DIGITS = len(str(_SIZE_MAX))

#: Refused publishes are reported at most this often per topic (and broker
#: reason code), with a count of those in between. A 30 Hz depth stream that
#: goes over the limit would otherwise write thirty errors a second.
DROPPED_PUBLISH_LOG_INTERVAL_S = 10.0

#: How many topics :class:`ReportThrottle` remembers. A driver publishing to
#: per-request reply topics would otherwise grow its map without end; the
#: least recently reported topic is forgotten first, which at worst lets one
#: extra report through.
REPORT_THROTTLE_MAX_KEYS = 256


def resolve_max_payload_bytes(value: Optional[int]) -> int:
    """The configured payload ceiling in bytes, where 0 turns the check off.

    A malformed argument raises. A malformed environment value is logged and
    the default used instead, as the C++ SDK does with the same variable: one
    typo in a fleet-wide setting must not stop the Python drivers while the
    C++ ones keep running.
    """
    if value is None:
        # The whitespace the C++ SDK trims; str.strip() would also take "\v"
        # and a no-break space, turning "0\v" into a silent "no limit".
        source = os.getenv(MAX_PAYLOAD_BYTES_ENV, "")
        if not source:
            return DEFAULT_MAX_PAYLOAD_BYTES
        raw = source.strip(" \t\r\n")
        # The values std::from_chars accepts: ASCII digits only, so no sign,
        # no "1_000" and no other script's digits, all of which int() takes,
        # and nothing over SIZE_MAX, which int() takes as a limit nothing hits.
        # Leading zeros go first, as from_chars skips them: CPython counts
        # them towards the 4300 digits it refuses to convert past.
        digits = raw.lstrip("0") or "0"
        if (
            raw.isascii()
            and raw.isdecimal()
            and len(digits) <= _SIZE_MAX_DIGITS
            and int(digits) <= _SIZE_MAX
        ):
            return int(digits)
        logger.warning(
            "Invalid %s=%r (want a byte count, or 0 for no limit); using %d",
            MAX_PAYLOAD_BYTES_ENV,
            source,
            DEFAULT_MAX_PAYLOAD_BYTES,
        )
        return DEFAULT_MAX_PAYLOAD_BYTES
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"max_payload_bytes must be a byte count of 0 or more, not {value!r}"
        )
    return value


def payload_bytes(payload: Any) -> int:
    """How many bytes paho will put on the wire for ``payload``.

    Serialized JSON is ASCII (``json.dumps`` escapes everything else), so the
    common case is answered without copying a payload that may be megabytes.
    """
    if payload is None:
        return 0
    if isinstance(payload, str):
        return len(payload) if payload.isascii() else len(payload.encode("utf-8"))
    if isinstance(payload, (bytes, bytearray)):
        return len(payload)
    if isinstance(payload, memoryview):
        return payload.nbytes
    if isinstance(payload, (int, float)):
        return len(str(payload))
    # paho refuses any other type itself, with an error ``publish`` reports.
    return 0


def properties_bytes(properties: Any) -> int:
    """How many bytes paho's MQTT 5 ``properties`` add to a PUBLISH, length included."""
    if properties is None:
        return 0
    try:
        return len(properties.pack())
    except Exception:
        # A property paho cannot pack fails the publish itself, reported there.
        return 0


def oversize_reason(
    topic: str,
    payload: Any,
    *,
    max_payload_bytes: int,
    broker_max_packet_size: int = 0,
    properties: Any = None,
) -> Optional[str]:
    """Why the broker would drop this publish for its size, or None if it fits.

    ``max_payload_bytes`` 0 skips the payload check; ``broker_max_packet_size``
    0 (no MQTT 5 CONNACK limit) skips the packet check.
    """
    size = payload_bytes(payload)
    if max_payload_bytes and size > max_payload_bytes:
        return (
            f"payload is {size} bytes, over the {max_payload_bytes}-byte "
            f"limit (set {MAX_PAYLOAD_BYTES_ENV} or max_payload_bytes only if "
            "the broker accepts more)"
        )
    if broker_max_packet_size:
        packet = (
            size
            + len(topic.encode("utf-8"))
            + PUBLISH_PACKET_OVERHEAD_BYTES
            + properties_bytes(properties)
        )
        if packet > broker_max_packet_size:
            return (
                f"packet is about {packet} bytes, over the broker's "
                f"{broker_max_packet_size}-byte Maximum Packet Size"
            )
    return None


def unrecorded_reason(topic: str, payload: Any) -> Optional[str]:
    """Why the cloud would not record this publish, though the broker delivers it.

    None for a payload it records; only point clouds have a smaller limit there.
    """
    if not topic.endswith(POINTCLOUD_TOPIC_SUFFIX):
        return None
    size = payload_bytes(payload)
    if size <= POINTCLOUD_INGEST_MAX_PAYLOAD_BYTES:
        return None
    return (
        f"payload is {size} bytes, over the {POINTCLOUD_INGEST_MAX_PAYLOAD_BYTES}-"
        "byte point cloud limit of Cyberwave's telemetry pipeline: live "
        "subscribers get it, but it is not recorded (downsample it to keep it)"
    )


class ReportThrottle:
    """Lets one report per key through each interval, counting those held back.

    Thread-safe, and bounded to :data:`REPORT_THROTTLE_MAX_KEYS` keys.
    """

    def __init__(
        self,
        interval_s: float = DROPPED_PUBLISH_LOG_INTERVAL_S,
        *,
        max_keys: int = REPORT_THROTTLE_MAX_KEYS,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._interval_s = interval_s
        self._max_keys = max_keys
        self._clock = clock
        self._entries: collections.OrderedDict[str, tuple[float, int]] = (
            collections.OrderedDict()
        )
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def admit(self, key: str) -> Optional[int]:
        """None to stay quiet, else how many reports of ``key`` were held back."""
        # time.monotonic is looked up per call, so a test that patches it sees it.
        now = self._clock() if self._clock is not None else time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                last, suppressed = entry
                if now - last < self._interval_s:
                    self._entries[key] = (last, suppressed + 1)
                    return None
                self._entries.move_to_end(key)
            else:
                suppressed = 0
                while len(self._entries) >= self._max_keys:
                    self._entries.popitem(last=False)
            self._entries[key] = (now, 0)
            return suppressed

    def report(
        self, key: str, text: str, log: Callable[[str], Any] = logger.error
    ) -> None:
        """Log ``text`` through ``log`` unless ``key`` was reported too recently."""
        suppressed = self.admit(key)
        if suppressed is None:
            return
        if suppressed:
            text = f"{text} ({suppressed} more since the last report)"
        log(text)
