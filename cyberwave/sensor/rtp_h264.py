"""H.264 over RTP/UDP ingest: RFC 3550 parsing and RFC 6184 depayloading.

Many video sources push H.264 over RTP/UDP: IP cameras, gimbal payloads, drone
ground stations, ``ffmpeg -f rtp``, GStreamer ``rtph264pay``. The SDK already
has a WebRTC path for pre-encoded H.264
(:class:`~cyberwave.sensor.camera_h264.H264PacketCameraStreamer`), which wants
whole Annex-B access units through a fast, non-blocking
``get_packet() -> (bytes, is_keyframe) | None``. This module is the bridge
between the two, with no re-encode: RTP parsing, H.264 depayloading, and a
receiver thread with a small bounded hand-over queue
(:class:`UdpRtpReceiver`). :class:`~cyberwave.sensor.camera_h264.RtpH264CameraStreamer`
wires a receiver to the WebRTC streamer.

It is deliberately stdlib-only, so it imports (and is tested) without PyAV,
aiortc or GStreamer, and the receive path is just a socket. That is also why it
does not reuse aiortc's ``RtpPacket``, ``H264PayloadDescriptor`` and
``JitterBuffer``: they assume a WebRTC peer with an RTCP channel back to the
sender, where a gap is answered with NACK/PLI. A one-way UDP push has no such
channel, so the useful policy is different (drop to the next IDR, below), and
importing aiortc would pull PyAV into a path that never decodes. NAL framing,
SPS parsing and the SPS/PPS cache are shared with the encoder side in
:mod:`cyberwave.sensor.h264_bitstream`.

Robustness choices, each because a forwarded radio-link stream needs it:

* Loss is detected from RTP sequence gaps (with 16-bit wrap). A lost packet
  makes the in-progress access unit and every following P-frame undecodable,
  so the depayloader drops them and waits for the next IDR instead of
  forwarding a smeared picture. There is no control channel back to the
  sender to request a keyframe, so the encoder's GOP is the recovery time.
* There is no reorder buffer: a late (reordered) packet is counted and
  dropped, and the gap it left already cost the picture. A source that
  reorders routinely, or speaks RTSP, is better served by GStreamer's jitter
  buffer and depayloader (``rtspsrc latency=N ! rtph264depay ! h264parse
  config-interval=-1``), as in the edge runtime's
  ``intelligence/ros2/rtsp-h264-passthrough/cyberwave_rtsp_h264/gst_pipeline.py``.
* SPS/PPS may arrive in-band periodically, once at start, or only out of band
  (SDP ``sprop-parameter-sets``). The latest ones seen (or configured) are
  cached and inserted into any IDR that lacks them (after a leading AUD, see
  :class:`~cyberwave.sensor.h264_bitstream.ParameterSetCache`), so a WebRTC
  viewer that joins mid-stream still gets a decodable keyframe.
* A sender may switch sources (new SPS, possibly new SSRC). An SSRC change is
  a new stream: state resets and the next IDR resynchronises.

Packetization modes 0 and 1 (single NAL, STAP-A, FU-A) are supported; the
interleaved mode's STAP-B/MTAP/FU-B are counted and force a resync.
"""

from __future__ import annotations

import base64
import contextlib
import logging
import random
import socket
import struct
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing_extensions import Self

from .h264_bitstream import (
    NAL_AUD,
    NAL_IDR,
    NAL_PPS,
    NAL_SEI,
    NAL_SPS,
    VCL_SLICES,
    ParameterSetCache,
    first_mb_in_slice,
    join_annexb,
    nal_type,
    parse_sps_dimensions,
    split_annexb,
)

logger = logging.getLogger(__name__)

RTP_VERSION = 2
RTP_HEADER_SIZE = 12
H264_CLOCK_RATE = 90_000
DEFAULT_PAYLOAD_TYPE = 96
DEFAULT_MTU = 1200  # RTP payload bytes; leaves room for IP/UDP/RTP on 1500 links

# RFC 6184 aggregation and fragmentation packets.
NAL_STAP_A = 24
NAL_STAP_B = 25
NAL_MTAP16 = 26
NAL_MTAP24 = 27
NAL_FU_A = 28
NAL_FU_B = 29

_UNSUPPORTED_AGGREGATES = frozenset({NAL_STAP_B, NAL_MTAP16, NAL_MTAP24, NAL_FU_B})
_AU_LEADING_NALS = frozenset({NAL_SEI, NAL_SPS, NAL_PPS, NAL_AUD})

# A sequence number this far *behind* the last one is not a late packet but a
# sender restart that kept its SSRC (otherwise every packet would be discarded
# as "reordered" until the counter caught up: up to 32768 packets of freeze).
RESYNC_BACKWARD_JUMP = 100


class RtpParseError(ValueError):
    """A datagram is not a well-formed RTP version 2 packet."""


# -- RTP (RFC 3550) -------------------------------------------------------------


@dataclass(frozen=True)
class RtpPacket:
    """One RTP packet with padding and header extension already stripped."""

    payload_type: int
    sequence: int
    timestamp: int
    ssrc: int
    payload: bytes
    marker: bool = False
    padding: bool = False
    extension: bool = False
    csrcs: tuple[int, ...] = ()
    extension_profile: int | None = None
    extension_data: bytes = b""

    @classmethod
    def parse(cls, data: bytes) -> RtpPacket:
        if len(data) < RTP_HEADER_SIZE:
            raise RtpParseError(f"datagram too short for RTP ({len(data)} bytes)")
        b0, b1, sequence, timestamp, ssrc = struct.unpack_from("!BBHII", data)
        version = b0 >> 6
        if version != RTP_VERSION:
            raise RtpParseError(f"RTP version {version}, expected 2")
        padding = bool(b0 & 0x20)
        extension = bool(b0 & 0x10)
        csrc_count = b0 & 0x0F
        marker = bool(b1 & 0x80)
        payload_type = b1 & 0x7F

        offset = RTP_HEADER_SIZE
        end = len(data)
        if offset + 4 * csrc_count > end:
            raise RtpParseError("CSRC list runs past the end of the datagram")
        csrcs = struct.unpack_from(f"!{csrc_count}I", data, offset)
        offset += 4 * csrc_count

        extension_profile: int | None = None
        extension_data = b""
        if extension:
            if offset + 4 > end:
                raise RtpParseError("header extension runs past the datagram")
            extension_profile, words = struct.unpack_from("!HH", data, offset)
            offset += 4
            if offset + 4 * words > end:
                raise RtpParseError("header extension runs past the datagram")
            extension_data = bytes(data[offset : offset + 4 * words])
            offset += 4 * words

        if padding:
            pad = data[-1]
            if pad == 0 or offset + pad > end:
                raise RtpParseError(f"invalid RTP padding length {pad}")
            end -= pad

        return cls(
            payload_type=payload_type,
            sequence=sequence,
            timestamp=timestamp,
            ssrc=ssrc,
            payload=bytes(data[offset:end]),
            marker=marker,
            padding=padding,
            extension=extension,
            csrcs=tuple(csrcs),
            extension_profile=extension_profile,
            extension_data=extension_data,
        )

    def serialize(self) -> bytes:
        """Encode without padding (the packetizer never needs it)."""
        b0 = (RTP_VERSION << 6) | (0x10 if self.extension else 0) | len(self.csrcs)
        b1 = (0x80 if self.marker else 0) | (self.payload_type & 0x7F)
        header = struct.pack(
            "!BBHII",
            b0,
            b1,
            self.sequence & 0xFFFF,
            self.timestamp & 0xFFFFFFFF,
            self.ssrc & 0xFFFFFFFF,
        )
        header += b"".join(struct.pack("!I", csrc) for csrc in self.csrcs)
        if self.extension:
            ext = self.extension_data
            if len(ext) % 4:
                raise ValueError("RTP header extension must be a multiple of 4 bytes")
            header += struct.pack("!HH", self.extension_profile or 0, len(ext) // 4)
            header += ext
        return header + self.payload


def sequence_delta(previous: int, current: int) -> int:
    """Signed distance between 16-bit RTP sequence numbers (handles wrap)."""
    delta = (current - previous) & 0xFFFF
    return delta - 0x10000 if delta >= 0x8000 else delta


def is_dynamic_payload_type(payload_type: int) -> bool:
    return 96 <= payload_type <= 127


# -- SDP and RFC 6184 helpers --------------------------------------------------


def parse_sprop_parameter_sets(value: str) -> list[bytes]:
    """Decode an SDP ``sprop-parameter-sets`` value (comma-separated base64)."""
    nals: list[bytes] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            nal = base64.b64decode(part + "=" * (-len(part) % 4), validate=True)
        except ValueError as exc:
            raise ValueError(f"invalid sprop-parameter-sets entry {part!r}") from exc
        if nal:
            nals.append(nal)
    return nals


def format_sprop_parameter_sets(nals: Iterable[bytes]) -> str:
    return ",".join(base64.b64encode(nal).decode("ascii") for nal in nals)


# An aggregate or fragment header never legitimately wraps another one
# (RFC 6184); deeper nesting is a malformed or hostile datagram.
_MAX_NESTED_HEADERS = 2


def payload_starts_access_unit(payload: bytes, _depth: int = 0) -> bool:
    """Whether an RTP payload certainly begins a new picture.

    True for a leading AUD/SEI/SPS/PPS, or a slice (single, first NAL of a
    STAP-A, or the start fragment of an FU-A) at macroblock 0. Bounded: a
    datagram claiming more nesting than a real stream can is not a picture
    start, and must not cost a recursion-depth error per packet.
    """
    if not payload or _depth > _MAX_NESTED_HEADERS:
        return False
    kind = nal_type(payload)
    if kind == NAL_STAP_A:
        if len(payload) < 4:
            return False
        (size,) = struct.unpack_from("!H", payload, 1)
        return payload_starts_access_unit(payload[3 : 3 + size], _depth + 1)
    if kind == NAL_FU_A:
        if len(payload) < 3 or not payload[1] & 0x80:
            return False
        nal = bytes([(payload[0] & 0xE0) | (payload[1] & 0x1F)]) + payload[2:]
        return payload_starts_access_unit(nal, _depth + 1)
    if kind in _AU_LEADING_NALS:
        return True
    return first_mb_in_slice(payload) == 0


# -- RFC 6184 depayloader ------------------------------------------------------


@dataclass(frozen=True)
class AccessUnit:
    """One coded picture as an Annex-B byte stream."""

    data: bytes
    is_keyframe: bool
    timestamp: int
    nal_types: tuple[int, ...]


@dataclass
class DepayloaderStats:
    packets: int = 0
    access_units: int = 0
    keyframes: int = 0
    losses: int = 0  # gap events
    lost_packets: int = 0
    duplicates: int = 0
    reordered: int = 0  # late packets discarded
    poisoned_packets: int = 0  # same picture as a lost packet, discarded
    incomplete_access_units: int = 0  # first slice not at macroblock 0
    unsupported_packets: int = 0  # STAP-B / MTAP / FU-B / reserved types
    malformed_packets: int = 0
    dropped_access_units: int = 0  # discarded while waiting for a keyframe
    parameter_sets_prepended: int = 0
    keyframes_without_parameter_sets: int = 0
    stream_resets: int = 0  # SSRC changes and sender restarts


class H264Depayloader:
    """RFC 6184 (packetization-mode 0/1) depayloader producing access units.

    Not thread-safe; the receiver owns one per socket thread.
    """

    def __init__(self, *, sprop_parameter_sets: str | None = None) -> None:
        self.stats = DepayloaderStats()
        self.parameter_sets = ParameterSetCache()
        self.width: int | None = None
        self.height: int | None = None
        self.ssrc: int | None = None
        self.waiting_for_keyframe = True
        self._last_sequence: int | None = None
        self._au_timestamp: int | None = None
        self._poisoned_timestamp: int | None = None
        self._au_nals: list[bytes] = []
        self._fu: bytearray | None = None
        if sprop_parameter_sets:
            for nal in parse_sprop_parameter_sets(sprop_parameter_sets):
                self._remember_parameter_set(nal)

    # -- public ------------------------------------------------------------

    def push(self, packet: RtpPacket) -> list[AccessUnit]:
        """Feed one RTP packet; return the access units it completed (0-2)."""
        self.stats.packets += 1
        if self.ssrc != packet.ssrc:
            if self.ssrc is not None:
                self.stats.stream_resets += 1
                logger.info(
                    "RTP SSRC changed %08x -> %08x; resynchronising on next IDR",
                    self.ssrc,
                    packet.ssrc,
                )
            self.ssrc = packet.ssrc
            self._resync()

        if self._last_sequence is not None:
            delta = sequence_delta(self._last_sequence, packet.sequence)
            if delta == 0:
                self.stats.duplicates += 1
                return []
            if delta < -RESYNC_BACKWARD_JUMP:
                self.stats.stream_resets += 1
                logger.info(
                    "RTP sequence jumped back %d (sender restart?); "
                    "resynchronising on next IDR",
                    -delta,
                )
                self._resync()
            elif delta < 0:
                self.stats.reordered += 1
                return []
            elif delta > 1:
                self.stats.losses += 1
                self.stats.lost_packets += delta - 1
                self._on_loss(packet)
        self._last_sequence = packet.sequence

        if self._poisoned_timestamp is not None:
            if packet.timestamp == self._poisoned_timestamp:
                # Same picture as the lost packet(s): never a whole picture.
                self.stats.poisoned_packets += 1
                return []
            self._poisoned_timestamp = None

        out: list[AccessUnit] = []
        if self._au_timestamp is not None and packet.timestamp != self._au_timestamp:
            # The sender omitted the marker (or it was on a lost packet).
            self._emit(out)

        nals = self._extract(packet.payload)
        if nals is None:
            return out
        if nals:
            if self._au_timestamp is None:
                self._au_timestamp = packet.timestamp
            for nal in nals:
                if nal_type(nal) in (NAL_SPS, NAL_PPS):
                    self._remember_parameter_set(nal)
            self._au_nals.extend(nals)
        if packet.marker and self._fu is None:
            self._emit(out)
        return out

    def flush(self) -> list[AccessUnit]:
        out: list[AccessUnit] = []
        self._emit(out)
        return out

    # -- internals ---------------------------------------------------------

    def _extract(self, payload: bytes) -> list[bytes] | None:
        """NAL units completed by this payload, or None if it was unusable."""
        if not payload:
            self.stats.malformed_packets += 1
            return None
        if payload[0] & 0x80:  # forbidden_zero_bit
            self.stats.malformed_packets += 1
            return None
        kind = nal_type(payload)
        if 1 <= kind <= 23:
            return [payload]
        if kind == NAL_STAP_A:
            return self._unpack_stap_a(payload)
        if kind == NAL_FU_A:
            return self._unpack_fu_a(payload)
        self.stats.unsupported_packets += 1
        if kind in _UNSUPPORTED_AGGREGATES:
            # Interleaved mode: we cannot reorder, so the picture is lost.
            self._corrupt()
        return None

    def _unpack_stap_a(self, payload: bytes) -> list[bytes] | None:
        nals: list[bytes] = []
        offset = 1
        while offset + 2 <= len(payload):
            (size,) = struct.unpack_from("!H", payload, offset)
            offset += 2
            if size == 0 or offset + size > len(payload):
                self.stats.malformed_packets += 1
                self._corrupt()
                return None
            nals.append(payload[offset : offset + size])
            offset += size
        if offset != len(payload) or not nals:
            self.stats.malformed_packets += 1
            self._corrupt()
            return None
        return nals

    def _unpack_fu_a(self, payload: bytes) -> list[bytes] | None:
        if len(payload) < 2:
            self.stats.malformed_packets += 1
            self._corrupt()
            return None
        indicator, header = payload[0], payload[1]
        start, end = bool(header & 0x80), bool(header & 0x40)
        if start:
            if self._fu is not None:
                # A new fragment began before the previous one ended.
                self._corrupt()
            self._fu = bytearray([(indicator & 0xE0) | (header & 0x1F)])
            self._fu += payload[2:]
        elif self._fu is None:
            # Middle/end fragment whose start we never saw (joined mid-NAL or
            # it was lost before a sequence baseline existed).
            self.waiting_for_keyframe = True
            return None
        else:
            self._fu += payload[2:]
        if end:
            nal = bytes(self._fu)
            self._fu = None
            return [nal]
        return []

    def _resync(self) -> None:
        self._last_sequence = None
        self._poisoned_timestamp = None
        self._discard_partial()
        self.waiting_for_keyframe = True

    def _on_loss(self, packet: RtpPacket) -> None:
        """Discard everything the lost packets may have belonged to.

        The lost packets were either the tail of the in-progress picture or the
        head of this packet's picture. The in-progress one is always dropped;
        this packet's picture is dropped too (all packets sharing its RTP
        timestamp) unless the packet provably starts a picture, since an IDR
        missing its first slices must not be forwarded as a keyframe.
        """
        same_picture = packet.timestamp == self._au_timestamp
        self._discard_partial()
        self.waiting_for_keyframe = True
        if same_picture or not payload_starts_access_unit(packet.payload):
            self._poisoned_timestamp = packet.timestamp

    def _remember_parameter_set(self, nal: bytes) -> None:
        if self.parameter_sets.remember(nal):
            try:
                self.width, self.height = parse_sps_dimensions(nal)
            except ValueError as exc:
                logger.warning("could not parse SPS dimensions: %s", exc)

    def _corrupt(self) -> None:
        self._discard_partial()
        self.waiting_for_keyframe = True

    def _discard_partial(self) -> None:
        if self._au_nals or self._fu is not None:
            self.stats.dropped_access_units += 1
        self._au_nals = []
        self._au_timestamp = None
        self._fu = None

    def _emit(self, out: list[AccessUnit]) -> None:
        nals, timestamp = self._au_nals, self._au_timestamp
        self._au_nals, self._au_timestamp, self._fu = [], None, None
        if not nals or timestamp is None:
            return
        types = [nal_type(nal) for nal in nals]
        first_slice = next((nal for nal in nals if nal_type(nal) in VCL_SLICES), None)
        if first_slice is not None and first_mb_in_slice(first_slice) not in (0, None):
            # Lost its first slice(s) without a visible gap (joined mid-picture).
            self.stats.incomplete_access_units += 1
            self.stats.dropped_access_units += 1
            self.waiting_for_keyframe = True
            return
        keyframe = NAL_IDR in types
        if self.waiting_for_keyframe and not keyframe:
            self.stats.dropped_access_units += 1
            return
        if keyframe:
            self.waiting_for_keyframe = False
            completed = self.parameter_sets.complete(nals)
            if len(completed) > len(nals):
                self.stats.parameter_sets_prepended += 1
            nals, types = completed, [nal_type(nal) for nal in completed]
            if NAL_SPS not in types or NAL_PPS not in types:
                self.stats.keyframes_without_parameter_sets += 1
            self.stats.keyframes += 1
        self.stats.access_units += 1
        out.append(AccessUnit(join_annexb(nals), keyframe, timestamp, tuple(types)))


# -- RFC 6184 packetizer (sender side, for simulation and tests) ----------------


class RtpH264Packetizer:
    """Packetization-mode 1 sender: single NAL units, FU-A above the MTU.

    For simulated sources and tests; mirrors what a typical hardware sender
    does (no STAP-A, marker on the last packet of each access unit).
    """

    def __init__(
        self,
        *,
        payload_type: int = DEFAULT_PAYLOAD_TYPE,
        ssrc: int | None = None,
        mtu: int = DEFAULT_MTU,
        initial_sequence: int | None = None,
    ) -> None:
        if mtu < 3:
            raise ValueError("MTU too small for FU-A")
        rng = random.SystemRandom()
        self.payload_type = payload_type
        self.ssrc = rng.getrandbits(32) if ssrc is None else ssrc
        self.mtu = mtu
        self.sequence = (
            rng.getrandbits(16) if initial_sequence is None else initial_sequence
        ) & 0xFFFF

    def packetize(self, access_unit: bytes, timestamp: int) -> list[bytes]:
        """RTP packets for one Annex-B access unit sharing ``timestamp`` (90 kHz)."""
        payloads: list[bytes] = []
        for nal in split_annexb(access_unit):
            payloads.extend(self._fragment(nal))
        packets = []
        for index, payload in enumerate(payloads):
            packet = RtpPacket(
                payload_type=self.payload_type,
                sequence=self.sequence,
                timestamp=timestamp & 0xFFFFFFFF,
                ssrc=self.ssrc,
                payload=payload,
                marker=index == len(payloads) - 1,
            )
            packets.append(packet.serialize())
            self.sequence = (self.sequence + 1) & 0xFFFF
        return packets

    def _fragment(self, nal: bytes) -> list[bytes]:
        if len(nal) <= self.mtu:
            return [nal]
        header = nal[0]
        indicator = (header & 0xE0) | NAL_FU_A
        kind = header & 0x1F
        body = nal[1:]
        chunk = self.mtu - 2
        fragments = []
        for offset in range(0, len(body), chunk):
            fu_header = kind
            if offset == 0:
                fu_header |= 0x80
            if offset + chunk >= len(body):
                fu_header |= 0x40
            fragments.append(
                bytes([indicator, fu_header]) + body[offset : offset + chunk]
            )
        return fragments


# -- UDP receiver --------------------------------------------------------------


@dataclass
class ReceiverStats:
    packets: int = 0
    bytes: int = 0
    rejected_packets: int = 0  # wrong source, not RTP, or non-dynamic PT
    access_units: int = 0
    keyframes: int = 0
    losses: int = 0
    lost_packets: int = 0
    dropped_access_units: int = 0
    queue_overflows: int = 0
    queued: int = 0
    last_packet_age_s: float | None = None
    ssrc: int | None = None
    payload_type: int | None = None
    source: str | None = None
    width: int | None = None
    height: int | None = None
    waiting_for_keyframe: bool = True
    depayloader: DepayloaderStats = field(default_factory=DepayloaderStats)


def parse_bind(value: str, *, default_port: int = 5600) -> tuple[str, int]:
    """``host:port`` / ``host`` / ``:port`` -> (host, port)."""
    value = value.strip()
    host, sep, port_text = value.rpartition(":")
    if not sep:
        host, port_text = value, ""
    host = host.strip("[]") or "0.0.0.0"  # noqa: S104 - listening for a push is the point
    port = int(port_text) if port_text else default_port
    if not 0 <= port <= 65535:
        raise ValueError(f"port out of range: {port}")
    return host, port


class UdpRtpReceiver:
    """Receive an H.264/RTP push on a socket thread and queue access units.

    ``get_packet`` matches the SDK's ``H264PacketCameraStreamer`` contract: it
    never blocks and hands each access unit over exactly once (encoded frames
    are not idempotent, so repeating one would corrupt the reference chain).
    """

    def __init__(
        self,
        host: str = "0.0.0.0",  # noqa: S104 - the sender pushes to this host
        port: int = 5600,
        *,
        allowed_source: str | None = None,
        payload_type: int | None = None,
        sprop_parameter_sets: str | None = None,
        queue_size: int = 30,
        receive_buffer_bytes: int = 4 * 1024 * 1024,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be >= 1")
        self.host = host
        self.port = port
        self.allowed_source = allowed_source
        self.payload_type = payload_type
        self.queue_size = queue_size
        self.receive_buffer_bytes = receive_buffer_bytes
        self.depayloader = H264Depayloader(sprop_parameter_sets=sprop_parameter_sets)
        self._clock = clock
        self._lock = threading.Lock()
        self._queue: deque[AccessUnit] = deque()
        self._gate_until_keyframe = False
        self._stats = ReceiverStats()
        self._last_packet_at: float | None = None
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._keyframe_seen = threading.Event()

    # -- lifecycle -----------------------------------------------------------

    @property
    def address(self) -> tuple[str, int]:
        """The bound address (useful with port 0)."""
        if self._sock is None:
            return self.host, self.port
        host, port = self._sock.getsockname()[:2]
        return host, port

    def start(self) -> None:
        if self._thread is not None:
            return
        family = socket.AF_INET6 if ":" in self.host else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        with contextlib.suppress(OSError):
            sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_RCVBUF, self.receive_buffer_bytes
            )
        sock.bind((self.host, self.port))
        sock.settimeout(0.2)  # so stop() is honoured promptly
        self._sock = sock
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="rtp-h264-rx", daemon=True
        )
        self._thread.start()
        effective = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        logger.info(
            "listening for H.264/RTP on udp://%s:%d (SO_RCVBUF=%d, source=%s)",
            *self.address,
            effective,
            self.allowed_source or "any",
        )

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def _run(self) -> None:
        sock = self._sock
        assert sock is not None
        while not self._stop.is_set():
            try:
                data, addr = sock.recvfrom(65535)
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                logger.exception("video socket error")
                time.sleep(0.1)
                continue
            try:
                self.handle_datagram(data, addr[0])
            except Exception:
                logger.exception("failed to process RTP datagram")

    # -- data path -----------------------------------------------------------

    def handle_datagram(self, data: bytes, source_ip: str) -> list[AccessUnit]:
        """Process one datagram (the socket thread's body; callable from tests)."""
        now = self._clock()
        if self.allowed_source is not None and source_ip != self.allowed_source:
            with self._lock:
                self._stats.rejected_packets += 1
            return []
        try:
            packet = RtpPacket.parse(data)
        except RtpParseError:
            with self._lock:
                self._stats.rejected_packets += 1
            return []
        if (
            packet.payload_type != self.payload_type
            if self.payload_type is not None
            else not is_dynamic_payload_type(packet.payload_type)
        ):
            # RTCP muxed on the same port lands here too (PT 72-76 range).
            with self._lock:
                self._stats.rejected_packets += 1
            return []

        with self._lock:
            units = self.depayloader.push(packet)
            self._stats.packets += 1
            self._stats.bytes += len(data)
            self._stats.ssrc = packet.ssrc
            self._stats.payload_type = packet.payload_type
            self._stats.source = source_ip
            self._last_packet_at = now
            for unit in units:
                self._enqueue(unit)
        if any(unit.is_keyframe for unit in units):
            self._keyframe_seen.set()
        return units

    def _enqueue(self, unit: AccessUnit) -> None:
        """Queue under the lock; overflow never leaves an undecodable chain."""
        if self._gate_until_keyframe:
            if not unit.is_keyframe:
                return
            self._gate_until_keyframe = False
        if len(self._queue) >= self.queue_size:
            # The consumer is far behind. Dropping just the oldest P-frame would
            # break every frame after it, so drop whole GOPs: restart from the
            # newest queued keyframe, or from the incoming one.
            self._stats.queue_overflows += 1
            keys = [i for i, queued in enumerate(self._queue) if queued.is_keyframe]
            if unit.is_keyframe:
                self._queue.clear()
            elif keys and keys[-1] > 0:
                for _ in range(keys[-1]):
                    self._queue.popleft()
            else:
                self._queue.clear()
                self._gate_until_keyframe = True
                return
        self._queue.append(unit)

    def get_packet(self) -> tuple[bytes, bool] | None:
        """Next access unit as ``(annexb_bytes, is_keyframe)``, or None."""
        with self._lock:
            if not self._queue:
                return None
            unit = self._queue.popleft()
        return unit.data, unit.is_keyframe

    def wait_for_keyframe(self, timeout: float | None = None) -> bool:
        return self._keyframe_seen.wait(timeout)

    @property
    def keyframe_seen(self) -> bool:
        return self._keyframe_seen.is_set()

    def stats(self) -> ReceiverStats:
        with self._lock:
            d = self.depayloader
            snapshot = ReceiverStats(**{**self._stats.__dict__})
            snapshot.access_units = d.stats.access_units
            snapshot.keyframes = d.stats.keyframes
            snapshot.losses = d.stats.losses
            snapshot.lost_packets = d.stats.lost_packets
            snapshot.dropped_access_units = d.stats.dropped_access_units
            snapshot.queued = len(self._queue)
            snapshot.width, snapshot.height = d.width, d.height
            snapshot.waiting_for_keyframe = d.waiting_for_keyframe
            snapshot.depayloader = DepayloaderStats(**d.stats.__dict__)
            if self._last_packet_at is not None:
                snapshot.last_packet_age_s = self._clock() - self._last_packet_at
        return snapshot


__all__ = [
    "DEFAULT_MTU",
    "DEFAULT_PAYLOAD_TYPE",
    "H264_CLOCK_RATE",
    "NAL_FU_A",
    "NAL_FU_B",
    "NAL_MTAP16",
    "NAL_MTAP24",
    "NAL_STAP_A",
    "NAL_STAP_B",
    "RESYNC_BACKWARD_JUMP",
    "RTP_HEADER_SIZE",
    "RTP_VERSION",
    "AccessUnit",
    "DepayloaderStats",
    "H264Depayloader",
    "ReceiverStats",
    "RtpH264Packetizer",
    "RtpPacket",
    "RtpParseError",
    "UdpRtpReceiver",
    "format_sprop_parameter_sets",
    "is_dynamic_payload_type",
    "parse_bind",
    "parse_sprop_parameter_sets",
    "payload_starts_access_unit",
    "sequence_delta",
]
