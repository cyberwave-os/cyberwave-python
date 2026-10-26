"""WebRTC video track/streamer for already-encoded H.264 access units.

``VirtualVideoTrack`` hands aiortc raw ``av.VideoFrame``s, which aiortc's
``H264Encoder.encode()`` then encodes via a PyAV ``CodecContext`` (libx264 or,
with :func:`cyberwave.sensor.hw_encoder.apply_h264_hw_patch`, a probed hardware
codec). This module instead hands aiortc already-encoded ``av.Packet``s.
aiortc's ``RTCRtpSender._next_encoded_frame`` routes anything that isn't an
``av.Frame`` through ``Encoder.pack()`` instead of ``Encoder.encode()`` --
pure NAL-unit splitting and RTP packetization, no PyAV ``CodecContext``
involved at all. Use this when encoding already happened upstream of this
process (e.g. a GStreamer ``nvv4l2h264enc`` pipeline feeding access units in
over MQTT/ROS/Zenoh/etc.), so the bytes aren't decoded and re-encoded in
software just to satisfy aiortc's default frame-in pipeline.

Framing: aiortc's ``H264Encoder.pack()`` splits NAL units on ``0x000001`` start
codes (Annex-B / byte-stream). Sources that hand over **AVCC / length-prefixed**
access units (a 4-byte length before each NAL, e.g. Kinova's RTSP depay path or
a GStreamer ``h264parse`` negotiated to ``stream-format=avc``) carry no start
codes, so ``pack()`` produces zero RTP payloads -- nothing ever reaches the
receiver and the SFU tears the transport down (``connectionState=closed`` a few
seconds after connect). :func:`to_annex_b` normalises AVCC to Annex-B before the
bytes reach aiortc; already-Annex-B input passes through untouched.

Detection is AVCC-first (try the parse, fall back to passthrough), never a
leading-bytes start-code sniff: an AVCC length prefix can *be* a start code --
a 256-511 byte first NAL encodes as ``00 00 01 xx`` -- so sniffing forwards
those unconverted and ``pack()`` emits one garbage NAL. Pass ``framing``
explicitly when the producer's framing is known.

Keyframe handling: this is a pure passthrough -- every live access unit is
forwarded as-is, so the upstream encoder's own keyframes (SPS/PPS + IDR) reach
the receiver natively at their periodic interval. We deliberately do NOT cache
and re-send keyframes on RTCP PLI/FIR: we have no control channel back to the
separate process producing the bytes, so the only keyframe available to re-send
is a *stale* one, and re-injecting it flashes an old frame (visible flicker)
without giving the decoder a correct resync point for the live P-frames that
follow. A receiver that joins mid-GOP or loses packets therefore waits for the
upstream's next natural keyframe; the right lever for faster recovery is a
shorter keyframe interval at the encoder, not stale re-injection here.

That is not the SPS/PPS caching done by the RTP ingest
(:mod:`cyberwave.sensor.rtp_h264`) and :class:`~cyberwave.sensor.hw_encoder.HardwareH264Encoder`
(:class:`~cyberwave.sensor.h264_bitstream.ParameterSetCache`): those insert
cached *parameter sets* (decoder configuration, no picture data) into the
*live* IDR that lacks them, when it is produced. No old picture is ever re-sent.
"""

from __future__ import annotations

import asyncio
import fractions
import logging
import time
from typing import TYPE_CHECKING, Callable, Optional, Tuple

from av import Packet

from . import BaseVideoStreamer, BaseVideoTrack
from .h264_bitstream import ANNEXB_START_CODE, nal_type

if TYPE_CHECKING:
    from ..mqtt_client import CyberwaveMQTTClient
    from ..utils import TimeReference
    from .rtp_h264 import UdpRtpReceiver

logger = logging.getLogger(__name__)

# Matches VirtualVideoTrack's own fallback so an unset fps behaves the same way.
DEFAULT_FPS = 15

#: Accepted values for the ``framing`` argument of :func:`to_annex_b` and the
#: ``framing`` constructor argument of :class:`H264PacketVideoTrack`.
FRAMING_AUTO = "auto"
FRAMING_ANNEXB = "annexb"
FRAMING_AVCC = "avcc"
_FRAMINGS = (FRAMING_AUTO, FRAMING_ANNEXB, FRAMING_AVCC)


def _avcc_to_annex_b(data: bytes) -> Optional[bytes]:
    """Reframe a 4-byte-length-prefixed AVCC buffer as Annex-B, or ``None`` if
    it isn't cleanly framed AVCC (so callers fall back instead of shipping
    mangled bytes).

    Validating each NAL header is what makes AVCC-first detection safe: an
    Annex-B buffer starting ``00 00 00 01`` parses as a 1-byte AVCC NAL, and
    requiring a well-formed header at every step makes that false positive
    vanishingly unlikely.
    """
    out = bytearray()
    i, n = 0, len(data)
    while i + 4 <= n:
        length = int.from_bytes(data[i : i + 4], "big")
        i += 4
        if length == 0 or i + length > n:
            return None
        nal = data[i : i + length]
        if nal[0] & 0x80 or not nal_type(nal):
            return None  # forbidden_zero_bit set, or nal_unit_type 0
        out += ANNEXB_START_CODE
        out += nal
        i += length
    if i != n or not out:
        return None  # trailing bytes, or nothing framed at all
    return bytes(out)


def to_annex_b(data: bytes, framing: str = FRAMING_AUTO) -> bytes:
    """Return an H.264 access unit as an Annex-B (start-code) byte stream.

    aiortc's ``H264Encoder.pack()`` splits NAL units on ``0x000001`` start
    codes. An **AVCC / length-prefixed** access unit -- each NAL preceded by a
    4-byte big-endian length, as produced by e.g. a GStreamer ``h264parse``
    negotiated to ``stream-format=avc``, or Kinova's RTSP depay path -- contains
    no start codes at all, so ``pack()`` yields zero NAL units and therefore
    zero RTP payloads. Nothing reaches the receiver and the SFU tears the
    transport down (observed as ``connectionState=closed`` seconds after
    connect). This normalises AVCC (4-byte length prefix) to Annex-B.

    Args:
        data: One access unit, AVCC- or Annex-B-framed.
        framing: ``"auto"`` (default) tries an AVCC parse and falls back to
            passthrough; ``"avcc"``/``"annexb"`` skip detection. Prefer an
            explicit value when the producer is known -- ``"avcc"`` also warns
            on a malformed access unit instead of silently forwarding it.

    Detection is AVCC-first, never a start-code sniff (see the module
    docstring). A buffer that doesn't frame as clean 4-byte AVCC is assumed
    Annex-B and returned unchanged, so an unrecognised framing stays visible
    rather than getting mangled.
    """
    if not data:
        return data
    if framing == FRAMING_ANNEXB:
        return data
    converted = _avcc_to_annex_b(data)
    if converted is not None:
        return converted
    if framing == FRAMING_AVCC:
        # Declared AVCC but doesn't parse as AVCC: forward unchanged, but say so.
        logger.warning(
            "framing='avcc' but access unit (%d bytes, starts %s) is not cleanly "
            "framed 4-byte AVCC; forwarding unconverted",
            len(data),
            data[:4].hex(" "),
        )
    return data


class H264PacketVideoTrack(BaseVideoTrack):
    """Video track that forwards pre-encoded H.264 access units.

    Args:
        get_packet: Callable returning the latest available
            ``(data: bytes, is_keyframe: bool)`` access unit, or ``None`` if
            none is available yet. Must be fast and non-blocking -- called
            from this track's own pacing loop, mirroring
            ``VirtualVideoTrack.get_frame``'s contract.
        width: Frame width in pixels, for ``get_stream_attributes`` only --
            purely informational, since this track never touches pixels.
        height: Frame height in pixels (see ``width``).
        fps: Target polling rate. Should match the upstream encoder's frame
            rate: unlike ``VirtualVideoTrack``'s raw frames, encoded access
            units are not idempotent, so a repeat or a drop breaks the P-frame
            reference chain until the next keyframe. Prefer a ``get_packet``
            that hands over each access unit exactly once.
        framing: Framing of the bytes ``get_packet`` returns -- ``"auto"``
            (default), ``"avcc"``, or ``"annexb"``. See :func:`to_annex_b`.
    """

    # Default for tracks built without ``__init__``; ``+=`` rebinds per instance.
    captured_frame_count: int = 0

    def __init__(
        self,
        get_packet: Callable[[], Optional[Tuple[bytes, bool]]],
        *,
        width: int,
        height: int,
        fps: int = DEFAULT_FPS,
        framing: str = FRAMING_AUTO,
        time_reference: Optional["TimeReference"] = None,
    ) -> None:
        super().__init__()
        if framing not in _FRAMINGS:
            raise ValueError(f"framing must be one of {_FRAMINGS}, got {framing!r}")
        self.get_packet = get_packet
        self.width = width
        self.height = height
        self.fps = fps
        self.framing = framing
        self.time_reference = time_reference
        self._last_time: Optional[float] = None

        # Only advances on a real access unit, unlike ``frame_count``.
        self.captured_frame_count: int = 0

    def get_stream_attributes(self) -> dict:
        """Get streaming attributes for the offer payload."""
        return {
            "camera_type": "h264_passthrough",
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }

    async def recv(self):
        """Return the next access unit as an av.Packet (encoded, not raw)."""
        now = time.time()
        if self._last_time is not None:
            elapsed = now - self._last_time
            wait = max(0.0, (1.0 / float(self.fps)) - elapsed)
            if wait > 0:
                await asyncio.sleep(wait)
        self._last_time = time.time()

        timestamp, timestamp_monotonic = self._capture_timestamp(self.time_reference)
        if self.frame_count == 0:
            self.frame_0_timestamp = timestamp
            self.frame_0_timestamp_monotonic = timestamp_monotonic

        data = b""
        try:
            result = self.get_packet()
            if result is not None:
                # The upstream is_keyframe flag is unused: this is a straight
                # passthrough, so keyframes flow through natively with the live
                # stream (we never re-send a cached/stale one -- see the module
                # docstring on why that only caused flicker).
                data, _upstream_is_keyframe = result
        except Exception as e:
            logger.warning("H264 packet provider error: %s", e)

        if data:
            # The empty-packet tick below still advances ``frame_count``, so
            # a stalled encoder would otherwise read as healthy forever.
            self.captured_frame_count += 1
            # aiortc's pack() requires Annex-B start codes; normalise AVCC input.
            data = to_annex_b(data, self.framing)

        # An empty packet round-trips to zero RTP payloads (H264Encoder.pack()
        # splits zero NAL units out of it), which RTCRtpSender treats as
        # "nothing to send this tick" -- a safe no-op while the upstream
        # encoder hasn't produced a first access unit yet.
        packet = Packet(data)
        time_base = fractions.Fraction(1, self.fps)
        packet.pts = self.frame_count
        packet.time_base = time_base

        self._store_frame_metadata_for_sync(
            frame_index=self.frame_count,
            pts=packet.pts,
            time_base_num=time_base.numerator,
            time_base_den=time_base.denominator,
            capture_wall_time=timestamp,
            capture_monotonic=timestamp_monotonic,
        )
        self._capture_sync_frame(
            timestamp,
            timestamp_monotonic,
            frame_index=self.frame_count,
            pts=packet.pts,
            time_base_num=time_base.numerator,
            time_base_den=time_base.denominator,
        )
        self.frame_count += 1
        return packet

    def close(self) -> None:
        """Release resources (no-op -- this track owns no encoder/device)."""


class H264PacketCameraStreamer(BaseVideoStreamer):
    """Stream pre-encoded H.264 access units to Cyberwave via WebRTC.

    Same signaling/reconnect/health-check machinery as ``VirtualCameraStreamer``;
    only ``initialize_track`` differs.
    """

    def __init__(
        self,
        client: "CyberwaveMQTTClient",
        get_packet: Callable[[], Optional[Tuple[bytes, bool]]],
        width: int,
        height: int,
        fps: int = DEFAULT_FPS,
        framing: str = FRAMING_AUTO,
        **kwargs,
    ) -> None:
        super().__init__(client=client, **kwargs)
        self.get_packet = get_packet
        self.width = width
        self.height = height
        self.fps = fps
        self.framing = framing

    def initialize_track(self) -> H264PacketVideoTrack:
        """Create the video track (called internally by BaseVideoStreamer)."""
        return H264PacketVideoTrack(
            self.get_packet,
            width=self.width,
            height=self.height,
            fps=self.fps,
            framing=self.framing,
            time_reference=self.time_reference,
        )


class RtpH264CameraStreamer(H264PacketCameraStreamer):
    """Forward an H.264/RTP/UDP push to Cyberwave over WebRTC, no re-encode.

    A :class:`~cyberwave.sensor.rtp_h264.UdpRtpReceiver` listens for the push
    (an IP camera, a payload or ground station, ``ffmpeg -f rtp``, GStreamer
    ``rtph264pay``) and hands whole access units to the passthrough track::

        streamer = RtpH264CameraStreamer(
            client.mqtt,
            bind="0.0.0.0:5600",
            twin_uuid=twin.uuid,
            camera_name="main",
        )
        await streamer.run_with_auto_reconnect(subscribe_to_commands=False)

    :meth:`run_with_auto_reconnect` starts the receiver, waits for the first
    keyframe (so the offer never advertises a stream that cannot show a
    picture, and carries the real resolution from the SPS), streams, and stops
    the receiver when it returns. Pass ``receiver=`` to manage an existing
    receiver yourself; it is then neither started nor stopped here.

    Args:
        client: The SDK MQTT client used for signalling.
        bind: ``host:port`` / ``:port`` to listen on (default ``0.0.0.0:5600``).
        allowed_source: Only accept datagrams from this IP.
        payload_type: Only accept this RTP payload type (default: any dynamic).
        sprop_parameter_sets: SDP ``sprop-parameter-sets`` for senders that
            never repeat SPS/PPS in-band.
        queue_size: Access units buffered between receiver and track.
        width, height: Fallback resolution when the SPS cannot be parsed.
        fps: Track polling rate; set at or above the sender's frame rate.
        receiver: Use this receiver instead of creating one.
        **kwargs: Passed to :class:`H264PacketCameraStreamer` (``twin_uuid``,
            ``camera_name``, ``turn_servers``, ...).
    """

    def __init__(
        self,
        client: "CyberwaveMQTTClient",
        *,
        bind: str = "0.0.0.0:5600",  # noqa: S104 - the sender pushes to this host
        allowed_source: Optional[str] = None,
        payload_type: Optional[int] = None,
        sprop_parameter_sets: Optional[str] = None,
        queue_size: int = 30,
        width: int = 1920,
        height: int = 1080,
        fps: int = 60,
        receiver: Optional["UdpRtpReceiver"] = None,
        **kwargs,
    ) -> None:
        from .rtp_h264 import UdpRtpReceiver, parse_bind

        self._owns_receiver = receiver is None
        if receiver is None:
            host, port = parse_bind(bind)
            receiver = UdpRtpReceiver(
                host,
                port,
                allowed_source=allowed_source,
                payload_type=payload_type,
                sprop_parameter_sets=sprop_parameter_sets,
                queue_size=queue_size,
            )
        self.receiver = receiver
        kwargs.setdefault("framing", FRAMING_ANNEXB)
        super().__init__(
            client=client,
            get_packet=receiver.get_packet,
            width=width,
            height=height,
            fps=fps,
            **kwargs,
        )

    def initialize_track(self) -> H264PacketVideoTrack:
        """Create the track, advertising the resolution the SPS reports."""
        depayloader = self.receiver.depayloader
        if depayloader.width and depayloader.height:
            self.width, self.height = depayloader.width, depayloader.height
        return super().initialize_track()

    async def wait_for_keyframe(
        self, stop_event: Optional[asyncio.Event] = None, poll_s: float = 0.2
    ) -> bool:
        """Wait for the receiver's first keyframe; ``False`` if stopped first."""
        while not self.receiver.keyframe_seen:
            if stop_event is not None and stop_event.is_set():
                return False
            await asyncio.sleep(poll_s)
        return True

    async def run_with_auto_reconnect(
        self,
        stop_event: Optional[asyncio.Event] = None,
        command_callback: Optional[Callable] = None,
        subscribe_to_commands: bool = True,
    ):
        """Receive, wait for a keyframe, then stream until ``stop_event``."""
        stop = stop_event or asyncio.Event()
        if self._owns_receiver:
            self.receiver.start()
        try:
            if not await self.wait_for_keyframe(stop):
                return
            depayloader = self.receiver.depayloader
            logger.info(
                "first H.264 keyframe received (%sx%s); starting WebRTC stream",
                depayloader.width or self.width,
                depayloader.height or self.height,
            )
            await super().run_with_auto_reconnect(
                stop_event=stop,
                command_callback=command_callback,
                subscribe_to_commands=subscribe_to_commands,
            )
        finally:
            if self._owns_receiver:
                self.receiver.stop()
