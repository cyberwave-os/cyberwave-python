"""End-to-end aiortc loopback test for the H.264 passthrough camera track.

``test_camera_h264.py`` covers :func:`to_annex_b` as a pure byte-transform in
isolation. It cannot catch the actual reported failure: aiortc's
``RTCRtpSender`` routes pre-encoded packets through ``H264Encoder.pack()``,
which splits NAL units on Annex-B start codes. AVCC (length-prefixed) access
units carry none, so ``pack()`` silently produced zero RTP payloads -- nothing
ever reached the receiver and the transport eventually closed.

This test wires :class:`H264PacketVideoTrack` into a real sender/receiver
``RTCPeerConnection`` pair (no backend, no media-service, no mocks) so
aiortc's own ``pack()``/RTP/jitter-buffer/decode path runs for real, and
asserts frames actually arrive decoded on the other end.
"""

from __future__ import annotations

import asyncio
from fractions import Fraction
from typing import Optional, Tuple

import numpy as np
import pytest

pytest.importorskip(
    "aiortc", reason="aiortc not installed (install with extras: camera)"
)

import av
from aiortc import RTCPeerConnection, RTCRtpSender

from cyberwave.sensor.camera_h264 import H264PacketVideoTrack

WIDTH, HEIGHT = 64, 48
FPS = 15
NUM_FRAMES = 8


def _encode_annex_b_access_units(num_frames: int) -> list[bytes]:
    """Encode real libx264 access units -- PyAV emits Annex-B (start-code
    framed), same as any raw encoder output before a muxer/depay reframes it.
    """
    cc = av.CodecContext.create("libx264", "w")
    cc.width = WIDTH
    cc.height = HEIGHT
    cc.pix_fmt = "yuv420p"
    cc.time_base = Fraction(1, FPS)
    cc.options = {"preset": "ultrafast", "tune": "zerolatency"}

    packets = []
    for i in range(num_frames):
        arr = np.full((HEIGHT, WIDTH, 3), (i * 30) % 256, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="bgr24").reformat(
            format="yuv420p"
        )
        frame.pts = i
        packets.extend(bytes(pkt) for pkt in cc.encode(frame))
    packets.extend(bytes(pkt) for pkt in cc.encode(None))
    return packets


def _annex_b_to_avcc(data: bytes) -> bytes:
    """Reframe an Annex-B access unit (one or more NALs) as AVCC: a 4-byte
    big-endian length prefix per NAL, no start codes -- the framing produced
    by e.g. Kinova's RTSP depay path, which is what triggered the bug.
    """
    nals = []
    i, n = 0, len(data)
    start: Optional[int] = None
    while i < n:
        if data[i : i + 4] == b"\x00\x00\x00\x01":
            if start is not None:
                nals.append(data[start:i])
            i += 4
            start = i
        elif data[i : i + 3] == b"\x00\x00\x01":
            if start is not None:
                nals.append(data[start:i])
            i += 3
            start = i
        else:
            i += 1
    if start is not None:
        nals.append(data[start:n])
    return b"".join(len(nal).to_bytes(4, "big") + nal for nal in nals)


class _QueuePacketSource:
    """``get_packet`` callable that yields queued access units in order,
    holding the last one once exhausted (mirrors a live upstream encoder that
    keeps producing frames after this test's fixed sample runs out).
    """

    def __init__(self, access_units: list[bytes]) -> None:
        self._units = access_units
        self._idx = 0

    def __call__(self) -> Optional[Tuple[bytes, bool]]:
        if not self._units:
            return None
        unit = self._units[min(self._idx, len(self._units) - 1)]
        self._idx += 1
        return unit, False


async def _run_loopback(access_units: list[bytes]) -> Tuple[str, int]:
    """Send `access_units` over a real aiortc sender/receiver pair.

    Returns (final connectionState of the sender pc, number of frames decoded
    by the receiver).
    """
    pc_sender = RTCPeerConnection()
    pc_receiver = RTCPeerConnection()

    track = H264PacketVideoTrack(
        _QueuePacketSource(access_units), width=WIDTH, height=HEIGHT, fps=FPS
    )
    transceiver = pc_sender.addTransceiver(track, direction="sendonly")
    # H264Encoder.pack() (the code path under test) only runs for the H264
    # codec; without pinning it, aiortc may negotiate VP8/VP9 first and this
    # test would pass for the wrong reason. Mirrors the SFU-forced H264 path
    # base_video.py's _filter_sdp() achieves via SDP surgery in production.
    h264_codecs = [
        c
        for c in RTCRtpSender.getCapabilities("video").codecs
        if c.mimeType == "video/H264"
    ]
    transceiver.setCodecPreferences(h264_codecs)

    track_ready: "asyncio.Future" = asyncio.get_event_loop().create_future()

    @pc_receiver.on("track")
    def on_track(remote_track):
        if not track_ready.done():
            track_ready.set_result(remote_track)

    try:
        await pc_sender.setLocalDescription(await pc_sender.createOffer())
        await pc_receiver.setRemoteDescription(pc_sender.localDescription)
        await pc_receiver.setLocalDescription(await pc_receiver.createAnswer())
        await pc_sender.setRemoteDescription(pc_receiver.localDescription)

        remote_track = await asyncio.wait_for(track_ready, timeout=10)

        received_frames = []
        try:
            for _ in range(len(access_units)):
                frame = await asyncio.wait_for(remote_track.recv(), timeout=10)
                received_frames.append(frame)
        except asyncio.TimeoutError:
            pass

        return pc_sender.connectionState, len(received_frames)
    finally:
        await pc_sender.close()
        await pc_receiver.close()


async def test_avcc_passthrough_survives_real_rtp_round_trip() -> None:
    """The regression this fix addresses: AVCC-framed access units used to
    pack into zero RTP payloads, so no frame ever reached the receiver.
    """
    annex_b_units = _encode_annex_b_access_units(NUM_FRAMES)
    avcc_units = [_annex_b_to_avcc(u) for u in annex_b_units]
    assert all(u[:3] != b"\x00\x00\x01" for u in avcc_units), (
        "test fixture must be AVCC-framed, not Annex-B"
    )

    state, num_received = await _run_loopback(avcc_units)

    assert state == "connected", (
        f"sender connection should stay connected, got {state!r}"
    )
    assert num_received > 0, (
        "receiver decoded zero frames -- AVCC framing broke aiortc's RTP "
        "packetization (the exact failure this fix addresses)"
    )


async def test_already_annex_b_passthrough_still_works() -> None:
    """Control: already-Annex-B input (e.g. some GStreamer h264parse
    configs) must keep working -- to_annex_b() is a no-op passthrough there.
    """
    annex_b_units = _encode_annex_b_access_units(NUM_FRAMES)

    state, num_received = await _run_loopback(annex_b_units)

    assert state == "connected"
    assert num_received > 0
