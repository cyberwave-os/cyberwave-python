"""Pre-encoded H.264 GStreamer source (``cyberwave.sensor.camera_gstreamer``)."""

from __future__ import annotations

import sys
import time

import pytest

# The streamer needs the optional [camera] extra (av).
cg = pytest.importorskip("cyberwave.sensor.camera_gstreamer", reason="needs cyberwave[camera] (av)")

SYNC_SEI_UUID = bytes.fromhex("f47ac9e13b2d4f8a9c1ed68f4a2b7c3d")
TEST_SOURCE = (
    "videotestsrc is-live=true "
    "! video/x-raw,width=320,height=240,framerate=15/1 "
    "! x264enc tune=zerolatency key-int-max=15"
)


def test_pipeline_tail_is_appended() -> None:
    description = cg.build_pipeline_description("rtspsrc location=rtsp://x ! rtph264depay ! ")

    assert description.startswith("rtspsrc location=rtsp://x ! rtph264depay ! h264parse")
    assert "stream-format=byte-stream,alignment=au" in description
    assert "cw_raw_appsink" not in description


def test_decode_branch_is_teed() -> None:
    description = cg.build_pipeline_description("videotestsrc ! x264enc", decode=True)

    assert "tee name=cw_tee" in description
    assert "cw_raw_appsink" in description


def test_empty_source_is_rejected() -> None:
    with pytest.raises(ValueError):
        cg.build_pipeline_description(" ! ")


@pytest.fixture
def gst():
    gi = pytest.importorskip("gi")
    try:
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
    except (ValueError, ImportError):
        pytest.skip("GStreamer introspection data not installed")
    Gst.init(None)
    for element in ("videotestsrc", "x264enc", "h264parse", "appsink", "decodebin"):
        if Gst.ElementFactory.find(element) is None:
            pytest.skip(f"GStreamer element {element} not installed")
    return Gst


def _drain(source: cg.GstH264Source, count: int, timeout: float = 10.0) -> list[tuple[bytes, bool]]:
    packets: list[tuple[bytes, bool]] = []
    deadline = time.monotonic() + timeout
    while len(packets) < count and time.monotonic() < deadline:
        packet = source.get_packet()
        if packet is None:
            time.sleep(0.01)
        else:
            packets.append(packet)
    return packets


def test_source_hands_out_access_units_from_a_keyframe(gst) -> None:
    decoded: list = []
    source = cg.GstH264Source(TEST_SOURCE, frame_callback=lambda f, n: decoded.append(f))
    source.start()
    try:
        packets = _drain(source, 5)
    finally:
        source.stop()

    assert len(packets) == 5
    assert packets[0][1] is True, "first access unit must be a keyframe"
    assert all(data.startswith(b"\x00\x00\x00\x01") or data.startswith(b"\x00\x00\x01") for data, _ in packets)
    assert (source.width, source.height, source.fps) == (320, 240, 15)
    assert decoded and decoded[0].shape == (240, 320, 3)


def test_source_writes_sync_sei_when_the_probe_is_installed(gst) -> None:
    sync = pytest.importorskip("cyberwave_video_sync")
    if not hasattr(sync, "attach_gst_probe"):
        pytest.skip("installed cyberwave_video_sync predates attach_gst_probe")
    source = cg.GstH264Source(TEST_SOURCE, track_id=lambda: None, require_sync=True)
    source.start()
    try:
        packets = _drain(source, 3)
    finally:
        source.stop()

    assert source.sync_attached
    assert all(data.count(SYNC_SEI_UUID) == 1 for data, _ in packets)


def test_require_sync_without_the_probe_refuses_to_start(gst, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cyberwave_video_sync", None)
    source = cg.GstH264Source(TEST_SOURCE, require_sync=True)

    with pytest.raises(cg.VideoSyncUnavailableError):
        source.start()
    source.stop()


def test_streamer_track_takes_the_pipeline_resolution(gst) -> None:
    pytest.importorskip("aiortc")

    class _Mqtt:
        topic_prefix = ""

    streamer = cg.GStreamerH264CameraStreamer(_Mqtt(), source=TEST_SOURCE, twin_uuid="t")
    streamer.source.start()
    try:
        assert _drain(streamer.source, 1)
        track = streamer.initialize_track()
    finally:
        streamer.source.stop()

    assert (track.width, track.height, track.fps) == (320, 240, 15)
    assert track.framing == "annexb"


def test_sync_sei_reaches_the_rtp_payloads(gst) -> None:
    """What aiortc puts on the wire for a pre-encoded track carries the SEI."""
    sync = pytest.importorskip("cyberwave_video_sync")
    if not hasattr(sync, "attach_gst_probe"):
        pytest.skip("installed cyberwave_video_sync predates attach_gst_probe")
    pytest.importorskip("aiortc")
    import asyncio

    from aiortc.codecs.h264 import H264Encoder

    from cyberwave.sensor.camera_h264 import H264PacketVideoTrack

    source = cg.GstH264Source(TEST_SOURCE, require_sync=True)
    track = H264PacketVideoTrack(source.get_packet, width=320, height=240, fps=15)
    source.track_id = lambda: track.id
    source.start()
    encoder = H264Encoder()

    async def _payloads(frames: int) -> list[bytes]:
        out: list[bytes] = []
        sent = 0
        deadline = time.monotonic() + 10.0
        while sent < frames and time.monotonic() < deadline:
            packet = await track.recv()
            if bytes(packet):
                payloads, _timestamp = encoder.pack(packet)
                out.append(b"".join(payloads))
                sent += 1
        return out

    try:
        frames = asyncio.run(_payloads(5))
    finally:
        source.stop()

    assert len(frames) == 5
    assert all(SYNC_SEI_UUID in frame for frame in frames)
