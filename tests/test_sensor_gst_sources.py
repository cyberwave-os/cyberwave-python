"""Attach to existing H.264 streams and push H.264 in-process, without decoding.

The URL mapping is pure; the rest runs real GStreamer pipelines (skipped when
PyGObject or the plugins are missing): a sender encodes once with x264enc, and
the Cyberwave source must hand out that H.264 untouched plus the sync SEI.
"""

from __future__ import annotations

import socket
import struct
import threading
import time

import pytest

from cyberwave.sensor.gst_sources import h264_source_for_url, redact_credentials

try:
    from cyberwave.sensor import camera_gstreamer as cg
except ImportError:  # the streamer needs the optional [camera] extra (av)
    cg = None  # type: ignore[assignment]

needs_streamer = pytest.mark.skipif(cg is None, reason="needs the cyberwave[camera] extra (av)")

SYNC_SEI_UUID = bytes.fromhex("f47ac9e13b2d4f8a9c1ed68f4a2b7c3d")
ENCODER = (
    "videotestsrc is-live=true ! video/x-raw,width=320,height=240,framerate=15/1 "
    "! x264enc tune=zerolatency key-int-max=15"
)


class TestUrlMapping:
    def test_rtsp_is_depayloaded_over_tcp(self) -> None:
        fragment = h264_source_for_url("rtsp://admin:secret@10.0.0.9:554/ch1")

        assert fragment.startswith('rtspsrc location="rtsp://admin:secret@10.0.0.9:554/ch1"')
        assert "protocols=tcp ! rtph264depay" in fragment

    def test_rtp_listens_with_h264_caps(self) -> None:
        fragment = h264_source_for_url("rtp://0.0.0.0:5600")

        assert "udpsrc address=0.0.0.0 port=5600" in fragment
        assert "encoding-name=H264" in fragment
        assert fragment.endswith("rtph264depay")

    def test_mpegts_over_udp_and_tcp_is_demuxed(self) -> None:
        assert h264_source_for_url("udp://239.1.1.1:5000").endswith("! tsdemux")
        assert h264_source_for_url("tcp://cam.local:8000").startswith(
            "tcpclientsrc host=cam.local port=8000"
        )

    @pytest.mark.parametrize(
        ("url", "match"),
        [("http://cam/x", "Unsupported"), ("rtp://0.0.0.0", "port"), ("tcp://:8000", "host")],
    )
    def test_bad_urls(self, url: str, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            h264_source_for_url(url)

    def test_credentials_never_reach_the_logs(self) -> None:
        text = redact_credentials('rtspsrc location="rtsp://admin:s3cr3t@10.0.0.9/ch1"')

        assert "s3cr3t" not in text and "rtsp://***@10.0.0.9/ch1" in text

    @pytest.mark.parametrize("password", ["p@ss", "pa/ss", "p@s/s:w"])
    def test_passwords_with_at_or_slash_are_fully_masked(self, password: str) -> None:
        text = redact_credentials(f'rtspsrc location="rtsp://admin:{password}@cam.lan:554/s" latency=50')

        assert password not in text
        assert 'location="rtsp://***@cam.lan:554/s" latency=50' in text

    @needs_streamer
    def test_push_source_expands_to_appsrc(self) -> None:
        description = cg.build_pipeline_description("appsrc")

        assert description.startswith("appsrc name=cw_appsrc is-live=true format=time")
        assert cg.GstH264Source("appsrc").accepts_push
        assert not cg.GstH264Source("videotestsrc ! x264enc").accepts_push


@pytest.fixture
def gst():
    if cg is None:
        pytest.skip("needs the cyberwave[camera] extra (av)")
    gi = pytest.importorskip("gi")
    try:
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
    except (ValueError, ImportError):
        pytest.skip("GStreamer introspection data not installed")
    Gst.init(None)
    for element in (
        "videotestsrc", "x264enc", "h264parse", "appsrc", "appsink", "rtph264pay",
        "rtph264depay", "udpsink", "udpsrc", "mpegtsmux", "tsdemux", "tcpserversink",
    ):
        if Gst.ElementFactory.find(element) is None:
            pytest.skip(f"GStreamer element {element} not installed")
    return Gst


@pytest.fixture
def sync_probe():
    sync = pytest.importorskip("cyberwave_video_sync")
    if not hasattr(sync, "attach_gst_probe"):
        pytest.skip("installed cyberwave_video_sync predates attach_gst_probe")
    return sync


def _free_port(kind: int) -> int:
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Sender:
    """A separate pipeline standing in for the camera on the network."""

    def __init__(self, gst, description: str) -> None:
        self.pipeline = gst.parse_launch(description)
        self.gst = gst

    def __enter__(self) -> _Sender:  # noqa: PYI034
        self.pipeline.set_state(self.gst.State.PLAYING)
        return self

    def __exit__(self, *exc) -> None:
        self.pipeline.set_state(self.gst.State.NULL)


def _drain(source: cg.GstH264Source, count: int, timeout: float = 10.0) -> list[bytes]:
    out: list[bytes] = []
    deadline = time.monotonic() + timeout
    while len(out) < count and time.monotonic() < deadline:
        packet = source.get_packet()
        if packet is None:
            time.sleep(0.01)
        else:
            out.append(packet[0])
    return out


def _sei_wall_time(access_unit: bytes) -> float:
    """capture_wall_time from the Cyberwave SEI payload, in seconds."""
    start = access_unit.index(SYNC_SEI_UUID) + len(SYNC_SEI_UUID)
    raw, zeros = bytearray(), 0
    for byte in access_unit[start : start + 120]:
        if zeros >= 2 and byte == 3:  # emulation prevention byte
            zeros = 0
            continue
        raw.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    (wall_ns,) = struct.unpack(">Q", bytes(raw[26:34]))
    return wall_ns / 1e9


def _assert_muxed_with_sei(aus: list[bytes], count: int) -> None:
    assert len(aus) == count, f"expected {count} access units, got {len(aus)}"
    assert all(au.count(SYNC_SEI_UUID) == 1 for au in aus)


def test_rtp_over_udp_is_muxed_without_decoding(gst, sync_probe) -> None:
    port = _free_port(socket.SOCK_DGRAM)
    source = cg.GstH264Source(h264_source_for_url(f"rtp://127.0.0.1:{port}"), require_sync=True)
    source.start()
    try:
        with _Sender(gst, f"{ENCODER} ! rtph264pay config-interval=-1 pt=96 ! udpsink host=127.0.0.1 port={port}"):
            aus = _drain(source, 5)
    finally:
        source.stop()

    _assert_muxed_with_sei(aus, 5)
    assert (source.width, source.height) == (320, 240)


def test_mpegts_over_udp_is_muxed_without_decoding(gst, sync_probe) -> None:
    port = _free_port(socket.SOCK_DGRAM)
    source = cg.GstH264Source(h264_source_for_url(f"udp://127.0.0.1:{port}"), require_sync=True)
    source.start()
    try:
        with _Sender(gst, f"{ENCODER} ! h264parse ! mpegtsmux alignment=7 ! udpsink host=127.0.0.1 port={port}"):
            aus = _drain(source, 5)
    finally:
        source.stop()

    _assert_muxed_with_sei(aus, 5)


def test_mpegts_over_tcp_is_muxed_without_decoding(gst, sync_probe) -> None:
    port = _free_port(socket.SOCK_STREAM)
    with _Sender(gst, f"{ENCODER} ! h264parse ! mpegtsmux ! tcpserversink host=127.0.0.1 port={port}"):
        time.sleep(0.3)  # server socket up before the client connects
        source = cg.GstH264Source(h264_source_for_url(f"tcp://127.0.0.1:{port}"), require_sync=True)
        source.start()
        try:
            aus = _drain(source, 5)
        finally:
            source.stop()

    _assert_muxed_with_sei(aus, 5)


def test_pushed_h264_keeps_the_callers_capture_time(gst, sync_probe) -> None:
    """The Go2 case: H.264 already decoded nowhere, handed over in-process."""
    source = cg.GstH264Source("appsrc", require_sync=True)
    assert source.push(b"\x00\x00\x00\x01\x65") is False, "push before start must drop"
    source.start()

    # The robot's encoder: a separate pipeline whose access units we push.
    producer = gst.parse_launch(
        f"{ENCODER} ! h264parse config-interval=-1 "
        "! video/x-h264,stream-format=byte-stream,alignment=au ! appsink name=out sync=false"
    )
    sink = producer.get_by_name("out")
    producer.set_state(gst.State.PLAYING)
    capture_lag_s = 0.4
    stop = threading.Event()

    def _robot_receive_loop() -> None:
        while not stop.is_set():
            sample = sink.emit("try-pull-sample", 200 * gst.MSECOND)
            if sample is None:
                continue
            buf = sample.get_buffer()
            data = buf.extract_dup(0, buf.get_size())
            # Frames arrive late; the robot says when they were captured.
            source.push(data, capture_time=time.time() - capture_lag_s)

    pusher = threading.Thread(target=_robot_receive_loop, daemon=True)
    pusher.start()
    try:
        aus = _drain(source, 5)
    finally:
        stop.set()
        pusher.join(timeout=2)
        producer.set_state(gst.State.NULL)
        source.stop()

    _assert_muxed_with_sei(aus, 5)
    for au in aus:
        age = time.time() - _sei_wall_time(au)
        assert capture_lag_s - 0.05 < age < capture_lag_s + 2.0, age
