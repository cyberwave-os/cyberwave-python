"""Opt-in hardware integration tests for the H264 encoder selection layer.

These run only where real encode hardware is reachable:
- Raspberry Pi 4: /dev/video11 (V4L2 M2M encoder)
- Jetson image variant: av built with h264_nvmpi
They are part of the parent-level edge test script, not CI.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("av", reason="pyav not installed")
pytest.importorskip("aiortc", reason="aiortc not installed")

from cyberwave.sensor import hw_encoder  # noqa: E402


def _codec_present(name: str) -> bool:
    try:
        from av.codec import Codec

        Codec(name, "w")
        return True
    except Exception:
        return False


requires_pi_encoder = pytest.mark.skipif(
    not os.path.exists("/dev/video11"),
    reason="no V4L2 M2M encoder device (/dev/video11)",
)
requires_nvmpi = pytest.mark.skipif(
    not _codec_present("h264_nvmpi"),
    reason="av not built with h264_nvmpi (jetson image variant only)",
)


def _stream_and_force_idr(codec_name: str) -> None:
    enc = hw_encoder.HardwareH264Encoder(codec_name=codec_name)
    got_packets = False
    saw_idr_with_params = False
    for i in range(30):
        payloads = list(
            enc._encode_frame(
                hw_encoder._black_frame(640, 480, i), force_keyframe=(i == 15)
            )
        )
        got_packets = got_packets or bool(payloads)
        types = [p[0] & 0x1F for p in payloads if p]
        # Every IDR access unit must carry SPS (7) + PPS (8) so a mid-stream
        # WebRTC consumer can initialise its decoder — this is the CYB-2835 fix.
        if 5 in types:
            assert 7 in types and 8 in types, (
                f"{codec_name}: IDR emitted without SPS/PPS ({types}) — "
                "mid-stream consumers would render black"
            )
            saw_idr_with_params = True
    assert got_packets, f"{codec_name}: no packets produced"
    assert saw_idr_with_params, f"{codec_name}: no self-contained keyframe seen"
    assert enc._codec_name == codec_name, "encoder silently fell back to software"


@requires_pi_encoder
def test_v4l2m2m_probe_passes() -> None:
    assert hw_encoder.probe_encoder("h264_v4l2m2m") is True


@requires_pi_encoder
def test_v4l2m2m_streams_and_honors_forced_keyframe() -> None:
    _stream_and_force_idr("h264_v4l2m2m")


@requires_nvmpi
def test_nvmpi_probe_passes() -> None:
    assert hw_encoder.probe_encoder("h264_nvmpi") is True


@requires_nvmpi
def test_nvmpi_streams_and_honors_forced_keyframe() -> None:
    _stream_and_force_idr("h264_nvmpi")
