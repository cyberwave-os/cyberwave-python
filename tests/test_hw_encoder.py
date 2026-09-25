"""Tests for the hardware H264 encoder selection layer (hw_encoder.py).

Pins: per-encoder option maps (libx264 must stay byte-identical to aiortc
upstream so software selection is a behavioral no-op), the real-test-encode
probe semantics (presence is not enough — the context must produce packets),
and the env-driven selection/caching contract.
"""

from __future__ import annotations

import hashlib
import inspect

import pytest

pytest.importorskip("av", reason="pyav not installed")
pytest.importorskip("aiortc", reason="aiortc not installed")

import av  # noqa: E402
from aiortc.codecs.h264 import H264Encoder  # noqa: E402

from cyberwave.sensor import hw_encoder  # noqa: E402

# sha256 of aiortc H264Encoder._encode_frame's source. HardwareH264Encoder
# copies that method's reset/keyframe/encode structure; if upstream changes
# it, this guard fails so the override is re-reviewed for divergence (bump
# the hash after re-syncing).
_UPSTREAM_ENCODE_FRAME_SHA256 = (
    "d93057b16ce7b32d317cf7aad92cae43d72b842be8e84a25fd6035e733117559"
)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CYBERWAVE_VIDEO_ENCODER", raising=False)
    monkeypatch.delenv("CYBERWAVE_KEYFRAME_INTERVAL", raising=False)
    hw_encoder.reset_selection_cache()
    yield
    hw_encoder.reset_selection_cache()


# ---------------------------------------------------------------- options


def test_libx264_options_match_aiortc_upstream() -> None:
    # aiortc 1.14 H264Encoder._encode_frame options — software selection
    # must be a behavioral no-op.
    assert hw_encoder.encoder_options("libx264") == {
        "level": "31",
        "tune": "zerolatency",
    }


def test_hw_encoders_get_explicit_gop() -> None:
    for name in ("h264_v4l2m2m", "h264_nvmpi", "h264_nvenc"):
        assert hw_encoder.encoder_options(name)["g"] == "60"


def test_gop_respects_keyframe_interval_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CYBERWAVE_KEYFRAME_INTERVAL", "30")
    assert hw_encoder.encoder_options("h264_v4l2m2m")["g"] == "30"


def test_upstream_encode_frame_unchanged() -> None:
    # HardwareH264Encoder._encode_frame copies aiortc's reset/keyframe/encode
    # structure. If upstream changes, re-review the override for divergence,
    # then bump _UPSTREAM_ENCODE_FRAME_SHA256.
    src = inspect.getsource(H264Encoder._encode_frame)
    digest = hashlib.sha256(src.encode()).hexdigest()
    assert digest == _UPSTREAM_ENCODE_FRAME_SHA256, (
        f"aiortc H264Encoder._encode_frame changed (got {digest}). Re-review "
        "HardwareH264Encoder._encode_frame for divergence, then update "
        "_UPSTREAM_ENCODE_FRAME_SHA256."
    )


# ---------------------------------------------------------------- probe


def test_probe_unknown_codec_returns_false() -> None:
    assert hw_encoder.probe_encoder("h264_does_not_exist") is False


def test_probe_libx264_succeeds() -> None:
    # libx264 is always present in the av wheel; the probe must really
    # encode and see packets.
    assert hw_encoder.probe_encoder("libx264") is True


def test_probe_rejects_encoder_with_empty_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SilentCtx:
        def open(self) -> None: ...
        def encode(self, frame=None):  # noqa: ANN001
            return []

    monkeypatch.setattr(
        hw_encoder, "_build_codec_context", lambda *a, **k: _SilentCtx()
    )
    assert hw_encoder.probe_encoder("h264_silent") is False


def test_probe_rejects_encoder_that_raises_on_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ExplodingCtx:
        def open(self) -> None: ...
        def encode(self, frame=None):  # noqa: ANN001
            raise av.FFmpegError(22, "boom")

    monkeypatch.setattr(
        hw_encoder, "_build_codec_context", lambda *a, **k: _ExplodingCtx()
    )
    assert hw_encoder.probe_encoder("h264_exploding") is False


# ---------------------------------------------------------------- selection


def _probe_map(monkeypatch: pytest.MonkeyPatch, results: dict[str, bool]) -> list[str]:
    """Fake probe_encoder from a name->bool map; records call order."""
    calls: list[str] = []

    def fake_probe(name: str) -> bool:
        calls.append(name)
        return results.get(name, False)

    monkeypatch.setattr(hw_encoder, "probe_encoder", fake_probe)
    return calls


def test_software_pin_skips_probing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _probe_map(monkeypatch, {})
    for pin in ("libx264", "software", "SOFTWARE"):
        hw_encoder.reset_selection_cache()
        monkeypatch.setenv("CYBERWAVE_VIDEO_ENCODER", pin)
        sel = hw_encoder.select_h264_encoder()
        assert sel == hw_encoder.EncoderSelection("libx264", "software")
    assert calls == []


def test_explicit_pin_is_probed_and_honored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _probe_map(monkeypatch, {"h264_v4l2m2m": True})
    monkeypatch.setenv("CYBERWAVE_VIDEO_ENCODER", "h264_v4l2m2m")
    sel = hw_encoder.select_h264_encoder()
    assert sel == hw_encoder.EncoderSelection("h264_v4l2m2m", "pinned")


def test_failed_pin_warns_and_falls_back(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _probe_map(monkeypatch, {})
    monkeypatch.setenv("CYBERWAVE_VIDEO_ENCODER", "h264_nvmpi")
    with caplog.at_level("WARNING"):
        sel = hw_encoder.select_h264_encoder()
    assert sel == hw_encoder.EncoderSelection("libx264", "fallback")
    assert any("h264_nvmpi" in r.message for r in caplog.records)


def test_auto_takes_first_probing_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _probe_map(monkeypatch, {"h264_qsv": True, "h264_v4l2m2m": True})
    sel = hw_encoder.select_h264_encoder()
    assert sel == hw_encoder.EncoderSelection("h264_qsv", "probed")
    # order respected, stops at first success, never probes libx264
    assert calls == ["h264_nvenc", "h264_nvmpi", "h264_qsv"]


def test_auto_all_fail_falls_back_to_software(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _probe_map(monkeypatch, {})
    sel = hw_encoder.select_h264_encoder()
    assert sel == hw_encoder.EncoderSelection("libx264", "fallback")
    assert "libx264" not in calls


def test_selection_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _probe_map(monkeypatch, {"h264_v4l2m2m": True})
    first = hw_encoder.select_h264_encoder()
    second = hw_encoder.select_h264_encoder()
    assert first is second
    assert calls == ["h264_nvenc", "h264_nvmpi", "h264_qsv", "h264_v4l2m2m"]


# ------------------------------------------------------- HardwareH264Encoder


def _frames(count: int, width: int = 320, height: int = 240):
    return [hw_encoder._black_frame(width, height, i) for i in range(count)]


def _nal_types(payloads: list[bytes]) -> list[int]:
    return [p[0] & 0x1F for p in payloads if p]


def test_software_encoder_produces_packets_and_initial_idr() -> None:
    enc = hw_encoder.HardwareH264Encoder(codec_name="libx264")
    payloads = list(enc._encode_frame(_frames(1)[0], force_keyframe=False))
    assert payloads
    assert 5 in _nal_types(payloads)  # first frame is an IDR


def test_forced_keyframe_yields_idr_mid_stream() -> None:
    enc = hw_encoder.HardwareH264Encoder(codec_name="libx264")
    frames = _frames(6)
    for f in frames[:5]:
        list(enc._encode_frame(f, force_keyframe=False))
    payloads = list(enc._encode_frame(frames[5], force_keyframe=True))
    assert 5 in _nal_types(payloads)


def test_bitrate_change_rebuilds_context() -> None:
    enc = hw_encoder.HardwareH264Encoder(codec_name="libx264")
    list(enc._encode_frame(_frames(1)[0], force_keyframe=False))
    first_ctx = enc.codec
    enc.target_bitrate = 2_000_000  # >10% change vs 1 Mbps default
    list(enc._encode_frame(hw_encoder._black_frame(320, 240, 1), False))
    assert enc.codec is not first_ctx
    assert enc.codec.bit_rate == 2_000_000


def test_zero_bitrate_readback_does_not_crash_on_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A hardware wrapper may report bit_rate == 0 after open; the reset check
    # must not ZeroDivisionError on the next frame.
    class _ZeroBitrateCtx:
        width = 320
        height = 240
        bit_rate = 0

        def encode(self, frame=None):  # noqa: ANN001
            return []

    monkeypatch.setattr(
        hw_encoder, "_build_codec_context", lambda *a, **k: _ZeroBitrateCtx()
    )
    enc = hw_encoder.HardwareH264Encoder(codec_name="h264_v4l2m2m")
    list(enc._encode_frame(_frames(1)[0], force_keyframe=True))
    # Second frame hits the reset branch, which reads codec.bit_rate (== 0).
    list(enc._encode_frame(hw_encoder._black_frame(320, 240, 1), force_keyframe=False))


def test_hw_failure_mid_stream_falls_back_to_libx264(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class _ExplodingCtx:
        width = 320
        height = 240
        bit_rate = 1_000_000

        def encode(self, frame=None):  # noqa: ANN001
            raise av.FFmpegError(22, "hw died")

    real_build = hw_encoder._build_codec_context

    def fake_build(name: str, w: int, h: int, br: int):
        if name == "h264_fake_hw":
            return _ExplodingCtx()
        return real_build(name, w, h, br)

    monkeypatch.setattr(hw_encoder, "_build_codec_context", fake_build)
    enc = hw_encoder.HardwareH264Encoder(codec_name="h264_fake_hw")
    with caplog.at_level("ERROR"):
        payloads = list(enc._encode_frame(_frames(1)[0], force_keyframe=False))
    assert enc._codec_name == "libx264"  # permanent fallback
    assert payloads  # the stream survived
    assert any("h264_fake_hw" in r.message for r in caplog.records)


def test_default_codec_name_comes_from_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hw_encoder,
        "select_h264_encoder",
        lambda refresh=False: hw_encoder.EncoderSelection("h264_v4l2m2m", "probed"),
    )
    enc = hw_encoder.HardwareH264Encoder()
    assert enc._codec_name == "h264_v4l2m2m"


# ------------------------------------------- SPS/PPS re-injection (CYB-2835)
#
# Hardware encoders (v4l2m2m on Pi, nvmpi on Jetson) emit SPS/PPS once at
# stream start and don't repeat them before later IDRs, so WebRTC consumers
# that join mid-stream never get decoder config and render black. The encoder
# must cache SPS/PPS and re-inject them before any bare IDR.

# Minimal NAL units: first byte's low 5 bits carry the type.
_SPS = bytes([0x67, 0x42, 0x00, 0x1F])  # type 7
_PPS = bytes([0x68, 0xCE, 0x3C, 0x80])  # type 8
_IDR = bytes([0x65, 0x88, 0x84, 0x00])  # type 5
_NON_IDR = bytes([0x41, 0x9A, 0x00])  # type 1


def _fresh_encoder() -> "hw_encoder.HardwareH264Encoder":
    return hw_encoder.HardwareH264Encoder(codec_name="h264_v4l2m2m")


def test_first_keyframe_with_params_is_cached_not_modified() -> None:
    enc = _fresh_encoder()
    out = enc._maybe_inject_parameter_sets([_SPS, _PPS, _IDR])
    assert out == [_SPS, _PPS, _IDR]
    assert enc._sps == _SPS and enc._pps == _PPS


def test_bare_idr_gets_params_prepended_after_caching() -> None:
    enc = _fresh_encoder()
    enc._maybe_inject_parameter_sets([_SPS, _PPS, _IDR])  # cache
    out = enc._maybe_inject_parameter_sets([_IDR])
    assert out == [_SPS, _PPS, _IDR]


def test_non_idr_frames_are_untouched() -> None:
    enc = _fresh_encoder()
    enc._maybe_inject_parameter_sets([_SPS, _PPS, _IDR])  # cache
    out = enc._maybe_inject_parameter_sets([_NON_IDR])
    assert out == [_NON_IDR]


def test_bare_idr_without_cached_params_passes_through() -> None:
    # No SPS/PPS seen yet — must not crash, just emit the IDR as-is.
    enc = _fresh_encoder()
    out = enc._maybe_inject_parameter_sets([_IDR])
    assert out == [_IDR]


def test_idr_with_only_sps_still_gets_both_reinjected() -> None:
    enc = _fresh_encoder()
    enc._maybe_inject_parameter_sets([_SPS, _PPS, _IDR])  # cache both
    # A later AU that somehow carries SPS but not PPS is still incomplete.
    out = enc._maybe_inject_parameter_sets([_SPS, _IDR])
    assert out[-3:] == [_SPS, _PPS, _IDR]


def test_encode_frame_reinjects_params_on_bare_idr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate a hw encoder: first frame yields SPS+PPS+IDR, later forced
    # keyframes yield a bare IDR. Every emitted keyframe AU must carry params.
    enc = hw_encoder.HardwareH264Encoder(codec_name="h264_v4l2m2m")

    class _FakeCtx:
        width = 320
        height = 240
        bit_rate = 1_000_000

        def __init__(self) -> None:
            self._n = 0

        def encode(self, frame=None):  # noqa: ANN001
            self._n += 1
            sc = b"\x00\x00\x00\x01"
            if self._n == 1:
                return [_Pkt(sc + _SPS + sc + _PPS + sc + _IDR)]
            return [_Pkt(sc + _IDR)]

    class _Pkt:
        def __init__(self, data: bytes) -> None:
            self._d = data

        def __bytes__(self) -> bytes:
            return self._d

    monkeypatch.setattr(hw_encoder, "_build_codec_context", lambda *a, **k: _FakeCtx())

    first = list(enc._encode_frame(_frames(1)[0], force_keyframe=True))
    assert _nal_types(first) == [7, 8, 5]

    second = list(enc._encode_frame(_frames(1)[0], force_keyframe=True))
    assert _nal_types(second) == [7, 8, 5]  # params re-injected before bare IDR
