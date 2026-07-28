"""Hardware-accelerated H264 encoder selection for aiortc streaming.

aiortc 1.14 hardcodes ``av.CodecContext.create("libx264", "w")`` in
``H264Encoder._encode_frame`` — every WebRTC frame is software-encoded.
This module probes for usable hardware H264 encoders at runtime (a real
2-frame test-encode, not a presence check: ``h264_v4l2m2m`` exists on every
Linux av wheel but only opens on a real V4L2 encoder device) and provides
``HardwareH264Encoder`` + ``apply_h264_hw_patch()`` so streamers transparently
use hardware where available and libx264 everywhere else.

See docs/superpowers/specs/2026-07-21-aiortc-hw-h264-encoding-design.md.
"""

from __future__ import annotations

import fractions
import logging
import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional, cast

import av
from aiortc.codecs.h264 import H264Encoder

logger = logging.getLogger(__name__)

ENV_VAR = "CYBERWAVE_VIDEO_ENCODER"
SOFTWARE_ENCODER = "libx264"
_SOFTWARE_ALIASES = frozenset({SOFTWARE_ENCODER, "software"})

# Probed in order; first candidate that passes a real test-encode wins.
# Missing codecs (not compiled into the linked FFmpeg) fail the probe in
# microseconds at codec lookup, so one list serves every platform.
AUTO_CANDIDATES = (
    "h264_nvenc",  # desktop NVIDIA dGPU (needs custom av build — future)
    "h264_nvmpi",  # Jetson Orin via jetson-ffmpeg (jetson image variant)
    "h264_qsv",  # Intel QuickSync (needs custom av build — future)
    "h264_v4l2m2m",  # Raspberry Pi 4 — present in the stock Linux av wheel
    "h264_videotoolbox",  # macOS dev machines — present in the macOS wheel
    SOFTWARE_ENCODER,
)

# Mirror aiortc's encoder defaults so hardware contexts behave like upstream.
MAX_FRAME_RATE = 30
_PROBE_WIDTH = 320
_PROBE_HEIGHT = 240
_PROBE_FRAMES = 2
_PROBE_BITRATE = 500_000
_DEFAULT_GOP = 60  # 2s at 30fps; overridden by CYBERWAVE_KEYFRAME_INTERVAL


def _default_gop() -> int:
    env_value = os.environ.get("CYBERWAVE_KEYFRAME_INTERVAL")
    if env_value:
        try:
            interval = int(env_value)
            if interval > 0:
                return interval
        except ValueError:
            pass
    return _DEFAULT_GOP


def encoder_options(codec_name: str) -> Dict[str, str]:
    """Per-encoder CodecContext options.

    libx264 must stay byte-identical to aiortc upstream (software selection
    is a behavioral no-op). Hardware encoders get an explicit GOP because
    their forced-keyframe handling varies; a bounded GOP caps PLI recovery
    latency either way.
    """
    if codec_name == SOFTWARE_ENCODER:
        return {"level": "31", "tune": "zerolatency"}
    gop = str(_default_gop())
    if codec_name == "h264_nvenc":
        return {"preset": "p1", "tune": "ull", "delay": "0", "g": gop}
    if codec_name == "h264_videotoolbox":
        return {"realtime": "1", "g": gop}
    return {"g": gop}


def _build_codec_context(
    codec_name: str, width: int, height: int, bitrate: int
) -> "av.VideoCodecContext":
    # av.CodecContext.create is typed as CodecContext; the write-mode H264
    # context is really a VideoCodecContext with width/height/pix_fmt/etc.
    ctx = cast(
        "av.video.codeccontext.VideoCodecContext",
        av.CodecContext.create(codec_name, "w"),
    )
    ctx.width = width
    ctx.height = height
    ctx.bit_rate = bitrate
    ctx.pix_fmt = "yuv420p"
    ctx.framerate = fractions.Fraction(MAX_FRAME_RATE, 1)
    ctx.time_base = fractions.Fraction(1, MAX_FRAME_RATE)
    ctx.options = encoder_options(codec_name)
    if codec_name == SOFTWARE_ENCODER:
        ctx.profile = "Baseline"
    return ctx


def _black_frame(width: int, height: int, pts: int) -> av.VideoFrame:
    frame = av.VideoFrame(width, height, "yuv420p")
    for plane in frame.planes:
        plane.update(bytes(plane.buffer_size))
    frame.pts = pts
    frame.time_base = fractions.Fraction(1, MAX_FRAME_RATE)
    return frame


def probe_encoder(codec_name: str) -> bool:
    """True iff ``codec_name`` really encodes: open a context, encode two
    synthetic frames, flush, and require at least one non-empty packet."""
    try:
        ctx = _build_codec_context(
            codec_name, _PROBE_WIDTH, _PROBE_HEIGHT, _PROBE_BITRATE
        )
        ctx.open()
        packets = []
        for i in range(_PROBE_FRAMES):
            packets.extend(ctx.encode(_black_frame(_PROBE_WIDTH, _PROBE_HEIGHT, i)))
        packets.extend(ctx.encode(None))
        return any(len(bytes(p)) > 0 for p in packets)
    except Exception as exc:
        logger.debug("H264 encoder probe failed for %s: %s", codec_name, exc)
        return False


@dataclass(frozen=True)
class EncoderSelection:
    """Outcome of encoder selection.

    reason: "pinned" (env pin passed probe), "probed" (auto probe winner),
    "fallback" (pin or all candidates failed → libx264),
    "software" (explicitly pinned to libx264/software — probing skipped).
    """

    codec_name: str
    reason: str


_selection_lock = threading.Lock()
_selection: Optional[EncoderSelection] = None


def _select_uncached() -> EncoderSelection:
    pin = os.environ.get(ENV_VAR, "").strip().lower()
    if pin in _SOFTWARE_ALIASES:
        return EncoderSelection(SOFTWARE_ENCODER, "software")
    if pin and pin != "auto":
        if probe_encoder(pin):
            return EncoderSelection(pin, "pinned")
        logger.warning(
            "%s=%s failed the encode probe; falling back to %s",
            ENV_VAR,
            pin,
            SOFTWARE_ENCODER,
        )
        return EncoderSelection(SOFTWARE_ENCODER, "fallback")
    for candidate in AUTO_CANDIDATES:
        if candidate == SOFTWARE_ENCODER:
            break
        if probe_encoder(candidate):
            return EncoderSelection(candidate, "probed")
    return EncoderSelection(SOFTWARE_ENCODER, "fallback")


def select_h264_encoder(refresh: bool = False) -> EncoderSelection:
    """Process-wide cached H264 encoder selection (thread-safe)."""
    global _selection
    with _selection_lock:
        if _selection is None or refresh:
            _selection = _select_uncached()
            logger.info(
                "H264 encoder selected: %s (reason=%s)",
                _selection.codec_name,
                _selection.reason,
            )
        return _selection


def reset_selection_cache() -> None:
    """Testing hook: drop the cached selection."""
    global _selection
    with _selection_lock:
        _selection = None


class HardwareH264Encoder(H264Encoder):
    """aiortc H264Encoder that builds its codec context from the runtime
    encoder selection.

    Only codec-context creation differs from upstream: NAL packetization,
    ``target_bitrate`` adaptation (REMB) and keyframe forcing are inherited.
    If the hardware context fails mid-stream the encoder logs once and
    permanently rebuilds as libx264 — a dying encoder degrades, never kills
    the track.
    """

    def __init__(self, codec_name: Optional[str] = None) -> None:
        super().__init__()
        self._codec_name = codec_name or select_h264_encoder().codec_name
        # Cached parameter sets for re-injection. Hardware encoders
        # (v4l2m2m, nvmpi) emit SPS/PPS once at stream start and don't repeat
        # them before later IDRs, so WebRTC consumers that join mid-stream get
        # no decoder config and render black. libx264 repeats them in-band, so
        # this is a no-op there. See CYB-2835.
        self._sps: Optional[bytes] = None
        self._pps: Optional[bytes] = None

    def _fall_back_to_software(self, why: str) -> None:
        logger.error(
            "H264 hardware encoder %s %s; permanently falling back to %s",
            self._codec_name,
            why,
            SOFTWARE_ENCODER,
        )
        self._codec_name = SOFTWARE_ENCODER
        self.buffer_data = b""
        self.buffer_pts = None
        self.codec = None

    def _encode_frame(self, frame: av.VideoFrame, force_keyframe: bool):
        # Mirror upstream reset conditions (aiortc 1.14 h264.py), but guard
        # bit_rate truthiness: unlike libx264, hardware wrappers (v4l2m2m,
        # nvmpi) may report bit_rate as 0 after open, which would make the
        # upstream ``/ self.codec.bit_rate`` a ZeroDivisionError.
        if self.codec and (
            frame.width != self.codec.width
            or frame.height != self.codec.height
            or (
                self.codec.bit_rate
                and abs(self.target_bitrate - self.codec.bit_rate) / self.codec.bit_rate
                > 0.1
            )
        ):
            self.buffer_data = b""
            self.buffer_pts = None
            self.codec = None

        if force_keyframe:
            frame.pict_type = av.video.frame.PictureType.I
        else:
            frame.pict_type = av.video.frame.PictureType.NONE

        if self.codec is None:
            try:
                self.codec = _build_codec_context(
                    self._codec_name, frame.width, frame.height, self.target_bitrate
                )
            except Exception:
                if self._codec_name == SOFTWARE_ENCODER:
                    raise
                self._fall_back_to_software("context creation failed")
                self.codec = _build_codec_context(
                    self._codec_name, frame.width, frame.height, self.target_bitrate
                )

        try:
            packages = self.codec.encode(frame)
        except av.FFmpegError:
            if self._codec_name == SOFTWARE_ENCODER:
                raise
            self._fall_back_to_software("failed to encode")
            self.codec = _build_codec_context(
                self._codec_name, frame.width, frame.height, self.target_bitrate
            )
            packages = self.codec.encode(frame)

        data_to_send = b"".join(bytes(p) for p in packages)
        if data_to_send:
            nals = list(self._split_bitstream(data_to_send))
            yield from self._maybe_inject_parameter_sets(nals)

    def _maybe_inject_parameter_sets(self, nals: "list[bytes]") -> "list[bytes]":
        """Cache SPS/PPS and re-inject them before any bare IDR.

        ``nals`` are the access unit's NAL units (no start codes), as produced
        by :meth:`_split_bitstream`. The first byte's low 5 bits are the NAL
        type: 7=SPS, 8=PPS, 5=IDR. When an IDR arrives without both parameter
        sets in the same access unit, we prepend the last-seen SPS/PPS so the
        keyframe is self-contained for mid-stream consumers.
        """
        has_idr = has_sps = has_pps = False
        for nal in nals:
            if not nal:
                continue
            nal_type = nal[0] & 0x1F
            if nal_type == 7:
                self._sps = bytes(nal)
                has_sps = True
            elif nal_type == 8:
                self._pps = bytes(nal)
                has_pps = True
            elif nal_type == 5:
                has_idr = True

        if not has_idr or (has_sps and has_pps):
            return nals
        if self._sps is None or self._pps is None:
            # Never saw parameter sets yet — nothing to inject. Emit as-is.
            return nals

        out: "list[bytes]" = []
        injected = False
        for nal in nals:
            nal_type = nal[0] & 0x1F if nal else -1
            if nal_type == 5 and not injected:
                out.append(self._sps)
                out.append(self._pps)
                injected = True
            out.append(nal)
        return out


_patch_lock = threading.Lock()


def apply_h264_hw_patch() -> EncoderSelection:
    """Idempotently route aiortc's H264 encoding through the runtime selection.

    aiortc's ``get_encoder()`` resolves ``H264Encoder`` as a module global in
    ``aiortc.codecs`` at call time, so rebinding that one symbol is the whole
    patch. A libx264/software selection applies no patch at all — upstream
    behavior stays byte-identical.
    """
    selection = select_h264_encoder()
    if selection.codec_name == SOFTWARE_ENCODER:
        return selection
    import aiortc.codecs as aiortc_codecs

    with _patch_lock:
        if aiortc_codecs.H264Encoder is not HardwareH264Encoder:
            setattr(aiortc_codecs, "H264Encoder", HardwareH264Encoder)
    return selection
