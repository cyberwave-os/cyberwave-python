"""Unit tests for the pre-encoded H.264 passthrough helpers.

Covers :func:`cyberwave.sensor.camera_h264.to_annex_b`, the AVCC -> Annex-B
normalisation that is the fix for the reported failure: aiortc's
``H264Encoder.pack()`` splits NAL units on ``0x000001`` start codes, so an
AVCC (length-prefixed) access unit carries no start codes and packs into zero
RTP payloads -- silently tearing the WebRTC transport down. ``to_annex_b``
rewrites AVCC framing to Annex-B, and deliberately passes anything it can't
cleanly frame through untouched rather than risk corrupting it.
"""

from __future__ import annotations

import pytest

# Importing the module pulls the base video track (aiortc). Skip cleanly where
# the camera extras aren't installed, matching the rest of the video suite.
pytest.importorskip(
    "aiortc", reason="aiortc not installed (install with extras: camera)"
)

from cyberwave.sensor.camera_h264 import (
    _ANNEXB_START,
    FRAMING_ANNEXB,
    FRAMING_AVCC,
    to_annex_b,
)

# Small, realistic NAL bodies (valid headers: forbidden_zero_bit clear,
# nal_unit_type non-zero).
_SPS = b"\x67\x42\x00\x1e"
_SLICE = b"\x65\x88\x84\x21"


def _slice_of(length: int) -> bytes:
    """A non-IDR slice NAL (header 0x41) of exactly `length` bytes."""
    return b"\x41" + b"\xAA" * (length - 1)


def _avcc(*nals: bytes) -> bytes:
    """Frame NALs as AVCC: a 4-byte big-endian length before each NAL."""
    return b"".join(len(n).to_bytes(4, "big") + n for n in nals)


def test_empty_buffer_returned_unchanged():
    assert to_annex_b(b"") == b""


def test_already_annexb_4byte_start_code_passthrough():
    data = _ANNEXB_START + _SLICE
    # Returned identical (identity), not re-framed.
    assert to_annex_b(data) is data


def test_already_annexb_3byte_start_code_passthrough():
    data = b"\x00\x00\x01" + _SLICE
    assert to_annex_b(data) is data


def test_avcc_single_nal_converted_to_annexb():
    assert to_annex_b(_avcc(_SLICE)) == _ANNEXB_START + _SLICE


def test_avcc_multiple_nals_each_get_start_code():
    assert (
        to_annex_b(_avcc(_SPS, _SLICE)) == _ANNEXB_START + _SPS + _ANNEXB_START + _SLICE
    )


def test_avcc_zero_length_prefix_left_untouched():
    # A declared length of 0 is not clean AVCC -- leave it rather than mangle.
    data = b"\x00\x00\x00\x00" + _SLICE
    assert to_annex_b(data) == data


def test_avcc_length_overrun_left_untouched():
    # Declares 255 bytes but only 2 follow -> framing mismatch, pass through.
    data = b"\x00\x00\x00\xff" + b"\xaa\xbb"
    assert to_annex_b(data) == data


def test_trailing_bytes_after_valid_nal_left_untouched():
    # One cleanly framed NAL followed by 2 dangling bytes -> not clean AVCC.
    data = _avcc(_SLICE) + b"\xaa\xbb"
    assert to_annex_b(data) == data


# --- AVCC length prefixes that masquerade as Annex-B start codes -------------
#
# A first NAL of 256-511 bytes encodes as b"\x00\x00\x01\xNN" and a 1-byte one
# as b"\x00\x00\x00\x01" -- both valid Annex-B start codes. Sniffing the
# leading bytes therefore forwarded these unconverted and pack() emitted one
# garbage NAL. Detection is AVCC-first precisely so these convert.


@pytest.mark.parametrize("length", [256, 257, 300, 400, 510, 511])
def test_avcc_length_prefix_that_looks_like_3byte_start_code_is_converted(length):
    nal = _slice_of(length)
    assert _avcc(nal)[:3] == b"\x00\x00\x01", "fixture must reproduce the collision"
    assert to_annex_b(_avcc(nal)) == _ANNEXB_START + nal


def test_avcc_single_byte_nal_looks_like_4byte_start_code_is_converted():
    nal = b"\x41"  # length 1 -> prefix is exactly b"\x00\x00\x00\x01"
    assert _avcc(nal)[:4] == _ANNEXB_START
    assert to_annex_b(_avcc(nal)) == _ANNEXB_START + nal


def test_avcc_multi_nal_with_colliding_first_nal_is_converted():
    # The collision is on the *first* prefix, so multi-NAL AUs hit it too.
    first, second = _slice_of(300), _slice_of(5000)
    assert to_annex_b(_avcc(first, second)) == (
        _ANNEXB_START + first + _ANNEXB_START + second
    )


def test_annexb_buffer_starting_with_4byte_start_code_not_misread_as_avcc():
    # AVCC-first must not corrupt real Annex-B: this parses as a 1-byte AVCC
    # NAL, then fails to chain, so it falls back to passthrough.
    data = _ANNEXB_START + _SLICE
    assert to_annex_b(data) is data


def test_nal_header_with_forbidden_zero_bit_set_is_not_treated_as_avcc():
    # Header 0x80 has forbidden_zero_bit set -> not a valid NAL, so not AVCC.
    data = _avcc(b"\x80\xaa\xbb\xcc")
    assert to_annex_b(data) == data


# --- explicit framing -------------------------------------------------------


def test_framing_annexb_skips_conversion_entirely():
    # Bytes that *would* convert are passed through when the caller says so.
    data = _avcc(_slice_of(300))
    assert to_annex_b(data, FRAMING_ANNEXB) is data


def test_framing_avcc_converts():
    nal = _slice_of(300)
    assert to_annex_b(_avcc(nal), FRAMING_AVCC) == _ANNEXB_START + nal


def test_framing_avcc_warns_when_input_is_not_avcc(caplog):
    data = _ANNEXB_START + _SLICE
    with caplog.at_level("WARNING"):
        assert to_annex_b(data, FRAMING_AVCC) is data
    assert "not cleanly framed 4-byte AVCC" in caplog.text
