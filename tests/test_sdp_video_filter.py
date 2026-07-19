"""Tests for the VP8/VP9 SDP filter used by the video streamers."""

from __future__ import annotations

import pytest

pytest.importorskip("aiortc", reason="aiortc not installed (install with extras: camera)")

from cyberwave.sensor.base_video import _strip_vp8_video


def _sdp(*lines: str) -> str:
    return "\r\n".join(lines) + "\r\n"


def test_strips_vp8_rtpmap_fmtp_and_rtcp_fb() -> None:
    sdp = _sdp(
        "v=0",
        "m=video 9 UDP/TLS/RTP/SAVPF 97 98 106",
        "a=rtpmap:97 VP8/90000",
        "a=rtcp-fb:97 nack",
        "a=rtcp-fb:97 nack pli",
        "a=fmtp:97 max-fr=30",
        "a=rtpmap:98 rtx/90000",
        "a=fmtp:98 apt=97",
        "a=rtpmap:106 H264/90000",
        "a=fmtp:106 profile-level-id=42e01f",
    )

    filtered = _strip_vp8_video(sdp)

    assert "VP8" not in filtered
    assert "97" not in filtered.split("m=video")[1].split("\r\n", 1)[0]
    assert "98" not in filtered.split("m=video")[1].split("\r\n", 1)[0]
    assert "a=rtpmap:97" not in filtered
    assert "a=fmtp:97" not in filtered  # orphan fmtp for VP8 must go
    assert "a=rtcp-fb:97" not in filtered
    assert "a=rtpmap:98" not in filtered  # RTX riding on VP8 must go too
    assert "a=fmtp:98 apt=97" not in filtered
    assert "a=rtpmap:106 H264/90000" in filtered


def test_leaves_audio_section_untouched() -> None:
    sdp = _sdp(
        "v=0",
        "m=video 9 UDP/TLS/RTP/SAVPF 106",
        "a=rtpmap:106 H264/90000",
        "m=audio 9 UDP/TLS/RTP/SAVPF 97 111",
        "a=rtpmap:97 PCMU/8000",
        "a=rtpmap:111 opus/48000/2",
    )

    filtered = _strip_vp8_video(sdp)

    audio_section = filtered.split("m=audio", 1)[1]
    assert "a=rtpmap:97 PCMU/8000" in audio_section
    assert "97" in audio_section.split("\r\n", 1)[0]  # PCMU PT stays in m=audio


def test_survives_aiortc_pt_reshuffle_where_h264_lands_on_97() -> None:
    """If aiortc ever renumbers so H264 gets PT 97, the old hardcoded filter
    would strip it and Safari would end up with no compatible codec. The
    codec-name-driven filter must preserve H264 regardless of PT."""
    sdp = _sdp(
        "v=0",
        "m=video 9 UDP/TLS/RTP/SAVPF 97 100",
        "a=rtpmap:97 H264/90000",
        "a=fmtp:97 profile-level-id=42e01f",
        "a=rtpmap:100 VP8/90000",
        "a=rtcp-fb:100 nack",
    )

    filtered = _strip_vp8_video(sdp)

    assert "a=rtpmap:97 H264/90000" in filtered
    assert "a=fmtp:97 profile-level-id=42e01f" in filtered
    assert "a=rtpmap:100" not in filtered
    assert "VP8" not in filtered
    m_video = filtered.split("m=video", 1)[1].split("\r\n", 1)[0]
    assert "97" in m_video
    assert "100" not in m_video


def test_preserves_line_terminator() -> None:
    sdp = "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 106\r\na=rtpmap:106 H264/90000\r\n"
    assert _strip_vp8_video(sdp).endswith("\r\n")

    sdp_lf = "v=0\nm=video 9 UDP/TLS/RTP/SAVPF 106\na=rtpmap:106 H264/90000\n"
    filtered = _strip_vp8_video(sdp_lf)
    assert "\r\n" not in filtered


def test_bails_out_when_stripping_would_leave_no_video_pt() -> None:
    """If aiortc failed to negotiate H264, the filter would otherwise emit
    an m=video line with orphan PTs. Instead, we return the SDP untouched
    so the SFU rejects the offer with a clear error."""
    sdp = _sdp(
        "v=0",
        "m=video 9 UDP/TLS/RTP/SAVPF 97 98",
        "a=rtpmap:97 VP8/90000",
        "a=rtpmap:98 rtx/90000",
        "a=fmtp:98 apt=97",
    )
    assert _strip_vp8_video(sdp) == sdp


def test_bails_out_when_any_video_section_would_be_emptied() -> None:
    """Even if one m=video section would still have PTs, if another would
    be emptied we bail so no downstream parser sees an orphan m-line."""
    sdp = _sdp(
        "v=0",
        "m=video 9 UDP/TLS/RTP/SAVPF 97 106",
        "a=rtpmap:97 VP8/90000",
        "a=rtpmap:106 H264/90000",
        "m=video 9 UDP/TLS/RTP/SAVPF 100",
        "a=rtpmap:100 VP9/90000",
    )
    assert _strip_vp8_video(sdp) == sdp


def test_rtx_chain_reaches_fixpoint() -> None:
    """Multiple passes are needed if an fmtp for the outer RTX appears
    before the fmtp for the inner one that binds it to VP8."""
    sdp = _sdp(
        "v=0",
        "m=video 9 UDP/TLS/RTP/SAVPF 97 98 99 106",
        "a=fmtp:99 apt=98",
        "a=fmtp:98 apt=97",
        "a=rtpmap:97 VP8/90000",
        "a=rtpmap:98 rtx/90000",
        "a=rtpmap:99 rtx/90000",
        "a=rtpmap:106 H264/90000",
    )
    filtered = _strip_vp8_video(sdp)
    m_video = filtered.split("m=video", 1)[1].split("\r\n", 1)[0]
    assert "97" not in m_video
    assert "98" not in m_video
    assert "99" not in m_video
    assert "106" in m_video


def test_no_op_when_no_vp8_present() -> None:
    sdp = _sdp(
        "v=0",
        "m=video 9 UDP/TLS/RTP/SAVPF 106",
        "a=rtpmap:106 H264/90000",
        "a=fmtp:106 profile-level-id=42e01f",
    )
    assert _strip_vp8_video(sdp) == sdp
