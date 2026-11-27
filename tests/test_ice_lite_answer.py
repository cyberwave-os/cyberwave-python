"""ICE-lite answers are marked complete, so ICE can fail and the stream re-offer."""

from __future__ import annotations

import asyncio
import re

import pytest

from cyberwave.sensor.ice import complete_ice_lite_answer

ANSWER = "\r\n".join(
    [
        "v=0",
        "o=mediasoup-client 1 1 IN IP4 0.0.0.0",
        "s=-",
        "t=0 0",
        "a=ice-lite",
        "a=group:BUNDLE 0 1",
        "m=video 7 UDP/TLS/RTP/SAVPF 102",
        "c=IN IP4 127.0.0.1",
        "a=mid:0",
        "a=ice-ufrag:u",
        "a=ice-pwd:p",
        "a=candidate:udpcandidate 1 udp 1076302079 203.0.113.7 40000 typ host",
        "a=candidate:tcpcandidate 1 tcp 1076276479 203.0.113.7 40000 typ host tcptype passive",
        "a=rtpmap:102 H264/90000",
        "m=application 7 UDP/DTLS/SCTP webrtc-datachannel",
        "c=IN IP4 127.0.0.1",
        "a=mid:1",
        "a=candidate:udpcandidate 1 udp 1076302079 203.0.113.7 40000 typ host",
        "m=audio 0 UDP/TLS/RTP/SAVPF 111",
        "a=mid:2",
        "",
    ]
)


def _sections(sdp: str) -> list[str]:
    return re.split(r"\r\n(?=m=)", sdp)


def test_adds_end_of_candidates_to_each_section_with_candidates() -> None:
    out = complete_ice_lite_answer(ANSWER)
    session, video, application, audio = _sections(out)
    assert "a=end-of-candidates" not in session
    assert video.endswith("a=rtpmap:102 H264/90000\r\na=end-of-candidates")
    assert application.count("a=end-of-candidates") == 1
    assert "a=end-of-candidates" not in audio  # no candidates, nothing to end
    assert out.endswith("\r\n")
    assert out.replace("a=end-of-candidates\r\n", "") == ANSWER


def test_leaves_other_answers_alone() -> None:
    full_ice = ANSWER.replace("a=ice-lite\r\n", "")
    assert complete_ice_lite_answer(full_ice) == full_ice
    once = complete_ice_lite_answer(ANSWER)
    assert complete_ice_lite_answer(once) == once
    assert complete_ice_lite_answer("") == ""


def test_keeps_lf_line_endings() -> None:
    lf = ANSWER.replace("\r\n", "\n")
    out = complete_ice_lite_answer(lf)
    assert "\r" not in out
    assert out.count("a=end-of-candidates\n") == 2


async def _final_ice_state(complete: bool, monkeypatch: pytest.MonkeyPatch) -> str:
    """Offer from aiortc; answer as an ICE-lite SFU whose only candidate is unreachable."""
    aiortc = pytest.importorskip("aiortc")
    from aioice import stun
    from aiortc.mediastreams import VideoStreamTrack

    # Fail each connectivity check after 2 tries (0.5 + 1 s) instead of 7.
    monkeypatch.setattr(stun, "RETRY_MAX", 1)

    # No external STUN/TURN: the test only needs local candidate gathering.
    offerer = aiortc.RTCPeerConnection(aiortc.RTCConfiguration(iceServers=[]))
    answerer = aiortc.RTCPeerConnection(aiortc.RTCConfiguration(iceServers=[]))
    try:
        offerer.addTrack(VideoStreamTrack())
        await offerer.setLocalDescription(await offerer.createOffer())
        await answerer.setRemoteDescription(offerer.localDescription)
        # Only the answer's SDP is used; the answerer never starts ICE, so
        # nothing can reach the offerer through a peer-reflexive candidate.
        sdp = (await answerer.createAnswer()).sdp
        sdp = sdp.replace("a=end-of-candidates\r\n", "")
        sdp = re.sub(r"a=candidate:[^\r\n]*\r\n", "", sdp)
        # One host candidate nobody answers on (TEST-NET-1), as a lost SFU would be.
        sdp = sdp.replace(
            "a=mid:0\r\n",
            "a=mid:0\r\na=candidate:1 1 udp 2130706431 192.0.2.1 9 typ host\r\n",
        )
        sdp = sdp.replace("t=0 0\r\n", "t=0 0\r\na=ice-lite\r\n")
        if complete:
            sdp = complete_ice_lite_answer(sdp)
        await offerer.setRemoteDescription(
            aiortc.RTCSessionDescription(sdp=sdp, type="answer")
        )

        for _ in range(80):  # up to 8 s
            if offerer.iceConnectionState in ("failed", "closed"):
                break
            await asyncio.sleep(0.1)
        return offerer.iceConnectionState
    finally:
        await offerer.close()
        await answerer.close()


async def test_unreachable_ice_lite_answer_fails_when_marked_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert await _final_ice_state(True, monkeypatch) == "failed"


async def test_aioice_011_waits_forever_without_end_of_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aioice = pytest.importorskip("aioice")
    if tuple(int(x) for x in aioice.__version__.split(".")[:2]) < (0, 11):
        pytest.skip("aioice < 0.11 declares ICE failed without end-of-candidates")
    assert await _final_ice_state(False, monkeypatch) == "checking"


def test_completes_remaining_sections_when_one_is_already_complete() -> None:
    partial = ANSWER.replace("m=application", "a=end-of-candidates\r\nm=application")
    out = complete_ice_lite_answer(partial)
    assert out.count("a=end-of-candidates") == 2
    assert complete_ice_lite_answer(out) == out


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("base_video", "BaseVideoStreamer"),
        ("microphone", "BaseAudioStreamer"),
        ("audio_microphone", "BaseAudioStreamer"),
        ("av_streamer", "MultimediaStreamer"),
    ],
)
@pytest.mark.parametrize("ice_lite", [True, False])
async def test_streamer_answer_handlers_complete_ice_lite(
    module_name, class_name, ice_lite
):
    import importlib
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    # The streamer modules import aiortc at module level. The standalone CI job
    # installs no extras, so skip there like the other WebRTC tests do.
    pytest.importorskip("aiortc", reason="aiortc not installed (install with extras: camera)")

    cls = getattr(
        importlib.import_module(f"cyberwave.sensor.{module_name}"), class_name
    )
    sdp = ANSWER if ice_lite else ANSWER.replace("a=ice-lite\r\n", "")
    pc = SimpleNamespace(
        signalingState="have-local-offer", setRemoteDescription=AsyncMock()
    )
    streamer = SimpleNamespace(
        _answer_received=True, _answer_data={"sdp": sdp, "type": "answer"}, pc=pc
    )
    await cls._wait_for_answer(streamer)
    applied = pc.setRemoteDescription.call_args.args[0]
    assert applied.sdp == complete_ice_lite_answer(sdp)
    assert applied.type == "answer"


@pytest.mark.parametrize(
    "kinds,direction",
    [
        (["video"], "recvonly"),
        (["video"], "sendonly"),
        (["audio"], "recvonly"),
        (["audio"], "sendonly"),
        (["video", "audio"], "sendonly"),
    ],
)
async def test_reachable_ice_lite_delivers_media_and_data(kinds, direction):
    """Exercise codec/DTLS/RTP and SCTP, not just successful SDP parsing."""
    aiortc = pytest.importorskip("aiortc")
    from aiortc.mediastreams import AudioStreamTrack, VideoStreamTrack

    offerer = aiortc.RTCPeerConnection(aiortc.RTCConfiguration(iceServers=[]))
    answerer = aiortc.RTCPeerConnection(aiortc.RTCConfiguration(iceServers=[]))
    frames = []
    data_received = asyncio.Event()
    receiver = offerer if direction == "recvonly" else answerer

    @receiver.on("track")
    def on_track(track):
        frames.append(asyncio.create_task(track.recv()))

    @answerer.on("datachannel")
    def on_channel(channel):
        @channel.on("message")
        def on_message(message):
            if message == "ice-regression":
                data_received.set()

    channel = offerer.createDataChannel("regression")

    @channel.on("open")
    def on_open():
        channel.send("ice-regression")

    try:
        for kind in kinds:
            track_type = VideoStreamTrack if kind == "video" else AudioStreamTrack
            if direction == "sendonly":
                offerer.addTransceiver(track_type(), direction=direction)
            else:
                offerer.addTransceiver(kind, direction=direction)
                answerer.addTrack(track_type())
        # Match the SFU's H264/Opus codec selection, not aiortc's VP8 default.
        for pc in (offerer, answerer):
            for transceiver in pc.getTransceivers():
                mime = "video/H264" if transceiver.kind == "video" else "audio/opus"
                transceiver.setCodecPreferences(
                    [
                        codec
                        for codec in aiortc.RTCRtpSender.getCapabilities(
                            transceiver.kind
                        ).codecs
                        if codec.mimeType == mime
                    ]
                )
        await offerer.setLocalDescription(await offerer.createOffer())
        await answerer.setRemoteDescription(offerer.localDescription)
        await answerer.setLocalDescription(await answerer.createAnswer())
        sdp = answerer.localDescription.sdp.replace("a=end-of-candidates\r\n", "")
        sdp = sdp.replace("t=0 0\r\n", "t=0 0\r\na=ice-lite\r\n")
        await offerer.setRemoteDescription(
            aiortc.RTCSessionDescription(
                sdp=complete_ice_lite_answer(sdp), type="answer"
            )
        )
        assert len(frames) == len(kinds)
        decoded = await asyncio.wait_for(asyncio.gather(*frames), timeout=10)
        assert all(frame is not None for frame in decoded)
        await asyncio.wait_for(data_received.wait(), timeout=10)
        assert offerer.connectionState == "connected"
    finally:
        for task in frames:
            task.cancel()
        await asyncio.gather(*frames, return_exceptions=True)
        await offerer.close()
        await answerer.close()
