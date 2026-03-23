"""Unit tests for cyberwave.sensor.audio_microphone.

All tests run fully offline — no network, no MQTT broker, no robot required.

Coverage:
  - MicrophoneAudioTrack: frame format, PTS progression, silence padding, closed guard
  - BaseAudioStreamer / MicrophoneAudioStreamer: offer payload fields, answer routing
    (accept matching sensor, reject wrong sensor, reject no-audio SDP, reject offers)
"""

from __future__ import annotations

import asyncio
import fractions
import json
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

aiortc = pytest.importorskip("aiortc", reason="aiortc not installed (install with extras: camera)")
from aiortc.mediastreams import MediaStreamError

from cyberwave.sensor.audio_microphone import (
    AUDIO_PTIME,
    DEFAULT_LAYOUT,
    DEFAULT_SAMPLE_RATE,
    MicrophoneAudioStreamer,
    MicrophoneAudioTrack,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLES_PER_FRAME = int(AUDIO_PTIME * DEFAULT_SAMPLE_RATE)  # 960
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2  # 1920


def _make_mqtt_client(topic_prefix: str = "") -> MagicMock:
    client = MagicMock()
    client.topic_prefix = topic_prefix
    client.subscribe = MagicMock()
    client.publish = MagicMock()
    return client


def _silent_get_audio() -> bytes | None:
    return None


def _chunk_get_audio() -> bytes:
    return bytes(range(256)) * (BYTES_PER_FRAME // 256) + bytes(BYTES_PER_FRAME % 256)


# ===========================================================================
# MicrophoneAudioTrack
# ===========================================================================


class TestMicrophoneAudioTrack:
    def test_stream_attributes_defaults(self):
        track = MicrophoneAudioTrack(_silent_get_audio)
        attrs = track.get_stream_attributes()
        assert attrs["audio_type"] == "microphone"
        assert attrs["sample_rate"] == DEFAULT_SAMPLE_RATE
        assert attrs["layout"] == DEFAULT_LAYOUT
        assert attrs["ptime_ms"] == int(AUDIO_PTIME * 1000)

    @pytest.mark.asyncio
    async def test_recv_returns_correct_frame_format(self):
        track = MicrophoneAudioTrack(_chunk_get_audio)
        frame = await track.recv()
        assert frame.format.name == "s16"
        assert frame.layout.name == DEFAULT_LAYOUT
        assert frame.samples == SAMPLES_PER_FRAME
        assert frame.sample_rate == DEFAULT_SAMPLE_RATE
        assert frame.time_base == fractions.Fraction(1, DEFAULT_SAMPLE_RATE)

    @pytest.mark.asyncio
    async def test_pts_advances_by_samples_per_frame(self):
        track = MicrophoneAudioTrack(_chunk_get_audio)
        frame0 = await track.recv()
        frame1 = await track.recv()
        assert frame0.pts == 0
        assert frame1.pts == SAMPLES_PER_FRAME

    @pytest.mark.asyncio
    async def test_none_get_audio_produces_silence(self):
        track = MicrophoneAudioTrack(_silent_get_audio)
        frame = await track.recv()
        data = bytes(frame.planes[0])
        assert data == bytes(BYTES_PER_FRAME)

    @pytest.mark.asyncio
    async def test_short_chunk_padded_with_silence(self):
        # Callback returns fewer bytes than a full frame
        track = MicrophoneAudioTrack(lambda: bytes(10))
        frame = await track.recv()
        data = bytes(frame.planes[0])
        assert data == bytes(BYTES_PER_FRAME)

    @pytest.mark.asyncio
    async def test_recv_raises_on_closed_track(self):
        track = MicrophoneAudioTrack(_silent_get_audio)
        track.close()
        with pytest.raises(MediaStreamError):
            await track.recv()

    @pytest.mark.asyncio
    async def test_frame_data_matches_callback_output(self):
        payload = bytes(i % 256 for i in range(BYTES_PER_FRAME))
        track = MicrophoneAudioTrack(lambda: payload)
        frame = await track.recv()
        assert bytes(frame.planes[0]) == payload


# ===========================================================================
# BaseAudioStreamer / MicrophoneAudioStreamer — offer payload
# ===========================================================================


class TestOfferPayload:
    """Verify that _send_offer builds the correct MQTT payload."""

    def _make_streamer(self, sensor_name: str = "mic") -> MicrophoneAudioStreamer:
        client = _make_mqtt_client(topic_prefix="")
        streamer = MicrophoneAudioStreamer(
            client,
            get_audio=_silent_get_audio,
            twin_uuid="twin-123",
            sensor_name=sensor_name,
        )
        # Set up a minimal fake PC + track so _send_offer can read them
        fake_pc = MagicMock()
        fake_pc.localDescription.type = "offer"
        fake_pc.localDescription.sdp = "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
        streamer.pc = fake_pc
        streamer.streamer = MicrophoneAudioTrack(_silent_get_audio)
        return streamer

    def test_offer_payload_required_fields(self):
        s = self._make_streamer()
        s._send_offer("v=0\r\n")
        s.client.publish.assert_called_once()
        _, payload = s.client.publish.call_args[0]
        assert payload["frontend_type"] == "audio"
        assert payload["track_type"] == "audio"
        assert payload["target"] == "backend"
        assert payload["sender"] == "edge"
        assert payload["sensor"] == "mic"
        assert payload["recording"] is False

    def test_offer_published_to_correct_topic(self):
        s = self._make_streamer()
        s._send_offer("v=0\r\n")
        topic = s.client.publish.call_args[0][0]
        assert topic == "cyberwave/twin/twin-123/webrtc-offer"

    def test_offer_includes_stream_attributes(self):
        s = self._make_streamer()
        s._send_offer("v=0\r\n")
        _, payload = s.client.publish.call_args[0]
        attrs = payload["stream_attributes"]
        assert attrs["audio_type"] == "microphone"
        assert attrs["sample_rate"] == DEFAULT_SAMPLE_RATE

    def test_topic_prefix_respected(self):
        client = _make_mqtt_client(topic_prefix="env/")
        s = MicrophoneAudioStreamer(
            client, get_audio=_silent_get_audio, twin_uuid="twin-123", sensor_name="mic"
        )
        fake_pc = MagicMock()
        fake_pc.localDescription.type = "offer"
        s.pc = fake_pc
        s.streamer = MicrophoneAudioTrack(_silent_get_audio)
        s._send_offer("v=0\r\n")
        topic = s.client.publish.call_args[0][0]
        assert topic == "env/cyberwave/twin/twin-123/webrtc-offer"


# ===========================================================================
# BaseAudioStreamer — answer routing (_subscribe_to_answer / on_answer)
# ===========================================================================


def _extract_on_answer(streamer: MicrophoneAudioStreamer) -> callable:
    """Call _subscribe_to_answer and return the registered on_answer callback."""
    streamer._subscribe_to_answer()
    # First subscribe call is for the answer topic
    return streamer.client.subscribe.call_args_list[0][0][1]


VALID_ANSWER = {
    "type": "answer",
    "target": "edge",
    "sdp": "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
    "sensor": "mic",
}


class TestAnswerRouting:
    def _make_streamer(self, sensor_name: str = "mic") -> MicrophoneAudioStreamer:
        client = _make_mqtt_client()
        return MicrophoneAudioStreamer(
            client,
            get_audio=_silent_get_audio,
            twin_uuid="twin-123",
            sensor_name=sensor_name,
        )

    def test_matching_answer_accepted(self):
        s = self._make_streamer()
        on_answer = _extract_on_answer(s)
        on_answer(VALID_ANSWER)
        assert s._answer_received is True
        assert s._answer_data == VALID_ANSWER

    def test_answer_with_wrong_sensor_rejected(self):
        s = self._make_streamer(sensor_name="mic")
        on_answer = _extract_on_answer(s)
        wrong = {**VALID_ANSWER, "sensor": "front_camera"}
        on_answer(wrong)
        assert s._answer_received is False

    def test_answer_without_audio_sdp_rejected(self):
        s = self._make_streamer()
        on_answer = _extract_on_answer(s)
        no_audio = {**VALID_ANSWER, "sdp": "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n"}
        on_answer(no_audio)
        assert s._answer_received is False

    def test_offer_type_message_ignored(self):
        s = self._make_streamer()
        on_answer = _extract_on_answer(s)
        on_answer({**VALID_ANSWER, "type": "offer"})
        assert s._answer_received is False

    def test_answer_missing_sensor_field_accepted_for_any_streamer(self):
        """sensor=None in answer payload → accepted regardless of streamer sensor_name."""
        s = self._make_streamer(sensor_name="mic")
        on_answer = _extract_on_answer(s)
        no_sensor = {k: v for k, v in VALID_ANSWER.items() if k != "sensor"}
        on_answer(no_sensor)
        assert s._answer_received is True

    def test_json_string_answer_parsed_correctly(self):
        s = self._make_streamer()
        on_answer = _extract_on_answer(s)
        on_answer(json.dumps(VALID_ANSWER))
        assert s._answer_received is True
        assert isinstance(s._answer_data, dict)

    def test_answer_not_for_edge_ignored(self):
        s = self._make_streamer()
        on_answer = _extract_on_answer(s)
        on_answer({**VALID_ANSWER, "target": "frontend"})
        assert s._answer_received is False

    def test_subscribes_to_both_answer_and_candidate_topics(self):
        s = self._make_streamer()
        s._subscribe_to_answer()
        topics = [c[0][0] for c in s.client.subscribe.call_args_list]
        assert "cyberwave/twin/twin-123/webrtc-answer" in topics
        assert "cyberwave/twin/twin-123/webrtc-candidate" in topics
