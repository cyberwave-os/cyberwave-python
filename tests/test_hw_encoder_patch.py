"""Pin the aiortc monkeypatch and its wiring into BaseVideoStreamer.

aiortc's RTCRtpSender resolves the encoder via aiortc.codecs.get_encoder(),
which looks up the module-global ``H264Encoder`` at call time — rebinding
that symbol is the whole patch. These tests pin: get_encoder returns our
subclass after patching, idempotency, software selection leaving upstream
untouched, and start() applying the patch + stamping the offer attribute.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("av", reason="pyav not installed")
pytest.importorskip("aiortc", reason="aiortc not installed")
pytest.importorskip("cv2", reason="OpenCV not installed")

import aiortc.codecs as aiortc_codecs  # noqa: E402
from aiortc.codecs.h264 import H264Encoder  # noqa: E402
from aiortc.rtcrtpparameters import RTCRtpCodecParameters  # noqa: E402

from cyberwave.sensor import base_video, hw_encoder  # noqa: E402
from cyberwave.sensor.camera_cv2 import CV2CameraStreamer  # noqa: E402

_H264_PARAMS = RTCRtpCodecParameters(
    mimeType="video/H264", clockRate=90000, payloadType=106
)


@pytest.fixture(autouse=True)
def _restore_aiortc(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CYBERWAVE_VIDEO_ENCODER", raising=False)
    hw_encoder.reset_selection_cache()
    original = aiortc_codecs.H264Encoder
    yield
    aiortc_codecs.H264Encoder = original
    hw_encoder.reset_selection_cache()


def _force_selection(monkeypatch: pytest.MonkeyPatch, name: str, reason: str):
    monkeypatch.setattr(
        hw_encoder,
        "select_h264_encoder",
        lambda refresh=False: hw_encoder.EncoderSelection(name, reason),
    )


def test_patch_routes_get_encoder_to_hardware_subclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_selection(monkeypatch, "h264_v4l2m2m", "probed")
    sel = hw_encoder.apply_h264_hw_patch()
    assert sel.codec_name == "h264_v4l2m2m"
    encoder = aiortc_codecs.get_encoder(_H264_PARAMS)
    assert isinstance(encoder, hw_encoder.HardwareH264Encoder)


def test_patch_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_selection(monkeypatch, "h264_v4l2m2m", "probed")
    hw_encoder.apply_h264_hw_patch()
    first = aiortc_codecs.H264Encoder
    hw_encoder.apply_h264_hw_patch()
    assert aiortc_codecs.H264Encoder is first


def test_software_selection_leaves_aiortc_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_selection(monkeypatch, "libx264", "software")
    sel = hw_encoder.apply_h264_hw_patch()
    assert sel.codec_name == "libx264"
    encoder = aiortc_codecs.get_encoder(_H264_PARAMS)
    assert type(encoder) is H264Encoder


class _FakeMQTT:
    topic_prefix = ""
    client_id = "test-client"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any], qos: int = 0) -> None:
        del qos
        self.calls.append((topic, dict(payload)))


@pytest.mark.asyncio
async def test_start_applies_patch_and_stamps_offer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied: list[str] = []

    def fake_apply() -> hw_encoder.EncoderSelection:
        applied.append("yes")
        return hw_encoder.EncoderSelection("h264_v4l2m2m", "probed")

    monkeypatch.setattr(base_video, "apply_h264_hw_patch", fake_apply)

    streamer = CV2CameraStreamer(
        client=_FakeMQTT(), camera_id=0, twin_uuid="twin-1", auto_reconnect=False
    )
    streamer.enable_health_check = False

    async def _noop(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(streamer, "_setup_webrtc", _noop)
    monkeypatch.setattr(streamer, "_perform_signaling", _noop)
    monkeypatch.setattr(streamer, "_subscribe_to_answer", lambda: None)
    monkeypatch.setattr(
        streamer, "_wait_and_publish_camera_sync_frame", _noop
    )
    monkeypatch.setattr(base_video.asyncio, "sleep", _noop)

    await streamer.start()

    assert applied == ["yes"]
    assert streamer._video_encoder_name == "h264_v4l2m2m"


def test_send_offer_includes_video_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeMQTT()
    streamer = CV2CameraStreamer(
        client=client, camera_id=0, twin_uuid="twin-1", auto_reconnect=False
    )
    streamer._video_encoder_name = "h264_v4l2m2m"

    class _FakePC:
        class localDescription:  # noqa: N801
            type = "offer"

    streamer.pc = _FakePC()
    streamer._send_offer("v=0\r\n")

    assert len(client.calls) == 1
    _, payload = client.calls[0]
    assert payload["stream_attributes"]["video_encoder"] == "h264_v4l2m2m"
