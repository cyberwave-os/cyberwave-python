import time

import numpy as np
import pytest

from cyberwave.exceptions import CyberwaveError, NoOngoingVideoStreamAvailable
from cyberwave.consumers.video import IncomingVideoStream


def test_no_ongoing_video_stream_available_is_cyberwave_error():
    assert issubclass(NoOngoingVideoStreamAvailable, CyberwaveError)
    err = NoOngoingVideoStreamAvailable("no stream")
    assert str(err) == "no stream"


class _FakeMqtt:
    """Minimal CyberwaveMQTTClient stand-in for unit tests."""

    def __init__(self, prefix="", client_id="sdk_test"):
        self.topic_prefix = prefix
        self.client_id = client_id
        self.subscribed = []
        self.unsubscribed = []
        self.published = []

    def subscribe(self, topic, handler=None, qos=0, *, subscriber_key=None, **kw):
        self.subscribed.append((topic, subscriber_key))

    def unsubscribe(self, topic, subscriber_key=None):
        self.unsubscribed.append((topic, subscriber_key))

    def publish(self, topic, message, qos=0):
        self.published.append((topic, message, qos))


def _make_stream(prefix=""):
    return IncomingVideoStream(
        _FakeMqtt(prefix=prefix),
        "twin-123",
        sensor_id="wrist_cam",
        stream_source=None,
        stream_instance_id=None,
        frontend_type="rgb",
        timeout=3.0,
    )


def test_offer_payload_matches_contract():
    s = _make_stream(prefix="dev/")
    payload = s._build_offer_payload("v=0\r\n")
    assert payload["target"] == "backend"
    assert payload["sender"] == "client_python_sdk"
    assert payload["type"] == "offer"
    assert payload["sdp"] == "v=0\r\n"
    assert payload["frontend_type"] == "rgb"
    assert payload["sensor"] == "wrist_cam"
    assert payload["session_id"] == s._session_id
    assert "timestamp" in payload
    # Optional identity fields omitted when unset
    assert "stream_source" not in payload
    assert "stream_instance_id" not in payload
    # Topics carry the env prefix
    assert s._offer_topic == "dev/cyberwave/twin/twin-123/webrtc-offer"
    assert s._answer_topic == "dev/cyberwave/twin/twin-123/webrtc-answer"


def test_offer_payload_sender_defaults_and_override():
    default = IncomingVideoStream(_FakeMqtt(), "t", sensor_id="cam")
    assert default._build_offer_payload("sdp")["sender"] == "client_python_sdk"

    override = IncomingVideoStream(_FakeMqtt(), "t", sensor_id="cam", sender="frontend")
    assert override._build_offer_payload("sdp")["sender"] == "frontend"


def test_offer_payload_includes_stream_identity_when_set():
    s = IncomingVideoStream(
        _FakeMqtt(), "twin-9", sensor_id="cam", stream_source="live",
        stream_instance_id="default", frontend_type="rgb",
    )
    payload = s._build_offer_payload("sdp")
    assert payload["stream_source"] == "live"
    assert payload["stream_instance_id"] == "default"


def test_classify_reply_answer():
    assert IncomingVideoStream._classify_reply(
        {"type": "answer", "sdp": "v=0", "session_id": "S"}, "S"
    ) == ("answer", "v=0")


def test_classify_reply_wait_and_error_are_unavailable():
    assert IncomingVideoStream._classify_reply(
        {"type": "wait", "message": "no producer", "session_id": "S"}, "S"
    ) == ("unavailable", "no producer")
    assert IncomingVideoStream._classify_reply(
        {"type": "error", "error": "boom", "session_id": "S"}, "S"
    ) == ("unavailable", "boom")


def test_classify_reply_ignores_other_session_and_unknown_types():
    assert IncomingVideoStream._classify_reply(
        {"type": "answer", "sdp": "v=0", "session_id": "OTHER"}, "S"
    ) is None
    assert IncomingVideoStream._classify_reply(
        {"type": "offer", "session_id": "S"}, "S"
    ) is None
    assert IncomingVideoStream._classify_reply("not-a-dict", "S") is None
    # answer without sdp is not actionable
    assert IncomingVideoStream._classify_reply(
        {"type": "answer", "session_id": "S"}, "S"
    ) is None


def test_format_frame_numpy_returns_copy():
    s = _make_stream()
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    out = s._format_frame(arr, "numpy")
    assert out is not arr
    assert np.array_equal(out, arr)


def test_format_frame_bytes_is_jpeg(monkeypatch):
    """Doesn't require real cv2/opencv-python (an optional extra) to be installed —
    fakes cv2.imencode so this runs in any environment, matching the fake-cv2
    pattern used by the show()-related tests below."""
    import sys
    import types

    fake_jpeg = np.frombuffer(b"\xff\xd8fake-jpeg-body", dtype=np.uint8)
    fake_cv2 = types.SimpleNamespace(imencode=lambda ext, arr: (True, fake_jpeg))
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    s = _make_stream()
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    out = s._format_frame(arr, "bytes")
    assert isinstance(out, (bytes, bytearray))
    assert bytes(out[:2]) == b"\xff\xd8"  # JPEG SOI marker


import asyncio
import threading


class _FakePC:
    """Patched RTCPeerConnection: records transceivers, exposes a canned SDP."""

    instances = []

    def __init__(self, *a, **kw):
        self.localDescription = type("D", (), {"sdp": "v=0\r\ncanned", "type": "offer"})()
        self.iceGatheringState = "complete"
        self.connectionState = "new"
        self.transceivers = []
        self.closed = False
        self._handlers = {}
        _FakePC.instances.append(self)

    def on(self, event):
        def deco(fn):
            self._handlers[event] = fn
            return fn
        return deco

    def addTransceiver(self, kind, direction=None):
        self.transceivers.append((kind, direction))

    def getReceivers(self):
        return []

    async def createOffer(self):
        return self.localDescription

    async def setLocalDescription(self, desc):
        self.localDescription = desc

    async def setRemoteDescription(self, desc):
        self.remote = desc

    async def close(self):
        self.closed = True


@pytest.fixture
def patch_pc(monkeypatch):
    _FakePC.instances.clear()
    import cyberwave.consumers.video as v
    monkeypatch.setattr(v, "RTCPeerConnection", _FakePC)
    return _FakePC


def test_start_publishes_offer_then_answer_connects(patch_pc):
    mqtt = _FakeMqtt(prefix="")
    s = IncomingVideoStream(mqtt, "twin-1", sensor_id="cam", timeout=3.0)

    # Deliver the answer shortly after start() subscribes+publishes.
    def deliver():
        # busy-wait until the offer has been published and handler registered
        for _ in range(200):
            if mqtt.published and s._reply_handler is not None:
                break
            time.sleep(0.01)
        s._reply_handler(
            {"type": "answer", "sdp": "v=0\r\nanswer", "session_id": s._session_id}
        )

    t = threading.Thread(target=deliver)
    t.start()
    s.start()
    t.join()

    # Offer published to the right topic with the contract payload
    topic, payload, _ = mqtt.published[0]
    assert topic == "cyberwave/twin/twin-1/webrtc-offer"
    assert payload["sender"] == "client_python_sdk"
    # Subscribed to the answer topic under our unique key
    assert (s._answer_topic, s._subscriber_key) in mqtt.subscribed
    # recvonly video transceiver was added
    assert ("video", "recvonly") in patch_pc.instances[0].transceivers
    s.stop()


def test_start_raises_no_ongoing_on_wait(patch_pc):
    mqtt = _FakeMqtt()
    s = IncomingVideoStream(mqtt, "twin-2", sensor_id="cam", timeout=3.0)

    def deliver():
        for _ in range(200):
            if mqtt.published and s._reply_handler is not None:
                break
            time.sleep(0.01)
        s._reply_handler(
            {"type": "wait", "message": "no producer", "session_id": s._session_id}
        )

    t = threading.Thread(target=deliver)
    t.start()
    with pytest.raises(NoOngoingVideoStreamAvailable):
        s.start()
    t.join()
    s.stop()


def test_start_times_out_without_reply(patch_pc):
    mqtt = _FakeMqtt()
    s = IncomingVideoStream(mqtt, "twin-3", sensor_id="cam", timeout=0.3)
    with pytest.raises(TimeoutError):
        s.start()
    s.stop()


def test_stop_is_idempotent_and_unsubscribes(patch_pc):
    mqtt = _FakeMqtt()
    s = IncomingVideoStream(mqtt, "twin-4", sensor_id="cam", timeout=0.3)
    try:
        s.start()
    except TimeoutError:
        pass
    s.stop()
    s.stop()  # no raise on second call
    assert (s._answer_topic, s._subscriber_key) in mqtt.unsubscribed


def test_get_frame_none_before_first_frame(patch_pc):
    mqtt = _FakeMqtt()
    s = IncomingVideoStream(mqtt, "twin-5", sensor_id="cam", timeout=0.3)
    assert s.get_frame() is None


def test_show_renders_frames_and_exits_on_quit_key(monkeypatch):
    """show() should imshow available frames and return when 'q' is pressed."""
    import sys
    import types

    s = IncomingVideoStream(_FakeMqtt(), "twin-show", sensor_id="cam")
    with s._frame_lock:
        s._current_frame = np.zeros((4, 4, 3), dtype=np.uint8)

    calls = {"imshow": 0, "destroyed": []}

    fake_cv2 = types.SimpleNamespace(
        WND_PROP_VISIBLE=0,
        imshow=lambda *a: calls.__setitem__("imshow", calls["imshow"] + 1),
        # First poll returns a non-quit key, second returns 'q' to exit.
        waitKey=lambda *_: (0 if calls["imshow"] < 2 else ord("q")),
        getWindowProperty=lambda *a: 1.0,
        destroyWindow=lambda w: calls["destroyed"].append(w),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    s.show(window_name="test-window")

    assert calls["imshow"] >= 1
    assert calls["destroyed"] == ["test-window"]


def test_show_pumps_waitkey_after_destroy(monkeypatch):
    """On macOS/Cocoa the native window is torn down by the HighGUI event loop,
    not by destroyWindow() itself — show() must pump waitKey() after destroying
    so the window actually closes instead of lingering as a ghost."""
    import sys
    import types

    s = IncomingVideoStream(_FakeMqtt(), "twin-show", sensor_id="cam")
    with s._frame_lock:
        s._current_frame = np.zeros((4, 4, 3), dtype=np.uint8)

    events: list[str] = []
    polls = {"n": 0}

    def _waitkey(*_):
        events.append("waitKey")
        polls["n"] += 1
        # Quit on the first poll so we can assert the post-destroy pump.
        return ord("q") if polls["n"] == 1 else 0

    fake_cv2 = types.SimpleNamespace(
        WND_PROP_VISIBLE=0,
        imshow=lambda *a: None,
        waitKey=_waitkey,
        getWindowProperty=lambda *a: 1.0,
        destroyWindow=lambda w: events.append("destroy"),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    s.show(window_name="w")

    assert "destroy" in events
    # At least one waitKey must run AFTER destroy to flush the Cocoa event loop.
    assert events[-1] == "waitKey"
    assert events.index("destroy") < len(events) - 1


def test_get_video_constructs_and_starts_stream(monkeypatch):
    import cyberwave.consumers.video as v
    from cyberwave.twin.sensors.camera import (
        CAMERA_HANDLE_PUBLIC_METHODS,
        TwinCameraHandle,
    )

    assert "get_video" in CAMERA_HANDLE_PUBLIC_METHODS

    captured = {}

    class _StubStream:
        def __init__(self, mqtt, twin_uuid, **kwargs):
            captured["mqtt"] = mqtt
            captured["twin_uuid"] = twin_uuid
            captured["kwargs"] = kwargs

        def start(self):
            captured["started"] = True
            return self

    monkeypatch.setattr(v, "IncomingVideoStream", _StubStream)

    fake_mqtt = object()
    twin = type(
        "T", (), {
            "client": type("C", (), {"mqtt": fake_mqtt})(),
            "uuid": "twin-xyz",
            "_resolve_sensor_id": lambda self, s: s or "cam0",
            "_ensure_mqtt_connected": lambda self: None,
            "_ensure_simulation_support": lambda self, level, **kw: None,
        },
    )()
    handle = TwinCameraHandle(twin, sensor_id="wrist")

    stream = handle.get_video(timeout=4.0)

    assert isinstance(stream, _StubStream)
    assert captured["started"] is True
    assert captured["mqtt"] is fake_mqtt
    assert captured["twin_uuid"] == "twin-xyz"
    assert captured["kwargs"]["sensor_id"] == "wrist"
    assert captured["kwargs"]["timeout"] == 4.0
