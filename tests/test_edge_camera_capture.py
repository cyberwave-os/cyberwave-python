"""Tests for TwinCameraHandle.edge_photo and edge_photos."""

import base64
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cyberwave.exceptions import CyberwaveError
from cyberwave.twin import Twin, TwinCameraHandle

FAKE_JPEG = b"\xff\xd8fake-jpeg-payload\xff\xd9"
FAKE_B64 = base64.b64encode(FAKE_JPEG).decode()


def _make_mqtt_client():
    """Build a mock MQTT client with subscribe/publish/unsubscribe."""
    mqtt = MagicMock()
    mqtt.connected = True

    # Track subscribed handlers so tests can trigger them
    _handlers: dict[str, callable] = {}

    def _subscribe(topic, handler, qos=0):
        _handlers[topic] = handler

    def _unsubscribe(topic):
        _handlers.pop(topic, None)

    mqtt.subscribe = MagicMock(side_effect=_subscribe)
    mqtt.unsubscribe = MagicMock(side_effect=_unsubscribe)
    mqtt._test_handlers = _handlers  # expose for test injection
    return mqtt


def _make_twin_with_mqtt():
    """Create a Twin with a mock client that has MQTT support.

    Mirrors the real SDK architecture: the outer mqtt_client.py wrapper
    delegates to an inner _client that actually has subscribe/unsubscribe.
    """
    twins_manager = MagicMock()
    twins_manager.get_latest_frame.return_value = FAKE_JPEG

    # Inner client (what actually has subscribe/unsubscribe)
    inner_mqtt = _make_mqtt_client()

    # Outer wrapper (delegates subscribe/publish to inner)
    outer_mqtt = MagicMock()
    outer_mqtt.connected = True
    outer_mqtt._client = inner_mqtt
    outer_mqtt.subscribe = MagicMock(side_effect=inner_mqtt.subscribe)
    outer_mqtt.publish = MagicMock(side_effect=inner_mqtt.publish)

    config = SimpleNamespace(topic_prefix="")
    client = SimpleNamespace(twins=twins_manager, mqtt=outer_mqtt, config=config)
    twin = Twin(client, SimpleNamespace(uuid="twin-uuid", name="TestTwin"))
    return twin, inner_mqtt


# ---------------------------------------------------------------------------
# edge_photo tests
# ---------------------------------------------------------------------------


class TestEdgePhoto:
    def test_sends_take_photo_command(self):
        twin, mqtt = _make_twin_with_mqtt()
        camera = TwinCameraHandle(twin)

        # Simulate photo response in a background thread
        def respond():
            time.sleep(0.05)
            handler = mqtt._test_handlers.get(
                "cyberwave/twin/twin-uuid/camera/photo"
            )
            if handler:
                handler(
                    json.dumps({
                        "image": FAKE_B64,
                        "format": "jpeg",
                        "width": 640,
                        "height": 480,
                    })
                )

        t = threading.Thread(target=respond)
        t.start()

        result = camera.edge_photo(format="bytes")
        t.join()

        # Verify command was published
        mqtt.publish.assert_called_once()
        call_args = mqtt.publish.call_args
        topic = call_args[0][0]
        payload = call_args[0][1]
        assert topic == "cyberwave/twin/twin-uuid/command"
        assert payload["command"] == "take_photo"
        assert payload["source_type"] == "tele"
        assert result == FAKE_JPEG

    def test_returns_numpy_frame(self):
        twin, mqtt = _make_twin_with_mqtt()
        camera = TwinCameraHandle(twin)

        mock_np = MagicMock()
        mock_cv2 = MagicMock()
        sentinel = MagicMock(name="numpy_frame")
        mock_np.frombuffer.return_value = MagicMock()
        mock_np.uint8 = "uint8"
        mock_cv2.imdecode.return_value = sentinel
        mock_cv2.IMREAD_COLOR = 1

        def respond():
            time.sleep(0.05)
            handler = mqtt._test_handlers.get(
                "cyberwave/twin/twin-uuid/camera/photo"
            )
            if handler:
                handler(json.dumps({"image": FAKE_B64, "format": "jpeg"}))

        t = threading.Thread(target=respond)
        t.start()

        with patch.dict("sys.modules", {"numpy": mock_np, "cv2": mock_cv2}):
            result = camera.edge_photo(format="numpy")

        t.join()
        assert result is sentinel

    def test_raises_on_timeout(self):
        twin, mqtt = _make_twin_with_mqtt()
        camera = TwinCameraHandle(twin)

        # No response sent — should timeout
        with pytest.raises(CyberwaveError, match="Timed out"):
            camera.edge_photo(timeout=0.2)

    def test_raises_on_edge_error_response(self):
        twin, mqtt = _make_twin_with_mqtt()
        camera = TwinCameraHandle(twin)

        def respond():
            time.sleep(0.05)
            handler = mqtt._test_handlers.get(
                "cyberwave/twin/twin-uuid/camera/photo"
            )
            if handler:
                handler(
                    json.dumps({
                        "status": "error",
                        "message": "No camera frame available",
                    })
                )

        t = threading.Thread(target=respond)
        t.start()

        with pytest.raises(CyberwaveError, match="No camera frame available"):
            camera.edge_photo(timeout=2.0)

        t.join()

    def test_raises_on_missing_image_field(self):
        twin, mqtt = _make_twin_with_mqtt()
        camera = TwinCameraHandle(twin)

        def respond():
            time.sleep(0.05)
            handler = mqtt._test_handlers.get(
                "cyberwave/twin/twin-uuid/camera/photo"
            )
            if handler:
                handler(json.dumps({"format": "jpeg"}))

        t = threading.Thread(target=respond)
        t.start()

        with pytest.raises(CyberwaveError, match="missing"):
            camera.edge_photo(timeout=2.0)

        t.join()

    def test_unsubscribes_after_response(self):
        twin, mqtt = _make_twin_with_mqtt()
        camera = TwinCameraHandle(twin)

        def respond():
            time.sleep(0.05)
            handler = mqtt._test_handlers.get(
                "cyberwave/twin/twin-uuid/camera/photo"
            )
            if handler:
                handler(json.dumps({"image": FAKE_B64, "format": "jpeg"}))

        t = threading.Thread(target=respond)
        t.start()

        camera.edge_photo(format="bytes")
        t.join()

        mqtt.unsubscribe.assert_called_once_with(
            "cyberwave/twin/twin-uuid/camera/photo"
        )

    def test_unsubscribes_on_timeout(self):
        twin, mqtt = _make_twin_with_mqtt()
        camera = TwinCameraHandle(twin)

        with pytest.raises(CyberwaveError):
            camera.edge_photo(timeout=0.1)

        mqtt.unsubscribe.assert_called_once_with(
            "cyberwave/twin/twin-uuid/camera/photo"
        )

    def test_respects_topic_prefix(self):
        twin, mqtt = _make_twin_with_mqtt()
        twin.client.config.topic_prefix = "custom/"
        camera = TwinCameraHandle(twin)

        def respond():
            time.sleep(0.05)
            handler = mqtt._test_handlers.get(
                "custom/cyberwave/twin/twin-uuid/camera/photo"
            )
            if handler:
                handler(json.dumps({"image": FAKE_B64, "format": "jpeg"}))

        t = threading.Thread(target=respond)
        t.start()

        camera.edge_photo(format="bytes")
        t.join()

        call_args = mqtt.publish.call_args
        assert call_args[0][0] == "custom/cyberwave/twin/twin-uuid/command"


# ---------------------------------------------------------------------------
# edge_photos tests
# ---------------------------------------------------------------------------


class TestEdgePhotos:
    def test_captures_multiple_frames(self):
        twin, mqtt = _make_twin_with_mqtt()
        camera = TwinCameraHandle(twin)

        call_count = 0

        def respond():
            nonlocal call_count
            time.sleep(0.05)
            handler = mqtt._test_handlers.get(
                "cyberwave/twin/twin-uuid/camera/photo"
            )
            if handler:
                call_count += 1
                handler(
                    json.dumps({"image": FAKE_B64, "format": "jpeg"})
                )

        frames = []
        for _ in range(3):
            t = threading.Thread(target=respond)
            t.start()
            frames.append(camera.edge_photo(format="bytes", timeout=2.0))
            t.join()

        assert len(frames) == 3
        assert all(f == FAKE_JPEG for f in frames)

    def test_edge_photos_convenience(self):
        twin, mqtt = _make_twin_with_mqtt()
        camera = TwinCameraHandle(twin)

        # Use a polling responder that keeps trying until it finds a handler
        def poll_respond(stop_event):
            while not stop_event.is_set():
                handler = mqtt._test_handlers.get(
                    "cyberwave/twin/twin-uuid/camera/photo"
                )
                if handler:
                    handler(json.dumps({"image": FAKE_B64, "format": "jpeg"}))
                time.sleep(0.01)

        stop = threading.Event()
        t = threading.Thread(target=poll_respond, args=(stop,))
        t.start()

        try:
            frames = camera.edge_photos(count=2, interval_ms=50, format="bytes", timeout=2.0)
            assert len(frames) == 2
        finally:
            stop.set()
            t.join()
