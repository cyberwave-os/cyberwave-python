"""Unit tests for the camera overlay helpers (``cyberwave.driver.sensors.overlays``).

These helpers are pure-Python and do not depend on a running camera, Zenoh
router, or the Cyberwave backend, so we can exercise them directly.
"""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("cv2")

from cyberwave.driver.sensors.overlays import (  # noqa: E402
    DetectionBox,
    DetectionCache,
    OverlayCache,
    draw_detections,
    draw_overlay,
    overlay_color_from_box,
    parse_detections_payload,
    parse_overlay_payload,
    validate_overlay_payload,
)

# ── parse_detections_payload ──────────────────────────────────────


def test_parse_valid_payload() -> None:
    payload = json.dumps(
        {
            "detections": [
                {"label": "person", "confidence": 0.92, "x1": 10, "y1": 20, "x2": 100, "y2": 200},
                {"label": "car", "confidence": 0.77, "x1": 150, "y1": 30, "x2": 400, "y2": 280},
            ],
            "frame_width": 1920,
            "frame_height": 1080,
        }
    ).encode()

    result = parse_detections_payload(payload)
    assert result is not None
    boxes, w, h = result

    assert (w, h) == (1920, 1080)
    assert [b.label for b in boxes] == ["person", "car"]
    assert boxes[0] == DetectionBox("person", 0.92, 10, 20, 100, 200)


def test_parse_empty_detections_list() -> None:
    payload = json.dumps({"detections": [], "frame_width": 640, "frame_height": 480}).encode()
    result = parse_detections_payload(payload)
    assert result == ([], 640, 480)


def test_parse_float_coordinates_truncates() -> None:
    # ONNX exports sometimes emit float pixel coords; matches OBSBOT C++ behavior.
    payload = json.dumps(
        {
            "detections": [
                {"label": "person", "confidence": 0.9, "x1": 10.7, "y1": 20.3, "x2": 100.9, "y2": 200.1}
            ],
            "frame_width": 640,
            "frame_height": 480,
        }
    ).encode()

    result = parse_detections_payload(payload)
    assert result is not None
    boxes, _, _ = result
    assert boxes[0] == DetectionBox("person", 0.9, 10, 20, 100, 200)


def test_parse_invalid_json_returns_none() -> None:
    assert parse_detections_payload(b"not json") is None


def test_parse_missing_fields_uses_defaults() -> None:
    payload = json.dumps({"detections": [{"confidence": 0.5}]}).encode()
    result = parse_detections_payload(payload)
    assert result is not None
    boxes, w, h = result
    assert (w, h) == (0, 0)
    assert boxes[0] == DetectionBox("?", 0.5, 0, 0, 0, 0)


def test_parse_skips_non_dict_detections() -> None:
    payload = json.dumps(
        {"detections": ["not a dict", 42, {"label": "person", "x1": 1, "y1": 2, "x2": 3, "y2": 4}]}
    ).encode()
    result = parse_detections_payload(payload)
    assert result is not None
    boxes, _, _ = result
    assert len(boxes) == 1
    assert boxes[0].label == "person"


# ── DetectionCache ────────────────────────────────────────────────


def test_cache_empty_returns_none() -> None:
    assert DetectionCache().snapshot() is None


def test_cache_returns_fresh_batch() -> None:
    cache = DetectionCache()
    box = DetectionBox("dog", 0.8, 1, 2, 3, 4)
    cache.update([box], 640, 480)

    snap = cache.snapshot()
    assert snap is not None
    boxes, w, h = snap
    assert boxes == [box]
    assert (w, h) == (640, 480)


def test_cache_returns_none_when_stale() -> None:
    cache = DetectionCache()
    cache.update([DetectionBox("dog", 0.8, 1, 2, 3, 4)], 640, 480)

    # Cache staleness is hardcoded to 1000 ms; fake the clock to jump 2s.
    with patch("cyberwave.driver.sensors.overlays.time.monotonic", return_value=time.monotonic() + 2.0):
        assert cache.snapshot() is None


def test_cache_update_replaces_prior_batch() -> None:
    cache = DetectionCache()
    cache.update([DetectionBox("dog", 0.5, 0, 0, 10, 10)], 640, 480)
    new_box = DetectionBox("cat", 0.9, 5, 5, 15, 15)
    cache.update([new_box], 1920, 1080)

    snap = cache.snapshot()
    assert snap is not None
    boxes, w, h = snap
    assert boxes == [new_box]
    assert (w, h) == (1920, 1080)


# ── draw_detections ───────────────────────────────────────────────


def test_draw_is_noop_on_empty_list() -> None:
    pytest.importorskip("cv2")
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    original = frame.copy()
    draw_detections(frame, [], 100, 100)
    assert np.array_equal(frame, original)


def test_draw_marks_bbox_corners_green() -> None:
    pytest.importorskip("cv2")
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    box = DetectionBox("person", 0.9, 50, 60, 150, 160)
    draw_detections(frame, [box], 300, 200)

    # All four bbox corners should be green (OBSBOT C++ draw color, BGR).
    green = np.array([0, 255, 0], dtype=np.uint8)
    for x, y in [(box.x1, box.y1), (box.x2, box.y1), (box.x1, box.y2), (box.x2, box.y2)]:
        assert np.array_equal(frame[y, x], green), f"pixel at ({x},{y}) not green: {frame[y, x]}"


def test_draw_keeps_label_on_frame_for_near_top_bboxes() -> None:
    # Regression: a bbox with y1=0 used to draw its label background at
    # y = -text_h - 6, silently clipped by OpenCV.  The label must now land
    # inside the frame.
    pytest.importorskip("cv2")
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    box = DetectionBox("person", 0.9, 10, 0, 100, 80)

    draw_detections(frame, [box], 200, 120)

    # Label area sits immediately below y1=0 when there's no room above, so a
    # horizontal strip of green pixels should exist in the first ~20 rows.
    green_mask = np.all(frame[: 25, box.x1 + 2 : box.x1 + 20] == (0, 255, 0), axis=-1)
    assert green_mask.any(), "label background not rendered inside frame for top-edge bbox"


def test_cache_snapshot_returns_defensive_copy() -> None:
    # Regression: snapshot() used to return the cache's internal list by
    # reference, so a caller could corrupt the cache by mutating the result.
    cache = DetectionCache()
    cache.update([DetectionBox("dog", 0.8, 1, 2, 3, 4)], 640, 480)

    snap = cache.snapshot()
    assert snap is not None
    boxes, _, _ = snap
    boxes.clear()

    snap_again = cache.snapshot()
    assert snap_again is not None
    assert len(snap_again[0]) == 1, "mutation through snapshot leaked into cache"


def test_draw_scales_from_detection_frame_to_capture_frame() -> None:
    pytest.importorskip("cv2")
    import cv2

    # Detection frame 300x200; capture frame 600x400 (2x scale).
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    box = DetectionBox("person", 0.9, 50, 60, 150, 160)

    with patch.object(cv2, "rectangle", wraps=cv2.rectangle) as mock_rect:
        draw_detections(frame, [box], 300, 200)

    # First call is the bbox rectangle — should be scaled 2x.
    args, _kwargs = mock_rect.call_args_list[0]
    _frame_arg, pt1, pt2, _color, _thickness = args
    assert pt1 == (100, 120)
    assert pt2 == (300, 320)



# ── Workflow overlay (annotate-node) ───────────────────────────────


def _overlay_payload_bytes(
    *,
    boxes: list[dict] | None = None,
    line_width: int = 4,
    font_scale: float = 0.5,
    show_confidence: bool = True,
) -> bytes:
    """Build a v1-schema overlay payload matching what
    ``cyberwave.vision.build_overlay_payload`` produces. Kept inline
    so the test doesn't have to import the SDK."""
    return json.dumps(
        {
            "v": 1,
            "boxes": boxes
            if boxes is not None
            else [{"box_2d": [10.0, 20.0, 110.0, 120.0], "label": "person", "conf": 0.92}],
            "style": {
                "line_width": line_width,
                "font_scale": font_scale,
                "show_confidence": show_confidence,
            },
        }
    ).encode()


def test_parse_overlay_valid_payload_round_trips_style() -> None:
    parsed = parse_overlay_payload(_overlay_payload_bytes())
    assert parsed is not None
    assert parsed["v"] == 1
    assert parsed["boxes"][0]["label"] == "person"
    assert parsed["style"]["line_width"] == 4


def test_parse_overlay_rejects_unknown_schema_version() -> None:
    # Forward-compat: a future bump must not crash, just stop drawing.
    payload = json.dumps({"v": 99, "boxes": [], "style": {}}).encode()
    assert parse_overlay_payload(payload) is None


def test_parse_overlay_rejects_invalid_json_or_shape() -> None:
    assert parse_overlay_payload(b"not json") is None
    assert parse_overlay_payload(b"[]") is None  # not a dict
    assert parse_overlay_payload(b'{"v":1,"boxes":"oops"}') is None


def test_overlay_cache_returns_fresh_then_stale() -> None:
    cache = OverlayCache()
    assert cache.snapshot() is None
    cache.update({"v": 1, "boxes": [], "style": {}})
    assert cache.snapshot() is not None
    with patch("cyberwave.driver.sensors.overlays.time.monotonic", return_value=time.monotonic() + 5.0):
        assert cache.snapshot() is None


def test_draw_overlay_respects_payload_line_width() -> None:
    pytest.importorskip("cv2")
    import cv2

    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    payload = json.loads(_overlay_payload_bytes(line_width=7, font_scale=0))
    with patch.object(cv2, "rectangle", wraps=cv2.rectangle) as mock_rect:
        draw_overlay(frame, payload)
    # ``font_scale=0`` skips the caption pass, so the only rectangle
    # call is the bbox itself — and its thickness must come from the
    # payload, not a driver-side default.
    args, _kwargs = mock_rect.call_args_list[0]
    _frame_arg, pt1, pt2, _color, thickness = args
    assert pt1 == (10, 20)
    assert pt2 == (110, 120)
    assert thickness == 7


def test_draw_overlay_renders_polygon_fill_inside_polygon() -> None:
    pytest.importorskip("cv2")
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    payload = {
        "v": 1,
        "boxes": [
            {
                "box_2d": [20.0, 20.0, 60.0, 60.0],
                "label": "person",
                "conf": 0.9,
                "polygon": [[20, 20], [60, 20], [60, 60], [20, 60]],
            }
        ],
        "style": {
            "line_width": 0,
            "font_scale": 0.0,
            "show_confidence": False,
            "mask_alpha": 0.5,
            "mask_outline": False,
        },
    }
    draw_overlay(frame, payload)
    assert frame[40, 40].any(), "Pixel inside polygon should be coloured by mask fill"
    assert not frame[5, 5].any(), "Pixel outside polygon should stay zero"


def test_draw_overlay_polygon_no_fill_when_alpha_zero_but_outline_draws() -> None:
    pytest.importorskip("cv2")
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    payload = {
        "v": 1,
        "boxes": [
            {
                "box_2d": [20.0, 20.0, 60.0, 60.0],
                "label": "person",
                "conf": 0.9,
                "polygon": [[20, 20], [60, 20], [60, 60], [20, 60]],
            }
        ],
        "style": {
            "line_width": 1,
            "font_scale": 0.0,
            "show_confidence": False,
            "mask_alpha": 0.0,
            "mask_outline": True,
        },
    }
    draw_overlay(frame, payload)
    # Interior should be zero (no fill), edge should have a pixel from polylines.
    assert not frame[40, 40].any()
    assert frame[20, 30:50].any()


def test_overlay_color_fallback_matches_sdk_label_color() -> None:
    """``overlay_color_from_box`` fallback (no ``color`` field) must return
    the same colour as ``cyberwave.vision.annotate.label_color``.

    The driver no longer keeps a local palette copy — it delegates directly
    to the SDK, so parity is structural.  This test guards against the
    import path breaking or the SDK's public API being renamed.
    """
    pytest.importorskip("cyberwave.vision.annotate")

    from cyberwave.vision.annotate import label_color

    for label in ("person", "car", "Bicycle", "traffic light", "café"):
        box_without_color = {"label": label, "conf": 0.9}
        assert overlay_color_from_box(box_without_color) == label_color(label), label


def test_overlay_color_falls_back_to_default_when_label_color_missing() -> None:
    """Driver must degrade to green (not raise) when the installed SDK
    predates ``cyberwave.vision.annotate.label_color``."""
    import builtins


    real_import = builtins.__import__

    def _import_without_label_color(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "cyberwave.vision.annotate" and "label_color" in (fromlist or ()):
            raise ImportError(
                "cannot import name 'label_color' from 'cyberwave.vision.annotate'"
            )
        return real_import(name, globals, locals, fromlist, level)

    with patch.object(builtins, "__import__", _import_without_label_color):
        assert overlay_color_from_box({"label": "person", "conf": 0.9}) == (0, 200, 0)
        assert overlay_color_from_box({"label": "car", "conf": 0.5}) == (0, 200, 0)
        # A pre-resolved ``color`` short-circuits before the import.
        assert overlay_color_from_box(
            {"label": "person", "color": [10, 20, 30]}
        ) == (10, 20, 30)


def test_draw_overlay_skips_zero_area_and_unknown_box_shape() -> None:
    pytest.importorskip("cv2")
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    payload = json.loads(
        _overlay_payload_bytes(
            boxes=[
                {"box_2d": [10, 10, 10, 10], "label": "a", "conf": 0.5},  # zero area
                {"box_2d": [10, 10, 5, 5], "label": "b", "conf": 0.5},  # inverted
                {"box_2d": [10, 10], "label": "c", "conf": 0.5},  # malformed
            ]
        )
    )
    draw_overlay(frame, payload)
    # No drawable boxes → frame stays untouched.
    assert (frame == 0).all()


def test_overlay_cache_snapshot_returns_defensive_copy() -> None:
    """Mirrors ``test_cache_snapshot_returns_defensive_copy`` for the
    detection cache. A caller that mutates the snapshot dict (e.g.
    ``parsed["boxes"].sort(...)``) must not corrupt the cache that the
    encoder thread reads on the next frame."""
    cache = OverlayCache()
    cache.update({"v": 1, "boxes": [{"label": "person"}], "style": {}})

    snap = cache.snapshot()
    assert snap is not None
    snap.clear()  # would corrupt the cache if returned by reference

    snap_again = cache.snapshot()
    assert snap_again is not None
    assert snap_again["v"] == 1, "mutation through snapshot leaked into cache"


# ── Subscriber callback contract: bus delivers decoded dicts ───────


def testvalidate_overlay_payload_accepts_decoded_dict() -> None:
    """Documents the contract: ``data_bus.subscribe`` (default
    ``raw=False``) hands the callback the **decoded dict**, not a
    ``Sample`` with raw bytes. Validate accepts the dict directly so
    the subscriber wiring matches the bus's actual delivery format."""
    decoded = {
        "v": 1,
        "boxes": [{"box_2d": [10, 20, 100, 200], "label": "person", "conf": 0.9}],
        "style": {"line_width": 2, "font_scale": 0.5, "show_confidence": True},
    }
    parsed = validate_overlay_payload(decoded)
    assert parsed is not None
    assert parsed["boxes"][0]["label"] == "person"


def testvalidate_overlay_payload_rejects_non_dict_and_unknown_schema() -> None:
    assert validate_overlay_payload("not a dict") is None
    assert validate_overlay_payload(None) is None
    assert validate_overlay_payload({"v": 99, "boxes": []}) is None
    assert validate_overlay_payload({"v": 1, "boxes": "oops"}) is None


def test_subscribe_publish_round_trip_updates_cache() -> None:
    """End-to-end regression for the wire contract: a worker calling
    ``cw.data.publish(FRAME_OVERLAY_CHANNEL, build_overlay_payload(...))``
    must drive the driver's overlay cache via the SAME callback
    signature ``CameraStreamMixin`` registers.

    This test would have caught the original bug where the callback
    expected ``sample.payload`` (raw bytes) but the bus delivers the
    already-decoded dict."""
    pytest.importorskip("cyberwave.data")

    from cyberwave.data import FRAME_OVERLAY_CHANNEL
    from cyberwave.data.api import DataBus
    from cyberwave.data.backend import DataBackend, Sample, Subscription

    class _MemBackend(DataBackend):
        """Synchronous in-process backend — invokes subscribers inline
        on publish so the round-trip is deterministic with no thread."""

        def __init__(self) -> None:
            self._subs: dict[str, list] = {}

        def publish(self, channel, payload, *, metadata=None):
            for cb in self._subs.get(channel, []):
                cb(Sample(channel=channel, payload=payload, metadata=metadata))

        def subscribe(self, channel, callback, *, policy="latest"):
            self._subs.setdefault(channel, []).append(callback)
            return Subscription()

        def latest(self, channel, *, timeout_s=1.0):
            return None

        def close(self) -> None: ...

    bus = DataBus(_MemBackend(), twin_uuid="00000000-0000-0000-0000-000000000c01")
    cache = OverlayCache()

    # Register the SAME callback shape the driver uses (decoded dict).
    def _on_decoded(decoded: object) -> None:
        parsed = validate_overlay_payload(decoded)
        if parsed is not None:
            cache.update(parsed)

    bus.subscribe(FRAME_OVERLAY_CHANNEL, _on_decoded)

    # Publish exactly what the codegen emits.
    bus.publish(
        FRAME_OVERLAY_CHANNEL,
        {
            "v": 1,
            "boxes": [{"box_2d": [1, 2, 3, 4], "label": "person", "conf": 0.9}],
            "style": {"line_width": 2, "font_scale": 0.5, "show_confidence": True},
        },
    )

    snap = cache.snapshot()
    assert snap is not None, "callback never updated cache — broken wire contract"
    assert snap["boxes"][0]["label"] == "person"
