"""Tests for cyberwave.vision.annotate."""

import json

import numpy as np
import pytest

from cyberwave.models.types import BoundingBox, Detection
from cyberwave.vision.annotate import (
    OVERLAY_PAYLOAD_VERSION,
    _contrast_text_colour,
    _default_color_for,
    annotate_detections,
    build_overlay_payload,
)

cv2 = pytest.importorskip("cv2")


def _make_det(
    *,
    label: str = "person",
    bbox: tuple[float, float, float, float] = (10.0, 10.0, 50.0, 80.0),
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        label=label,
        confidence=confidence,
        bbox=BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
    )


class TestAnnotateDetections:
    def test_does_not_mutate_input_by_default(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = _make_det()
        out = annotate_detections(frame, [det])
        assert out is not frame
        # Source frame still all-zero.
        assert (frame == 0).all()

    def test_inplace_true_mutates_input(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = _make_det()
        out = annotate_detections(frame, [det], inplace=True)
        assert out is frame

    def test_draws_a_bounding_box(self):
        """A coloured rectangle should appear along the bbox edges."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = _make_det(bbox=(20, 20, 60, 60))
        out = annotate_detections(frame, [det], font_scale=0)  # box only
        # Top edge (y=20) along the bbox should have non-zero pixels.
        assert np.any(out[20, 25:55] > 0)
        # Centre of the box is hollow.
        assert np.array_equal(out[40, 40], np.zeros(3, dtype=np.uint8))

    def test_line_width_zero_skips_box(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = _make_det(bbox=(20, 20, 60, 60))
        out = annotate_detections(frame, [det], line_width=0, font_scale=0)
        # No box, no caption → frame still empty.
        assert (out == 0).all()

    def test_caption_drawn_above_box_when_room(self):
        """The caption sits above the box. Pixels just above y1 should be
        non-zero (background fill) when the box has room above."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        det = _make_det(bbox=(40, 80, 120, 150))  # plenty of headroom
        out = annotate_detections(frame, [det])
        # A pixel a few rows above the box top should be coloured (caption bg).
        assert np.any(out[75, 40:120] > 0)

    def test_caption_clamped_inside_when_box_at_top(self):
        """When the box hugs the top of the frame, the caption is tucked
        inside the box instead of falling off-frame."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        det = _make_det(bbox=(40, 0, 120, 80))
        out = annotate_detections(frame, [det])
        # Top row is the bbox edge; rows just below y=0 should be coloured
        # by the caption background since there is no headroom.
        assert np.any(out[5, 40:120] > 0)

    def test_font_scale_zero_skips_caption(self):
        """Caption-suppression mode: only the box is drawn."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        det = _make_det(bbox=(40, 80, 120, 150))
        out = annotate_detections(frame, [det], font_scale=0)
        # No caption background just above the box.
        assert not np.any(out[75, 40:120] > 0)
        # But the box edge is still drawn.
        assert np.any(out[80, 40:120] > 0)

    def test_labels_filter_skips_unlisted_detections(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        person = _make_det(label="person", bbox=(10, 10, 30, 30))
        car = _make_det(label="car", bbox=(50, 50, 80, 80))
        out = annotate_detections(frame, [person, car], labels=["person"])
        # Person box drawn.
        assert np.any(out[10, 10:30] > 0)
        # Car box not drawn (no pixels along its edges).
        assert not np.any(out[50, 50:80] > 0)

    def test_labels_none_draws_every_detection(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        person = _make_det(label="person", bbox=(10, 10, 30, 30))
        car = _make_det(label="car", bbox=(50, 50, 80, 80))
        out = annotate_detections(frame, [person, car])  # labels=None default
        assert np.any(out[10, 10:30] > 0)
        assert np.any(out[50, 50:80] > 0)

    def test_labels_empty_iterable_draws_nothing(self):
        """Empty target set is a valid noop — the codegen path may pass
        ``target_classes=[]`` and we shouldn't raise or draw."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        det = _make_det(bbox=(10, 10, 30, 30))
        out = annotate_detections(frame, [det], labels=[])
        assert (out == 0).all()

    def test_labels_accepts_arbitrary_iterable(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        person = _make_det(label="person", bbox=(10, 10, 30, 30))
        car = _make_det(label="car", bbox=(50, 50, 80, 80))

        def _gen():
            yield "person"
            yield "car"

        out = annotate_detections(frame, [person, car], labels=_gen())
        assert np.any(out[10, 10:30] > 0)
        assert np.any(out[50, 50:80] > 0)

    def test_color_fn_overrides_palette(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        det = _make_det(bbox=(40, 40, 80, 80))
        out = annotate_detections(
            frame,
            [det],
            font_scale=0,
            line_width=2,
            color_fn=lambda _det: (0, 255, 0),
        )
        # Top edge of the box should contain green pixels.
        top_row = out[40, 40:80]
        assert (top_row == [0, 255, 0]).all(axis=-1).any()

    def test_show_confidence_false_hides_score(self):
        """Caption width changes with the text, so suppressing the score
        changes the caption-background extent. We compare the painted
        widths between show_confidence=True/False."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        det = _make_det(bbox=(40, 80, 120, 150), confidence=0.99, label="person")
        with_score = annotate_detections(frame, [det], show_confidence=True)
        without_score = annotate_detections(frame, [det], show_confidence=False)
        # Caption background sits above the box (row ~73). Without the
        # confidence number the caption is narrower.
        wide = int(np.any(with_score[75] > 0, axis=-1).sum())
        narrow = int(np.any(without_score[75] > 0, axis=-1).sum())
        assert narrow < wide
        assert narrow > 0

    def test_skips_zero_area_bbox(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = _make_det(bbox=(20, 20, 20, 20))
        out = annotate_detections(frame, [det])
        np.testing.assert_array_equal(out, frame)
        assert out is not frame  # still copied

    def test_clamps_oversize_bbox(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = _make_det(bbox=(-50, -50, 200, 200))
        out = annotate_detections(frame, [det], font_scale=0)
        # Clamped box edge somewhere along the frame border.
        assert np.any(out[0, :] > 0) or np.any(out[:, 0] > 0)

    def test_invalid_frame_shape_raises(self):
        with pytest.raises(ValueError, match="at least 2-D"):
            annotate_detections(np.zeros(10, dtype=np.uint8), [])

    def test_empty_detections_returns_unchanged_copy(self):
        frame = np.full((50, 50, 3), 42, dtype=np.uint8)
        out = annotate_detections(frame, [])
        np.testing.assert_array_equal(out, frame)
        assert out is not frame


class TestDefaultColorFor:
    def test_same_label_yields_same_colour(self):
        """Visual continuity guarantee — the codegen template relies on
        each class being painted the same colour every frame."""
        assert _default_color_for("person") == _default_color_for("person")
        assert _default_color_for("car") == _default_color_for("car")

    def test_different_labels_likely_yield_different_colours(self):
        """Not strictly guaranteed by hashing but should hold for the
        common YOLO classes — collisions here would be surprising in
        production review screens."""
        seen = {
            _default_color_for(label)
            for label in ("person", "car", "truck", "bicycle", "dog", "cat")
        }
        assert len(seen) >= 4  # tolerate up to two collisions

    def test_returns_palette_member(self):
        from cyberwave.vision.annotate import _DEFAULT_PALETTE

        assert _default_color_for("anything") in _DEFAULT_PALETTE


class TestContrastTextColour:
    def test_dark_background_picks_white_text(self):
        assert _contrast_text_colour((0, 0, 0)) == (255, 255, 255)
        assert _contrast_text_colour((50, 50, 50)) == (255, 255, 255)

    def test_light_background_picks_black_text(self):
        assert _contrast_text_colour((255, 255, 255)) == (0, 0, 0)
        # Yellow (0, 255, 255 in BGR) is high-luma → black text.
        assert _contrast_text_colour((0, 255, 255)) == (0, 0, 0)


class TestBuildOverlayPayload:
    """``build_overlay_payload`` is the JSON wire-contract the
    ``annotate`` workflow node publishes on
    ``FRAME_OVERLAY_CHANNEL`` for the camera driver to composite. The
    cases below pin schema shape and the label-filter behaviour
    codegen relies on (the only behaviour the driver actually
    branches on)."""

    def test_schema_shape_and_versioning(self):
        person = _make_det(label="person", bbox=(10, 10, 50, 80), confidence=0.9)
        car = _make_det(label="car", bbox=(60, 60, 120, 100), confidence=0.5)
        payload = build_overlay_payload([person, car])

        assert payload["v"] == OVERLAY_PAYLOAD_VERSION
        assert payload["style"] == {
            "line_width": 2,
            "font_scale": 0.5,  # font_size=14 / 28
            "show_confidence": True,
        }
        assert payload["boxes"] == [
            {"box_2d": [10.0, 10.0, 50.0, 80.0], "label": "person", "conf": 0.9},
            {"box_2d": [60.0, 60.0, 120.0, 100.0], "label": "car", "conf": 0.5},
        ]
        # Round-trips cleanly through json — the camera driver
        # consumes it as raw JSON bytes off Zenoh.
        assert json.loads(json.dumps(payload)) == payload

    def test_labels_filter_excludes_other_classes(self):
        person = _make_det(label="person")
        car = _make_det(label="car", bbox=(60, 60, 120, 100))
        payload = build_overlay_payload([person, car], labels=["person"])
        assert [b["label"] for b in payload["boxes"]] == ["person"]

    def test_labels_empty_iterable_excludes_everything(self):
        # Mirrors the ``annotate_detections`` behaviour — codegen may
        # pass ``target_classes=[]`` and we shouldn't raise or include.
        payload = build_overlay_payload([_make_det()], labels=[])
        assert payload["boxes"] == []

    def test_font_scale_overrides_font_size(self):
        payload = build_overlay_payload([], font_size=14, font_scale=1.25)
        assert payload["style"]["font_scale"] == 1.25
