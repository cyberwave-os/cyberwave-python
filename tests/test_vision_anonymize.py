"""Tests for cyberwave.vision.anonymize."""

import numpy as np
import pytest

from cyberwave.models.types import BoundingBox, Detection
from cyberwave.vision.anonymize import (
    COCO17_EDGE_GROUPS,
    COCO17_SEGMENT_COLORS,
    COCO17_SKELETON,
    anonymize_frame,
    blank_persons,
    draw_skeleton,
)

# Skip all tests if cv2 isn't installed in this environment.
cv2 = pytest.importorskip("cv2")


def _make_det(
    *,
    label: str = "person",
    bbox: tuple[float, float, float, float] = (10.0, 10.0, 50.0, 80.0),
    keypoints: np.ndarray | None = None,
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        label=label,
        confidence=confidence,
        bbox=BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
        keypoints=keypoints,
    )


class TestBlankPersons:
    def test_does_not_mutate_input(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        det = _make_det()
        out = blank_persons(frame, [det], mode="bbox")
        # Input should still be all white.
        assert frame[20, 20, 0] == 255
        assert out[20, 20, 0] == 0  # bbox region blacked out

    def test_bbox_mode_fills_color(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        det = _make_det()
        out = blank_persons(frame, [det], mode="bbox", color=(10, 20, 30))
        # Inside bbox: filled.
        assert tuple(out[20, 20]) == (10, 20, 30)
        # Outside bbox: untouched.
        assert tuple(out[5, 5]) == (255, 255, 255)
        assert tuple(out[90, 90]) == (255, 255, 255)

    def test_blur_mode_changes_pixels(self):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, size=(100, 100, 3), dtype=np.uint8)
        before = frame[20, 20].copy()
        det = _make_det()
        out = blank_persons(frame, [det], mode="blur", blur_kernel=15)
        # With high noise + heavy blur, inside-bbox pixels should differ.
        # Compare the centre of the bbox region.
        assert not np.array_equal(out[40, 30], before)
        # Outside bbox: untouched.
        np.testing.assert_array_equal(out[5, 5], frame[5, 5])

    def test_pixelate_mode_quantises_pixels(self):
        """Pixelate must reduce the number of unique colour values inside
        the bbox region. Random noise → at most one colour per mosaic block."""
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, size=(100, 100, 3), dtype=np.uint8)
        det = _make_det(bbox=(10, 10, 90, 90))  # 80x80 ROI
        out = blank_persons(frame, [det], mode="pixelate", pixel_size=10)
        roi = out[10:90, 10:90]
        n_unique_in = len(np.unique(frame[10:90, 10:90].reshape(-1, 3), axis=0))
        n_unique_out = len(np.unique(roi.reshape(-1, 3), axis=0))
        # 80x80 / (10x10) = 64 blocks → at most 64 unique colours.
        assert n_unique_out <= 64
        # But strictly less than the original noise.
        assert n_unique_out < n_unique_in
        # Outside untouched.
        np.testing.assert_array_equal(out[5, 5], frame[5, 5])

    def test_pixelate_mode_adaptive_pixel_size(self):
        """When pixel_size is None, the block size scales with the bbox."""
        rng = np.random.default_rng(1)
        frame = rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8)
        # Should not raise on either tiny or large bboxes.
        det_small = _make_det(bbox=(10, 10, 30, 30))  # 20x20
        det_large = _make_det(bbox=(20, 20, 180, 180))  # 160x160
        out = blank_persons(frame, [det_small, det_large], mode="pixelate")
        # Both regions modified.
        assert not np.array_equal(out[20, 20], frame[20, 20])
        assert not np.array_equal(out[100, 100], frame[100, 100])

    def test_pixelate_mode_falls_back_to_fill_for_tiny_bbox(self):
        """A 2x2 bbox cannot be meaningfully pixelated — the helper falls
        back to a solid fill so the pixels are still hidden."""
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        det = _make_det(bbox=(20, 20, 22, 22))  # 2x2 ROI
        out = blank_persons(frame, [det], mode="pixelate", color=(11, 22, 33))
        assert tuple(out[20, 20]) == (11, 22, 33)
        assert tuple(out[23, 23]) == (255, 255, 255)

    def test_redact_mode_paints_solid_color_with_visible_grid(self):
        """Redact must (a) destroy underlying pixel info and (b) leave
        a visible grid so the result is distinguishable from a flat bbox."""
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, size=(100, 100, 3), dtype=np.uint8)
        det = _make_det(bbox=(10, 10, 90, 90))  # 80x80 ROI
        out = blank_persons(frame, [det], mode="redact", color=(0, 0, 0), pixel_size=10)
        roi = out[10:90, 10:90]

        unique = np.unique(roi.reshape(-1, 3), axis=0)
        # Block fill (black) + grid fill (40,40,40).
        assert len(unique) == 2
        assert (np.array([0, 0, 0]) == unique).all(axis=1).any()
        assert (np.array([40, 40, 40]) == unique).all(axis=1).any()

        # Grid lines run every `pixel_size` pixels inside the ROI, so a
        # pixel one block in must be on a separator while the centre of
        # the first block must be solid color.
        assert tuple(out[10, 20]) == (40, 40, 40)  # vertical grid line
        assert tuple(out[20, 10]) == (40, 40, 40)  # horizontal grid line
        assert tuple(out[14, 14]) == (0, 0, 0)  # interior of a block

        # Outside the bbox is untouched.
        np.testing.assert_array_equal(out[5, 5], frame[5, 5])

    def test_redact_mode_falls_back_to_fill_for_tiny_bbox(self):
        """A 2x2 ROI is below the pixelate/redact min-side guard and must
        fall back to the solid-fill path without raising."""
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        det = _make_det(bbox=(20, 20, 22, 22))
        out = blank_persons(
            frame, [det], mode="redact", color=(11, 22, 33), pixel_size=10
        )
        assert tuple(out[20, 20]) == (11, 22, 33)
        assert tuple(out[23, 23]) == (255, 255, 255)

    def test_only_processes_matching_label(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        person = _make_det(label="person", bbox=(10, 10, 50, 50))
        car = _make_det(label="car", bbox=(60, 60, 90, 90))
        out = blank_persons(frame, [person, car], mode="bbox", label="person")
        # Person bbox: filled.
        assert tuple(out[20, 20]) == (0, 0, 0)
        # Car bbox: untouched.
        assert tuple(out[70, 70]) == (255, 255, 255)

    def test_clamps_oversize_bbox(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        # Bbox exceeds frame; should not raise.
        det = _make_det(bbox=(-50, -50, 200, 200))
        out = blank_persons(frame, [det], mode="bbox")
        assert tuple(out[50, 50]) == (0, 0, 0)

    def test_skips_zero_area_bbox(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        det = _make_det(bbox=(20, 20, 20, 20))  # zero-width
        out = blank_persons(frame, [det])
        np.testing.assert_array_equal(out, frame)

    def test_blur_mode_falls_back_to_fill_for_tiny_bbox(self):
        """A 2x2 bbox is smaller than any usable Gaussian kernel, so the
        blur path must not call cv2.GaussianBlur (which would raise) and
        must still hide the pixels."""
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        det = _make_det(bbox=(20, 20, 22, 22))  # 2x2 ROI
        out = blank_persons(frame, [det], mode="blur", color=(7, 8, 9))
        assert tuple(out[20, 20]) == (7, 8, 9)
        # Pixel just outside the ROI is untouched.
        assert tuple(out[23, 23]) == (255, 255, 255)

    def test_blur_kernel_is_clamped_to_odd_roi_side(self):
        """Even-sized ROI must not feed an even kernel into cv2.GaussianBlur."""
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 255, size=(100, 100, 3), dtype=np.uint8)
        det = _make_det(bbox=(10, 10, 16, 16))  # 6x6 even ROI
        # Should not raise even though 6 is even and < default blur_kernel=51.
        blank_persons(frame, [det], mode="blur")

    def test_invalid_mode_raises(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="mode must be"):
            blank_persons(frame, [], mode="garbage")

    def test_empty_detections_returns_unchanged_copy(self):
        frame = np.full((10, 10, 3), 123, dtype=np.uint8)
        out = blank_persons(frame, [])
        np.testing.assert_array_equal(out, frame)
        # But it should be a copy, not the same array.
        assert out is not frame

    def test_inplace_true_mutates_input_and_returns_it(self):
        """inplace=True skips the per-frame copy — the workhorse path for
        edge workers that own the buffer."""
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        det = _make_det()
        out = blank_persons(frame, [det], mode="bbox", inplace=True)
        assert out is frame
        assert tuple(frame[20, 20]) == (0, 0, 0)

    def test_inplace_false_default_does_not_mutate(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        det = _make_det()
        out = blank_persons(frame, [det], mode="bbox")  # default inplace=False
        assert out is not frame
        assert tuple(frame[20, 20]) == (255, 255, 255)


class TestBlankPersonsMultiLabel:
    """Coverage for the ``labels=`` multi-class API and its precedence over
    the legacy ``label=`` shim."""

    def test_labels_processes_every_listed_class(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        person = _make_det(label="person", bbox=(10, 10, 30, 30))
        car = _make_det(label="car", bbox=(40, 40, 60, 60))
        bike = _make_det(label="bicycle", bbox=(70, 70, 90, 90))
        out = blank_persons(
            frame,
            [person, car, bike],
            mode="bbox",
            labels=["person", "car"],
        )
        # Both listed classes blanked.
        assert tuple(out[20, 20]) == (0, 0, 0)
        assert tuple(out[50, 50]) == (0, 0, 0)
        # Bicycle (not listed) untouched.
        assert tuple(out[80, 80]) == (255, 255, 255)

    def test_labels_takes_precedence_over_label(self):
        """If both kwargs are given, ``labels`` wins; the legacy ``label``
        is silently ignored. This is what makes ``labels`` a clean shim."""
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        person = _make_det(label="person", bbox=(10, 10, 30, 30))
        car = _make_det(label="car", bbox=(40, 40, 60, 60))
        out = blank_persons(
            frame,
            [person, car],
            mode="bbox",
            label="person",  # would normally only blank persons
            labels=["car"],  # but this wins
        )
        assert tuple(out[20, 20]) == (255, 255, 255)  # person untouched
        assert tuple(out[50, 50]) == (0, 0, 0)  # car blanked

    def test_label_alone_still_works_for_backwards_compat(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        person = _make_det(label="person", bbox=(10, 10, 30, 30))
        car = _make_det(label="car", bbox=(40, 40, 60, 60))
        out = blank_persons(frame, [person, car], mode="bbox", label="car")
        assert tuple(out[20, 20]) == (255, 255, 255)  # person untouched
        assert tuple(out[50, 50]) == (0, 0, 0)  # car blanked

    def test_labels_accepts_arbitrary_iterable_not_just_list(self):
        """Tuples, sets, generators all need to work — the API hint is
        ``Iterable[str]`` so the implementation must not assume list."""
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        person = _make_det(label="person", bbox=(10, 10, 30, 30))
        car = _make_det(label="car", bbox=(40, 40, 60, 60))

        def _gen():
            yield "person"
            yield "car"

        out = blank_persons(frame, [person, car], mode="bbox", labels=_gen())
        assert tuple(out[20, 20]) == (0, 0, 0)
        assert tuple(out[50, 50]) == (0, 0, 0)

    def test_labels_empty_iterable_is_a_noop(self):
        """An empty target set is a valid request — return an unchanged copy
        without erroring so codegen / dynamic config can pass through."""
        frame = np.full((100, 100, 3), 123, dtype=np.uint8)
        person = _make_det(label="person", bbox=(10, 10, 30, 30))
        out = blank_persons(frame, [person], mode="bbox", labels=[])
        np.testing.assert_array_equal(out, frame)
        assert out is not frame


class TestDrawSkeleton:
    def test_handles_none_or_empty_keypoints(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        # None: returned unchanged.
        out = draw_skeleton(frame, None)  # type: ignore[arg-type]
        np.testing.assert_array_equal(out, frame)
        # Empty array: returned unchanged.
        out = draw_skeleton(frame, np.empty((0, 3)))
        np.testing.assert_array_equal(out, frame)

    def test_draws_lines_when_visibility_above_threshold(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        # All 17 keypoints visible at the same point — at least joint dots
        # should appear at (25, 25).
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[:, 0] = 25
        kps[:, 1] = 25
        kps[:, 2] = 0.9
        out = draw_skeleton(frame, kps, color=(0, 255, 0))
        assert (out[25, 25] == [0, 255, 0]).all()

    def test_skips_low_visibility_keypoints(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[:, 0] = 25
        kps[:, 1] = 25
        kps[:, 2] = 0.1  # below default threshold (0.3)
        out = draw_skeleton(frame, kps)
        np.testing.assert_array_equal(out, frame)

    def test_handles_fewer_keypoints_than_default_skeleton(self):
        # 5 keypoints, only edge (0,1) is fully present.
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        kps = np.array(
            [
                [10, 10, 0.9],
                [40, 40, 0.9],
                [25, 25, 0.9],
                [25, 25, 0.9],
                [25, 25, 0.9],
            ],
            dtype=np.float32,
        )
        out = draw_skeleton(frame, kps, color=(255, 0, 0), thickness=1, radius=0)
        # The line (0->1) traverses the diagonal — pixel at (25,25) should be set.
        assert (out[25, 25] == [255, 0, 0]).all()

    def test_keypoints_without_visibility_column(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        # (K, 2) — no visibility; all kps treated as visible.
        kps = np.full((17, 2), 25.0, dtype=np.float32)
        out = draw_skeleton(frame, kps, color=(0, 0, 255))
        assert (out[25, 25] == [0, 0, 255]).all()

    def test_does_not_mutate_input_by_default(self):
        """Match :func:`blank_persons` — the standalone helper must copy so
        callers don't accidentally clobber the source frame."""
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[:, 0] = 25
        kps[:, 1] = 25
        kps[:, 2] = 0.9
        out = draw_skeleton(frame, kps, color=(0, 255, 0))
        assert out is not frame
        assert (frame == 0).all()
        assert (out[25, 25] == [0, 255, 0]).all()

    def test_inplace_true_mutates_input(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[:, 0] = 25
        kps[:, 1] = 25
        kps[:, 2] = 0.9
        out = draw_skeleton(frame, kps, color=(0, 255, 0), inplace=True)
        assert out is frame
        assert (frame[25, 25] == [0, 255, 0]).all()

    def test_skips_nan_and_out_of_frame_keypoints(self):
        """Non-finite or negative coords would crash ``int()`` / ``cv2.circle``;
        we drop them silently so a single bad keypoint can't break the whole
        anonymisation pipeline."""
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        kps = np.array(
            [
                [np.nan, 25, 0.9],  # NaN x
                [25, np.inf, 0.9],  # inf y
                [-5, 25, 0.9],  # off-frame negative
                [200, 25, 0.9],  # off-frame positive
                [25, 25, 0.9],  # this one is fine
            ],
            dtype=np.float32,
        )
        out = draw_skeleton(
            frame, kps, edges=[(0, 1), (2, 3), (4, 4)], color=(0, 255, 0)
        )
        # The only valid joint is (25, 25); the rest were dropped.
        assert (out[25, 25] == [0, 255, 0]).all()

    def test_per_segment_colour_palette_is_default(self):
        """When color=None (default), edges are coloured by body part."""
        from cyberwave.vision.anonymize import COCO17_SEGMENT_COLORS

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        kps = np.zeros((17, 3), dtype=np.float32)
        # Place left + right shoulders far apart so the (5,6) torso edge
        # leaves a visible coloured streak.
        kps[5] = (40, 100, 0.9)
        kps[6] = (160, 100, 0.9)
        out = draw_skeleton(frame, kps, edges=[(5, 6)], thickness=4, radius=0)
        # Torso colour somewhere along the line.
        assert tuple(out[100, 100]) == COCO17_SEGMENT_COLORS["torso"]

    def test_thickness_zero_disables_overlay_entirely(self):
        """thickness=0 should suppress both lines and joints, not just lines.

        Otherwise a caller asking for "no skeleton" still sees joint dots,
        which is surprising.
        """
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[:, :2] = 50
        kps[:, 2] = 0.9
        out = draw_skeleton(frame, kps, thickness=0)
        # Frame should be entirely untouched.
        assert (out == 0).all()

    def test_thickness_zero_with_explicit_radius_still_draws_joints(self):
        """If the caller is explicit about radius, honour it — the auto-zero
        only kicks in when radius is left to its default."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[:, :2] = 50
        kps[:, 2] = 0.9
        out = draw_skeleton(frame, kps, thickness=0, radius=3)
        # Joints drawn even though lines aren't.
        assert (out[50, 50] > 0).any()

    def test_thickness_auto_scales_with_frame(self):
        """thickness=None → derived from frame height; bigger frame → thicker."""
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[5] = (40, 50, 0.9)
        kps[6] = (60, 50, 0.9)

        small = np.zeros((100, 100, 3), dtype=np.uint8)
        big = np.zeros((1080, 1920, 3), dtype=np.uint8)
        kps_big = kps.copy()
        kps_big[5, :2] *= 5
        kps_big[6, :2] *= 5

        small_out = draw_skeleton(small, kps, edges=[(5, 6)], radius=0)
        big_out = draw_skeleton(big, kps_big, edges=[(5, 6)], radius=0)

        # Count non-zero pixels along the edge — bigger frame → thicker line
        # → more painted pixels (per unit edge length).
        small_painted = int(np.any(small_out > 0, axis=-1).sum())
        big_painted = int(np.any(big_out > 0, axis=-1).sum())
        assert big_painted > small_painted * 5


class TestSkeletonPaletteContract:
    """Lock the structural contract between the three module-level constants.

    A typo in either ``COCO17_SKELETON``, ``COCO17_EDGE_GROUPS``, or
    ``COCO17_SEGMENT_COLORS`` would silently degrade the per-segment
    colouring to the white-joint fallback at runtime — these tests catch
    it at import time instead.
    """

    def test_every_skeleton_edge_has_a_palette_colour(self):
        for edge in COCO17_SKELETON:
            group = COCO17_EDGE_GROUPS.get(edge) or COCO17_EDGE_GROUPS.get(edge[::-1])
            assert group is not None, f"edge {edge} missing from COCO17_EDGE_GROUPS"
            assert group in COCO17_SEGMENT_COLORS, (
                f"edge {edge} maps to group {group!r} which has no palette colour"
            )

    def test_no_orphan_edge_groups(self):
        """Every entry in COCO17_EDGE_GROUPS should refer to an edge that
        actually exists in COCO17_SKELETON (in either orientation)."""
        canonical = {tuple(sorted(e)) for e in COCO17_SKELETON}
        for edge in COCO17_EDGE_GROUPS:
            assert tuple(sorted(edge)) in canonical, (
                f"COCO17_EDGE_GROUPS lists {edge} but it's not in COCO17_SKELETON"
            )


class TestAnonymizeFrame:
    def test_returns_new_array_with_blanked_persons(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        det = _make_det()
        out = anonymize_frame(frame, [det], mode="bbox")
        assert out is not frame
        # Bbox region: blanked to default colour (0, 0, 0).
        assert tuple(out[20, 20]) == (0, 0, 0)

    def test_default_mode_is_pixelate(self):
        """The default mode changed from 'bbox' to 'pixelate' so the bluntest
        possible mask is no longer the out-of-the-box behaviour."""
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, size=(100, 100, 3), dtype=np.uint8)
        det = _make_det(bbox=(10, 10, 90, 90))
        out = anonymize_frame(frame, [det])  # no mode → pixelate
        # In-bbox pixels modified.
        assert not np.array_equal(out[40, 40], frame[40, 40])
        # But the colour palette is reduced, not all-black, not all-white.
        roi = out[10:90, 10:90]
        n_unique = len(np.unique(roi.reshape(-1, 3), axis=0))
        assert 2 < n_unique < 1000  # quantised but not collapsed to one colour

    def test_overlays_skeleton_on_top_of_blanked_region(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        # Single visible keypoint inside the bbox region.
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[0, 0] = 30  # x inside bbox (10..50)
        kps[0, 1] = 30  # y inside bbox (10..80)
        kps[0, 2] = 0.9
        det = _make_det(keypoints=kps)
        out = anonymize_frame(
            frame,
            [det],
            mode="bbox",
            skeleton_color=(0, 255, 0),
        )
        # The skeleton joint dot at (30, 30) should be green (single-colour
        # mode collapses joints to the line colour for back-compat).
        assert (out[30, 30] == [0, 255, 0]).all()
        # Other in-bbox pixels remain blanked.
        assert tuple(out[20, 20]) == (0, 0, 0)

    def test_skips_skeleton_when_keypoints_missing(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        det = _make_det(keypoints=None)
        # Should not raise — and bbox mode keeps the bluntest assertion.
        out = anonymize_frame(frame, [det], mode="bbox")
        assert tuple(out[20, 20]) == (0, 0, 0)

    def test_non_person_detections_left_alone(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        car = _make_det(label="car", bbox=(60, 60, 90, 90))
        out = anonymize_frame(frame, [car], label="person")
        # Car bbox untouched (we only anonymise people).
        assert tuple(out[70, 70]) == (255, 255, 255)

    def test_inplace_true_mutates_input(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        det = _make_det()
        out = anonymize_frame(frame, [det], mode="bbox", inplace=True)
        assert out is frame
        assert tuple(frame[20, 20]) == (0, 0, 0)

    def test_labels_anonymises_multiple_classes(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        person = _make_det(label="person", bbox=(10, 10, 30, 30))
        car = _make_det(label="car", bbox=(40, 40, 60, 60))
        out = anonymize_frame(
            frame, [person, car], mode="bbox", labels=["person", "car"]
        )
        assert tuple(out[20, 20]) == (0, 0, 0)
        assert tuple(out[50, 50]) == (0, 0, 0)

    def test_labels_skeleton_overlay_covers_every_listed_class(self):
        """Skeleton overlay must follow the same active-label set as the
        obscuring path — otherwise non-person poses would render over a
        plain background."""
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[0] = (50, 50, 0.9)
        rider = _make_det(label="cyclist", bbox=(40, 40, 60, 60), keypoints=kps)
        out = anonymize_frame(
            frame,
            [rider],
            mode="bbox",
            labels=["cyclist"],
            skeleton_color=(0, 255, 0),
        )
        assert (out[50, 50] == [0, 255, 0]).all()
        assert tuple(out[45, 45]) == (0, 0, 0)

    def test_labels_takes_precedence_over_label(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        person = _make_det(label="person", bbox=(10, 10, 30, 30))
        car = _make_det(label="car", bbox=(40, 40, 60, 60))
        out = anonymize_frame(
            frame,
            [person, car],
            mode="bbox",
            label="person",
            labels=["car"],
        )
        assert tuple(out[20, 20]) == (255, 255, 255)
        assert tuple(out[50, 50]) == (0, 0, 0)

    def test_draw_skeleton_false_suppresses_overlay(self):
        """Callers on a plain-detector pipeline may still pass pose-model
        keypoints defensively (e.g. from a mixed-model fallback); they
        must be able to turn the overlay off without removing keypoints."""
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[0, 0] = 30
        kps[0, 1] = 30
        kps[0, 2] = 0.9
        det = _make_det(keypoints=kps)
        out = anonymize_frame(
            frame,
            [det],
            mode="bbox",
            skeleton_color=(0, 255, 0),
            draw_skeleton=False,
        )
        # No skeleton drawn — the pixel is still the bbox fill colour.
        assert tuple(out[30, 30]) == (0, 0, 0)

    def test_draw_skeleton_true_default_still_draws_when_keypoints_present(self):
        """Regression guard: the new kwarg must default to the old behaviour."""
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[0, 0] = 30
        kps[0, 1] = 30
        kps[0, 2] = 0.9
        det = _make_det(keypoints=kps)
        out = anonymize_frame(
            frame,
            [det],
            mode="bbox",
            skeleton_color=(0, 255, 0),
        )
        assert (out[30, 30] == [0, 255, 0]).all()
