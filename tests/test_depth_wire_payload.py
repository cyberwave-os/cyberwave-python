"""Producer wire payload: ``build_depth_mqtt_payload`` round-trips through ``_decode_depth``.

The depth wire is self-describing via ``dtype`` — ``uint16`` carries millimetres,
``float32`` carries absolute metres. These tests lock the producer helper and the
SDK consumer to the same contract so an fp32 ``/depth`` stream decodes verbatim.
"""

import base64
from typing import Any

import numpy as np
import pytest

from cyberwave.twin.sensors.depth import _decode_depth
from cyberwave.utils.depth import (
    DEPTH_OUTPUT_MODE_METRIC_MM,
    DEPTH_OUTPUT_MODE_NORMALIZED_UINT16,
    build_depth_mqtt_payload,
    depth_to_colored_pointcloud,
    depth_to_uint16,
)


def _as_depth_payload(wire: dict) -> dict:
    """Wrap a wire dict as a ``/depth`` MQTT payload for ``_decode_depth``."""
    return {"type": "depth_data", "data": wire}


def test_uint16_payload_is_tagged_and_default() -> None:
    arr = np.array([[0, 500], [1000, 2000]], dtype=np.uint16)  # millimetres
    wire = build_depth_mqtt_payload(arr)
    assert wire["dtype"] == "uint16"
    assert (wire["height"], wire["width"]) == (2, 2)
    out = _decode_depth(_as_depth_payload(wire))
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, arr)


def test_uint16_mode_coerces_non_uint16_input() -> None:
    # Legacy contract: a non-uint16 array under the default mode is cast to uint16.
    wire = build_depth_mqtt_payload(np.array([[1.0, 2.0]], dtype=np.float32))
    assert wire["dtype"] == "uint16"
    out = _decode_depth(_as_depth_payload(wire))
    np.testing.assert_array_equal(out, np.array([[1, 2]], dtype=np.uint16))


def test_float32_payload_carries_metres_verbatim() -> None:
    metres = np.array([[0.0, 0.3, 0.611], [1.04, 2.5, 70.0]], dtype=np.float32)
    wire = build_depth_mqtt_payload(metres, wire_dtype="float32")
    assert wire["dtype"] == "float32"
    assert (wire["height"], wire["width"]) == (2, 3)
    out = _decode_depth(_as_depth_payload(wire))
    assert out.dtype == np.float32
    # No quantisation: values (incl. 70 m, beyond uint16 mm range) survive exactly.
    np.testing.assert_array_equal(out, metres)


def test_float32_tag_always_matches_bytes() -> None:
    # float64 input is normalised to float32 bytes with a matching dtype tag.
    wire = build_depth_mqtt_payload(
        np.array([[1.5, 2.5]], dtype=np.float64), wire_dtype="float32"
    )
    assert wire["dtype"] == "float32"
    raw = np.frombuffer(base64.b64decode(wire["depth_binary"]), dtype=np.float32)
    np.testing.assert_array_equal(raw, np.array([1.5, 2.5], dtype=np.float32))


def test_unsupported_wire_dtype_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported wire_dtype"):
        build_depth_mqtt_payload(np.zeros((1, 1), dtype=np.float32), wire_dtype="int8")


def test_metric_mm_producer_path_roundtrips_via_payload() -> None:
    # metric_mm producer path: metres → uint16 mm → payload → decode → mm.
    metres = np.array([[0.1, 0.25], [1.0, 3.0]], dtype=np.float32)
    u16 = depth_to_uint16(metres, output_mode=DEPTH_OUTPUT_MODE_METRIC_MM)
    wire = build_depth_mqtt_payload(u16)
    out = _decode_depth(_as_depth_payload(wire))
    np.testing.assert_array_equal(
        out, np.array([[100, 250], [1000, 3000]], dtype=np.uint16)
    )


# ---------------------------------------------------------------------------
# output_mode is opt-in: a publisher that never declared one must not be
# labelled, or consumers rescale its frames with the wrong affine.
# ---------------------------------------------------------------------------


def test_output_mode_absent_unless_declared() -> None:
    # The RealSense path: raw uint16 millimetres, no output_mode argument.
    wire = build_depth_mqtt_payload(np.array([[1000, 2000]], dtype=np.uint16))
    assert "output_mode" not in wire


def test_output_mode_stamped_when_declared() -> None:
    wire = build_depth_mqtt_payload(
        np.array([[1000, 2000]], dtype=np.uint16),
        output_mode=DEPTH_OUTPUT_MODE_METRIC_MM,
        depth_scale=0.001,
    )
    assert wire["output_mode"] == DEPTH_OUTPUT_MODE_METRIC_MM
    assert wire["depth_scale"] == 0.001


def test_metric_mode_without_known_scale_omits_depth_scale() -> None:
    """A publisher that does not know its metres-per-unit must not invent one.

    Raw uint16 is millimetres only by convention — RealSense ``depth_units`` is
    configurable — so stamping 0.001 would override a twin's schema-declared
    calibration with a guess and misplace the cloud by that ratio.
    """
    wire = build_depth_mqtt_payload(
        np.array([[1000, 2000]], dtype=np.uint16),
        output_mode=DEPTH_OUTPUT_MODE_METRIC_MM,
        depth_scale=None,
    )
    assert wire["output_mode"] == DEPTH_OUTPUT_MODE_METRIC_MM
    assert "depth_scale" not in wire


# ---------------------------------------------------------------------------
# normalized_uint16 encodes against the *declared* window, so the published
# min/max/depth_scale are the true decode affine and the scale is stable.
# ---------------------------------------------------------------------------


def _decode_normalized(u16: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + u16.astype(np.float64) / 65535.0 * (hi - lo)


def test_normalized_window_roundtrips_to_original_metres() -> None:
    metres = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    u16 = depth_to_uint16(
        metres,
        output_mode=DEPTH_OUTPUT_MODE_NORMALIZED_UINT16,
        min_depth=0.1,
        max_depth=10.0,
    )
    np.testing.assert_allclose(_decode_normalized(u16, 0.1, 10.0), metres, atol=1e-3)


def test_normalized_window_scale_is_stable_across_frames() -> None:
    # Two frames whose own depth ranges differ decode to their true metres —
    # under per-frame auto-ranging both would have hit the same u16 endpoints.
    a = np.array([[2.0, 5.0]], dtype=np.float32)
    b = np.array([[1.0, 9.0]], dtype=np.float32)
    kw = dict(
        output_mode=DEPTH_OUTPUT_MODE_NORMALIZED_UINT16, min_depth=0.1, max_depth=10.0
    )
    np.testing.assert_allclose(
        _decode_normalized(depth_to_uint16(a, **kw), 0.1, 10.0), a, atol=1e-3
    )
    np.testing.assert_allclose(
        _decode_normalized(depth_to_uint16(b, **kw), 0.1, 10.0), b, atol=1e-3
    )


def test_normalized_without_window_keeps_legacy_auto_range() -> None:
    metres = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    u16 = depth_to_uint16(metres, output_mode=DEPTH_OUTPUT_MODE_NORMALIZED_UINT16)
    # 1, not 0: valid samples start above the "no reading" sentinel so the
    # consumers' raw>0 guard cannot mistake the nearest surface for a hole.
    assert u16.min() == 1
    assert u16.max() == 65535


# ---------------------------------------------------------------------------
# Relative checkpoints emit disparity (larger = nearer); encoding it as depth
# without flipping turns the reconstructed scene inside-out.
# ---------------------------------------------------------------------------


def test_disparity_is_inverted_so_nearest_decodes_nearest() -> None:
    # Disparity: 8.0 is the closest surface, 1.0 the farthest.
    disparity = np.array([[8.0, 4.0, 1.0]], dtype=np.float32)
    u16 = depth_to_uint16(
        disparity,
        output_mode=DEPTH_OUTPUT_MODE_NORMALIZED_UINT16,
        min_depth=0.5,
        max_depth=10.0,
        input_is_disparity=True,
    )
    decoded = _decode_normalized(u16, 0.5, 10.0)
    assert decoded[0][0] < decoded[0][1] < decoded[0][2]
    assert decoded[0][0] == pytest.approx(0.5, abs=1e-3)
    assert decoded[0][2] == pytest.approx(10.0, abs=1e-3)


def test_disparity_invalid_pixels_stay_zero() -> None:
    # 0.0 means "no reading"; inversion must not promote it to the far plane.
    disparity = np.array([[8.0, 0.0, 1.0]], dtype=np.float32)
    u16 = depth_to_uint16(
        disparity,
        output_mode=DEPTH_OUTPUT_MODE_NORMALIZED_UINT16,
        min_depth=0.5,
        max_depth=10.0,
        input_is_disparity=True,
    )
    assert u16[0][1] == 0


def test_metric_mm_rejects_disparity_input() -> None:
    with pytest.raises(ValueError, match="cannot encode a disparity map"):
        depth_to_uint16(
            np.array([[1.0, 2.0]], dtype=np.float32),
            output_mode=DEPTH_OUTPUT_MODE_METRIC_MM,
            input_is_disparity=True,
        )


# ---------------------------------------------------------------------------
# u16 == 0 is "no reading", not a distance. The published cloud has to drop it
# on the raw sample: under normalized_uint16 the affine decodes 0 to min_depth,
# so a decoded-depth guard passes every invalid pixel through.
# ---------------------------------------------------------------------------


def _pointcloud_z(cloud: np.ndarray) -> np.ndarray:
    return cloud[2::6]


def test_pointcloud_drops_no_reading_pixels_in_normalized_mode() -> None:
    """The defect this pins: sky rendered as a wall 0.1 m from the lens.

    A relative checkpoint's ``relu`` output is exactly 0 over sky and holes, and
    ``depth_to_uint16`` sends every invalid pixel to u16=0. Decoding that as
    ``min_depth + 0`` put all of them at the near plane, so the ``/pointcloud``
    topic showed a solid plane the ``/depth`` renderer did not.
    """
    disparity = np.zeros((4, 4), dtype=np.float32)
    disparity[0][0] = 8.0
    disparity[0][1] = 4.0
    u16 = depth_to_uint16(
        disparity,
        output_mode=DEPTH_OUTPUT_MODE_NORMALIZED_UINT16,
        min_depth=0.1,
        max_depth=10.0,
        input_is_disparity=True,
    )
    # Exactly the 14 invalid pixels. This assertion used to read 15, counting
    # "14 invalid + the nearest surface" — the sentinel collision that deleted
    # real geometry, pinned as if it were the contract. Both valid samples now
    # encode above 0.
    assert int((u16 == 0).sum()) == 14

    cloud = depth_to_colored_pointcloud(
        u16,
        output_mode=DEPTH_OUTPUT_MODE_NORMALIZED_UINT16,
        min_depth=0.1,
        max_depth=10.0,
        step=1,
    )
    assert cloud is not None
    # Both real samples survive, and no near-plane wall from the 14 holes.
    assert cloud.size // 6 == 2
    # The near sample (disparity 8.0, the largest) decodes at the window floor
    # and the far one (4.0) at its ceiling. The near point was previously
    # missing entirely — its u16 landed on the sentinel and both consumers
    # dropped it, which is the whole defect this pair of assertions now pins.
    np.testing.assert_allclose(_pointcloud_z(cloud), [0.1, 10.0], atol=1e-3)


def test_pointcloud_returns_none_when_every_sample_is_a_sentinel() -> None:
    u16 = np.zeros((4, 4), dtype=np.uint16)
    assert (
        depth_to_colored_pointcloud(
            u16,
            output_mode=DEPTH_OUTPUT_MODE_NORMALIZED_UINT16,
            min_depth=0.1,
            max_depth=10.0,
            step=1,
        )
        is None
    )


def test_pointcloud_metric_mm_still_drops_zero_samples() -> None:
    """The metric path already worked (0 x scale = 0); keep it that way."""
    u16 = np.array([[0, 2000], [0, 0]], dtype=np.uint16)
    cloud = depth_to_colored_pointcloud(
        u16, output_mode=DEPTH_OUTPUT_MODE_METRIC_MM, depth_scale=0.001, step=1
    )
    assert cloud is not None
    assert cloud.size // 6 == 1
    np.testing.assert_allclose(_pointcloud_z(cloud), [2.0], atol=1e-6)


# ---------------------------------------------------------------------------
# max_points: a resolution-independent density cap. A bare pixel stride costs
# 6.75x more at 1080p than at 480p, and the workflow emitter cannot pick one
# because a depth model may resample the frame away from the declared size.
# ---------------------------------------------------------------------------


def _filled(h: int, w: int) -> np.ndarray:
    return np.full((h, w), 30000, dtype=np.uint16)


def _point_count(cloud: Any) -> int:
    return 0 if cloud is None else cloud.size // 6


@pytest.mark.parametrize(
    "width,height", [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)]
)
def test_max_points_caps_every_resolution(width: int, height: int) -> None:
    """One budget holds from 480p to 4K — that is the whole point of the knob."""
    cloud = depth_to_colored_pointcloud(
        _filled(height, width),
        output_mode=DEPTH_OUTPUT_MODE_METRIC_MM,
        depth_scale=0.001,
        step=1,
        max_points=20000,
    )
    assert 0 < _point_count(cloud) <= 20000


@pytest.mark.parametrize(
    "width,height,budget",
    [(160, 120, 100), (640, 480, 500), (100, 100, 50), (33, 17, 7), (640, 480, 1)],
)
def test_max_points_is_a_guarantee_not_an_estimate(
    width: int, height: int, budget: int
) -> None:
    """Small budgets are where the closed-form stride overshoots.

    ``ceil(sqrt(h*w/budget))`` sizes the grid only in the continuous case;
    rounding each axis up independently can land over the cap (160x120 at 100
    points gives a 9x12 = 108-point grid). These cases pin the tightening step
    that turns the cap into a real bound.
    """
    cloud = depth_to_colored_pointcloud(
        _filled(height, width),
        output_mode=DEPTH_OUTPUT_MODE_METRIC_MM,
        depth_scale=0.001,
        step=1,
        max_points=budget,
    )
    assert 0 < _point_count(cloud) <= budget


def test_default_budget_leaves_640x480_at_its_historical_density() -> None:
    """20 000 is chosen so the common case does not change.

    640x480 at the old hardcoded ``step=4`` is exactly 19 200 points, under the
    budget — so an existing 480p deployment publishes the same cloud it always
    did, and only larger frames get coarsened.
    """
    budgeted = depth_to_colored_pointcloud(
        _filled(480, 640),
        output_mode=DEPTH_OUTPUT_MODE_METRIC_MM,
        depth_scale=0.001,
        step=1,
        max_points=20000,
    )
    legacy = depth_to_colored_pointcloud(
        _filled(480, 640),
        output_mode=DEPTH_OUTPUT_MODE_METRIC_MM,
        depth_scale=0.001,
        step=4,
    )
    assert _point_count(legacy) == 19200
    np.testing.assert_array_equal(budgeted, legacy)


def test_explicit_coarser_step_still_wins() -> None:
    """The budget is a ceiling, not a target — it never *densifies* a cloud."""
    cloud = depth_to_colored_pointcloud(
        _filled(480, 640),
        output_mode=DEPTH_OUTPUT_MODE_METRIC_MM,
        depth_scale=0.001,
        step=16,
        max_points=1_000_000,
    )
    assert _point_count(cloud) == 30 * 40  # ceil(480/16) * ceil(640/16)


def test_max_points_omitted_leaves_step_alone() -> None:
    cloud = depth_to_colored_pointcloud(
        _filled(1080, 1920),
        output_mode=DEPTH_OUTPUT_MODE_METRIC_MM,
        depth_scale=0.001,
        step=4,
    )
    assert _point_count(cloud) == 129600


class TestSentinelIsReservedForInvalidPixels:
    """``u16 == 0`` must mean "no reading" and nothing else.

    Both consumers drop it — ``depth_to_colored_pointcloud``'s ``raw > 0`` guard
    and the frontend's ``raw === 0`` gate in ``depthToPoints`` — because a zero
    decodes to ``min_depth`` under the affine, which would paint sky and holes
    as a wall right in front of the camera. That guard is only correct if the
    encoder never puts a *valid* sample on 0.
    """

    def test_nearest_surface_is_not_encoded_as_the_no_reading_sentinel(self):
        # A flat near plane over the top 30% of the frame: a wall, table or
        # floor close to the camera. The inverted branch maps the largest
        # disparity to unit 0, so the whole surface used to land on the
        # sentinel and vanish from the cloud.
        near_plane = np.full((72, 320), 5.0, dtype=np.float32)
        ramp = np.linspace(0.2, 4.9, 168 * 320).reshape(168, 320).astype(np.float32)
        disparity = np.concatenate([near_plane, ramp])

        encoded = depth_to_uint16(
            disparity,
            output_mode=DEPTH_OUTPUT_MODE_NORMALIZED_UINT16,
            min_depth=0.1,
            max_depth=10.0,
            input_is_disparity=True,
        )

        lost = int(((encoded == 0) & (disparity > 0)).sum())
        assert lost == 0, f"{lost} valid pixels collided with the sentinel"

    def test_depth_at_the_window_floor_is_not_the_sentinel(self):
        # Same collision in the windowed branch: anything at or below
        # ``min_depth`` clips to unit 0.
        depth = np.array([[0.1, 0.05, 2.0, 10.0]], dtype=np.float32)
        encoded = depth_to_uint16(
            depth,
            output_mode=DEPTH_OUTPUT_MODE_NORMALIZED_UINT16,
            min_depth=0.1,
            max_depth=10.0,
        )
        assert encoded[0, 0] != 0
        assert encoded[0, 1] != 0

    def test_invalid_pixels_still_encode_to_zero(self):
        depth = np.array([[0.0, np.nan, 2.0]], dtype=np.float32)
        encoded = depth_to_uint16(
            depth,
            output_mode=DEPTH_OUTPUT_MODE_NORMALIZED_UINT16,
            min_depth=0.1,
            max_depth=10.0,
        )
        assert encoded[0, 0] == 0
        assert encoded[0, 1] == 0
        assert encoded[0, 2] != 0

    def test_the_far_end_of_the_window_still_saturates(self):
        """The offset must not cost the top of the range."""
        depth = np.array([[10.0]], dtype=np.float32)
        encoded = depth_to_uint16(
            depth,
            output_mode=DEPTH_OUTPUT_MODE_NORMALIZED_UINT16,
            min_depth=0.1,
            max_depth=10.0,
        )
        assert encoded[0, 0] == 65535

    def test_the_nearest_surface_survives_into_the_cloud(self):
        """End to end: encode a near plane, decode it, and count the points."""
        near_plane = np.full((24, 32), 5.0, dtype=np.float32)
        ramp = np.linspace(0.2, 4.9, 24 * 32).reshape(24, 32).astype(np.float32)
        disparity = np.concatenate([near_plane, ramp])

        encoded = depth_to_uint16(
            disparity,
            output_mode=DEPTH_OUTPUT_MODE_NORMALIZED_UINT16,
            min_depth=0.1,
            max_depth=10.0,
            input_is_disparity=True,
        )
        cloud = depth_to_colored_pointcloud(
            encoded,
            output_mode=DEPTH_OUTPUT_MODE_NORMALIZED_UINT16,
            min_depth=0.1,
            max_depth=10.0,
            fx_normalized=0.8,
            fy_normalized=1.06,
            step=1,
        )
        assert cloud is not None
        assert cloud.size // 6 == disparity.size, "the near plane was dropped"
