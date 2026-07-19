"""Point-cloud payload decode."""

import base64
import logging

import numpy as np
import pytest

from cyberwave.exceptions import CyberwaveError
from cyberwave.twin.sensors import pointcloud as _pointcloud
from cyberwave.twin.sensors.pointcloud import _decode_pointcloud


@pytest.fixture(autouse=True)
def _reset_stride_warning_guard():
    """The missing-``point_stride`` advisory warns once per guessed stride
    (process-global) to avoid flooding on streamed clouds. Reset it so the
    warning-assertion tests here are deterministic regardless of run order."""
    _pointcloud._warned_missing_stride.clear()
    yield
    _pointcloud._warned_missing_stride.clear()


def _payload(points: np.ndarray, **extra) -> dict:
    payload = {
        "type": "pointcloud",
        "data": base64.b64encode(points.astype(np.float32).tobytes()).decode(),
    }
    payload.update(extra)
    return payload


def test_decode_pointcloud_returns_nx3() -> None:
    pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    out = _decode_pointcloud(_payload(pts))
    assert out.shape == (2, 3)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, pts)


def test_decode_pointcloud_ignores_wrong_type() -> None:
    assert _decode_pointcloud({"type": "depth_data", "data": "x"}) is None


def test_decode_pointcloud_rejects_misaligned_buffer() -> None:
    bad = {
        "type": "pointcloud",
        "data": base64.b64encode(np.zeros(4, dtype=np.float32).tobytes()).decode(),
    }
    with pytest.raises(CyberwaveError, match="not divisible by stride"):
        _decode_pointcloud(bad)


def test_decode_pointcloud_ignores_non_string_data() -> None:
    assert _decode_pointcloud({"type": "pointcloud", "data": 123}) is None
    assert _decode_pointcloud({"type": "pointcloud"}) is None


def test_decode_pointcloud_rejects_unsupported_explicit_stride() -> None:
    pts = np.zeros((2, 4), dtype=np.float32)  # 8 floats, divisible by 4
    with pytest.raises(CyberwaveError, match="unsupported point_stride"):
        _decode_pointcloud(_payload(pts, point_stride=4))


def test_decode_pointcloud_explicit_stride6_strips_rgb() -> None:
    raw = np.array([[1, 2, 3, 200, 100, 50], [4, 5, 6, 10, 20, 30]], dtype=np.float32)
    out = _decode_pointcloud(_payload(raw, point_stride=6))
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, raw[:, :3])


def test_infer_stride6_from_normalized_rgb(caplog) -> None:
    """No point_stride + colors in [0, 1] -> inferred stride 6, with a warning."""
    raw = np.array(
        [[1, 2, 3, 0.9, 0.1, 0.5], [4, 5, 6, 0.2, 0.8, 0.3]], dtype=np.float32
    )
    with caplog.at_level(logging.WARNING):
        out = _decode_pointcloud(_payload(raw))
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, raw[:, :3])
    assert any("point_stride" in r.message for r in caplog.records)


def test_infer_stride3_when_rgb_columns_out_of_range() -> None:
    """No point_stride + values > 1 -> stride 3, so no points are dropped."""
    pts = np.array([[1, 2, 3], [40, 50, 60]], dtype=np.float32)
    out = _decode_pointcloud(_payload(pts))
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, pts)


def test_infer_stride3_all_in_unit_cube_is_lossy_but_documented() -> None:
    """Known ambiguity: an even-count XYZ cloud fully within [0, 1] is misread as
    stride 6 (half the points dropped). Locks the documented behavior so a future
    change to the heuristic is a conscious decision, not an accident."""
    pts = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    out = _decode_pointcloud(_payload(pts))
    # Misclassified as stride 6 -> single point returned instead of two.
    assert out.shape == (1, 3)
