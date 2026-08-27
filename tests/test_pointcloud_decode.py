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


# ---------------------------------------------------------------------------
# Producer -> consumer round trip
#
# ``publish_pointcloud`` is the only Python producer for this topic. It knows
# the stride exactly, so it must stamp ``point_stride`` and never leave the
# consumer above guessing — every heuristic branch in this file is a bug when
# the sender could simply have said.
# ---------------------------------------------------------------------------


@pytest.fixture
def publishing_client():
    from unittest.mock import patch

    from cyberwave.mqtt import CyberwaveMQTTClient

    with patch("cyberwave.mqtt.mqtt.Client"):
        client = CyberwaveMQTTClient(
            mqtt_broker="localhost",
            mqtt_port=1883,
            mqtt_username="user",
            api_key="api_key_secret",
            auto_connect=False,
        )
    published: list[tuple[str, dict]] = []
    client.publish = lambda topic, message, **kw: published.append((topic, message))
    client._handle_twin_update_with_telemetry = lambda *_a, **_kw: None
    client.published = published  # type: ignore[attr-defined]
    return client


def test_publish_flat_colored_cloud_roundtrips_without_guessing(
    publishing_client, caplog
) -> None:
    """The flat buffer ``depth_to_colored_pointcloud`` returns must survive."""
    pts = np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3], [4.0, 5.0, 6.0, 0.4, 0.5, 0.6]])
    flat = pts.astype(np.float32).reshape(-1)

    publishing_client.publish_pointcloud("twin-uuid", flat)

    _topic, message = publishing_client.published[-1]
    assert message["point_stride"] == 6
    assert message["cols"] == 6
    assert message["rows"] == 2

    with caplog.at_level(logging.WARNING):
        out = _decode_pointcloud(message)
    np.testing.assert_allclose(out, pts[:, :3])
    assert not any("point_stride" in r.message for r in caplog.records)


def test_publish_xyz_cloud_declares_stride3(publishing_client, caplog) -> None:
    """A stride-3 cloud inside the unit cube is the documented worst case for the
    heuristic (see above) — declaring the stride is what makes it survive."""
    pts = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

    publishing_client.publish_pointcloud("twin-uuid", pts)

    _topic, message = publishing_client.published[-1]
    assert message["point_stride"] == 3
    assert message["rows"] == 2

    with caplog.at_level(logging.WARNING):
        out = _decode_pointcloud(message)
    # Both points survive, where the guess would have dropped one.
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, pts)
    assert not any("point_stride" in r.message for r in caplog.records)


def test_publish_flat_cloud_honours_explicit_stride3(publishing_client) -> None:
    flat = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)

    publishing_client.publish_pointcloud("twin-uuid", flat, stride=3)

    _topic, message = publishing_client.published[-1]
    assert message["point_stride"] == 3
    assert message["rows"] == 2


@pytest.mark.parametrize(
    "cloud, kwargs",
    [
        (np.zeros((2, 4), dtype=np.float32), {}),  # unsupported width
        (np.zeros(7, dtype=np.float32), {}),  # 7 % 6 != 0
        (np.zeros(6, dtype=np.float32), {"stride": 4}),  # unsupported stride
        (np.zeros((1, 2, 6), dtype=np.float32), {}),  # 3-D
    ],
)
def test_publish_rejects_malformed_clouds(publishing_client, cloud, kwargs) -> None:
    with pytest.raises(ValueError):
        publishing_client.publish_pointcloud("twin-uuid", cloud, **kwargs)
    assert not publishing_client.published
