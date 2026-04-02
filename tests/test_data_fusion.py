"""Tests for fusion primitives — at(), window(), interpolation strategies."""

from __future__ import annotations

import math
import threading
from unittest.mock import patch

import pytest

from cyberwave.data.fusion import (
    ChannelBuffer,
    FusionLayer,
    Quaternion,
    WindowResult,
    interpolate_linear,
    interpolate_nearest,
    interpolate_slerp,
)
from cyberwave.data.ring_buffer import TimestampedSample


# ---------------------------------------------------------------------------
# Helper: approximate float comparison
# ---------------------------------------------------------------------------


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


def _quat_approx(
    a: Quaternion | list[float], b: Quaternion | list[float], tol: float = 1e-5
) -> bool:
    """Quaternion comparison — handles sign ambiguity (q ≡ -q)."""
    a_l = a.as_list() if isinstance(a, Quaternion) else a
    b_l = b.as_list() if isinstance(b, Quaternion) else b
    dot = sum(ai * bi for ai, bi in zip(a_l, b_l))
    return abs(abs(dot) - 1.0) < tol


# ---------------------------------------------------------------------------
# Linear interpolation
# ---------------------------------------------------------------------------


class TestInterpolateLinear:
    def test_scalar(self) -> None:
        s1 = TimestampedSample(ts=0.0, value=0.0)
        s2 = TimestampedSample(ts=1.0, value=10.0)
        assert _approx(interpolate_linear(s1, s2, 0.5), 5.0)
        assert _approx(interpolate_linear(s1, s2, 0.0), 0.0)
        assert _approx(interpolate_linear(s1, s2, 1.0), 10.0)
        assert _approx(interpolate_linear(s1, s2, 0.25), 2.5)

    def test_integer_values(self) -> None:
        s1 = TimestampedSample(ts=0.0, value=0)
        s2 = TimestampedSample(ts=1.0, value=100)
        result = interpolate_linear(s1, s2, 0.5)
        assert _approx(result, 50.0)

    def test_list_of_floats(self) -> None:
        s1 = TimestampedSample(ts=0.0, value=[0.0, 0.0, 0.0])
        s2 = TimestampedSample(ts=1.0, value=[10.0, 20.0, 30.0])
        result = interpolate_linear(s1, s2, 0.5)
        assert len(result) == 3
        assert _approx(result[0], 5.0)
        assert _approx(result[1], 10.0)
        assert _approx(result[2], 15.0)

    def test_dict_with_numeric_values(self) -> None:
        s1 = TimestampedSample(ts=0.0, value={"x": 0.0, "y": 0.0, "label": "arm"})
        s2 = TimestampedSample(ts=1.0, value={"x": 10.0, "y": 20.0, "label": "arm"})
        result = interpolate_linear(s1, s2, 0.5)
        assert _approx(result["x"], 5.0)
        assert _approx(result["y"], 10.0)
        assert result["label"] == "arm"

    def test_dict_with_list_values(self) -> None:
        s1 = TimestampedSample(ts=0.0, value={"pos": [0.0, 0.0], "name": "a"})
        s2 = TimestampedSample(ts=1.0, value={"pos": [10.0, 20.0], "name": "a"})
        result = interpolate_linear(s1, s2, 0.5)
        assert _approx(result["pos"][0], 5.0)
        assert _approx(result["pos"][1], 10.0)

    def test_same_timestamp_returns_first(self) -> None:
        s1 = TimestampedSample(ts=1.0, value=42.0)
        s2 = TimestampedSample(ts=1.0, value=99.0)
        assert interpolate_linear(s1, s2, 1.0) == 42.0

    def test_non_interpolatable_type_returns_first(self) -> None:
        s1 = TimestampedSample(ts=0.0, value="hello")
        s2 = TimestampedSample(ts=1.0, value="world")
        assert interpolate_linear(s1, s2, 0.5) == "hello"

    def test_numpy_array(self) -> None:
        np = pytest.importorskip("numpy")
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([10.0, 20.0, 30.0])
        s1 = TimestampedSample(ts=0.0, value=a)
        s2 = TimestampedSample(ts=1.0, value=b)
        result = interpolate_linear(s1, s2, 0.5)
        np.testing.assert_allclose(result, [5.0, 10.0, 15.0])


# ---------------------------------------------------------------------------
# SLERP interpolation
# ---------------------------------------------------------------------------


class TestInterpolateSlerp:
    def test_identity_quaternion(self) -> None:
        q = Quaternion(0.0, 0.0, 0.0, 1.0)
        s1 = TimestampedSample(ts=0.0, value=q)
        s2 = TimestampedSample(ts=1.0, value=q)
        result = interpolate_slerp(s1, s2, 0.5)
        assert isinstance(result, Quaternion)
        assert _quat_approx(result, q)

    def test_90_degree_rotation(self) -> None:
        q1 = Quaternion(0.0, 0.0, 0.0, 1.0)
        q2 = Quaternion(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
        s1 = TimestampedSample(ts=0.0, value=q1)
        s2 = TimestampedSample(ts=1.0, value=q2)

        result = interpolate_slerp(s1, s2, 0.5)
        assert isinstance(result, Quaternion)
        expected = [0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8)]
        for a, b in zip(result.as_list(), expected):
            assert _approx(a, b, tol=1e-4)

    def test_antipodal_quaternions(self) -> None:
        q1 = Quaternion(0.0, 0.0, 0.0, 1.0)
        q2 = Quaternion(0.0, 0.0, 0.0, -1.0)
        s1 = TimestampedSample(ts=0.0, value=q1)
        s2 = TimestampedSample(ts=1.0, value=q2)
        result = interpolate_slerp(s1, s2, 0.5)
        assert isinstance(result, Quaternion)
        assert _quat_approx(result, q1)

    def test_nearly_parallel_fallback_to_lerp(self) -> None:
        q1 = Quaternion(0.0, 0.0, 0.0, 1.0)
        q2 = Quaternion(0.0, 0.0, 1e-6, 1.0)
        s1 = TimestampedSample(ts=0.0, value=q1)
        s2 = TimestampedSample(ts=1.0, value=q2)
        result = interpolate_slerp(s1, s2, 0.5)
        assert isinstance(result, Quaternion)
        norm = math.sqrt(sum(c * c for c in result.as_list()))
        assert _approx(norm, 1.0, tol=1e-4)

    def test_same_timestamp(self) -> None:
        q = Quaternion(0.0, 0.0, 0.0, 1.0)
        s1 = TimestampedSample(ts=1.0, value=q)
        s2 = TimestampedSample(ts=1.0, value=Quaternion(0.0, 0.0, 0.7071, 0.7071))
        assert interpolate_slerp(s1, s2, 0.5) == q

    def test_dict_with_quaternion_field(self) -> None:
        s1 = TimestampedSample(
            ts=0.0,
            value={
                "orientation": Quaternion(0.0, 0.0, 0.0, 1.0),
                "position": [0.0, 0.0, 0.0],
            },
        )
        s2 = TimestampedSample(
            ts=1.0,
            value={
                "orientation": Quaternion(
                    0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)
                ),
                "position": [10.0, 0.0, 0.0],
            },
        )
        result = interpolate_slerp(s1, s2, 0.5)
        assert isinstance(result, dict)
        assert isinstance(result["orientation"], Quaternion)
        norm = math.sqrt(sum(c * c for c in result["orientation"].as_list()))
        assert _approx(norm, 1.0, tol=1e-4)
        assert _approx(result["position"][0], 5.0)

    def test_raw_list_not_slerped(self) -> None:
        """A plain list[float] of length 4 is NOT treated as a quaternion."""
        s1 = TimestampedSample(ts=0.0, value=[0.0, 0.0, 0.0, 1.0])
        s2 = TimestampedSample(ts=1.0, value=[1.0, 0.0, 0.0, 0.0])
        with pytest.warns(UserWarning, match="not quaternion-like"):
            result = interpolate_slerp(s1, s2, 0.5)
        assert isinstance(result, list)

    def test_non_quaternion_fallback_to_linear(self) -> None:
        s1 = TimestampedSample(ts=0.0, value=0.0)
        s2 = TimestampedSample(ts=1.0, value=10.0)
        with pytest.warns(UserWarning, match="not quaternion-like"):
            result = interpolate_slerp(s1, s2, 0.5)
        assert _approx(result, 5.0)


# ---------------------------------------------------------------------------
# Nearest interpolation
# ---------------------------------------------------------------------------


class TestInterpolateNearest:
    def test_closer_to_before(self) -> None:
        s1 = TimestampedSample(ts=0.0, value="a")
        s2 = TimestampedSample(ts=1.0, value="b")
        assert interpolate_nearest(s1, s2, 0.3) == "a"

    def test_closer_to_after(self) -> None:
        s1 = TimestampedSample(ts=0.0, value="a")
        s2 = TimestampedSample(ts=1.0, value="b")
        assert interpolate_nearest(s1, s2, 0.7) == "b"

    def test_equidistant_prefers_before(self) -> None:
        s1 = TimestampedSample(ts=0.0, value="a")
        s2 = TimestampedSample(ts=1.0, value="b")
        assert interpolate_nearest(s1, s2, 0.5) == "a"

    def test_same_timestamp(self) -> None:
        s1 = TimestampedSample(ts=1.0, value="a")
        s2 = TimestampedSample(ts=1.0, value="b")
        assert interpolate_nearest(s1, s2, 1.0) == "a"


# ---------------------------------------------------------------------------
# ChannelBuffer.at()
# ---------------------------------------------------------------------------


class TestChannelBufferAt:
    def test_empty_buffer_returns_none(self) -> None:
        cb = ChannelBuffer("test", capacity=10)
        assert cb.at(1.0) is None

    def test_exact_match(self) -> None:
        cb = ChannelBuffer("test", capacity=10)
        cb.ingest(1.0, 100.0)
        cb.ingest(2.0, 200.0)
        assert cb.at(2.0) == 200.0

    def test_linear_interpolation(self) -> None:
        cb = ChannelBuffer("test", capacity=10)
        cb.ingest(0.0, 0.0)
        cb.ingest(1.0, 10.0)
        result = cb.at(0.5, "linear")
        assert _approx(result, 5.0)

    def test_nearest_interpolation(self) -> None:
        cb = ChannelBuffer("test", capacity=10)
        cb.ingest(0.0, "a")
        cb.ingest(1.0, "b")
        assert cb.at(0.3, "nearest") == "a"
        assert cb.at(0.7, "nearest") == "b"

    def test_none_interpolation(self) -> None:
        cb = ChannelBuffer("test", capacity=10)
        cb.ingest(0.0, 100.0)
        cb.ingest(1.0, 200.0)
        assert cb.at(0.5, "none") is None
        assert cb.at(0.0, "none") == 100.0

    def test_before_range_returns_first(self) -> None:
        cb = ChannelBuffer("test", capacity=10)
        cb.ingest(5.0, 50.0)
        cb.ingest(6.0, 60.0)
        result = cb.at(1.0, "linear")
        assert result == 50.0

    def test_after_range_returns_last(self) -> None:
        cb = ChannelBuffer("test", capacity=10)
        cb.ingest(1.0, 10.0)
        cb.ingest(2.0, 20.0)
        result = cb.at(5.0, "linear")
        assert result == 20.0

    def test_invalid_interpolation_raises(self) -> None:
        cb = ChannelBuffer("test", capacity=10)
        cb.ingest(0.0, 0.0)
        with pytest.raises(ValueError, match="Invalid interpolation"):
            cb.at(0.5, "cubic")


# ---------------------------------------------------------------------------
# ChannelBuffer.window()
# ---------------------------------------------------------------------------


class TestChannelBufferWindow:
    def test_explicit_range(self) -> None:
        cb = ChannelBuffer("test", capacity=100)
        for i in range(10):
            cb.ingest(float(i), i * 10)
        result = cb.window(from_t=3.0, to_t=6.0)
        assert isinstance(result, WindowResult)
        assert len(result) == 4
        assert result.values == [30, 40, 50, 60]

    def test_empty_range(self) -> None:
        cb = ChannelBuffer("test", capacity=100)
        cb.ingest(1.0, 10)
        cb.ingest(2.0, 20)
        result = cb.window(from_t=5.0, to_t=10.0)
        assert len(result) == 0

    def test_duration_ms_goes_through_fusion_layer(self) -> None:
        """duration_ms is resolved by FusionLayer using its injected clock."""
        fixed_now = 1000.0
        fl = FusionLayer(clock=lambda: fixed_now)
        for i in range(10):
            fl.ingest("ch", fixed_now - 0.01 * (9 - i), i)
        result = fl.window("ch", duration_ms=50)
        assert len(result) == 6


# ---------------------------------------------------------------------------
# WindowResult
# ---------------------------------------------------------------------------


class TestWindowResult:
    def test_len(self) -> None:
        wr = WindowResult(
            samples=[TimestampedSample(ts=i, value=i) for i in range(5)],
            from_t=0.0,
            to_t=4.0,
        )
        assert len(wr) == 5

    def test_bool_nonempty(self) -> None:
        wr = WindowResult(
            samples=[TimestampedSample(ts=1.0, value="x")],
            from_t=0.0,
            to_t=2.0,
        )
        assert bool(wr) is True

    def test_bool_empty(self) -> None:
        wr = WindowResult(samples=[], from_t=0.0, to_t=1.0)
        assert bool(wr) is False

    def test_values(self) -> None:
        wr = WindowResult(
            samples=[
                TimestampedSample(ts=1.0, value="a"),
                TimestampedSample(ts=2.0, value="b"),
            ],
            from_t=1.0,
            to_t=2.0,
        )
        assert wr.values == ["a", "b"]

    def test_timestamps(self) -> None:
        wr = WindowResult(
            samples=[
                TimestampedSample(ts=1.0, value="a"),
                TimestampedSample(ts=2.0, value="b"),
            ],
            from_t=1.0,
            to_t=2.0,
        )
        assert wr.timestamps == [1.0, 2.0]


# ---------------------------------------------------------------------------
# FusionLayer — end-to-end
# ---------------------------------------------------------------------------


class TestFusionLayer:
    def test_at_unknown_channel_returns_none(self) -> None:
        fl = FusionLayer()
        assert fl.at("nonexistent", t=1.0) is None

    def test_at_linear(self) -> None:
        fl = FusionLayer()
        fl.ingest("joint", 0.0, [0.0, 0.0])
        fl.ingest("joint", 1.0, [10.0, 20.0])
        result = fl.at("joint", t=0.5, interpolation="linear")
        assert _approx(result[0], 5.0)
        assert _approx(result[1], 10.0)

    def test_at_slerp(self) -> None:
        fl = FusionLayer()
        q1 = Quaternion(0.0, 0.0, 0.0, 1.0)
        q2 = Quaternion(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
        fl.ingest("orient", 0.0, q1)
        fl.ingest("orient", 1.0, q2)
        result = fl.at("orient", t=0.5, interpolation="slerp")
        assert isinstance(result, Quaternion)
        norm = math.sqrt(sum(c * c for c in result.as_list()))
        assert _approx(norm, 1.0, tol=1e-4)

    def test_at_defaults_to_now(self) -> None:
        fixed_now = 1000.0
        fl = FusionLayer(clock=lambda: fixed_now)
        fl.ingest("ch", fixed_now - 0.1, 0.0)
        fl.ingest("ch", fixed_now + 0.1, 10.0)
        result = fl.at("ch")
        assert result is not None
        assert _approx(result, 5.0)

    def test_window_explicit(self) -> None:
        fl = FusionLayer()
        for i in range(10):
            fl.ingest("imu", float(i), {"ax": float(i)})
        wr = fl.window("imu", from_t=3.0, to_t=6.0)
        assert len(wr) == 4
        assert wr.from_t == 3.0
        assert wr.to_t == 6.0

    def test_window_duration(self) -> None:
        fixed_now = 1000.0
        fl = FusionLayer(clock=lambda: fixed_now)
        for i in range(10):
            fl.ingest("ft", fixed_now - 0.01 * (9 - i), i)
        wr = fl.window("ft", duration_ms=50)
        assert len(wr) == 6

    def test_window_unknown_channel(self) -> None:
        fl = FusionLayer()
        wr = fl.window("unknown", from_t=0.0, to_t=1.0)
        assert len(wr) == 0

    def test_window_unknown_channel_duration(self) -> None:
        fixed_now = 1000.0
        fl = FusionLayer(clock=lambda: fixed_now)
        wr = fl.window("unknown", duration_ms=100)
        assert len(wr) == 0
        assert _approx(wr.to_t, fixed_now)
        assert _approx(wr.from_t, fixed_now - 0.1)

    def test_window_unknown_channel_no_params_raises(self) -> None:
        fl = FusionLayer()
        with pytest.raises(ValueError, match="Must provide"):
            fl.window("unknown")

    def test_configure_channel_capacity(self) -> None:
        fl = FusionLayer(default_capacity=100)
        fl.configure_channel("small", 5)
        for i in range(20):
            fl.ingest("small", float(i), i)
        wr = fl.window("small", from_t=0.0, to_t=100.0)
        assert len(wr) == 5

    def test_configure_channel_after_ingest_raises(self) -> None:
        fl = FusionLayer()
        fl.ingest("ch", 1.0, 10)
        with pytest.raises(ValueError, match="must be called before"):
            fl.configure_channel("ch", 500)

    def test_channels_property(self) -> None:
        fl = FusionLayer()
        fl.ingest("a", 1.0, None)
        fl.ingest("b", 1.0, None)
        assert sorted(fl.channels) == ["a", "b"]

    def test_clear_single_channel(self) -> None:
        fl = FusionLayer()
        fl.ingest("a", 1.0, 10)
        fl.ingest("b", 1.0, 20)
        fl.clear("a")
        assert fl.at("a", t=1.0) is None
        assert fl.at("b", t=1.0) == 20

    def test_clear_all_channels(self) -> None:
        fl = FusionLayer()
        fl.ingest("a", 1.0, 10)
        fl.ingest("b", 1.0, 20)
        fl.clear()
        assert fl.at("a", t=1.0) is None
        assert fl.at("b", t=1.0) is None

    def test_multi_channel_isolation(self) -> None:
        fl = FusionLayer()
        fl.ingest("joint", 0.0, [0.0])
        fl.ingest("joint", 1.0, [10.0])
        fl.ingest("force", 0.0, 0.0)
        fl.ingest("force", 1.0, 100.0)
        assert _approx(fl.at("joint", t=0.5, interpolation="linear")[0], 5.0)
        assert _approx(fl.at("force", t=0.5, interpolation="linear"), 50.0)


# ---------------------------------------------------------------------------
# Edge cases for the full at() + ring buffer pipeline
# ---------------------------------------------------------------------------


class TestAtEdgeCases:
    def test_single_sample_exact(self) -> None:
        fl = FusionLayer()
        fl.ingest("ch", 5.0, 50)
        assert fl.at("ch", t=5.0) == 50

    def test_single_sample_before(self) -> None:
        fl = FusionLayer()
        fl.ingest("ch", 5.0, 50)
        result = fl.at("ch", t=3.0, interpolation="linear")
        assert result == 50

    def test_single_sample_after(self) -> None:
        fl = FusionLayer()
        fl.ingest("ch", 5.0, 50)
        result = fl.at("ch", t=7.0, interpolation="linear")
        assert result == 50

    def test_extrapolation_returns_boundary(self) -> None:
        fl = FusionLayer()
        fl.ingest("ch", 1.0, 10.0)
        fl.ingest("ch", 2.0, 20.0)
        assert fl.at("ch", t=0.0, interpolation="linear") == 10.0
        assert fl.at("ch", t=5.0, interpolation="linear") == 20.0


# ---------------------------------------------------------------------------
# Use-case driven tests (from the README)
# ---------------------------------------------------------------------------


class TestReadmeUseCases:
    def test_force_reactive_grasping(self) -> None:
        """Force reading interpolated to arm pose at moment of contact."""
        fl = FusionLayer()
        fl.ingest("force_torque", 0.0, {"force": [0.0, 0.0, 1.0]})
        fl.ingest("force_torque", 0.01, {"force": [0.0, 0.0, 2.0]})
        fl.ingest("force_torque", 0.02, {"force": [0.0, 0.0, 3.0]})
        fl.ingest("force_torque", 0.03, {"force": [0.0, 0.0, 4.0]})

        result = fl.at("force_torque", t=0.015, interpolation="linear")
        assert _approx(result["force"][2], 2.5)

    def test_vio_preintegration_window(self) -> None:
        """Full IMU series between camera keyframes."""
        fl = FusionLayer()
        for i in range(100):
            fl.ingest(
                "imu",
                i * 0.005,
                {
                    "ax": float(i) * 0.1,
                    "ay": 0.0,
                    "az": 9.81,
                },
            )

        prev_frame_ts = 0.1
        curr_frame_ts = 0.133
        imu_samples = fl.window("imu", from_t=prev_frame_ts, to_t=curr_frame_ts)
        assert len(imu_samples) >= 6

    def test_force_filtering_window(self) -> None:
        """Moving average over recent F/T samples."""
        fixed_now = 1000.0
        fl = FusionLayer(clock=lambda: fixed_now)
        for i in range(100):
            fl.ingest("ft", fixed_now - 0.001 * (99 - i), float(i))

        recent = fl.window("ft", duration_ms=50)
        assert len(recent) == 51
        avg = sum(recent.values) / len(recent)
        assert avg > 0

    def test_conveyor_pick_and_place(self) -> None:
        """Object position at detection time projected forward."""
        fl = FusionLayer()
        fl.ingest("encoder", 0.0, {"speed_mps": 0.5})
        fl.ingest("encoder", 0.1, {"speed_mps": 0.5})
        fl.ingest("encoder", 0.2, {"speed_mps": 0.5})

        detection_time = 0.05
        result = fl.at("encoder", t=detection_time, interpolation="linear")
        assert _approx(result["speed_mps"], 0.5)


# ---------------------------------------------------------------------------
# Schema drift
# ---------------------------------------------------------------------------


class TestSchemaDrift:
    def test_lerp_dict_warns_on_drift(self) -> None:
        s1 = TimestampedSample(ts=0.0, value={"x": 0.0, "y": 0.0})
        s2 = TimestampedSample(ts=1.0, value={"x": 10.0, "z": 30.0})
        with pytest.warns(UserWarning, match="Schema drift"):
            result = interpolate_linear(s1, s2, 0.5)
        assert _approx(result["x"], 5.0)
        assert result["y"] == 0.0  # only in before → passed through
        assert result["z"] == 30.0  # only in after → passed through

    def test_slerp_dict_warns_on_drift(self) -> None:
        s1 = TimestampedSample(
            ts=0.0, value={"q": Quaternion(), "only_before": 1.0}
        )
        s2 = TimestampedSample(
            ts=1.0, value={"q": Quaternion(), "only_after": 2.0}
        )
        with pytest.warns(UserWarning, match="Schema drift"):
            result = interpolate_slerp(s1, s2, 0.5)
        assert isinstance(result["q"], Quaternion)
        assert result["only_before"] == 1.0
        assert result["only_after"] == 2.0

    def test_no_warning_when_keys_match(self) -> None:
        s1 = TimestampedSample(ts=0.0, value={"x": 0.0, "y": 0.0})
        s2 = TimestampedSample(ts=1.0, value={"x": 10.0, "y": 20.0})
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error")
            result = interpolate_linear(s1, s2, 0.5)
        assert _approx(result["x"], 5.0)
        assert _approx(result["y"], 10.0)


# ---------------------------------------------------------------------------
# List length mismatch warning
# ---------------------------------------------------------------------------


class TestLerpListLengthMismatch:
    def test_warns_on_length_mismatch(self) -> None:
        s1 = TimestampedSample(ts=0.0, value=[0.0, 0.0, 0.0])
        s2 = TimestampedSample(ts=1.0, value=[10.0, 20.0])  # shorter
        with pytest.warns(UserWarning, match="length mismatch"):
            result = interpolate_linear(s1, s2, 0.5)
        assert len(result) == 2  # common prefix length

    def test_no_warning_when_lengths_match(self) -> None:
        s1 = TimestampedSample(ts=0.0, value=[0.0, 0.0])
        s2 = TimestampedSample(ts=1.0, value=[10.0, 20.0])
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error")
            result = interpolate_linear(s1, s2, 0.5)
        assert _approx(result[0], 5.0)
        assert _approx(result[1], 10.0)


# ---------------------------------------------------------------------------
# FusionLayer.at — interpolation validation on unknown channels
# ---------------------------------------------------------------------------


class TestFusionLayerValidation:
    def test_invalid_interpolation_on_known_channel_raises(self) -> None:
        fl = FusionLayer()
        fl.ingest("ch", 0.0, 0.0)
        with pytest.raises(ValueError, match="Invalid interpolation"):
            fl.at("ch", t=0.0, interpolation="cubic")

    def test_invalid_interpolation_on_unknown_channel_also_raises(self) -> None:
        """Unknown channel must not silently swallow a bad interpolation arg."""
        fl = FusionLayer()
        with pytest.raises(ValueError, match="Invalid interpolation"):
            fl.at("nonexistent", t=0.0, interpolation="bad_value")

    def test_valid_interpolations_on_unknown_channel_return_none(self) -> None:
        fl = FusionLayer()
        for interp in ("linear", "slerp", "nearest", "none"):
            assert fl.at("nonexistent", t=0.0, interpolation=interp) is None


# ---------------------------------------------------------------------------
# FusionLayer thread safety
# ---------------------------------------------------------------------------


class TestFusionLayerThreadSafety:
    def test_concurrent_ingest_and_read(self) -> None:
        """Multiple writers to different channels + concurrent readers."""
        fl = FusionLayer()
        errors: list[Exception] = []
        stop = threading.Event()

        def writer(channel: str, base_ts: float) -> None:
            try:
                for i in range(200):
                    fl.ingest(channel, base_ts + i * 0.001, float(i))
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                while not stop.is_set():
                    fl.at("ch_0", t=0.5)
                    fl.window("ch_1", from_t=0.0, to_t=1.0)
                    _ = fl.channels
            except Exception as e:
                errors.append(e)

        writers = [
            threading.Thread(target=writer, args=(f"ch_{i}", float(i * 10)))
            for i in range(4)
        ]
        readers = [threading.Thread(target=reader, daemon=True) for _ in range(2)]

        for r in readers:
            r.start()
        for w in writers:
            w.start()
        for w in writers:
            w.join(timeout=5.0)

        stop.set()
        for r in readers:
            r.join(timeout=2.0)

        assert not errors, f"Thread errors: {errors}"
        assert len(fl.channels) == 4

    def test_concurrent_ingest_same_channel(self) -> None:
        """Two writers to the same channel via FusionLayer."""
        fl = FusionLayer()
        errors: list[Exception] = []

        def writer(base_ts: float) -> None:
            for i in range(100):
                try:
                    fl.ingest("shared", base_ts + i * 0.0001, float(i))
                except ValueError:
                    pass
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=writer, args=(0.0,))
        t2 = threading.Thread(target=writer, args=(0.0,))
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert not errors, f"Thread errors: {errors}"
        assert "shared" in fl.channels
