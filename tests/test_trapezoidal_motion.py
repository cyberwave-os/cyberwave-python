"""trapezoidal_motion: duration formula, sampling, monotonic approach."""

from __future__ import annotations

import pytest

pytest.importorskip("scipy")

from cyberwave.driver.control.motion import trapezoidal_motion
from cyberwave.driver.control.types import MotionRequest


def _req(current, target, velocities, rate_hz=50.0):
    return MotionRequest(current=current, target=target, velocities=velocities, rate_hz=rate_hz)


def test_empty_target_returns_no_waypoints():
    assert trapezoidal_motion(_req({}, {}, {})) == ()


def test_satisfied_target_returns_no_waypoints():
    assert trapezoidal_motion(_req({"j1": 0.5}, {"j1": 0.5}, {"j1": 1.0})) == ()


def test_duration_is_max_delta_over_velocity():
    # j1: |1.0|/1.0 = 1.0s ; j2: |0.5|/2.0 = 0.25s -> duration 1.0s
    wps = trapezoidal_motion(
        _req({"j1": 0.0, "j2": 0.0}, {"j1": 1.0, "j2": 0.5}, {"j1": 1.0, "j2": 2.0})
    )
    assert wps[-1].time_from_start == pytest.approx(1.0)


def test_waypoint_count_matches_rate():
    wps = trapezoidal_motion(_req({"j1": 0.0}, {"j1": 1.0}, {"j1": 1.0}, rate_hz=50.0))
    # 1.0s at 50Hz -> 51 samples (inclusive endpoints)
    assert len(wps) == 51
    assert wps[0].time_from_start == pytest.approx(0.0)


def test_positions_start_at_current_end_at_target_monotonic():
    wps = trapezoidal_motion(_req({"j1": 0.2}, {"j1": 1.2}, {"j1": 1.0}))
    pos = [w.positions["j1"] for w in wps]
    assert pos[0] == pytest.approx(0.2, abs=1e-9)
    assert pos[-1] == pytest.approx(1.2, abs=1e-9)
    assert all(b >= a - 1e-12 for a, b in zip(pos, pos[1:]))  # monotonic up


def test_negative_delta_descends():
    wps = trapezoidal_motion(_req({"j1": 1.0}, {"j1": 0.0}, {"j1": 2.0}))
    pos = [w.positions["j1"] for w in wps]
    assert pos[0] == pytest.approx(1.0, abs=1e-9)
    assert pos[-1] == pytest.approx(0.0, abs=1e-9)
    assert wps[-1].time_from_start == pytest.approx(0.5)


def test_velocities_zero_at_endpoints_positive_mid():
    wps = trapezoidal_motion(_req({"j1": 0.0}, {"j1": 1.0}, {"j1": 1.0}))
    assert wps[0].velocities["j1"] == pytest.approx(0.0, abs=1e-9)
    assert wps[-1].velocities["j1"] == pytest.approx(0.0, abs=1e-9)
    mid = wps[len(wps) // 2].velocities["j1"]
    assert mid > 1.0  # trapezoid peak exceeds avg velocity (duration contract)


def test_short_move_still_two_waypoints_minimum():
    wps = trapezoidal_motion(_req({"j1": 0.0}, {"j1": 0.001}, {"j1": 10.0}, rate_hz=10.0))
    assert len(wps) >= 2
    assert wps[-1].positions["j1"] == pytest.approx(0.001, abs=1e-12)
