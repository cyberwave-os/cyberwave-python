"""FusionLayer demo: multi-sensor time-aware interpolation.

This example demonstrates ``cyberwave.data.fusion.FusionLayer`` with two
simulated sensor streams:

``joint_states`` — published at 100 Hz
    ``{"j1": float, "j2": float, "j3": float}`` (joint angles in radians)

``imu`` — published at 200 Hz
    ``{"ax": float, "ay": float, "az": float, "q": Quaternion(...)}``
    (linear acceleration + orientation quaternion)

What is shown
-------------
1. ``fusion.ingest(channel, ts, value)`` — feeding raw samples into per-channel
   ring buffers.
2. ``fusion.at(channel, t=, interpolation="linear")`` — interpolated joint pose
   at an arbitrary query timestamp.
3. ``fusion.at(channel, t=, interpolation="slerp")`` — SLERP-interpolated
   orientation at the same timestamp.
4. ``fusion.window(channel, duration_ms=50)`` — sliding window of recent IMU
   samples.

No Zenoh or network connection is required — the example is entirely
self-contained.

Prerequisites
-------------
::

    pip install cyberwave

Usage
-----
::

    python examples/zenoh_data_fusion.py
    python examples/zenoh_data_fusion.py --duration 0.2 --query-count 5
"""

from __future__ import annotations

import argparse
import math
import time

from cyberwave.data.fusion import FusionLayer, Quaternion


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def _joint_states_at(t: float) -> dict:
    """Return simulated joint angles at time *t* (sinusoidal trajectories)."""
    return {
        "j1": math.sin(2 * math.pi * t),
        "j2": math.cos(2 * math.pi * t * 0.5),
        "j3": math.sin(2 * math.pi * t * 1.5),
    }


def _imu_at(t: float) -> dict:
    """Return simulated IMU reading at time *t*.

    Orientation is a quaternion rotating about the Z-axis at 45 °/s.
    """
    angle = math.pi / 4 * t  # 45 deg/s
    q = Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(angle / 2),
        w=math.cos(angle / 2),
    )
    return {
        "ax": math.sin(2 * math.pi * t * 2),
        "ay": math.cos(2 * math.pi * t * 2),
        "az": 9.81,
        "q": q,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FusionLayer demo: multi-sensor interpolation"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.5,
        metavar="S",
        help="Simulated sensor data duration in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--query-count",
        type=int,
        default=4,
        metavar="N",
        help="Number of interpolation queries to run (default: 4)",
    )
    args = parser.parse_args()

    duration = args.duration
    joint_hz = 100.0
    imu_hz = 200.0

    # Use a synthetic clock that starts at t=0 so timestamps are readable.
    t0_wall = time.time()

    def synthetic_clock() -> float:
        return time.time() - t0_wall

    fusion = FusionLayer(default_capacity=2000, clock=synthetic_clock)

    # ── Ingest joint_states ──────────────────────────────────────────────────
    joint_interval = 1.0 / joint_hz
    t_joint = 0.0
    joint_count = 0
    while t_joint <= duration:
        fusion.ingest("joint_states", t_joint, _joint_states_at(t_joint))
        t_joint += joint_interval
        joint_count += 1

    # ── Ingest imu ───────────────────────────────────────────────────────────
    imu_interval = 1.0 / imu_hz
    t_imu = 0.0
    imu_count = 0
    while t_imu <= duration:
        fusion.ingest("imu", t_imu, _imu_at(t_imu))
        t_imu += imu_interval
        imu_count += 1

    print(f"Ingested {joint_count} joint_states samples at {joint_hz:.0f} Hz")
    print(f"Ingested {imu_count}  imu samples          at {imu_hz:.0f} Hz")
    print(f"Simulation window: 0.000 s → {duration:.3f} s\n")

    # ── Point queries ────────────────────────────────────────────────────────
    print("── at() — interpolated point reads ─────────────────────────────────")
    print(
        f"{'t (s)':>8}  {'j1 (interp)':>12}  {'j1 (true)':>10}"
        f"  {'ax (IMU)':>10}  {'qz (slerp)':>10}"
    )
    print("-" * 65)

    step = duration / (args.query_count + 1)
    for i in range(1, args.query_count + 1):
        query_t = i * step

        joints = fusion.at("joint_states", t=query_t, interpolation="linear")
        imu = fusion.at("imu", t=query_t, interpolation="slerp")

        j1_interp = joints["j1"] if joints else float("nan")
        j1_true = _joint_states_at(query_t)["j1"]
        ax = imu["ax"] if imu else float("nan")
        qz = imu["q"].z if (imu and isinstance(imu.get("q"), Quaternion)) else float("nan")

        print(
            f"{query_t:8.4f}  {j1_interp:+12.6f}  {j1_true:+10.6f}"
            f"  {ax:+10.6f}  {qz:+10.6f}"
        )

    # ── Window query ─────────────────────────────────────────────────────────
    window_ms = 50.0
    mid_t = duration / 2.0
    w_from = mid_t - window_ms / 2000.0
    w_to = mid_t + window_ms / 2000.0

    imu_window = fusion.window("imu", from_t=w_from, to_t=w_to)

    print(f"\n── window() — trailing {window_ms:.0f} ms IMU window around t={mid_t:.3f} s ──")
    print(f"  from_t : {imu_window.from_t:.4f} s")
    print(f"  to_t   : {imu_window.to_t:.4f} s")
    print(f"  samples: {len(imu_window)}")
    expected_in_window = int(window_ms / 1000.0 * imu_hz) + 1
    print(f"  expected ≈ {expected_in_window} (at {imu_hz:.0f} Hz over {window_ms:.0f} ms)")

    if imu_window:
        first = imu_window.samples[0]
        last = imu_window.samples[-1]
        print(
            f"  first ts: {first.ts:.4f} s  ax={first.value['ax']:+.4f}"
        )
        print(
            f"  last  ts: {last.ts:.4f} s  ax={last.value['ax']:+.4f}"
        )

    # ── Duration-based trailing window ───────────────────────────────────────
    # Rewind the synthetic clock so duration_ms=50 captures real data.
    # Because the fusion layer uses `synthetic_clock()` and all samples were
    # ingested at timestamps 0 … duration, we query with explicit from_t/to_t
    # for the last 50 ms of the recorded window.
    trailing_window = fusion.window(
        "imu", from_t=duration - window_ms / 1000.0, to_t=duration
    )
    print(
        f"\n── window() — last {window_ms:.0f} ms of recording ──────────────────────"
    )
    print(f"  samples in last {window_ms:.0f} ms: {len(trailing_window)}")

    print("\nFusionLayer demo complete.")


if __name__ == "__main__":
    main()
