"""
FusionLayer — multi-sensor time-aware interpolation.

Ingests simulated joint_states (100 Hz) and IMU (200 Hz), then queries
interpolated values at arbitrary timestamps.

Requirements:
    pip install cyberwave
"""

from __future__ import annotations

import math
import time

from cyberwave.data.fusion import FusionLayer, Quaternion

fusion = FusionLayer(default_capacity=2000, clock=time.time)

DURATION = 0.5

# Ingest joint_states at 100 Hz
t = 0.0
while t <= DURATION:
    fusion.ingest(
        "joint_states",
        t,
        {
            "j1": math.sin(2 * math.pi * t),
            "j2": math.cos(2 * math.pi * t * 0.5),
        },
    )
    t += 1 / 100

# Ingest IMU at 200 Hz
t = 0.0
while t <= DURATION:
    angle = math.pi / 4 * t
    fusion.ingest(
        "imu",
        t,
        {
            "ax": math.sin(2 * math.pi * t * 2),
            "az": 9.81,
            "q": Quaternion(x=0, y=0, z=math.sin(angle / 2), w=math.cos(angle / 2)),
        },
    )
    t += 1 / 200

# Query interpolated values
print("Interpolated point reads:")
for i in range(1, 5):
    query_t = i * DURATION / 5
    joints = fusion.at("joint_states", t=query_t, interpolation="linear")
    imu = fusion.at("imu", t=query_t, interpolation="slerp")
    j1 = joints["j1"] if joints else float("nan")
    qz = imu["q"].z if (imu and isinstance(imu.get("q"), Quaternion)) else float("nan")
    print(f"  t={query_t:.3f}  j1={j1:+.4f}  qz={qz:+.4f}")

# Window query
w = fusion.window("imu", from_t=0.2, to_t=0.25)
print(f"\nIMU window [0.2, 0.25]: {len(w)} samples")
