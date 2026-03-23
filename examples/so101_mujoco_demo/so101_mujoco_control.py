"""
SO-101 Sine-Wave Controller
============================

This is the EDITABLE controller for the SO-101 MuJoCo demo.

By default, ``sine_control`` sweeps every joint through a gentle sine wave so
the arm moves visibly the moment you run the demo.

Feel free to replace the body of ``sine_control`` with your own logic.
The only contract is the signature::

    sine_control(model, data, sim_time: float) -> None

``model`` and ``data`` are standard MuJoCo objects.
Write joint commands to ``data.ctrl[i]`` for actuator *i*.
Return value is ignored.

----

If you want *no* automated motion at all and prefer to play with the arm
manually via the MuJoCo viewer's slider panel (Ctrl+M), use manual mode::

    just run-manual
    # or:
    CONTROL_MODE=manual python so101_mujoco_demo.py run
"""

import numpy as np
import mujoco


def sine_control(model, data, sim_time: float) -> None:
    """Default controller: sweep each joint through a slow sine wave.

    Each actuator receives an independent phase offset so the arm makes a
    flowing, wave-like motion that sweeps through most of the joint range.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  This function is intentionally simple and meant to be replaced.    │
    │  Edit freely — the demo runner calls it every simulation step.      │
    └─────────────────────────────────────────────────────────────────────┘
    """
    n = max(1, model.nu)

    for i in range(model.nu):
        # Skip non-joint actuators (e.g. tendon, site)
        if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
            continue

        jnt_id = int(model.actuator_trnid[i, 0])
        if jnt_id < 0 or jnt_id >= model.njnt:
            continue

        # Joint limits
        if model.jnt_limited[jnt_id]:
            lo, hi = model.jnt_range[jnt_id]
        else:
            lo, hi = -np.pi, np.pi

        center = 0.5 * (lo + hi)
        amp    = 0.4 * 0.5 * (hi - lo)    # ±40 % of the half-range
        phase  = 2.0 * np.pi * i / n       # stagger each joint
        value  = np.sin(0.3 * sim_time + phase)   # ~0.3 rad/s oscillation

        is_position_actuator = model.actuator_biastype[i] == 1  # mjBIAS_AFFINE

        if is_position_actuator:
            data.ctrl[i] = float(np.clip(center + amp * value, lo, hi))
        else:
            # Torque actuator: scale by forcerange when available
            fr    = model.actuator_forcerange[i]
            scale = 0.3 * max(abs(fr[0]), abs(fr[1])) if fr[1] > fr[0] else 0.3
            data.ctrl[i] = float(scale * value)
