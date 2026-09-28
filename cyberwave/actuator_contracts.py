# !! GENERATED — do not edit directly. Run python-sdk-gen.sh to regenerate.

"""Admission contract for reusable learned joint actuators."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

ACTUATOR_LSTM_CONTRACT = "control.actuator_lstm.v1"


def validate_actuator_config(config: Mapping[str, Any]) -> None:
    """Validate declarations; runtime separately verifies graph, timing and plant.

    PD remains the implicit legacy actuator. Learned actuators consume position
    error (rad), velocity (rad/s), and explicit per-joint recurrent state, and
    return torque (Nm). No robot-name inference or substitution is allowed.
    """
    kind = config.get("actuator_model", "pd")
    if kind == "pd":
        return
    if kind != "actuator_net_lstm":
        raise ValueError(f"Unsupported actuator model {kind!r}.")
    network = config.get("actuator_network")
    if (
        not isinstance(network, Mapping)
        or network.get("contract") != ACTUATOR_LSTM_CONTRACT
    ):
        raise ValueError(
            "actuator_net_lstm needs its converted ONNX actuator network "
            f"with contract {ACTUATOR_LSTM_CONTRACT}."
        )
    path = network.get("model_path")
    if (
        not isinstance(path, str)
        or not path.endswith(".onnx")
        or "\\" in path
        or PurePosixPath(path).is_absolute()
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ValueError("Actuator model_path must be a relative ONNX artifact path.")
    digest = network.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Actuator network needs a SHA-256 checksum.")
    dt = network.get("step_dt")
    if (
        isinstance(dt, bool)
        or not isinstance(dt, int | float)
        or not 0 < dt <= 0.1
        or not math.isfinite(dt)
    ):
        raise ValueError(
            "Actuator step_dt must declare its trained timestep in seconds."
        )
    limit_model = network.get("effort_model")
    if limit_model not in ("constant", "dc_motor"):
        raise ValueError(
            "Actuator effort_model must declare constant or dc_motor clipping."
        )
    fields = (
        ("effort_limit", "saturation_effort", "velocity_limit")
        if limit_model == "dc_motor"
        else ("effort_limit",)
    )
    for field in fields:
        value = config.get(field)
        values = value if isinstance(value, list | tuple) else [value]
        if not values or any(
            type(item) not in (int, float)
            or not 0 < item <= 1e308
            or not math.isfinite(item)
            for item in values
        ):
            raise ValueError(f"Learned actuator {field} needs finite positive limits.")
