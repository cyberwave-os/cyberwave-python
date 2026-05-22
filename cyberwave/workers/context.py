"""Per-sample context passed to every hook callback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HookContext:
    """Per-sample context passed as the second argument to every hook callback.

    Attributes:
        timestamp: Sample time from the data layer (not wall-clock at dispatch).
        channel: Hook-level channel name — ``"frames"`` for a wildcard
            sensor hook, ``"frames/<sensor>"`` for a pinned sensor, or a
            bare name like ``"imu"`` for sensor-less channels.
        sensor_name: The actual sensor that produced this sample, as
            declared in the twin's asset (e.g. ``"color_camera"``).  For
            pinned hooks this equals the ``sensor=`` arg; for wildcard
            hooks it's taken from the wire key so downstream code can
            route per-sensor.  Falls back to ``"default"`` for bare
            channels and when the wire key is missing.
        twin_uuid: Set by the runtime from ``cw.config.twin_uuid``.
            ``None`` when not yet bound to a twin.
        metadata: Extra fields the data layer attached (e.g. frame
            dimensions, encoding).
    """

    timestamp: float
    channel: str
    sensor_name: str = "default"
    twin_uuid: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
