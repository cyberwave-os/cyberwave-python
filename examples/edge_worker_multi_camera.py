"""
Edge Worker — multi-camera detection with cross-twin synchronization.

Handles frames from two camera twins and routes detections to each twin's channel.
This file is loaded by the worker runtime (`cyberwave worker start`), not run directly.

Env vars:
    CAMERA_LEFT_TWIN     UUID of the left camera twin
    CAMERA_RIGHT_TWIN    UUID of the right camera twin

Requirements:
    pip install cyberwave[ml]
"""

import os

# ``cw`` is injected by the worker runtime.
# For IDE support: from cyberwave import Cyberwave; cw: Cyberwave

CAMERA_LEFT = os.environ["CAMERA_LEFT_TWIN"]
CAMERA_RIGHT = os.environ["CAMERA_RIGHT_TWIN"]

model = cw.models.load("yolov8n")  # type: ignore[name-defined]  # noqa: F821


@cw.on_frame(CAMERA_LEFT, sensor="default")  # type: ignore[name-defined]  # noqa: F821
def on_left_frame(frame, ctx):
    model.predict(frame, confidence=0.5, twin_uuid=ctx.twin_uuid)


@cw.on_frame(CAMERA_RIGHT, sensor="default")  # type: ignore[name-defined]  # noqa: F821
def on_right_frame(frame, ctx):
    model.predict(frame, confidence=0.5, twin_uuid=ctx.twin_uuid)


@cw.on_synchronized(  # type: ignore[name-defined]  # noqa: F821
    twin_channels={
        "left": (CAMERA_LEFT, "frames/default"),
        "right": (CAMERA_RIGHT, "frames/default"),
    },
    tolerance_ms=50.0,
)
def on_stereo_pair(samples, ctx):
    cw.publish_event(  # type: ignore[name-defined]  # noqa: F821
        CAMERA_LEFT,
        "stereo_pair_received",
        {
            "left_ts": samples["left"].timestamp,
            "right_ts": samples["right"].timestamp,
        },
    )
