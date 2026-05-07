"""
Example edge worker: multi-camera detection routing.

Demonstrates a single worker process handling frames from two cameras
(two digital twins) while routing detection results to the correct
twin's Zenoh channel.

Prerequisites:
  - Worker runtime injects ``cw`` as a builtin (no import needed).
  - Ultralytics installed: ``pip install cyberwave[ml]``.
  - Two camera twins provisioned on the same edge device.

Usage:
  This file is loaded by the worker runtime, not run directly.
  Set the twin UUIDs as environment variables before starting the worker::

    export CAMERA_LEFT_TWIN=<uuid-of-left-camera-twin>
    export CAMERA_RIGHT_TWIN=<uuid-of-right-camera-twin>
    cyberwave worker start
"""

import os

# ``cw`` is injected by the worker runtime — no import needed.
# For IDE support, uncomment the following:
# from cyberwave import Cyberwave; cw: Cyberwave

CAMERA_LEFT = os.environ["CAMERA_LEFT_TWIN"]
CAMERA_RIGHT = os.environ["CAMERA_RIGHT_TWIN"]

model = cw.models.load("yolov8n")  # type: ignore[name-defined]  # noqa: F821


# ── Per-camera detection hooks ───────────────────────────────────────
# Each hook receives frames from a specific twin.  Passing
# ``twin_uuid=ctx.twin_uuid`` to ``model.predict()`` ensures the
# detection bounding boxes are published back to that twin's
# ``detections/ultralytics`` channel — not to a single hardcoded twin.


@cw.on_frame(CAMERA_LEFT, sensor="default")  # type: ignore[name-defined]  # noqa: F821
def on_left_frame(frame, ctx):
    model.predict(frame, confidence=0.5, twin_uuid=ctx.twin_uuid)


@cw.on_frame(CAMERA_RIGHT, sensor="default")  # type: ignore[name-defined]  # noqa: F821
def on_right_frame(frame, ctx):
    model.predict(frame, confidence=0.5, twin_uuid=ctx.twin_uuid)


# ── Cross-twin synchronized hook (optional) ──────────────────────────
# Fires when both cameras have a frame within 50 ms of each other.
# Useful for stereo vision, sensor fusion, or cross-camera tracking.


@cw.on_synchronized(  # type: ignore[name-defined]  # noqa: F821
    twin_channels={
        "left": (CAMERA_LEFT, "frames/default"),
        "right": (CAMERA_RIGHT, "frames/default"),
    },
    tolerance_ms=50.0,
)
def on_stereo_pair(samples, ctx):
    left_sample = samples["left"]
    right_sample = samples["right"]

    cw.publish_event(  # type: ignore[name-defined]  # noqa: F821
        CAMERA_LEFT,
        "stereo_pair_received",
        {
            "left_ts": left_sample.timestamp,
            "right_ts": right_sample.timestamp,
            "twin_uuids": ctx.metadata.get("twin_uuids", []),
        },
    )
