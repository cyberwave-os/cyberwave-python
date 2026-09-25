"""
Security Pipeline Worker — privacy-preserving anonymisation for two cameras.

Runs person detection and publishes pixelated frames to the filtered channel.
Loaded by the worker runtime (`cyberwave worker start`), not run directly.

Env vars:
    CAMERA_1_TWIN    UUID of the first camera twin
    CAMERA_2_TWIN    UUID of the second camera twin

Requirements:
    pip install cyberwave[ml]
"""

import os

import numpy as np

# ``cw`` is injected by the worker runtime.
# For IDE support: from cyberwave import Cyberwave; cw: Cyberwave

from cyberwave.data import FILTERED_FRAME_CHANNEL  # noqa: E402
from cyberwave.vision import anonymize_frame  # noqa: E402

CAMERA_1 = os.environ["CAMERA_1_TWIN"]
CAMERA_2 = os.environ["CAMERA_2_TWIN"]

model = cw.models.load("yolov8n")  # type: ignore[name-defined]  # noqa: F821


def _anonymise(frame, ctx):
    result = model.predict(frame, classes=["person"], confidence=0.4)

    if any(d.label == "person" for d in result.detections):
        out = anonymize_frame(
            frame, result.detections, mode="pixelate", draw_skeleton=False
        )
    else:
        out = np.zeros_like(frame)

    cw.data.publish(FILTERED_FRAME_CHANNEL, out, twin_uuid=ctx.twin_uuid)  # type: ignore[name-defined]  # noqa: F821

    for det in result.detections:
        if det.area_ratio > 0.3:
            cw.publish_event(  # type: ignore[name-defined]  # noqa: F821
                ctx.twin_uuid,
                "person_too_close",
                {
                    "area_ratio": round(det.area_ratio, 3),
                    "confidence": round(det.confidence, 3),
                },
            )


@cw.on_frame(CAMERA_1, sensor="default")  # type: ignore[name-defined]  # noqa: F821
def on_camera_1_frame(frame, ctx):
    _anonymise(frame, ctx)


@cw.on_frame(CAMERA_2, sensor="default")  # type: ignore[name-defined]  # noqa: F821
def on_camera_2_frame(frame, ctx):
    _anonymise(frame, ctx)
