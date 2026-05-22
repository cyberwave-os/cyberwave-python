"""Multi-camera security pipeline: privacy-preserving anonymisation.

A single worker process consumes raw frames from two camera twins,
runs YOLOv8 person detection, and publishes an anonymised version
(every person pixelated) back to each camera's ``frames/filtered``
Zenoh channel (the SDK constant :data:`FILTERED_FRAME_CHANNEL`). The
generic-camera driver, when started with
``CYBERWAVE_METADATA_FRAME_FILTER_ENABLED=true`` on each camera twin,
substitutes that frame into the WebRTC stream BEFORE the bytes leave
the edge.

Why pixelate and not stick figures?

  - A pixelated person still reads as "a person, doing something" to a
    human reviewer — useful for retail-floor / warehouse / hospital
    monitoring where presence and motion matter.
  - Stick figures are visually fragile: occlusion, low light, or an
    unusual pose drops keypoints and the operator sees a floating
    hand or nothing at all.
  - Pixelate doesn't need a pose model, so per-frame cost is ~2× lower
    than pose+skeleton on CPU-bound edge devices.

Operator-visible behaviour:

  - Frontend WebRTC stream shows mosaicked persons, never raw faces or bodies.
  - High-level events (``person_detected``, ``person_too_close``) flow
    over MQTT to the cloud as usual.
  - Raw ``frames/*`` Zenoh channels stay LOCAL — the Zenoh→MQTT bridge
    does NOT forward them.

Environment:

  CAMERA_1_TWIN   UUID of the first camera twin.
  CAMERA_2_TWIN   UUID of the second camera twin.

Run via the standard worker container; the runtime injects ``cw``.
"""

import os

import numpy as np

# ``cw`` is injected by the worker runtime — no import needed.
# For IDE support, uncomment the following:
# from cyberwave import Cyberwave; cw: Cyberwave

# The driver subscribes to this exact channel name. Import the constant
# rather than hard-coding "frames/filtered" so the wire contract stays
# in one place — if the SDK ever renames it, this example follows.
from cyberwave.data import FILTERED_FRAME_CHANNEL  # noqa: E402
from cyberwave.vision import anonymize_frame  # noqa: E402

CAMERA_1 = os.environ["CAMERA_1_TWIN"]
CAMERA_2 = os.environ["CAMERA_2_TWIN"]

# Plain detector — no pose head. Cheaper per frame than the pose variant
# and all we need for "obscure every person" policy. If you later want
# posture overlays, swap this for "yolov8n-pose-onnx" and pass
# ``draw_skeleton=True`` below.
model = cw.models.load("yolov8n")  # type: ignore[name-defined]  # noqa: F821


def _anonymise(frame, ctx):
    """Shared per-frame logic — runs for both cameras.

    Steps:
      1. Run person detection on the raw frame.
      2. Build an anonymised frame by pixelating every person bbox.
         ``draw_skeleton=False`` is explicit (defensive) so anyone
         swapping in a pose model later doesn't accidentally re-enable
         the overlay.
      3. Publish the anonymised frame to ``FILTERED_FRAME_CHANNEL``
         (``"frames/filtered"``). The generic-camera driver subscribes
         to this exact channel per-twin when
         ``CYBERWAVE_METADATA_FRAME_FILTER_ENABLED=true``. Privacy
         fail-closed: when the model returns zero ``person`` matches
         (sub-threshold confidence, occluded subject, partial body)
         ``anonymize_frame`` returns the input frame *untouched* — so
         we substitute ``np.zeros_like(frame)`` and publish that
         instead. Without this gate, a transient detection miss on a
         single frame would leak raw pixels through a fresh, well-
         formed publish, and the driver — which has no way to tell
         filtered from raw — would substitute it into the WebRTC
         stream. Mirrors the gate the workflow ``anonymize`` node's
         codegen emits automatically.
      4. Emit a high-level event when a person is "too close" (large
         bbox area). The event flows to the cloud over MQTT.
    """
    result = model.predict(frame, classes=["person"], confidence=0.4)

    if any(d.label == "person" for d in result.detections):
        out = anonymize_frame(
            frame,
            result.detections,
            mode="pixelate",
            draw_skeleton=False,
        )
    else:
        out = np.zeros_like(frame)
    cw.data.publish(  # type: ignore[name-defined]  # noqa: F821
        FILTERED_FRAME_CHANNEL,
        out,
        twin_uuid=ctx.twin_uuid,
    )

    for det in result.detections:
        if det.area_ratio > 0.3:
            cw.publish_event(  # type: ignore[name-defined]  # noqa: F821
                ctx.twin_uuid,
                "person_too_close",
                {
                    "area_ratio": round(det.area_ratio, 3),
                    "confidence": round(det.confidence, 3),
                    "frame_ts": ctx.timestamp,
                },
            )


@cw.on_frame(CAMERA_1, sensor="default")  # type: ignore[name-defined]  # noqa: F821
def on_camera_1_frame(frame, ctx):
    _anonymise(frame, ctx)


@cw.on_frame(CAMERA_2, sensor="default")  # type: ignore[name-defined]  # noqa: F821
def on_camera_2_frame(frame, ctx):
    _anonymise(frame, ctx)
