"""
Example edge worker: detect people near the robot and publish events.

This file demonstrates the worker module pattern.  In production it would
live at ``/app/workers/detect_people.py`` inside the worker container.

Prerequisites:
  - Worker runtime injects ``cw`` as a builtin (no import needed).
  - Ultralytics installed: ``pip install cyberwave[ml]``.
  - Model weights available in the model cache.

Usage:
  This file is loaded by the worker runtime, not run directly.
  The runtime calls ``cw.run_edge_workers()`` after importing all worker modules.
"""

# ``cw`` is injected by the worker runtime — no import needed.
# For IDE support, uncomment the following:
# from cyberwave import Cyberwave; cw: Cyberwave

model = cw.models.load("yolov8n")  # type: ignore[name-defined]  # noqa: F821
twin_uuid = cw.config.twin_uuid  # type: ignore[name-defined]  # noqa: F821


@cw.on_frame(twin_uuid, sensor="front")  # type: ignore[name-defined]  # noqa: F821
def detect_people(frame, ctx):
    """Called for every new frame from the front camera."""
    results = model.predict(frame, classes=["person"], confidence=0.5)

    for det in results:
        if det.area_ratio > 0.3:
            cw.publish_event(  # type: ignore[name-defined]  # noqa: F821
                twin_uuid,
                "person_too_close",
                {
                    "distance_estimate": "near",
                    "detections": len(results),
                    "model": "yolov8n",
                    "frame_ts": ctx.timestamp,
                },
            )
