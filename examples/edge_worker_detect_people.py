"""
Edge Worker — detect people near the robot and publish events.

This file is loaded by the worker runtime (`cyberwave worker start`), not run directly.

Requirements:
    pip install cyberwave[ml]
"""

# ``cw`` is injected by the worker runtime.
# For IDE support: from cyberwave import Cyberwave; cw: Cyberwave

model = cw.models.load("yolov8n")  # type: ignore[name-defined]  # noqa: F821
twin_uuid = cw.config.twin_uuid  # type: ignore[name-defined]  # noqa: F821


@cw.on_frame(twin_uuid, sensor="front")  # type: ignore[name-defined]  # noqa: F821
def detect_people(frame, ctx):
    results = model.predict(frame, classes=["person"], confidence=0.5)
    for det in results:
        if det.area_ratio > 0.3:
            cw.publish_event(  # type: ignore[name-defined]  # noqa: F821
                twin_uuid,
                "person_too_close",
                {"detections": len(results), "frame_ts": ctx.timestamp},
            )
