"""
Edge Worker — Hailo-accelerated person detection.

Same shape as ``edge_worker_detect_people.py`` but targets a Hailo HEF
binary. Edge Core's model manager resolves the slug to a Hailo HEF in
the local cache, ``cw.models.load`` picks the ``hailo`` runtime from
the ``.hef`` extension, and inference runs on the Hailo-8 accelerator
via the ``hailo_platform`` Python bindings (shipped on the
``cyberwaveos/edge-ml-worker-hailo`` worker image, not bundled with
the SDK wheel).

This file is loaded by the worker runtime (``cyberwave worker start``),
not run directly.

Requirements (host):
    * Raspberry Pi 5 + AI HAT+ with ``/dev/hailo0`` present
    * HailoRT 4.23.0 driver installed (``apt install hailo-all``)

Requirements (container):
    * ``cyberwaveos/edge-ml-worker-hailo:<tag>`` — preinstalled HailoRT
      and ``hailo_platform`` matched to the host driver.
"""
from cyberwave import Cyberwave

cw = Cyberwave()

model = cw.models.load("yolov8s_h8")  # type: ignore[name-defined]  # noqa: F821
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
