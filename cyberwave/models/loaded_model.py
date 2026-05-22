"""User-facing wrapper around a loaded ML model.

Returned by ``ModelManager.load()`` / ``ModelManager.load_from_file()``.
Provides a stable ``.predict()`` API regardless of the underlying
runtime backend.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from cyberwave.models.runtimes.base import ModelRuntime
from cyberwave.models.types import PredictionResult

if TYPE_CHECKING:
    from cyberwave.data.api import DataBus

logger = logging.getLogger(__name__)

_LATENCY_WINDOW = 100  # rolling window size for latency percentiles


class LoadedModel:
    """A loaded ML model ready for inference."""

    def __init__(
        self,
        *,
        name: str,
        runtime: ModelRuntime,
        model_handle: Any,
        device: str = "cpu",
        model_path: str = "",
        data_bus: DataBus | None = None,
    ) -> None:
        self._name = name
        self._runtime = runtime
        self._model_handle = model_handle
        self._device = device
        self._model_path = model_path
        self._data_bus = data_bus

        # Inference latency tracking (thread-safe rolling window).
        self._latency_lock = threading.Lock()
        self._latency_window: collections.deque[float] = collections.deque(
            maxlen=_LATENCY_WINDOW
        )
        self._inference_count: int = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def runtime(self) -> str:
        return self._runtime.name

    @property
    def device(self) -> str:
        return self._device

    def predict(
        self,
        input_data: Any,
        *,
        confidence: float = 0.5,
        classes: list[str] | None = None,
        twin_uuid: str | None = None,
        **kwargs: Any,
    ) -> PredictionResult:
        """Run inference on *input_data*.

        Args:
            twin_uuid: Optional override to route published detections to a
                specific twin instead of the default one bound to the data bus.
        """
        t0 = time.monotonic()
        result = self._runtime.predict(
            self._model_handle,
            input_data,
            confidence=confidence,
            classes=classes,
            **kwargs,
        )
        latency_ms = (time.monotonic() - t0) * 1000.0
        with self._latency_lock:
            self._latency_window.append(latency_ms)
            self._inference_count += 1

        if result.detections:
            summary = ", ".join(
                f"{d.label} {d.confidence:.0%}" for d in result.detections
            )
            logger.info("[%s] %d detection(s): %s", self._name, len(result.detections), summary)

        # Publish every inference — including empty batches — so overlay
        # consumers (e.g. the camera driver's detection cache) see a
        # heartbeat at the worker's inference cadence and don't fall into
        # their staleness cutoff when the scene transiently has nothing
        # to detect.  Empty payloads naturally render as "no box" without
        # phantom last-known overlays lingering past object loss.
        self._publish_detections(result, input_data, twin_uuid=twin_uuid)
        return result

    def warm_up(
        self,
        input_shape: tuple[int, ...] | None = None,
        *,
        confidence: float = 0.5,
    ) -> tuple[float, float]:
        """Run two dummy inferences to warm up JIT / allocations.

        Returns ``(cold_ms, warm_ms)`` — the latency of the first and
        second inference passes on a zero-filled input.
        """
        import numpy as np

        shape = input_shape or (640, 640, 3)
        dummy = np.zeros(shape, dtype=np.uint8)

        t0 = time.monotonic()
        try:
            self._runtime.predict(
                self._model_handle, dummy, confidence=confidence,
            )
        except Exception:
            logger.debug("Warm-up cold pass raised (non-fatal)", exc_info=True)
        cold_ms = (time.monotonic() - t0) * 1000.0

        t1 = time.monotonic()
        try:
            self._runtime.predict(
                self._model_handle, dummy, confidence=confidence,
            )
        except Exception:
            logger.debug("Warm-up hot pass raised (non-fatal)", exc_info=True)
        warm_ms = (time.monotonic() - t1) * 1000.0

        logger.info(
            "[%s] Warm-up complete: cold=%.1f ms, warm=%.1f ms",
            self._name,
            cold_ms,
            warm_ms,
        )
        return cold_ms, warm_ms

    def inference_stats(self) -> dict[str, Any]:
        """Return inference latency statistics for monitoring.

        Returns a dict with ``count``, ``avg_ms``, ``p95_ms``, ``p99_ms``
        computed from a rolling window of recent inferences.
        """
        with self._latency_lock:
            count = self._inference_count
            if not self._latency_window:
                return {"name": self._name, "device": self._device, "count": count}
            samples = sorted(self._latency_window)
        n = len(samples)
        avg = sum(samples) / n
        p95 = samples[int(n * 0.95)] if n >= 2 else samples[-1]
        p99 = samples[int(n * 0.99)] if n >= 2 else samples[-1]
        return {
            "name": self._name,
            "device": self._device,
            "count": count,
            "avg_ms": round(avg, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
        }

    def _publish_detections(
        self,
        result: PredictionResult,
        input_data: Any,
        *,
        twin_uuid: str | None = None,
    ) -> None:
        """Publish detection results via Zenoh so the driver can draw overlays."""
        if self._data_bus is None:
            return
        try:
            import json as _json

            import numpy as np

            if isinstance(input_data, np.ndarray) and input_data.ndim >= 2:
                h, w = input_data.shape[:2]
            else:
                h, w = 0, 0

            payload_dict = {
                "detections": [
                    {
                        "label": d.label,
                        "confidence": round(d.confidence, 3),
                        "x1": round(d.bbox.x1),
                        "y1": round(d.bbox.y1),
                        "x2": round(d.bbox.x2),
                        "y2": round(d.bbox.y2),
                    }
                    for d in result.detections
                ],
                "frame_width": w,
                "frame_height": h,
                "timestamp": time.time(),
            }

            raw_bytes = _json.dumps(payload_dict, separators=(",", ":")).encode()
            topic = f"detections/{self._runtime.name}"
            self._data_bus.publish_raw(topic, raw_bytes, twin_uuid=twin_uuid)
        except Exception:
            logger.debug("Failed to publish detections via data bus", exc_info=True)

    def __repr__(self) -> str:
        return (
            f"LoadedModel(name={self._name!r}, "
            f"runtime={self.runtime!r}, "
            f"device={self._device!r})"
        )
