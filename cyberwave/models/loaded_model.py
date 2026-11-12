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
from cyberwave.models.types import DetectionResult, PredictionResult

if TYPE_CHECKING:
    from cyberwave.data.api import DataBus
    from cyberwave.models.tracking import TrackerConfig, TrackingModule

logger = logging.getLogger(__name__)

_LATENCY_WINDOW = 100  # rolling window size for latency percentiles
# Short silence clip for whisper.cpp warm-up (16 kHz mono int16).
_WHISPER_WARMUP_SAMPLES = 8000  # 0.5 s
# Drop polygons when the published JSON exceeds this size — a busy
# scene with 100+ instances would otherwise crowd the data bus.
_MAX_DETECTIONS_PAYLOAD_BYTES = 256 * 1024


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

        # Serializes predict() and warm_up() — backends such as whisper.cpp
        # are not safe for concurrent transcribe() on one model handle.
        self._inference_lock = threading.Lock()
        # Inference latency tracking (thread-safe rolling window).
        self._latency_lock = threading.Lock()
        self._latency_window: collections.deque[float] = collections.deque(
            maxlen=_LATENCY_WINDOW
        )
        self._inference_count: int = 0
        # Object tracking, created on the first predict() that asks for it.
        self._tracking: TrackingModule | None = None
        self._tracking_unavailable = False
        self._tracking_warned: set[str] = set()
        self._tracking_window: collections.deque[float] = collections.deque(
            maxlen=_LATENCY_WINDOW
        )

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
        tracker: str | dict[str, Any] | TrackerConfig | None = None,
        stream: str | None = None,
        track_scope: str | None = None,
        **kwargs: Any,
    ) -> PredictionResult:
        """Run inference on *input_data*.

        Args:
            twin_uuid: Optional override to route published detections to a
                specific twin instead of the default one bound to the data bus.
            tracker: Turn on object tracking for detection results:
                ``"bytetrack"``, ``"botsort"``, a mapping of
                :class:`~cyberwave.models.tracking.TrackerConfig` fields, or a
                config. Each confirmed detection then carries a ``track_id``.
                Needs the ``ml-tracking`` extra; without it, or if the tracker
                fails on a frame, the result is returned untracked and a
                warning is logged. ``"botsort"`` compensates camera motion only
                when *input_data* is a 3-channel image array.
            stream: The camera the frame came from (e.g. a sensor name).
                Tracker state is kept per ``(twin_uuid, stream)``, so cameras
                sharing this model never share ids. Defaults to ``"default"``.
            track_scope: Keeps separate tracker state for callers that share
                this model and stream, such as two workflow nodes; it is not
                published.
        """
        with self._inference_lock:
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

            if isinstance(result, DetectionResult):
                if tracker is not None:
                    self._track(
                        result,
                        input_data,
                        tracker=tracker,
                        twin_uuid=twin_uuid,
                        stream=stream,
                        track_scope=track_scope,
                    )
                if result.detections:
                    summary = ", ".join(
                        f"{d.label} {d.confidence:.0%}" for d in result.detections
                    )
                    logger.info(
                        "[%s] %d detection(s): %s",
                        self._name,
                        len(result.detections),
                        summary,
                    )

                # Publish every detection-shaped inference — including empty
                # batches — so overlay consumers see a heartbeat at the worker's
                # inference cadence.  Non-detection results (STT, embeddings,
                # etc.) are returned as-is without publishing to detections/*.
                self._publish_detections(
                    result,
                    input_data,
                    twin_uuid=twin_uuid,
                    sensor=(stream or "default") if tracker is not None else None,
                )
            elif tracker is not None:
                self._warn_once(
                    type(result).__name__,
                    "tracking needs detections; %s returned untracked",
                    type(result).__name__,
                )

            return result

    def _track(
        self,
        result: DetectionResult,
        input_data: Any,
        *,
        tracker: str | dict[str, Any] | TrackerConfig,
        twin_uuid: str | None,
        stream: str | None,
        track_scope: str | None,
    ) -> None:
        """Set ``track_id`` on *result*'s detections, in place.

        A bad *tracker* argument raises; any fault inside the tracker leaves
        the detections untracked rather than losing the inference.
        """
        if self._tracking_unavailable:
            return
        t0 = time.monotonic()
        try:
            from cyberwave.models.tracking import TrackerConfig, TrackingModule
        except ImportError as exc:
            self._tracking_unavailable = True
            logger.warning("[%s] object tracking disabled: %s", self._name, exc)
            return
        config = TrackerConfig.from_value(tracker)
        if self._tracking is None:
            self._tracking = TrackingModule()
        key = f"{twin_uuid or ''}/{stream or 'default'}"
        if track_scope:
            key = f"{key}/{track_scope}"
        # BoT-SORT's camera-motion compensation needs a 3-channel image; it
        # tracks without it, but fails on every grayscale frame.
        shape: tuple[int, ...] = getattr(input_data, "shape", ())
        frame = input_data if len(shape) == 3 and shape[-1] == 3 else None
        try:
            self._tracking.update(key, result, frame=frame, config=config)
        except ImportError as exc:
            self._tracking_unavailable = True
            logger.warning("[%s] object tracking disabled: %s", self._name, exc)
            return
        except Exception:
            for det in result.detections:
                det.track_id = None
            self._warn_once(
                f"failed:{key}",
                "tracking failed on stream %s; returning untracked",
                key,
                exc_info=True,
            )
            return
        with self._latency_lock:
            self._tracking_window.append((time.monotonic() - t0) * 1000.0)

    def _warn_once(self, topic: str, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a tracking warning once per *topic*, later repeats at debug."""
        level = logging.DEBUG if topic in self._tracking_warned else logging.WARNING
        self._tracking_warned.add(topic)
        logger.log(level, "[%s] " + message, self._name, *args, **kwargs)

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
        dummy, predict_kwargs = self._warm_up_input(input_shape)
        if self._runtime.name in {"whisper_cpp", "faster_whisper"}:
            logger.info(
                "[%s] Running STT warm-up inference (not triggered by wake word or workflow)",
                self._name,
            )

        with self._inference_lock:
            t0 = time.monotonic()
            try:
                self._runtime.predict(
                    self._model_handle,
                    dummy,
                    confidence=confidence,
                    **predict_kwargs,
                )
            except Exception:
                logger.debug("Warm-up cold pass raised (non-fatal)", exc_info=True)
            cold_ms = (time.monotonic() - t0) * 1000.0

            t1 = time.monotonic()
            try:
                self._runtime.predict(
                    self._model_handle,
                    dummy,
                    confidence=confidence,
                    **predict_kwargs,
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

    def _warm_up_input(
        self,
        input_shape: tuple[int, ...] | None,
    ) -> tuple[Any, dict[str, Any]]:
        """Return dummy input and runtime kwargs for warm-up inference."""
        import numpy as np

        if input_shape is not None:
            return np.zeros(input_shape, dtype=np.uint8), {}

        if self._runtime.name in {"whisper_cpp", "faster_whisper"}:
            return (
                np.zeros(_WHISPER_WARMUP_SAMPLES, dtype=np.int16),
                {"sample_rate_hz": 16000, "channels": 1},
            )

        return np.zeros((640, 640, 3), dtype=np.uint8), {}

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
            tracking = list(self._tracking_window)
        n = len(samples)
        avg = sum(samples) / n
        p95 = samples[int(n * 0.95)] if n >= 2 else samples[-1]
        p99 = samples[int(n * 0.99)] if n >= 2 else samples[-1]
        stats: dict[str, Any] = {
            "name": self._name,
            "device": self._device,
            "count": count,
            "avg_ms": round(avg, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
        }
        if tracking:
            # Tracking time is not part of avg_ms / p95_ms / p99_ms (inference).
            stats["tracking_avg_ms"] = round(sum(tracking) / len(tracking), 2)
        return stats

    def _publish_detections(
        self,
        result: PredictionResult,
        input_data: Any,
        *,
        twin_uuid: str | None = None,
        sensor: str | None = None,
    ) -> None:
        """Publish detection results via Zenoh so the driver can draw overlays.

        With tracking on (*sensor* set), each entry carries its ``track_id``
        when confirmed, and the payload names the ``model`` and ``sensor``.
        """
        if self._data_bus is None:
            return
        try:
            import json as _json

            import numpy as np

            if isinstance(input_data, np.ndarray) and input_data.ndim >= 2:
                h, w = input_data.shape[:2]
            else:
                h, w = 0, 0

            mask_to_polygon = None
            if any(getattr(d, "mask", None) is not None for d in result.detections):
                try:
                    from cyberwave.vision import mask_to_polygon as _mtp

                    mask_to_polygon = _mtp
                except Exception:
                    mask_to_polygon = None

            entries: list[dict[str, Any]] = []
            for d in result.detections:
                entry: dict[str, Any] = {
                    "label": d.label,
                    "confidence": round(d.confidence, 3),
                    "x1": round(d.bbox.x1),
                    "y1": round(d.bbox.y1),
                    "x2": round(d.bbox.x2),
                    "y2": round(d.bbox.y2),
                }
                if d.track_id is not None:
                    entry["track_id"] = d.track_id
                if mask_to_polygon is not None:
                    d_mask = getattr(d, "mask", None)
                    if d_mask is not None:
                        poly = mask_to_polygon(d_mask)
                        if poly is not None:
                            entry["polygon"] = poly
                entries.append(entry)

            payload_dict = {
                "detections": entries,
                "frame_width": w,
                "frame_height": h,
                "timestamp": time.time(),
            }
            if sensor is not None:
                payload_dict["model"] = self._name
                payload_dict["sensor"] = sensor

            raw_bytes = _json.dumps(payload_dict, separators=(",", ":")).encode()
            if len(raw_bytes) > _MAX_DETECTIONS_PAYLOAD_BYTES:
                for entry in entries:
                    entry.pop("polygon", None)
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
