"""Model-agnostic multi-object tracking for detection results.

The tracker only reads what every detection-shaped result carries (bbox,
confidence, label), so it works the same for every runtime: Ultralytics,
ONNX Runtime, TensorRT, Hailo, TFLite and any custom model that returns a
:class:`~cyberwave.models.types.DetectionResult`. Segmentation masks are
tracked through their bounding box.

State is kept per **stream**: one tracker instance per key the caller chooses,
typically ``"<twin_uuid>/<camera sensor>"``. One loaded model can serve several
cameras, so the state cannot live on the model itself::

    from cyberwave.models.tracking import TrackingModule

    tracking = TrackingModule()
    result = model.predict(frame)
    tracking.update("twin-1/front", result, frame=frame)
    for det in result:
        print(det.label, det.track_id)

``track_id`` stays ``None`` until a new object is confirmed
(:attr:`TrackerConfig.minimum_consecutive_frames`). Ids come from one counter
per stream shared by all labels, and are never reused within a stream until
:meth:`TrackingModule.reset` forgets it. After a frame gap (see
:attr:`TrackerConfig.reset_after_s`) the stream's tracker is reset and objects
get new ids, numbered after the ones already issued.

Requires the ``ml-tracking`` extra (``pip install 'cyberwave[ml-tracking]'``),
which provides ByteTrack and BoT-SORT through the ``trackers`` package.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol

try:
    from statemachine import State, StateMachine
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise ImportError(
        "Object tracking requires the 'ml-tracking' extra: "
        "pip install 'cyberwave[ml-tracking]'"
    ) from exc

if TYPE_CHECKING:
    from cyberwave.models.types import DetectionResult

logger = logging.getLogger(__name__)

TRACKER_KINDS: tuple[str, ...] = ("bytetrack", "botsort")

# Column carried through a ``trackers`` tracker so each output row maps back to
# its input row: those trackers return rows in their own order.
_INDEX_KEY = "_cw_index"

# A frame gap resets the tracker only when it is also this many times longer
# than the stream's previous frame interval, so a slow stream (one frame every
# few seconds) keeps its ids instead of resetting on every frame.
_GAP_FACTOR = 2.0


@dataclass(frozen=True)
class TrackerConfig:
    """Tracker choice and parameters.

    Field names follow the ``trackers`` package so they can be looked up in its
    documentation. The thresholds default lower than that package's (0.7 and
    0.6): detections reaching the tracker have already passed the model's own
    confidence threshold, and the higher values would leave objects scored just
    above it without an id.

    Attributes:
        kind: ``"bytetrack"`` (default) or ``"botsort"``. BoT-SORT also uses
            the frame for camera-motion compensation when one is passed.
        track_activation_threshold: Minimum confidence to start a new track.
        high_conf_det_threshold: Confidence splitting the high- and
            low-confidence association passes.
        minimum_consecutive_frames: Frames a new object must be seen before it
            gets a ``track_id``.
        lost_track_buffer: Frames a lost object keeps its id while unseen.
        reset_after_s: A frame gap longer than this, and more than twice the
            stream's previous frame interval, resets the stream's tracker.
            ``0`` disables the reset.
    """

    kind: str = "bytetrack"
    track_activation_threshold: float = 0.25
    high_conf_det_threshold: float = 0.25
    minimum_consecutive_frames: int = 2
    lost_track_buffer: int = 30
    reset_after_s: float = 2.0

    def __post_init__(self) -> None:
        if self.kind not in TRACKER_KINDS:
            raise ValueError(
                f"tracker kind must be one of {', '.join(TRACKER_KINDS)}, "
                f"got {self.kind!r}"
            )
        for name in ("track_activation_threshold", "high_conf_det_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.minimum_consecutive_frames < 1:
            raise ValueError(
                "minimum_consecutive_frames must be at least 1, "
                f"got {self.minimum_consecutive_frames}"
            )
        if self.lost_track_buffer < 0:
            raise ValueError(
                f"lost_track_buffer must be non-negative, got {self.lost_track_buffer}"
            )
        if self.reset_after_s < 0:
            raise ValueError(
                f"reset_after_s must be non-negative, got {self.reset_after_s}"
            )

    @classmethod
    def from_value(
        cls, value: "str | Mapping[str, Any] | TrackerConfig"
    ) -> "TrackerConfig":
        """Build a config from a kind name, a mapping of fields, or a config."""
        if isinstance(value, TrackerConfig):
            return value
        if isinstance(value, str):
            return cls(kind=value)
        if isinstance(value, Mapping):
            unknown = set(value) - set(cls.__dataclass_fields__)
            if unknown:
                raise ValueError(
                    f"unknown tracker option(s): {', '.join(sorted(unknown))}"
                )
            return cls(**value)
        raise TypeError(
            "tracker must be a kind name, a mapping or a TrackerConfig, "
            f"got {type(value).__name__}"
        )


class Tracker(Protocol):
    """What :class:`TrackingModule` needs from a tracker implementation.

    ``update`` receives one frame's detections as arrays (``xyxy`` of shape
    ``(N, 4)``, ``confidence`` and integer ``class_id`` of shape ``(N,)``) and
    returns one id per input row, in input order: ``-1`` while a row is not
    yet confirmed.
    """

    def update(
        self, xyxy: Any, confidence: Any, class_id: Any, frame: Any = None
    ) -> Any: ...

    def reset(self) -> None: ...


class _TrackersAdapter:
    """Adapts a ``trackers`` tracker, which reorders its rows, to :class:`Tracker`.

    Only trackers that use the image (BoT-SORT's camera-motion compensation)
    get the frame; ByteTrack warns on every call when handed one.
    """

    def __init__(self, tracker: Any, *, uses_frame: bool) -> None:
        self._tracker = tracker
        self._uses_frame = uses_frame

    def update(
        self, xyxy: Any, confidence: Any, class_id: Any, frame: Any = None
    ) -> Any:
        import numpy as np
        import supervision as sv

        count = len(xyxy)
        tracked = self._tracker.update(
            sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
                data={_INDEX_KEY: np.arange(count)},
            ),
            frame=frame if self._uses_frame else None,
        )
        ids = np.full(count, -1, dtype=int)
        # An empty result comes back as a fresh ``sv.Detections`` without the
        # index column.
        if len(tracked) and tracked.tracker_id is not None:
            ids[tracked.data[_INDEX_KEY]] = tracked.tracker_id
        return ids

    def reset(self) -> None:
        self._tracker.reset()


def make_tracker(config: TrackerConfig) -> Tracker:
    """Create a ``trackers`` ByteTrack or BoT-SORT tracker for *config*."""
    try:
        from trackers import BoTSORTTracker, ByteTrackTracker
    except ImportError as exc:
        raise ImportError(
            "Object tracking requires the 'ml-tracking' extra: "
            "pip install 'cyberwave[ml-tracking]'"
        ) from exc

    params: dict[str, Any] = {
        "lost_track_buffer": config.lost_track_buffer,
        "track_activation_threshold": config.track_activation_threshold,
        "minimum_consecutive_frames": config.minimum_consecutive_frames,
        "high_conf_det_threshold": config.high_conf_det_threshold,
    }
    if config.kind == "botsort":
        # BoT-SORT would otherwise id objects on a stream's first frame,
        # unlike ByteTrack; both wait for minimum_consecutive_frames.
        tracker = BoTSORTTracker(**params, instant_first_frame_activation=False)
        return _TrackersAdapter(tracker, uses_frame=True)
    return _TrackersAdapter(ByteTrackTracker(**params), uses_frame=False)


class StreamLifecycle(StateMachine):
    """Lifecycle of one tracked stream.

    Events are sent only on changes (the first frame, a frame gap, shutdown),
    never per frame. ``resume`` resets the stream's tracker through the
    listener's ``after_resume`` callback.
    """

    idle = State(initial=True)
    active = State()
    stale = State()
    closed = State(final=True)

    start = idle.to(active)
    gap = active.to(stale)
    resume = stale.to(active)
    close = idle.to(closed) | active.to(closed) | stale.to(closed)


class _Stream:
    """Tracker state for one stream, driven by its :class:`StreamLifecycle`."""

    def __init__(
        self,
        key: str,
        config: TrackerConfig,
        tracker_factory: Callable[[TrackerConfig], Tracker],
        *,
        first_id: int = 0,
    ) -> None:
        self.key = key
        self.config = config
        self.tracker = tracker_factory(config)
        self.class_ids: dict[str, int] = {}
        self.last_ts: float | None = None
        self.interval: float | None = None
        # A reset tracker numbers from 0 again; the offset keeps the ids this
        # stream publishes unique.
        self.id_offset = first_id
        self.next_id = first_id
        self.lifecycle = StreamLifecycle(listeners=[self])

    # -- lifecycle listener callbacks --------------------------------------

    def after_resume(self) -> None:
        self.tracker.reset()
        self.id_offset = self.next_id

    def after_transition(self, event: str, source: State, target: State) -> None:
        logger.debug(
            "tracking stream %s: %s -> %s (%s)", self.key, source.id, target.id, event
        )

    # -- per frame ---------------------------------------------------------

    def observe(self, ts: float) -> None:
        """Advance the lifecycle for a frame at *ts* before it is tracked."""
        if self.lifecycle.idle.is_active:
            self.lifecycle.send("start")
        elif self.last_ts is not None:
            gap = ts - self.last_ts
            if (
                self.config.reset_after_s > 0
                and self.interval is not None
                and gap > self.config.reset_after_s
                and gap > _GAP_FACTOR * self.interval
            ):
                self.lifecycle.send("gap")
                self.lifecycle.send("resume")
            self.interval = gap
        self.last_ts = ts

    def track(self, result: "DetectionResult", frame: Any) -> None:
        import numpy as np

        detections = result.detections
        count = len(detections)
        ids = self.tracker.update(
            np.array(
                [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections],
                dtype=float,
            ).reshape(count, 4),
            np.array([d.confidence for d in detections], dtype=float),
            np.array(
                [
                    self.class_ids.setdefault(d.label, len(self.class_ids))
                    for d in detections
                ],
                dtype=int,
            ),
            frame=frame,
        )
        for det, raw_id in zip(detections, ids):
            if raw_id < 0:
                det.track_id = None
                continue
            det.track_id = int(raw_id) + self.id_offset
            self.next_id = max(self.next_id, det.track_id + 1)

    def close(self) -> None:
        if not self.lifecycle.closed.is_active:
            self.lifecycle.send("close")


class TrackingModule:
    """Assigns ``track_id`` to detection results, one tracker per stream.

    Thread-safe. All tracking state lives here; the models and runtimes that
    produce the results stay stateless.

    Args:
        config: Default tracker for streams that do not pass their own.
        tracker_factory: Builds a tracker from a config. Defaults to
            :func:`make_tracker`; tests pass a fake.
        clock: Time source for frames updated without ``ts``.
    """

    def __init__(
        self,
        config: TrackerConfig | None = None,
        *,
        tracker_factory: Callable[[TrackerConfig], Tracker] = make_tracker,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config or TrackerConfig()
        self._tracker_factory = tracker_factory
        self._clock = clock
        self._streams: dict[str, _Stream] = {}
        self._lock = threading.Lock()

    @property
    def streams(self) -> tuple[str, ...]:
        """Keys of the streams currently tracked."""
        with self._lock:
            return tuple(self._streams)

    def update(
        self,
        stream: str,
        result: "DetectionResult",
        *,
        ts: float | None = None,
        frame: Any = None,
        config: "TrackerConfig | str | Mapping[str, Any] | None" = None,
    ) -> "DetectionResult":
        """Set ``track_id`` on every detection of *result*, in place.

        Args:
            stream: Key of the stream the frame came from.
            result: Detections of one frame.
            ts: Frame time in seconds (monotonic); defaults to now.
            frame: The image the detections came from; used by BoT-SORT.
            config: Tracker for this stream; defaults to the module's. A stream
                whose config changes starts a new tracker, numbering after the
                ids already issued.

        Returns:
            *result*, with ``track_id`` set on confirmed detections.
        """
        cfg = self._config if config is None else TrackerConfig.from_value(config)
        now = self._clock() if ts is None else ts
        with self._lock:
            state = self._streams.get(stream)
            first_id = 0
            if state is not None and state.config != cfg:
                state.close()
                first_id = state.next_id
                state = None
            if state is None:
                state = _Stream(stream, cfg, self._tracker_factory, first_id=first_id)
                self._streams[stream] = state
            state.observe(now)
            state.track(result, frame)
        return result

    def reset(self, stream: str | None = None) -> None:
        """Forget one stream, or every stream when *stream* is ``None``.

        A forgotten stream numbers its ids from 0 again.
        """
        with self._lock:
            keys = list(self._streams) if stream is None else [stream]
            for key in keys:
                state = self._streams.pop(key, None)
                if state is not None:
                    state.close()
