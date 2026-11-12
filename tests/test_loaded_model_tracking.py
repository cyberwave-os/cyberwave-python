"""Tests for object tracking in LoadedModel.predict (tracker= / stream=)."""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("statemachine")

from cyberwave.models.loaded_model import LoadedModel  # noqa: E402
from cyberwave.models.tracking import TrackerConfig, TrackingModule  # noqa: E402
from cyberwave.models.types import (  # noqa: E402
    BoundingBox,
    Detection,
    DetectionResult,
    TextResult,
)


class ConfirmingTracker:
    """Ids ``10 + i`` from the second frame on; ``-1`` on the first."""

    def __init__(self, config: TrackerConfig) -> None:
        self.frames = 0

    def update(
        self, xyxy: Any, confidence: Any, class_id: Any, frame: Any = None
    ) -> Any:
        self.frames += 1
        return np.full(len(xyxy), -1) if self.frames == 1 else np.arange(len(xyxy)) + 10

    def reset(self) -> None:
        self.frames = 0


def _model(data_bus: Any = None) -> LoadedModel:
    runtime = MagicMock()
    runtime.name = "onnxruntime"
    model = LoadedModel(
        name="detector",
        runtime=runtime,
        model_handle="handle",
        data_bus=data_bus,
    )
    model._tracking = TrackingModule(tracker_factory=ConfirmingTracker)
    return model


def _detections(*xs: float) -> DetectionResult:
    return DetectionResult(
        [
            Detection(
                label="person", confidence=0.9, bbox=BoundingBox(x, 10, x + 40, 80)
            )
            for x in xs
        ]
    )


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def test_tracker_kwargs_are_not_forwarded_to_the_runtime() -> None:
    model = _model()
    model._runtime.predict.return_value = _detections(0)
    model.predict(FRAME, tracker="bytetrack", stream="front", twin_uuid="t1")
    kwargs = model._runtime.predict.call_args.kwargs
    assert "tracker" not in kwargs and "stream" not in kwargs


def test_detections_get_track_ids_once_confirmed() -> None:
    model = _model()
    model._runtime.predict.side_effect = [_detections(0, 100), _detections(1, 101)]
    first = model.predict(FRAME, tracker="bytetrack", stream="front")
    second = model.predict(FRAME, tracker="bytetrack", stream="front")
    assert [d.track_id for d in first] == [None, None]
    assert [d.track_id for d in second] == [10, 11]


def test_state_is_kept_per_twin_and_stream() -> None:
    model = _model()
    model._runtime.predict.side_effect = lambda *a, **k: _detections(0)
    model.predict(FRAME, tracker="bytetrack", stream="front", twin_uuid="t1")
    model.predict(FRAME, tracker="bytetrack", stream="back", twin_uuid="t1")
    model.predict(FRAME, tracker="bytetrack", stream="front", twin_uuid="t2")
    model.predict(FRAME, tracker="bytetrack")
    assert model._tracking is not None
    assert model._tracking.streams == ("t1/front", "t1/back", "t2/front", "/default")


def test_payload_carries_track_id_model_and_sensor() -> None:
    data_bus = MagicMock()
    model = _model(data_bus)
    model._runtime.predict.side_effect = [_detections(0, 100), _detections(1, 101)]
    model.predict(FRAME, tracker="bytetrack", stream="front")
    model.predict(FRAME, tracker="bytetrack", stream="front")

    first, second = (
        json.loads(call.args[1].decode())
        for call in data_bus.publish_raw.call_args_list
    )
    assert "track_id" not in first["detections"][0]
    assert [d["track_id"] for d in second["detections"]] == [10, 11]
    assert second["model"] == "detector"
    assert second["sensor"] == "front"


def test_payload_without_tracker_is_unchanged() -> None:
    data_bus = MagicMock()
    model = _model(data_bus)
    model._runtime.predict.return_value = _detections(0)
    model.predict(FRAME)
    payload = json.loads(data_bus.publish_raw.call_args.args[1].decode())
    assert set(payload) == {"detections", "frame_width", "frame_height", "timestamp"}
    assert "track_id" not in payload["detections"][0]
    assert model._tracking is not None and model._tracking.streams == ()


def test_missing_extra_warns_once_and_returns_untracked(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = []

    def unavailable(config: TrackerConfig) -> Any:
        calls.append(config)
        raise ImportError("Object tracking requires the 'ml-tracking' extra")

    model = _model()
    model._tracking = TrackingModule(tracker_factory=unavailable)
    model._runtime.predict.side_effect = lambda *a, **k: _detections(0)
    with caplog.at_level(logging.WARNING):
        first = model.predict(FRAME, tracker="bytetrack")
        model.predict(FRAME, tracker="bytetrack")
    assert first[0].track_id is None
    assert len(calls) == 1
    assert sum("tracking disabled" in r.message for r in caplog.records) == 1


def test_tracking_a_non_detection_result_warns_and_returns_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _model()
    text = TextResult(text="hello")
    model._runtime.predict.return_value = text
    with caplog.at_level(logging.WARNING):
        assert model.predict("audio", tracker="bytetrack") is text
        model.predict("audio", tracker="bytetrack")
    assert sum("tracking needs detections" in r.message for r in caplog.records) == 1


def test_a_tracker_fault_returns_the_detections_untracked(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Failing(ConfirmingTracker):
        def update(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    data_bus = MagicMock()
    model = _model(data_bus)
    model._tracking = TrackingModule(tracker_factory=Failing)
    model._runtime.predict.side_effect = lambda *a, **k: _detections(0, 100)
    with caplog.at_level(logging.WARNING):
        first = model.predict(FRAME, tracker="bytetrack", stream="front")
        model.predict(FRAME, tracker="bytetrack", stream="front")
    assert [d.track_id for d in first] == [None, None]
    assert data_bus.publish_raw.call_count == 2
    assert sum("tracking failed" in r.message for r in caplog.records) == 1


def test_track_scope_separates_callers_on_one_stream() -> None:
    data_bus = MagicMock()
    model = _model(data_bus)
    model._runtime.predict.side_effect = lambda *a, **k: _detections(0)
    model.predict(
        FRAME, tracker="bytetrack", stream="front", twin_uuid="t1", track_scope="a"
    )
    model.predict(
        FRAME, tracker="bytetrack", stream="front", twin_uuid="t1", track_scope="b"
    )
    assert model._tracking is not None
    assert model._tracking.streams == ("t1/front/a", "t1/front/b")
    payload = json.loads(data_bus.publish_raw.call_args.args[1].decode())
    assert payload["sensor"] == "front"
    assert "track_scope" not in model._runtime.predict.call_args.kwargs


def test_invalid_tracker_raises() -> None:
    model = _model()
    model._runtime.predict.return_value = _detections(0)
    with pytest.raises(ValueError, match="tracker kind"):
        model.predict(FRAME, tracker="kalman")


def test_inference_stats_report_tracking_time_separately() -> None:
    model = _model()
    model._runtime.predict.side_effect = lambda *a, **k: _detections(0)
    model.predict(FRAME)
    assert "tracking_avg_ms" not in model.inference_stats()
    model.predict(FRAME, tracker="bytetrack")
    assert model.inference_stats()["tracking_avg_ms"] >= 0


def test_end_to_end_with_trackers_package() -> None:
    pytest.importorskip("trackers")
    runtime = MagicMock()
    runtime.name = "onnxruntime"
    model = LoadedModel(name="detector", runtime=runtime, model_handle="handle")
    runtime.predict.side_effect = [
        _detections(10 + 3 * i, 300 - 3 * i) for i in range(6)
    ]
    results = [
        model.predict(FRAME, tracker="bytetrack", stream="front") for _ in range(6)
    ]
    last = [d.track_id for d in results[-1]]
    assert None not in last and last[0] != last[1]
    assert [d.track_id for d in results[-2]] == last


@pytest.mark.parametrize(
    "frame",
    [np.zeros((480, 640), dtype=np.uint8), np.zeros((480, 640, 1), dtype=np.uint8)],
)
def test_botsort_tracks_grayscale_frames(frame: Any) -> None:
    pytest.importorskip("trackers")
    runtime = MagicMock()
    runtime.name = "onnxruntime"
    model = LoadedModel(name="detector", runtime=runtime, model_handle="handle")
    runtime.predict.side_effect = [
        _detections(10 + 3 * i, 300 - 3 * i) for i in range(4)
    ]
    results = [model.predict(frame, tracker="botsort", stream="ir") for _ in range(4)]
    last = [d.track_id for d in results[-1]]
    assert None not in last and last[0] != last[1]
