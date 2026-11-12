"""Tests for cyberwave.models.tracking (per-stream, model-agnostic tracking)."""

from __future__ import annotations

from typing import Any

import pytest

import numpy as np

pytest.importorskip("statemachine")

from cyberwave.models.tracking import (  # noqa: E402
    StreamLifecycle,
    TrackerConfig,
    TrackingModule,
)
from cyberwave.models.types import BoundingBox, Detection, DetectionResult  # noqa: E402

from _state_machine import assert_every_transition_runs  # noqa: E402


def _det(label: str, x: float, conf: float = 0.9) -> Detection:
    return Detection(label=label, confidence=conf, bbox=BoundingBox(x, 10, x + 40, 80))


def _result(*dets: Detection) -> DetectionResult:
    return DetectionResult(list(dets))


class FakeTracker:
    """Gives row *i* the id ``100 + i`` once seen, ``-1`` on its first frame."""

    instances: list["FakeTracker"] = []

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self.frames = 0
        self.resets = 0
        self.calls: list[dict[str, Any]] = []
        FakeTracker.instances.append(self)

    def update(
        self, xyxy: Any, confidence: Any, class_id: Any, frame: Any = None
    ) -> Any:
        self.frames += 1
        self.calls.append(
            {
                "xyxy": xyxy,
                "confidence": confidence,
                "class_id": class_id,
                "frame": frame,
            }
        )
        count = len(xyxy)
        return np.full(count, -1) if self.frames == 1 else np.arange(count) + 100

    def reset(self) -> None:
        self.resets += 1
        self.frames = 0


@pytest.fixture(autouse=True)
def _clear_fakes() -> None:
    FakeTracker.instances.clear()


def _module(**config: Any) -> TrackingModule:
    return TrackingModule(TrackerConfig(**config), tracker_factory=FakeTracker)


# ── TrackerConfig ─────────────────────────────────────────────────


class TestTrackerConfig:
    def test_defaults(self) -> None:
        cfg = TrackerConfig()
        assert cfg.kind == "bytetrack"
        assert cfg.minimum_consecutive_frames == 2
        assert cfg.reset_after_s == 2.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"kind": "sort"},
            {"track_activation_threshold": 1.5},
            {"high_conf_det_threshold": -0.1},
            {"minimum_consecutive_frames": 0},
            {"lost_track_buffer": -1},
            {"reset_after_s": -1},
        ],
    )
    def test_rejects_invalid_values(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            TrackerConfig(**kwargs)

    def test_from_value(self) -> None:
        assert TrackerConfig.from_value("botsort").kind == "botsort"
        assert TrackerConfig.from_value({"lost_track_buffer": 5}).lost_track_buffer == 5
        cfg = TrackerConfig()
        assert TrackerConfig.from_value(cfg) is cfg

    def test_from_value_rejects_unknown_options_and_types(self) -> None:
        with pytest.raises(ValueError, match="unknown tracker option"):
            TrackerConfig.from_value({"speed": 1})
        with pytest.raises(TypeError):
            TrackerConfig.from_value(3)  # type: ignore[arg-type]


# ── TrackingModule ────────────────────────────────────────────────


class TestTrackingModule:
    def test_assigns_ids_after_confirmation(self) -> None:
        tracking = _module()
        first = tracking.update(
            "cam", _result(_det("person", 0), _det("person", 100)), ts=0.0
        )
        assert [d.track_id for d in first] == [None, None]

        second = tracking.update(
            "cam", _result(_det("person", 1), _det("person", 101)), ts=0.1
        )
        assert [d.track_id for d in second] == [100, 101]

    def test_sends_boxes_scores_and_one_class_id_per_label(self) -> None:
        tracking = _module()
        tracking.update(
            "cam",
            _result(_det("person", 0, conf=0.6), _det("car", 50), _det("person", 100)),
            ts=0.0,
        )
        call = FakeTracker.instances[0].calls[0]
        assert call["xyxy"].tolist() == [
            [0, 10, 40, 80],
            [50, 10, 90, 80],
            [100, 10, 140, 80],
        ]
        assert call["confidence"].tolist() == [0.6, 0.9, 0.9]
        assert call["class_id"].tolist() == [0, 1, 0]

    def test_returns_the_same_result(self) -> None:
        result = _result(_det("car", 0))
        assert _module().update("cam", result, ts=0.0) is result

    def test_empty_result(self) -> None:
        result = _module().update("cam", _result(), ts=0.0)
        assert len(result) == 0

    def test_passes_the_frame_to_the_tracker(self) -> None:
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        _module().update("cam", _result(_det("car", 0)), ts=0.0, frame=frame)
        assert FakeTracker.instances[0].calls[0]["frame"] is frame

    def test_streams_are_isolated(self) -> None:
        tracking = _module()
        tracking.update("twin/front", _result(_det("person", 0)), ts=0.0)
        tracking.update("twin/back", _result(_det("person", 0)), ts=0.0)
        assert len(FakeTracker.instances) == 2
        assert tracking.streams == ("twin/front", "twin/back")

        tracking.reset("twin/front")
        assert tracking.streams == ("twin/back",)
        assert FakeTracker.instances[1].resets == 0

    def test_frame_gap_resets_the_tracker(self) -> None:
        tracking = _module(reset_after_s=2.0)
        for ts in (0.0, 0.2, 0.4):
            tracking.update("cam", _result(_det("car", 0)), ts=ts)
        assert FakeTracker.instances[0].resets == 0

        tracking.update("cam", _result(_det("car", 0)), ts=3.0)
        assert FakeTracker.instances[0].resets == 1

    def test_slow_stream_is_not_reset_and_gets_ids(self) -> None:
        tracking = _module(reset_after_s=2.0)
        results = [
            tracking.update("cam", _result(_det("car", 0)), ts=4.0 * i)
            for i in range(4)
        ]
        assert FakeTracker.instances[0].resets == 0
        assert results[-1][0].track_id == 100

    def test_slow_stream_is_reset_by_a_longer_stall(self) -> None:
        tracking = _module(reset_after_s=2.0)
        for ts in (0.0, 4.0, 8.0, 30.0):
            tracking.update("cam", _result(_det("car", 0)), ts=ts)
        assert FakeTracker.instances[0].resets == 1

    def test_ids_are_not_reused_after_a_gap_reset(self) -> None:
        tracking = _module(reset_after_s=2.0)
        before = [
            tracking.update("cam", _result(_det("car", 0), _det("car", 50)), ts=ts)
            for ts in (0.0, 0.1, 0.2)
        ][-1]
        after = [
            tracking.update("cam", _result(_det("car", 0)), ts=ts) for ts in (5.0, 5.1)
        ]
        assert [d.track_id for d in before] == [100, 101]
        assert after[0][0].track_id is None
        assert after[1][0].track_id == 102 + 100

    def test_zero_reset_after_disables_the_gap_reset(self) -> None:
        tracking = _module(reset_after_s=0)
        tracking.update("cam", _result(_det("car", 0)), ts=0.0)
        tracking.update("cam", _result(_det("car", 0)), ts=1000.0)
        assert FakeTracker.instances[0].resets == 0

    def test_config_change_starts_a_new_tracker(self) -> None:
        tracking = _module()
        tracking.update("cam", _result(_det("car", 0)), ts=0.0)
        tracking.update("cam", _result(_det("car", 0)), ts=0.1)
        tracking.update("cam", _result(_det("car", 0)), ts=0.2, config="botsort")
        last = tracking.update("cam", _result(_det("car", 0)), ts=0.3, config="botsort")
        assert [t.config.kind for t in FakeTracker.instances] == [
            "bytetrack",
            "botsort",
        ]
        # The new tracker numbers after the 100 already issued.
        assert last[0].track_id == 101 + 100

    def test_reset_all(self) -> None:
        tracking = _module()
        tracking.update("a", _result(_det("car", 0)), ts=0.0)
        tracking.update("b", _result(_det("car", 0)), ts=0.0)
        tracking.reset()
        assert tracking.streams == ()

    def test_uses_the_clock_without_ts(self) -> None:
        now = iter([0.0, 0.1, 10.0])
        tracking = TrackingModule(
            TrackerConfig(reset_after_s=2.0),
            tracker_factory=FakeTracker,
            clock=lambda: next(now),
        )
        for _ in range(3):
            tracking.update("cam", _result(_det("car", 0)))
        assert FakeTracker.instances[0].resets == 1


# ── StreamLifecycle ───────────────────────────────────────────────


class TestStreamLifecycle:
    def test_every_transition_runs(self) -> None:
        assert_every_transition_runs(StreamLifecycle)

    def test_closed_is_final(self) -> None:
        lifecycle = StreamLifecycle()
        lifecycle.send("close")
        assert lifecycle.is_terminated


# ── Adapter for the trackers package ──────────────────────────────


class TestTrackersAdapter:
    def test_maps_reordered_rows_back_to_input_order(self) -> None:
        pytest.importorskip("supervision")
        from cyberwave.models.tracking import _TrackersAdapter

        class Reversing:
            def update(self, detections: Any, frame: Any = None) -> Any:
                out = detections[::-1]
                out.tracker_id = np.array([7, -1, 5])
                return out

            def reset(self) -> None:
                pass

        ids = _TrackersAdapter(Reversing(), uses_frame=False).update(
            np.zeros((3, 4)), np.ones(3), np.zeros(3, dtype=int)
        )
        # Reversed rows were [2, 1, 0] -> ids 7, -1, 5.
        assert ids.tolist() == [5, -1, 7]

    @pytest.mark.parametrize("uses_frame", [True, False])
    def test_passes_the_frame_only_when_used(self, uses_frame: bool) -> None:
        pytest.importorskip("supervision")
        from cyberwave.models.tracking import _TrackersAdapter

        frames = []

        class Recording:
            def update(self, detections: Any, frame: Any = None) -> Any:
                frames.append(frame)
                detections.tracker_id = np.full(len(detections), -1)
                return detections

            def reset(self) -> None:
                pass

        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        _TrackersAdapter(Recording(), uses_frame=uses_frame).update(
            np.zeros((1, 4)), np.ones(1), np.zeros(1, dtype=int), frame=frame
        )
        assert (frames[0] is frame) is uses_frame


# ── With the real trackers package ────────────────────────────────


class TestWithTrackersPackage:
    @pytest.fixture(autouse=True)
    def _needs_trackers(self) -> None:
        pytest.importorskip("trackers")

    @pytest.mark.parametrize("kind", ["bytetrack", "botsort"])
    def test_same_class_objects_get_distinct_stable_ids(self, kind: str) -> None:
        tracking = TrackingModule(TrackerConfig(kind=kind))
        ids = []
        for i in range(6):
            result = tracking.update(
                "cam",
                _result(
                    _det("person", 10 + 2 * i), _det("person", 300 - 2 * i, conf=0.55)
                ),
                ts=i / 10,
            )
            ids.append([d.track_id for d in result])
        last = ids[-1]
        # Both kinds wait for a second frame before giving an id.
        assert ids[0] == [None, None]
        assert None not in last
        assert last[0] != last[1]
        assert ids[-2] == last

    @pytest.mark.parametrize("kind", ["bytetrack", "botsort"])
    def test_empty_frames_keep_tracking(self, kind: str) -> None:
        tracking = TrackingModule(TrackerConfig(kind=kind))
        boxes = (_det("person", 10), _det("person", 300))
        ids = [
            [d.track_id for d in tracking.update("cam", _result(*dets), ts=i / 10)]
            for i, dets in enumerate([(), boxes, boxes, (), (), boxes])
        ]
        assert ids[0] == ids[3] == ids[4] == []
        assert None not in ids[2] and ids[5] == ids[2]

    def test_a_label_change_keeps_the_id(self) -> None:
        tracking = TrackingModule()
        ids = [
            tracking.update("cam", _result(_det(label, 10 + i)), ts=i / 10)[0].track_id
            for i, label in enumerate(["truck", "truck", "car", "truck", "car"])
        ]
        assert ids[1] is not None
        assert len(set(ids[1:])) == 1
