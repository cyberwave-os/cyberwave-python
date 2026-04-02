"""Tests for BoundingBox, Detection, and PredictionResult output types."""

import pytest

from cyberwave.models.types import BoundingBox, Detection, PredictionResult


# ── BoundingBox ───────────────────────────────────────────────────


class TestBoundingBox:
    def test_width(self) -> None:
        bb = BoundingBox(x1=10, y1=20, x2=110, y2=70)
        assert bb.width == 100

    def test_height(self) -> None:
        bb = BoundingBox(x1=10, y1=20, x2=110, y2=70)
        assert bb.height == 50

    def test_area(self) -> None:
        bb = BoundingBox(x1=0, y1=0, x2=100, y2=200)
        assert bb.area == 20_000

    def test_center(self) -> None:
        bb = BoundingBox(x1=0, y1=0, x2=100, y2=200)
        assert bb.center == (50.0, 100.0)

    def test_zero_size_box(self) -> None:
        bb = BoundingBox(x1=5, y1=5, x2=5, y2=5)
        assert bb.width == 0
        assert bb.height == 0
        assert bb.area == 0

    def test_fractional_coords(self) -> None:
        bb = BoundingBox(x1=0.5, y1=0.5, x2=1.5, y2=2.5)
        assert bb.width == pytest.approx(1.0)
        assert bb.height == pytest.approx(2.0)
        assert bb.area == pytest.approx(2.0)

    def test_equality(self) -> None:
        a = BoundingBox(x1=1, y1=2, x2=3, y2=4)
        b = BoundingBox(x1=1, y1=2, x2=3, y2=4)
        assert a == b

    def test_inverted_x_raises(self) -> None:
        with pytest.raises(ValueError, match="Inverted x coordinates"):
            BoundingBox(x1=100, y1=0, x2=0, y2=100)

    def test_inverted_y_raises(self) -> None:
        with pytest.raises(ValueError, match="Inverted y coordinates"):
            BoundingBox(x1=0, y1=100, x2=100, y2=0)

    def test_inverted_both_raises(self) -> None:
        with pytest.raises(ValueError, match="Inverted x coordinates"):
            BoundingBox(x1=50, y1=50, x2=10, y2=10)


# ── Detection ─────────────────────────────────────────────────────


class TestDetection:
    def test_required_fields(self) -> None:
        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        det = Detection(label="person", confidence=0.95, bbox=bbox)
        assert det.label == "person"
        assert det.confidence == 0.95
        assert det.bbox is bbox

    def test_defaults(self) -> None:
        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        det = Detection(label="cat", confidence=0.8, bbox=bbox)
        assert det.area_ratio == 0.0
        assert det.mask is None
        assert det.keypoints is None
        assert det.metadata == {}

    def test_area_ratio_set(self) -> None:
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        det = Detection(
            label="car", confidence=0.7, bbox=bbox, area_ratio=0.35
        )
        assert det.area_ratio == 0.35

    def test_mask_and_keypoints(self) -> None:
        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        mask_stub = [[1, 0], [0, 1]]
        kp_stub = [(5, 5), (10, 10)]
        det = Detection(
            label="person",
            confidence=0.9,
            bbox=bbox,
            mask=mask_stub,
            keypoints=kp_stub,
        )
        assert det.mask == mask_stub
        assert det.keypoints == kp_stub

    def test_metadata_dict(self) -> None:
        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        det = Detection(
            label="person",
            confidence=0.9,
            bbox=bbox,
            metadata={"track_id": 42},
        )
        assert det.metadata["track_id"] == 42

    def test_confidence_too_high_raises(self) -> None:
        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        with pytest.raises(ValueError, match="confidence must be in"):
            Detection(label="person", confidence=1.5, bbox=bbox)

    def test_confidence_negative_raises(self) -> None:
        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        with pytest.raises(ValueError, match="confidence must be in"):
            Detection(label="person", confidence=-0.1, bbox=bbox)

    def test_confidence_boundary_zero(self) -> None:
        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        det = Detection(label="person", confidence=0.0, bbox=bbox)
        assert det.confidence == 0.0

    def test_confidence_boundary_one(self) -> None:
        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        det = Detection(label="person", confidence=1.0, bbox=bbox)
        assert det.confidence == 1.0


# ── PredictionResult ──────────────────────────────────────────────


class TestPredictionResult:
    @staticmethod
    def _make_det(label: str = "obj", conf: float = 0.9) -> Detection:
        return Detection(
            label=label,
            confidence=conf,
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        )

    def test_empty_result_is_falsy(self) -> None:
        result = PredictionResult()
        assert not result
        assert bool(result) is False

    def test_non_empty_result_is_truthy(self) -> None:
        result = PredictionResult(detections=[self._make_det()])
        assert result
        assert bool(result) is True

    def test_len(self) -> None:
        result = PredictionResult(detections=[self._make_det(), self._make_det()])
        assert len(result) == 2

    def test_len_empty(self) -> None:
        assert len(PredictionResult()) == 0

    def test_iter(self) -> None:
        dets = [self._make_det("a"), self._make_det("b")]
        result = PredictionResult(detections=dets)
        labels = [d.label for d in result]
        assert labels == ["a", "b"]

    def test_getitem(self) -> None:
        dets = [self._make_det("first"), self._make_det("second")]
        result = PredictionResult(detections=dets)
        assert result[0].label == "first"
        assert result[1].label == "second"

    def test_getitem_out_of_range(self) -> None:
        result = PredictionResult()
        with pytest.raises(IndexError):
            _ = result[0]

    def test_raw_field(self) -> None:
        sentinel = object()
        result = PredictionResult(raw=sentinel)
        assert result.raw is sentinel

    def test_raw_default_none(self) -> None:
        assert PredictionResult().raw is None

    def test_metadata(self) -> None:
        result = PredictionResult(metadata={"model": "yolov8n"})
        assert result.metadata["model"] == "yolov8n"

    def test_detections_default_empty(self) -> None:
        a = PredictionResult()
        b = PredictionResult()
        a.detections.append(self._make_det())
        assert len(b.detections) == 0
