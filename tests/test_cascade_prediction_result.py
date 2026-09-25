"""Tests for CascadePredictionResult mixed result types."""

import pytest

from cyberwave.models.cascade import CascadePredictionResult
from cyberwave.models.types import (
    BoundingBox,
    Detection,
    DetectionResult,
    TextResult,
)


class TestCascadePredictionResultMixed:
    @staticmethod
    def _det(label: str = "person") -> Detection:
        return Detection(
            label=label,
            confidence=0.9,
            bbox=BoundingBox(10, 20, 30, 40),
        )

    def test_total_detections_mixed_cascade(self) -> None:
        cascade = CascadePredictionResult(
            {
                "stt": TextResult(text="hello world"),
                "yolo": DetectionResult(detections=[self._det()]),
            },
            image_size=(640, 480),
            input_image=None,
        )
        assert cascade.total_detections() == 1

    def test_describe_mixed_cascade(self) -> None:
        cascade = CascadePredictionResult(
            {
                "stt": TextResult(text="hello world"),
                "yolo": DetectionResult(detections=[self._det("cup")]),
            },
            image_size=(640, 480),
            input_image=None,
        )
        text = cascade.describe()
        assert "TextResult" in text
        assert "hello" in text
        assert "1 detection(s)" in text
        assert "'cup'" in text

    def test_repr_mixed_cascade(self) -> None:
        cascade = CascadePredictionResult(
            {
                "stt": TextResult(text="x"),
                "yolo": DetectionResult(detections=[self._det(), self._det()]),
            },
            image_size=None,
            input_image=None,
        )
        assert "TextResult" in repr(cascade)
        assert "2" in repr(cascade)

    def test_draw_skips_non_detection_models(self) -> None:
        pil = pytest.importorskip("PIL")
        from PIL import Image

        cascade = CascadePredictionResult(
            {
                "stt": TextResult(text="ignored for draw"),
                "yolo": DetectionResult(detections=[self._det()]),
            },
            image_size=(100, 100),
            input_image=None,
        )
        base = Image.new("RGB", (100, 100), color=(0, 0, 0))
        out = cascade.draw_on_top(image=base)
        assert isinstance(out, pil.Image.Image)
