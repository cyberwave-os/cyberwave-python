"""Stable output types for model predictions.

These dataclasses define the contract that downstream codegen and runtime
adapters depend on.  They are intentionally simple value objects with no
framework dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates.

    Raises :class:`ValueError` if ``x2 < x1`` or ``y2 < y1``.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1:
            raise ValueError(
                f"Inverted x coordinates: x1={self.x1} > x2={self.x2}"
            )
        if self.y2 < self.y1:
            raise ValueError(
                f"Inverted y coordinates: y1={self.y1} > y2={self.y2}"
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class Detection:
    """A single object detection from a model prediction.

    Attributes:
        label: Class name (e.g. ``"person"``).
        confidence: Score in ``[0, 1]``.
        bbox: Pixel-coordinate bounding box.
        area_ratio: Bounding-box area divided by frame area (``0.0`` if
            frame dimensions are unknown).
        mask: Optional segmentation mask (numpy array).
        keypoints: Optional pose keypoints.
        metadata: Arbitrary extra fields.

    Raises :class:`ValueError` if *confidence* is outside ``[0, 1]``.
    """

    label: str
    confidence: float
    bbox: BoundingBox
    area_ratio: float = 0.0
    mask: Any | None = None
    keypoints: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )


@dataclass
class PredictionResult:
    """Container for model prediction output.

    Iterable over :class:`Detection` instances, truthy when non-empty, and
    supports ``len()``.
    """

    detections: list[Detection] = field(default_factory=list)
    raw: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Detection]:
        return iter(self.detections)

    def __len__(self) -> int:
        return len(self.detections)

    def __bool__(self) -> bool:
        return len(self.detections) > 0

    def __getitem__(self, index: int) -> Detection:
        return self.detections[index]
