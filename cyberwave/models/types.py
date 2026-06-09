"""Prediction output types for Cyberwave model runtimes.

Type hierarchy
--------------

Geometric primitives (immutable value objects):

    BoundingBox           — axis-aligned bounding box (AABB) in pixel space
    OrientedBoundingBox   — rotated bounding box for OBB detection models

Keypoints:

    Keypoint              — single landmark with (x, y, visibility)
    KeypointSet           — ordered collection of Keypoints with an optional
                            schema name (e.g. ``"coco_17"``); supports both
                            structured access and numpy round-trip via
                            ``to_array()`` / ``from_array()``

Segmentation:

    Mask                  — binary instance mask aligned to the original image;
                            wraps the raw ``np.ndarray`` and exposes
                            ``pixel_count``, ``area_ratio``, and ``describe()``

Classification:

    ClassificationCandidate   — single (label, confidence, class-index) triple
    ClassificationResult      — ranked top-K list with ``top1``, ``describe()``,
                                iteration, and length

Detection (bbox-based, covers detect / segment / pose / OBB):

    Detection             — base container: label, confidence, BoundingBox.
                            Optional ``mask``, ``keypoints`` (kept as raw
                            ``Any`` for backward compatibility with existing
                            runtimes), ``keypoint_set`` (structured), and
                            ``obb`` for oriented-bbox models.

Result containers (all subclass :class:`PredictionResult`):

    PredictionResult      — base with ``raw`` and ``metadata``; ``predict()``
                            returns a concrete subclass, not a wrapper.
    DetectionResult       — bbox-based detect / segment / pose / OBB lists.
    TextResult            — plain-text LLM / VLM playground output.
    JsonResult            — parsed JSON object or array from cloud models.
    ImageResult           — generated image payloads (data URLs / base64).

Backward compatibility
----------------------
The ``Detection.keypoints`` field is kept as ``Any | None`` (raw numpy array)
so that existing consumers — ``cyberwave.vision.anonymize``, ONNX runtime
tests, etc. — continue working without changes.  The ``keypoint_set`` field
is the structured, typed alternative; Ultralytics-backed models populate
**both** so callers can migrate gradually.

Similarly, ``Detection.mask`` stays as ``Any | None``; the Ultralytics adapter
now stores :class:`Mask` objects there, but other runtimes may still store raw
numpy arrays or ``None``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Standard COCO 17-keypoint schema used by YOLO-pose and many pose models.
COCO_KEYPOINT_SCHEMA: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


# ---------------------------------------------------------------------------
# Geometric primitives
# ---------------------------------------------------------------------------


@dataclass
class BoundingBox:
    """Axis-aligned bounding box (AABB) in pixel coordinates.

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


@dataclass(frozen=True)
class OrientedBoundingBox:
    """Rotated bounding box from an OBB detection model.

    Attributes:
        cx, cy: Center coordinates in pixel space.
        w, h:   Width and height of the box before rotation.
        angle_rad: Counter-clockwise rotation in radians.

    Use :meth:`axis_aligned` to get the tightest AABB containing this OBB.
    """

    cx: float
    cy: float
    w: float
    h: float
    angle_rad: float

    @property
    def area(self) -> float:
        return self.w * self.h

    def axis_aligned(self) -> BoundingBox:
        """Return the tightest AABB fully containing this OBB."""
        cos_a = abs(math.cos(self.angle_rad))
        sin_a = abs(math.sin(self.angle_rad))
        half_w = (self.w * cos_a + self.h * sin_a) / 2
        half_h = (self.w * sin_a + self.h * cos_a) / 2
        return BoundingBox(
            x1=self.cx - half_w,
            y1=self.cy - half_h,
            x2=self.cx + half_w,
            y2=self.cy + half_h,
        )

    def describe(self) -> str:
        deg = math.degrees(self.angle_rad)
        return f"obb(cx={self.cx:.1f},cy={self.cy:.1f},w={self.w:.1f},h={self.h:.1f},angle={deg:.1f}°)"


# ---------------------------------------------------------------------------
# Keypoints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Keypoint:
    """A single spatial landmark (body joint, face point, …).

    Attributes:
        x, y:       Pixel coordinates in the original image space.
        visibility: Score in ``[0, 1]``.  ``0`` means the point is not
                    visible / was not detected.  Defaults to ``1.0``.
    """

    x: float
    y: float
    visibility: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.visibility <= 1.0:
            raise ValueError(
                f"visibility must be in [0, 1], got {self.visibility}"
            )

    @property
    def visible(self) -> bool:
        """True when ``visibility >= 0.5``."""
        return self.visibility >= 0.5

    def as_tuple(self) -> tuple[float, float, float]:
        """Return ``(x, y, visibility)``."""
        return (self.x, self.y, self.visibility)


@dataclass
class KeypointSet:
    """Ordered collection of :class:`Keypoint` objects from a pose model.

    Attributes:
        points: Keypoints in schema order.
        schema: Optional name identifying the point layout (e.g.
                ``"coco_17"``).  Use :data:`COCO_KEYPOINT_SCHEMA` for the
                17-point COCO body layout.

    Structured access::

        kps = detection.keypoint_set
        nose = kps["nose"]          # named lookup when schema is set
        shoulder = kps[5]           # index lookup

        for name, kp in kps.named():
            if kp.visible:
                draw_point(kp.x, kp.y)

    Numpy interop::

        arr = kps.to_array()        # np.ndarray shape (K, 3)  x/y/visibility
        kps2 = KeypointSet.from_array(arr, schema=COCO_KEYPOINT_SCHEMA)
    """

    points: list[Keypoint]
    schema: tuple[str, ...] | None = None

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Iterator[Keypoint]:
        return iter(self.points)

    def __getitem__(self, key: int | str) -> Keypoint:
        """Index by position or by schema name."""
        if isinstance(key, int):
            return self.points[key]
        if self.schema is None:
            raise KeyError(
                f"Cannot look up keypoint by name {key!r}: no schema set."
            )
        try:
            idx = self.schema.index(key)
        except ValueError:
            raise KeyError(f"Keypoint name {key!r} not found in schema.") from None
        return self.points[idx]

    def named(self) -> Iterator[tuple[str | None, Keypoint]]:
        """Iterate over ``(name, keypoint)`` pairs.

        Name is ``None`` when no schema is set.
        """
        for i, kp in enumerate(self.points):
            name = self.schema[i] if self.schema and i < len(self.schema) else None
            yield name, kp

    def visible_points(self) -> list[tuple[str | None, Keypoint]]:
        """Return ``(name, keypoint)`` pairs for visible keypoints only."""
        return [(name, kp) for name, kp in self.named() if kp.visible]

    def to_array(self) -> Any:
        """Return ``np.ndarray`` of shape ``(K, 3)`` with ``(x, y, visibility)`` rows."""
        import numpy as np  # type: ignore[import-untyped]

        return np.array(
            [kp.as_tuple() for kp in self.points], dtype=np.float32
        )

    @classmethod
    def from_array(
        cls,
        array: Any,
        *,
        schema: tuple[str, ...] | None = None,
    ) -> KeypointSet:
        """Build from a ``(K, 2)`` or ``(K, 3)`` numpy array.

        Rows are ``(x, y)`` or ``(x, y, visibility)``.  Missing visibility
        defaults to ``1.0``.
        """
        points: list[Keypoint] = []
        for row in array:
            row_list = list(row)
            x, y = float(row_list[0]), float(row_list[1])
            raw_vis = float(row_list[2]) if len(row_list) > 2 else 1.0
            # Ultralytics visibility scores may exceed 1.0 (raw logits or
            # pixel-scale values).  Clamp to [0, 1] to satisfy the contract.
            vis = max(0.0, min(1.0, raw_vis))
            points.append(Keypoint(x=x, y=y, visibility=vis))
        return cls(points=points, schema=schema)

    def describe(self) -> str:
        n_visible = sum(1 for kp in self.points if kp.visible)
        schema_str = f", schema={self.schema[0]}…" if self.schema else ""
        return f"keypoints(n={len(self.points)}, visible={n_visible}{schema_str})"


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


@dataclass
class Mask:
    """Instance segmentation mask aligned to the original image.

    Attributes:
        data: Binary mask array of shape ``(H, W)``, dtype ``bool`` or
              ``uint8`` where ``1`` / ``True`` is foreground.  Coordinates
              are in the **original image space** (not the model's input crop).
        h, w: Original image height and width (0 if unknown).  Used for
              ``area_ratio`` and ``describe()``.
    """

    data: Any  # np.ndarray[H, W]
    h: int = 0
    w: int = 0

    @property
    def pixel_count(self) -> int:
        """Number of foreground (True / 1) pixels."""
        import numpy as np  # type: ignore[import-untyped]

        return int(np.count_nonzero(self.data))

    @property
    def area_ratio(self) -> float:
        """Fraction of image area occupied by the mask (``0.0`` if ``h``/``w`` unknown)."""
        total = self.h * self.w
        return self.pixel_count / total if total else 0.0

    def describe(self) -> str:
        shape = getattr(self.data, "shape", "?")
        return f"mask(shape={shape})"


# ---------------------------------------------------------------------------
# PredictionResult base
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """Base type for every ``model.predict()`` return value.

    ``predict()`` returns a **concrete subclass** — :class:`TextResult`,
    :class:`DetectionResult`, :class:`ImageResult`, … — not a wrapper around
    another object.

    Attributes:
        raw:       Unprocessed runtime / provider payload when available.
        metadata:  Transport context (``output_format``, ``model_slug``, …).
    """

    raw: Any | None = field(default=None, kw_only=True)
    metadata: dict[str, Any] = field(default_factory=dict, kw_only=True)

    def __new__(cls, *args: Any, **kwargs: Any) -> PredictionResult:
        """Legacy factory: ``PredictionResult(detections=...)`` → :class:`DetectionResult`."""
        if cls is not PredictionResult:
            return super().__new__(cls)

        detections = kwargs.pop("detections", None)
        output = kwargs.pop("output", None)
        raw = kwargs.pop("raw", None)
        metadata = kwargs.pop("metadata", None)
        meta = metadata if metadata is not None else {}

        if output is not None:
            if not isinstance(output, PredictionResult):
                raise TypeError(
                    "legacy PredictionResult(output=...) requires a "
                    f"PredictionResult subclass, got {type(output)!r}"
                )
            if raw is not None:
                output.raw = raw
            if metadata is not None:
                output.metadata = meta
            return output

        inst = object.__new__(DetectionResult)
        dets = list(detections) if detections is not None else []
        object.__setattr__(inst, "detections", dets)
        object.__setattr__(inst, "raw", raw)
        object.__setattr__(inst, "metadata", meta)
        return inst

    def describe(self) -> str:
        return "(empty prediction)"

    def describe_detections_lines(
        self,
        *,
        indent: str = "  ",
        empty_marker: str = "(no detections)",
    ) -> list[str]:
        """Fallback for non-detection results; :class:`DetectionResult` overrides."""
        _ = indent
        return [empty_marker]

    def describe_detections_text(
        self,
        *,
        indent: str = "  ",
        empty_marker: str = "(no detections)",
    ) -> str:
        """Fallback for non-detection results; :class:`DetectionResult` overrides."""
        if self:
            return self.describe()
        _ = indent
        return empty_marker

    def __len__(self) -> int:
        return 0

    def __bool__(self) -> bool:
        return False


@dataclass
class QueuedPredictionResult(PredictionResult):
    """Placeholder returned while an async cloud workload is still running."""

    def describe(self) -> str:
        wl = self.metadata.get("workload_uuid", "?")
        poll = self.metadata.get("poll_url", "")
        suffix = f", poll_url={poll!r}" if poll else ""
        return f"Queued(workload_uuid={wl!r}{suffix})"

    @property
    def workload_uuid(self) -> str | None:
        value = self.metadata.get("workload_uuid")
        return str(value) if value is not None else None

    @property
    def poll_url(self) -> str | None:
        value = self.metadata.get("poll_url")
        return str(value) if value is not None else None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassificationCandidate:
    """A single class prediction from a classification model.

    Attributes:
        label:      Class name.
        confidence: Probability in ``[0, 1]``.
        index:      Class index from the model's class list (``-1`` if unknown).
    """

    label: str
    confidence: float
    index: int = -1

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )

    def describe(self) -> str:
        return f"{self.label!r} conf={self.confidence:.3f}"


@dataclass
class ClassificationResult(PredictionResult):
    """Top-K output from a classification model.

    Attributes:
        top:  Candidates sorted **high → low** by confidence.

    Usage::

        result = model.predict(frame)
        print(result.top1.label)
        for c in result:
            print(c.describe())
    """

    top: list[ClassificationCandidate] = field(default_factory=list)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        top = kwargs.pop("top", None)
        if args:
            top = args[0]
        raw = kwargs.pop("raw", None)
        metadata = kwargs.pop("metadata", None)
        if kwargs:
            raise TypeError(
                f"unexpected keyword arguments: {list(kwargs)}"
            )
        if top is not None:
            self.top = list(top)
        elif not hasattr(self, "top"):
            self.top = []
        if raw is not None:
            self.raw = raw
        elif not hasattr(self, "raw"):
            self.raw = None
        if metadata is not None:
            self.metadata = metadata
        elif not hasattr(self, "metadata"):
            self.metadata = {}

    def __len__(self) -> int:
        return len(self.top)

    def __bool__(self) -> bool:
        return bool(self.top)

    def __iter__(self) -> Iterator[ClassificationCandidate]:
        return iter(self.top)

    def __getitem__(self, index: int) -> ClassificationCandidate:
        return self.top[index]

    @property
    def top1(self) -> ClassificationCandidate | None:
        """Highest-confidence candidate, or ``None`` when empty."""
        return self.top[0] if self.top else None

    def describe(self, *, n: int = 5) -> str:
        """Human-readable top-*n* summary."""
        if not self.top:
            return "(no classification)"
        lines = [f"  [{i}] {c.describe()}" for i, c in enumerate(self.top[:n])]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Detection (covers detect / segment / pose / OBB)
# ---------------------------------------------------------------------------


def _spatial_extra_label(name: str, value: Any) -> str:
    """Stable describe text for mask/keypoints fields."""
    # Prefer rich describe() on Mask / KeypointSet
    describe_fn = getattr(value, "describe", None)
    if describe_fn is not None:
        try:
            return describe_fn()
        except Exception:
            pass
    # Legacy: numpy array with .shape
    shape = getattr(value, "shape", None)
    if shape is not None:
        return f"{name} shape={tuple(int(d) for d in shape)}"
    return f"{name}=<{type(value).__name__}>"


@dataclass
class Detection:
    """A single object detection from a model prediction.

    Covers all bbox-based tasks: plain detection, instance segmentation, pose
    estimation, and oriented bounding box (OBB).

    Attributes:
        label:         Class name (e.g. ``"person"``).
        confidence:    Detection score in ``[0, 1]``.
        bbox:          Axis-aligned bounding box in pixel space.  For OBB
                       models this is the tightest AABB; see also ``obb``.
        area_ratio:    ``bbox.area / frame_area``  (``0.0`` if unknown).
        mask:          Segmentation mask (:class:`Mask` when produced by the
                       Ultralytics adapter; raw ``np.ndarray`` from other
                       runtimes; ``None`` for plain detection).
        keypoints:     Raw keypoint array ``(K, 2|3)`` kept as ``Any`` for
                       backward compatibility.  Prefer ``keypoint_set`` for
                       new code.
        keypoint_set:  Structured :class:`KeypointSet` populated by the
                       Ultralytics adapter.  Includes per-point visibility and
                       optional schema name (e.g. :data:`COCO_KEYPOINT_SCHEMA`).
        obb:           :class:`OrientedBoundingBox` for OBB tasks; ``None`` for
                       all other task types.
        metadata:      Arbitrary extra fields from the runtime.

    Raises :class:`ValueError` if *confidence* is outside ``[0, 1]``.
    """

    label: str
    confidence: float
    bbox: BoundingBox
    area_ratio: float = 0.0
    # mask: Any — raw numpy array or Mask object depending on runtime
    mask: Any | None = None
    # keypoints: Any — raw (K,2|3) numpy array; kept for backward compat
    keypoints: Any | None = None
    # keypoint_set: structured alternative populated by Ultralytics adapter
    keypoint_set: KeypointSet | None = None
    # obb: oriented bounding box for OBB tasks
    obb: OrientedBoundingBox | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )

    def describe_parts(
        self,
        *,
        include_keypoints: bool = True,
        include_mask: bool = True,
    ) -> list[str]:
        """Token list for CLI / notebooks (label, conf, bbox, optional extras)."""
        parts = [
            repr(self.label),
            f"conf={self.confidence:.3f}",
            f"bbox=({self.bbox.x1:.1f},{self.bbox.y1:.1f})-"
            f"({self.bbox.x2:.1f},{self.bbox.y2:.1f})",
        ]
        if include_mask and self.mask is not None:
            parts.append(_spatial_extra_label("mask", self.mask))
        if include_keypoints:
            if self.keypoint_set is not None:
                parts.append(self.keypoint_set.describe())
            elif self.keypoints is not None:
                parts.append(_spatial_extra_label("keypoints", self.keypoints))
        if self.obb is not None:
            parts.append(self.obb.describe())
        return parts

    def describe_line(self, index: int, *, indent: str = "  ") -> str:
        """Single human-readable row with bracketed index."""
        bits = [f"[{index}]"] + self.describe_parts()
        return indent + " ".join(bits)


# ---------------------------------------------------------------------------
# Detection-shaped results
# ---------------------------------------------------------------------------


@dataclass
class DetectionResult(PredictionResult):
    """An ordered list of :class:`Detection` objects from a single model run.

    Covers detection, instance segmentation, pose, and OBB tasks.

    Usage::

        result = model.predict(frame)
        for det in result:
            print(det.label, det.confidence)

        filtered = result.filter(min_confidence=0.6, labels={"person"})
        print(result.describe())
    """

    detections: list[Detection] = field(default_factory=list)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("output", None)
        kwargs.pop("classification", None)
        detections = kwargs.pop("detections", None)
        if args:
            detections = args[0]
        raw = kwargs.pop("raw", None)
        metadata = kwargs.pop("metadata", None)
        if kwargs:
            raise TypeError(
                f"unexpected keyword arguments: {list(kwargs)}"
            )
        if detections is not None:
            self.detections = list(detections)
        elif not hasattr(self, "detections"):
            self.detections = []
        if raw is not None:
            self.raw = raw
        elif not hasattr(self, "raw"):
            self.raw = None
        if metadata is not None:
            self.metadata = metadata
        elif not hasattr(self, "metadata"):
            self.metadata = {}

    def __iter__(self) -> Iterator[Detection]:
        return iter(self.detections)

    def __len__(self) -> int:
        return len(self.detections)

    def __bool__(self) -> bool:
        return bool(self.detections)

    def __getitem__(self, index: int) -> Detection:
        return self.detections[index]

    def __repr__(self) -> str:
        return f"DetectionResult({len(self.detections)} detections)"

    def filter(
        self,
        *,
        min_confidence: float = 0.0,
        labels: set[str] | None = None,
    ) -> "DetectionResult":
        """Return a same-type result containing only the matching detections.

        Preserves the concrete subtype — filtering a :class:`PoseResult` returns
        a :class:`PoseResult`, not a plain :class:`DetectionResult`.
        """
        items: list[Detection] = self.detections
        if min_confidence > 0.0:
            items = [d for d in items if d.confidence >= min_confidence]
        if labels is not None:
            items = [d for d in items if d.label in labels]
        return type(self)(items)  # type: ignore[return-value]

    def describe_lines(
        self,
        *,
        indent: str = "  ",
        empty_marker: str = "(no detections)",
    ) -> list[str]:
        """One string per detection, or ``[empty_marker]`` when empty."""
        if not self.detections:
            return [empty_marker]
        return [d.describe_line(i, indent=indent) for i, d in enumerate(self.detections)]

    def describe(
        self,
        *,
        indent: str = "  ",
        empty_marker: str = "(no detections)",
    ) -> str:
        """Multi-line human-readable summary."""
        return "\n".join(self.describe_lines(indent=indent, empty_marker=empty_marker))

    def describe_detections_lines(
        self,
        *,
        indent: str = "  ",
        empty_marker: str = "(no detections)",
    ) -> list[str]:
        """Backward-compat alias for :meth:`describe_lines`."""
        return self.describe_lines(indent=indent, empty_marker=empty_marker)

    def describe_detections_text(
        self,
        *,
        indent: str = "  ",
        empty_marker: str = "(no detections)",
    ) -> str:
        """Backward-compat alias for :meth:`describe`."""
        return self.describe(indent=indent, empty_marker=empty_marker)


# ---------------------------------------------------------------------------
# Specialised DetectionResult subtypes
# ---------------------------------------------------------------------------


@dataclass
class InstanceSegmentationResult(DetectionResult):
    """DetectionResult where every :class:`Detection` carries a :class:`Mask`.

    Returned by the Ultralytics adapter for ``*-seg`` models.
    """

    @property
    def masks(self) -> list[Mask]:
        """All non-null :class:`Mask` objects from the detection list."""
        return [d.mask for d in self.detections if d.mask is not None]


@dataclass
class PoseResult(DetectionResult):
    """DetectionResult where every :class:`Detection` carries a :class:`KeypointSet`.

    Returned by the Ultralytics adapter for ``*-pose`` models.
    """

    @property
    def skeletons(self) -> list[KeypointSet]:
        """All non-null :class:`KeypointSet` objects from the detection list."""
        return [d.keypoint_set for d in self.detections if d.keypoint_set is not None]


@dataclass
class OBBResult(DetectionResult):
    """DetectionResult where every :class:`Detection` carries an :class:`OrientedBoundingBox`.

    Returned by the Ultralytics adapter for ``*-obb`` models.
    """

    @property
    def oriented_boxes(self) -> list[OrientedBoundingBox]:
        """All non-null :class:`OrientedBoundingBox` objects from the detection list."""
        return [d.obb for d in self.detections if d.obb is not None]


# ---------------------------------------------------------------------------
# Semantic segmentation, embedding, and custom result types
# ---------------------------------------------------------------------------


@dataclass
class SemanticSegmentationResult(PredictionResult):
    """Full-image pixel-level class labelling (no per-instance bounding boxes).

    Produced by semantic segmentation models where every pixel is assigned the
    ID of the most likely class — contrast with :class:`InstanceSegmentationResult`
    which pairs per-object boxes with per-object masks.

    Attributes:
        mask:         ``np.ndarray`` of shape ``(H, W)`` with integer class IDs.
        class_names:  ``{class_id: name}`` mapping (empty when unknown).
        h, w:         Original image dimensions (``0`` if unknown).
    """

    mask: Any  # np.ndarray[H, W], dtype int
    class_names: dict[int, str] = field(default_factory=dict)
    h: int = 0
    w: int = 0

    def __len__(self) -> int:
        return len(self.class_names)

    def __bool__(self) -> bool:
        return self.mask is not None

    def present_classes(self) -> list[tuple[int, str]]:
        """Return ``(id, name)`` pairs for class IDs present in the mask."""
        import numpy as np  # type: ignore[import-untyped]

        return [
            (int(cid), self.class_names.get(int(cid), str(int(cid))))
            for cid in np.unique(self.mask).tolist()
        ]

    def area_of(self, class_id: int) -> float:
        """Fraction of image area ``[0, 1]`` occupied by *class_id*."""
        import numpy as np  # type: ignore[import-untyped]

        total = self.h * self.w
        if total == 0:
            return 0.0
        return float(np.sum(self.mask == class_id)) / total

    def describe(self) -> str:
        shape = getattr(self.mask, "shape", "?")
        return f"SemanticSegmentation(shape={shape}, classes={len(self.class_names)})"


@dataclass
class EmbeddingResult(PredictionResult):
    """Feature embedding vector(s) from an encoder model (CLIP, ViT, etc.).

    Attributes:
        vector: ``np.ndarray`` of shape ``(D,)`` for a single embedding or
                ``(N, D)`` for a batch.
        extras: Model-specific fields (e.g. ``source_layer``).
    """

    vector: Any = None  # np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        shape = getattr(self.vector, "shape", ())
        return int(shape[0]) if shape else 0

    def __bool__(self) -> bool:
        return self.vector is not None

    @property
    def dim(self) -> int:
        """Embedding dimension (last-axis size, or ``0`` if unknown)."""
        shape = getattr(self.vector, "shape", ())
        return int(shape[-1]) if shape else 0

    def normalize(self) -> "EmbeddingResult":
        """Return a copy with L2-normalised vector(s)."""
        import numpy as np  # type: ignore[import-untyped]

        norm = np.linalg.norm(self.vector, axis=-1, keepdims=True)
        return EmbeddingResult(
            vector=self.vector / (norm + 1e-8),
            extras=dict(self.extras),
            raw=self.raw,
            metadata=dict(self.metadata),
        )

    def describe(self) -> str:
        shape = getattr(self.vector, "shape", "?")
        return f"Embedding(shape={shape})"


# ---------------------------------------------------------------------------
# Cloud / LLM text, JSON, and image outputs
# ---------------------------------------------------------------------------


@dataclass
class TextResult(PredictionResult):
    """Plain-text model output from cloud LLM / VLM playground runs.

    Returned when the backend declares ``output_format="text"`` — free-form
    prompts, captions, Q&A, etc.
    """

    text: str = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("output", None)
        kwargs.pop("classification", None)
        kwargs.pop("detections", None)
        text = kwargs.pop("text", "")
        if args:
            text = args[0]
        raw = kwargs.pop("raw", None)
        metadata = kwargs.pop("metadata", None)
        if kwargs:
            raise TypeError(
                f"unexpected keyword arguments: {list(kwargs)}"
            )
        if text != "" or not hasattr(self, "text"):
            self.text = str(text)
        if raw is not None:
            self.raw = raw
        elif not hasattr(self, "raw"):
            self.raw = None
        if metadata is not None:
            self.metadata = metadata
        elif not hasattr(self, "metadata"):
            self.metadata = {}

    def __len__(self) -> int:
        return len(self.text)

    def __bool__(self) -> bool:
        return bool(self.text)

    def describe(self) -> str:
        preview = self.text[:120] + ("…" if len(self.text) > 120 else "")
        return f"Text({len(self.text)} chars): {preview!r}"

    def save(self, path: str | Path, *, encoding: str = "utf-8") -> str:
        """Write the text payload to *path* and return the resolved path."""
        target = Path(path)
        target.write_text(self.text, encoding=encoding)
        return str(target.resolve())


@dataclass
class JsonResult(PredictionResult):
    """Parsed JSON object or array from a cloud model run.

    :attr:`PredictionResult.raw` keeps the provider wire string when the
    backend supplies one; :attr:`data` holds the parsed canonical payload.
    """

    data: Any = None

    def __len__(self) -> int:
        try:
            return len(self.data)
        except TypeError:
            return 1

    def __bool__(self) -> bool:
        return self.data is not None

    def describe(self) -> str:
        kind = type(self.data).__name__
        return f"Json({kind}, n={len(self)})"

    def save(
        self,
        path: str | Path,
        *,
        indent: int | None = 2,
        encoding: str = "utf-8",
    ) -> str:
        """Serialize :attr:`data` as JSON to *path* and return the resolved path."""
        target = Path(path)
        with target.open("w", encoding=encoding) as handle:
            json.dump(self.data, handle, indent=indent)
        return str(target.resolve())


@dataclass
class ImageResult(PredictionResult):
    """Generated or returned image payload from a cloud model run.

    Normalizes ``{"data_url": ...}`` and ``{"image_url": ...}`` into
    :attr:`url`.  Use :meth:`save`, :meth:`to_ndarray`, and :meth:`to_pil`
    to materialize the image bytes.
    """

    data_url: str | None = None
    base64: str | None = None
    remote_url: str | None = None
    mime_type: str = "image/png"

    @classmethod
    def from_output(cls, output: Any) -> ImageResult:
        """Build from a playground ``output`` field (dict, data URL, or base64)."""
        if isinstance(output, str):
            if output.startswith("data:"):
                return cls(data_url=output)
            if output.startswith(("http://", "https://")):
                return cls(remote_url=output)
            return cls(base64=output)

        if isinstance(output, dict):
            for key in ("data_url", "image_url"):
                value = output.get(key)
                if isinstance(value, str) and value:
                    if value.startswith("data:"):
                        return cls(data_url=value)
                    if value.startswith(("http://", "https://")):
                        return cls(
                            remote_url=value,
                            mime_type=_mime_from_dict(output),
                        )
                    return cls(base64=value, mime_type=_mime_from_dict(output))

            b64 = output.get("base64")
            if isinstance(b64, str) and b64:
                mime = _mime_from_dict(output)
                if b64.startswith("data:"):
                    return cls(data_url=b64, mime_type=mime)
                return cls(base64=b64, mime_type=mime)

        if output is None:
            return cls()
        return cls(data_url=str(output))

    @property
    def url(self) -> str | None:
        """Best display URL — data URL, synthesized base64 URL, or remote http(s)."""
        if self.data_url:
            return self.data_url
        if self.remote_url:
            return self.remote_url
        if self.base64:
            payload = self.base64
            if payload.startswith("data:"):
                return payload
            return f"data:{self.mime_type};base64,{payload}"
        return None

    def __len__(self) -> int:
        return 1 if self.url else 0

    def __bool__(self) -> bool:
        return self.url is not None

    def describe(self) -> str:
        url = self.url
        if not url:
            return "Image(empty)"
        if url.startswith("data:"):
            return f"Image({self.mime_type}, {len(url)} chars)"
        return f"Image(url={url!r})"

    def bytes(self) -> bytes:
        """Return decoded image bytes (JPEG/PNG/WebP as returned by the provider)."""
        if self.remote_url:
            return _fetch_remote_image_bytes(self.remote_url)
        url = self.url
        if not url:
            raise ValueError("ImageResult has no image payload")
        from cyberwave.image import decode_image_base64

        return decode_image_base64(url)

    def save(self, path: str | Path) -> str:
        """Write decoded image bytes to *path* (extension selects container format).

        No re-encoding — the provider bytes are written as-is, so a JPEG
        payload saved to ``out.jpg`` stays JPEG. Returns the resolved path.
        """
        target = Path(path)
        target.write_bytes(self.bytes())
        return str(target.resolve())

    def save_jpg(self, path: str | Path) -> str:
        """Write bytes to *path*, appending ``.jpg`` when no suffix is present."""
        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(".jpg")
        return self.save(target)

    def save_png(self, path: str | Path) -> str:
        """Write bytes to *path*, appending ``.png`` when no suffix is present."""
        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(".png")
        return self.save(target)

    def to_pil(self) -> Any:
        """Return a ``PIL.Image`` (requires ``pip install cyberwave[image]``)."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "ImageResult.to_pil() requires Pillow. "
                "Install with: pip install cyberwave[image]"
            ) from exc

        import io

        return Image.open(io.BytesIO(self.bytes()))

    def to_ndarray(self) -> Any:
        """Return an ``H×W×C`` RGB ``numpy.ndarray`` (requires Pillow)."""
        import numpy as np

        return np.asarray(self.to_pil())


def _mime_from_dict(payload: dict[str, Any]) -> str:
    mime = payload.get("mime_type") or payload.get("content_type")
    return mime if isinstance(mime, str) and mime else "image/png"


def _fetch_remote_image_bytes(image_url: str) -> bytes:
    """Download image bytes from an http(s) URL (cloud playground payloads)."""
    import urllib3

    http = urllib3.PoolManager()
    response = http.request(
        "GET",
        image_url,
        timeout=urllib3.Timeout(connect=5, read=30),
    )
    if response.status >= 400:
        body = response.data.decode("utf-8", errors="replace")
        raise ValueError(
            f"Failed to download image from {image_url!r}: HTTP {response.status} {body}"
        )
    return bytes(response.data)


@dataclass
class CustomResult(PredictionResult):
    """Escape hatch for arbitrary model outputs.

    Use when the model produces data that doesn't fit any structured result
    type — multi-head networks, raw tensors, research outputs, etc.

    Attributes:
        data:  The raw model output — any Python object.
        label: A short human-readable tag identifying the output kind.
    """

    data: Any
    label: str = "custom"

    def __len__(self) -> int:
        try:
            return len(self.data)
        except TypeError:
            return 1

    def __bool__(self) -> bool:
        return self.data is not None

    def describe(self) -> str:
        return f"CustomResult(label={self.label!r}, type={type(self.data).__name__})"


# ---------------------------------------------------------------------------
# ModelOutput Protocol — common interface for all result types
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelOutput(Protocol):
    """Structural interface satisfied by every :class:`PredictionResult` subclass."""

    def __len__(self) -> int: ...
    def __bool__(self) -> bool: ...
    def describe(self) -> str: ...
