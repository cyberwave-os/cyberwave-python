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
    ClassificationPrediction  — ranked top-K list with ``top1``, ``describe()``,
                                iteration, and length

Detection (bbox-based, covers detect / segment / pose / OBB):

    Detection             — base container: label, confidence, BoundingBox.
                            Optional ``mask``, ``keypoints`` (kept as raw
                            ``Any`` for backward compatibility with existing
                            runtimes), ``keypoint_set`` (structured), and
                            ``obb`` for oriented-bbox models.

Result containers:

    PredictionResult      — holds a ``detections`` list plus an optional
                            ``classification`` for models that classify rather
                            than detect.  ``raw`` carries the unprocessed
                            runtime output for callers that need it.

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

import math
from dataclasses import dataclass, field
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
class ClassificationPrediction:
    """Top-K output from a classification model.

    A pure result type — no ``raw`` or ``metadata``; those belong on the
    :class:`PredictionResult` transport wrapper.

    Attributes:
        top:  Candidates sorted **high → low** by confidence.

    Usage::

        if pred.classification:
            print(pred.classification.top1.label)
            for c in pred.classification:
                print(c.describe())
    """

    top: list[ClassificationCandidate] = field(default_factory=list)

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
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class DetectionResult:
    """An ordered list of :class:`Detection` objects from a single model run.

    Covers detection, instance segmentation, pose, and OBB tasks.  The class
    owns all boilerplate so that :class:`PredictionResult` stays a thin wrapper.

    Usage::

        result = pred.output           # DetectionResult
        for det in result:
            print(det.label, det.confidence)

        filtered = result.filter(min_confidence=0.6, labels={"person"})
        print(result.describe())
    """

    detections: list[Detection] = field(default_factory=list)

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
class SemanticSegmentationResult:
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
class EmbeddingResult:
    """Feature embedding vector(s) from an encoder model (CLIP, ViT, etc.).

    Attributes:
        vector:   ``np.ndarray`` of shape ``(D,)`` for a single embedding or
                  ``(N, D)`` for a batch.
        metadata: Arbitrary extra fields (e.g. ``source_layer``).
    """

    vector: Any  # np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

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
        return EmbeddingResult(vector=self.vector / (norm + 1e-8), metadata=self.metadata)

    def describe(self) -> str:
        shape = getattr(self.vector, "shape", "?")
        return f"Embedding(shape={shape})"


@dataclass
class CustomResult:
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
    """Structural interface satisfied by every concrete result type.

    :class:`PredictionResult` holds any ``ModelOutput`` and can dispatch to it
    via ``describe()`` without branching on the concrete type.  Downstream code
    can use ``isinstance(pred.output, PoseResult)`` for task-specific access.
    """

    def __len__(self) -> int: ...
    def __bool__(self) -> bool: ...
    def describe(self) -> str: ...


class PredictionResult:
    """Thin transport wrapper over any :class:`ModelOutput`.

    Holds one concrete result type (e.g. :class:`DetectionResult`,
    :class:`PoseResult`, :class:`ClassificationPrediction`, …) as ``output``,
    plus ``raw`` (unprocessed runtime object) and ``metadata``.

    **Preferred construction** (new code)::

        PredictionResult(output=PoseResult(dets), raw=results)
        PredictionResult(output=ClassificationPrediction(candidates), raw=results)
        PredictionResult(output=EmbeddingResult(vector=vec), raw=results)

    **Legacy construction** (backward compat)::

        PredictionResult(detections=dets, raw=results)
        PredictionResult(classification=cls_pred, raw=results)

    **Type-check shortcuts**::

        pred.classification     # ClassificationPrediction | None
        pred.pose               # PoseResult | None
        pred.instance_segmentation  # InstanceSegmentationResult | None
        pred.obb_result         # OBBResult | None
        pred.embedding          # EmbeddingResult | None

    **Generic access** (unchanged from previous API)::

        for det in pred:    ...    # iterates detections (DetectionResult variants)
        len(pred)                  # item count
        bool(pred)                 # truthy when output is present and non-empty
        pred.detections            # list[Detection] ([] for non-detection types)
    """

    __slots__ = ("output", "raw", "metadata")

    def __init__(
        self,
        # Legacy positional/keyword — kept for backward compat
        detections: list[Detection] | None = None,
        *,
        output: ModelOutput | None = None,
        # Legacy keyword — kept for backward compat
        classification: ClassificationPrediction | None = None,
        raw: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if output is not None:
            self.output: ModelOutput | None = output
        elif classification is not None:
            self.output = classification
        elif detections is not None:
            self.output = DetectionResult(detections)
        else:
            self.output = None
        self.raw = raw
        self.metadata: dict[str, Any] = metadata if metadata is not None else {}

    # ------------------------------------------------------------------
    # Type-check convenience properties
    # ------------------------------------------------------------------

    @property
    def detections(self) -> list[Detection]:
        """All :class:`Detection` objects, or ``[]`` for non-detection models."""
        if isinstance(self.output, DetectionResult):
            return self.output.detections
        return []

    @property
    def classification(self) -> ClassificationPrediction | None:
        """Classification output, or ``None`` for other model types."""
        return self.output if isinstance(self.output, ClassificationPrediction) else None

    @property
    def pose(self) -> PoseResult | None:
        """Pose estimation output, or ``None`` for other model types."""
        return self.output if isinstance(self.output, PoseResult) else None

    @property
    def instance_segmentation(self) -> InstanceSegmentationResult | None:
        """Instance segmentation output, or ``None`` for other model types."""
        return self.output if isinstance(self.output, InstanceSegmentationResult) else None

    @property
    def obb_result(self) -> OBBResult | None:
        """Oriented bounding box output, or ``None`` for other model types."""
        return self.output if isinstance(self.output, OBBResult) else None

    @property
    def embedding(self) -> EmbeddingResult | None:
        """Embedding output, or ``None`` for other model types."""
        return self.output if isinstance(self.output, EmbeddingResult) else None

    # ------------------------------------------------------------------
    # Delegation — operate on whatever output type is stored
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Detection]:
        if isinstance(self.output, DetectionResult):
            return iter(self.output)
        return iter([])

    def __len__(self) -> int:
        return len(self.output) if self.output is not None else 0

    def __bool__(self) -> bool:
        return bool(self.output) if self.output is not None else False

    def __getitem__(self, index: int) -> Detection:
        if isinstance(self.output, DetectionResult):
            return self.output[index]
        raise IndexError(f"No detections — cannot index with {index!r}")

    def __repr__(self) -> str:
        return f"PredictionResult(output={self.output!r})"

    # ------------------------------------------------------------------
    # Describe helpers — backward compat names kept
    # ------------------------------------------------------------------

    def describe_detections_lines(
        self,
        *,
        indent: str = "  ",
        empty_marker: str = "(no detections)",
    ) -> list[str]:
        """Delegates to :meth:`DetectionResult.describe_lines`."""
        if isinstance(self.output, DetectionResult):
            return self.output.describe_lines(indent=indent, empty_marker=empty_marker)
        return [empty_marker]

    def describe_detections_text(
        self,
        *,
        indent: str = "  ",
        empty_marker: str = "(no detections)",
    ) -> str:
        """Concatenates :meth:`describe_detections_lines` with newlines."""
        return "\n".join(
            self.describe_detections_lines(indent=indent, empty_marker=empty_marker)
        )

    def describe(self) -> str:
        """Full summary — delegates to the stored output type's ``describe()``.

        No branching: every :class:`ModelOutput` owns its own ``describe()``.
        """
        return self.output.describe() if self.output is not None else "(empty prediction)"
