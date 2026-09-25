"""Multi-model cascade: run N models on the same input, collect results by name.

Returned by :meth:`~cyberwave.models.ModelManager.load` when a **list** of model
entries is passed::

    cascade = cw.models.load(
        [entry1, entry2, entry3],
        store_input=True,
    )
    pred = cascade.predict(image)

    # dict-style access — key is the display name of each catalog entry
    print(pred[entry1.name].describe_detections_text())

    # draw all predictions on the original image (store_input=True required when
    # no explicit image is passed)
    output_image = pred.draw_on_top()

    # draw all predictions on a black canvas, same size as the input frame
    output_image = pred.draw()

Drawing
-------
* :meth:`CascadePredictionResult.draw_on_top` — overlays every model's detections
  on the given (or stored) input image.  Each model is assigned a unique color so
  the predictions are distinguishable at a glance.
* :meth:`CascadePredictionResult.draw` — same as above but starts from a black
  background the same dimensions as the input instead of the original pixels.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator

from cyberwave.models.types import DetectionResult, PredictionResult

if TYPE_CHECKING:
    from cyberwave.models.cloud import CloudLoadedModel
    from cyberwave.models.loaded_model import LoadedModel

logger = logging.getLogger(__name__)

# Each model in the cascade is assigned one color from this palette (cycles if
# there are more than 8 models).
_PALETTE: list[str] = [
    "#EF4444",  # red
    "#3B82F6",  # blue
    "#22C55E",  # green
    "#F59E0B",  # amber
    "#8B5CF6",  # violet
    "#EC4899",  # pink
    "#14B8A6",  # teal
    "#F97316",  # orange
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_pil(image: Any) -> Any:
    """Convert a numpy array or PIL Image to a PIL ``Image.Image``."""
    try:
        from PIL import Image as PILImage
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for drawing. Install it: pip install pillow"
        ) from exc

    if isinstance(image, PILImage.Image):
        return image

    import numpy as np  # type: ignore[import-untyped]

    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return PILImage.fromarray(arr)


def _extract_image_size(image: Any) -> tuple[int, int] | None:
    """Return ``(width, height)`` from a PIL Image or numpy array, or ``None``."""
    try:
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            return image.size  # (width, height)
    except ImportError:
        pass

    try:
        import numpy as np  # type: ignore[import-untyped]

        arr = np.asarray(image)
        if arr.ndim >= 2:
            h, w = arr.shape[:2]
            return (w, h)
    except (ImportError, ValueError, TypeError):
        pass

    return None


# ---------------------------------------------------------------------------
# CascadePredictionResult
# ---------------------------------------------------------------------------


class CascadePredictionResult:
    """Prediction output from a :class:`CascadeModel` run.

    Maps model display name → :class:`~cyberwave.models.types.PredictionResult`
    for every model in the cascade.  Supports dict-style access by model name
    and iteration over ``(name, result)`` pairs.

    Drawing helpers:

    * :meth:`draw_on_top` — overlay all detections on the input image.
    * :meth:`draw` — overlay all detections on a black canvas the same size as
      the input.
    """

    def __init__(
        self,
        results: dict[str, PredictionResult],
        *,
        image_size: tuple[int, int] | None,
        input_image: Any | None,
    ) -> None:
        self._results = results
        self._image_size = image_size
        self._input_image = input_image

    # ------------------------------------------------------------------
    # Dict-like access
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> PredictionResult:
        """Access predictions by model display name: ``pred["YOLO26n"]``."""
        return self._results[key]

    def __iter__(self) -> Iterator[tuple[str, PredictionResult]]:
        """Iterate over ``(model_name, PredictionResult)`` pairs."""
        return iter(self._results.items())

    def __len__(self) -> int:
        return len(self._results)

    def __bool__(self) -> bool:
        return any(bool(r) for r in self._results.values())

    def __repr__(self) -> str:
        summary: dict[str, Any] = {}
        for name, r in self._results.items():
            if isinstance(r, DetectionResult):
                summary[name] = len(r.detections)
            else:
                summary[name] = type(r).__name__
        return f"CascadePredictionResult({summary})"

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Model display names in cascade order."""
        return list(self._results.keys())

    def total_detections(self) -> int:
        """Total number of detections across detection-shaped models only."""
        return sum(
            len(r.detections)
            for r in self._results.values()
            if isinstance(r, DetectionResult)
        )

    def describe(self) -> str:
        """Human-readable summary of all predictions (useful in notebooks)."""
        lines: list[str] = []
        for name, pred in self._results.items():
            if isinstance(pred, DetectionResult):
                lines.append(f"[{name}] {len(pred.detections)} detection(s)")
                if pred.detections:
                    lines.append(
                        pred.describe_detections_text(indent="    ")
                    )
            else:
                lines.append(f"[{name}] {type(pred).__name__}")
                if pred:
                    lines.append(f"    {pred.describe()}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _render_onto(self, canvas: Any) -> Any:
        """Draw all predictions onto *canvas* (PIL Image, RGB).

        Rendering strategy (tried in order per model):

        1. **Ultralytics** — when ``pred.raw[0]`` is an Ultralytics ``Results``
           object, call ``raw[0].plot(pil=True, img=current_bgr)`` so that
           segmentation masks, pose skeletons, and OBB all render correctly.
           Results are chained: each model paints on top of the previous one's
           output (matching the notebook ``plot_prediction_cascade`` helper).

        2. **PIL bounding boxes** — fallback for models without raw Ultralytics
           results (ONNX, TFLite, custom runtimes).  Each model gets a distinct
           color from :data:`_PALETTE`.

        Requires ``opencv-python[-headless]`` and ``numpy`` for path 1; falls
        back to path 2 gracefully if they are absent.

        Returns a PIL Image (RGB).
        """
        # ------------------------------------------------------------------
        # Path 1 — Ultralytics .plot() chaining (handles masks / poses / OBB)
        # ------------------------------------------------------------------
        _cv2_available = False
        try:
            import cv2  # type: ignore[import-untyped]
            import numpy as np  # type: ignore[import-untyped]
            from PIL import Image as PILImage

            _cv2_available = True
        except ImportError:
            pass

        if _cv2_available:
            current_bgr = cv2.cvtColor(
                np.asarray(canvas.convert("RGB")), cv2.COLOR_RGB2BGR
            ).copy()

            for idx, (name, pred) in enumerate(self._results.items()):
                if not isinstance(pred, DetectionResult):
                    continue

                if pred.raw is not None and len(pred.raw) > 0:
                    try:
                        plotted = pred.raw[0].plot(pil=True, img=current_bgr)
                        # plot() returns a PIL Image; convert to BGR for the next iteration
                        rgb = np.asarray(plotted.convert("RGB"))
                        current_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        continue
                    except Exception:
                        logger.debug(
                            "[cascade] Ultralytics plot failed for %r, falling back to bbox",
                            name,
                            exc_info=True,
                        )

                # Fallback for this model: draw bounding boxes with cv2
                color_hex = _PALETTE[idx % len(_PALETTE)]
                bgr_color = (
                    int(color_hex[5:7], 16),  # B
                    int(color_hex[3:5], 16),  # G
                    int(color_hex[1:3], 16),  # R
                )
                for detection in pred.detections:
                    b = detection.bbox
                    x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
                    cv2.rectangle(current_bgr, (x1, y1), (x2, y2), bgr_color, 2)
                    label = f"{name[:12]}: {detection.label} {detection.confidence:.0%}"
                    cv2.putText(
                        current_bgr,
                        label,
                        (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        bgr_color,
                        1,
                        cv2.LINE_AA,
                    )

            final_rgb = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2RGB)
            return PILImage.fromarray(final_rgb)

        # ------------------------------------------------------------------
        # Path 2 — pure PIL bounding boxes (no cv2/numpy)
        # ------------------------------------------------------------------
        try:
            from PIL import ImageDraw
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for drawing. Install it: pip install pillow"
            ) from exc

        draw = ImageDraw.Draw(canvas)
        for idx, (name, pred) in enumerate(self._results.items()):
            if not isinstance(pred, DetectionResult):
                continue
            color = _PALETTE[idx % len(_PALETTE)]
            short_name = name[:14]
            for detection in pred.detections:
                b = detection.bbox
                draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=color, width=2)
                label = f"{short_name} · {detection.label} {detection.confidence:.0%}"
                try:
                    text_y = b.y1 - 14
                    lb = draw.textbbox((b.x1, text_y), label)
                    draw.rectangle(lb, fill=color)
                    draw.text((b.x1, text_y), label, fill="white")
                except AttributeError:
                    draw.text((b.x1, b.y1), label, fill=color)
        return canvas

    def draw_on_top(self, image: Any | None = None) -> Any:
        """Draw all cascade predictions on top of the input image.

        Args:
            image: PIL Image or numpy array to draw on.  When ``None``, the
                stored input frame is used — requires the parent
                :class:`CascadeModel` to be created with ``store_input=True``.

        Returns:
            New PIL ``Image`` with bounding boxes and labels drawn for every
            model in the cascade, each in a distinct color.

        Raises:
            ValueError: when no image is available (none passed and the parent
                :class:`CascadeModel` was not created with ``store_input=True``).
        """
        base = image if image is not None else self._input_image
        if base is None:
            raise ValueError(
                "No input image available. Either pass image= explicitly or "
                "create the CascadeModel with store_input=True."
            )
        return self._render_onto(_to_pil(base).copy())

    def draw(self) -> Any:
        """Draw all cascade predictions on a black canvas (same size as input).

        The canvas dimensions are captured at :meth:`~CascadeModel.predict` time
        so this method never needs the original pixels — useful when you want a
        clean annotation layer.

        Returns:
            New black PIL ``Image`` with bounding boxes and labels for every
            model in the cascade.

        Raises:
            ValueError: when image size is unknown.
        """
        if self._image_size is None:
            raise ValueError(
                "Image size is unknown. This should not happen for results "
                "produced by CascadeModel.predict()."
            )
        try:
            from PIL import Image as PILImage
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for drawing. Install it: pip install pillow"
            ) from exc

        canvas = PILImage.new("RGB", self._image_size, (0, 0, 0))
        return self._render_onto(canvas)


# ---------------------------------------------------------------------------
# CascadeModel
# ---------------------------------------------------------------------------


class CascadeModel:
    """Run N models on the same input frame and collect predictions by model name.

    Created automatically by :meth:`~cyberwave.models.ModelManager.load` when
    a **list** of model entries is passed::

        edge_models = cw.models.list(filters=["edge", "image"])
        cascade = cw.models.load(
            [edge_models[0], edge_models[1], edge_models[2]],
            store_input=True,
        )

        pred = cascade.predict(image)

        # Access individual results by catalog display name
        print(pred[edge_models[0].name])

        # Compose all predictions onto the original frame
        output_image = pred.draw_on_top()   # uses stored input (store_input=True)
        output_image = pred.draw_on_top(other_frame)  # draw on an explicit image

        # Draw on a black canvas the same size as the input
        output_image = pred.draw()

    Each model in the cascade receives the **same** original input independently;
    outputs do not chain (model N+1 does not see model N's prediction).

    Args:
        models: Loaded models in cascade order (:class:`~cyberwave.models.LoadedModel`
            or :class:`~cyberwave.models.CloudLoadedModel`).
        names: Display names parallel to *models* (derived from the catalog
            ``name`` field when available, otherwise the string load ID).
        store_input: When ``True``, each :meth:`predict` call stores the input
            frame inside the returned :class:`CascadePredictionResult` so that
            :meth:`~CascadePredictionResult.draw_on_top` can be called without
            an explicit image argument.  Defaults to ``False`` to avoid
            accidental frame retention in memory-sensitive loops.
    """

    def __init__(
        self,
        models: list[LoadedModel | CloudLoadedModel],
        names: list[str],
        *,
        store_input: bool = False,
    ) -> None:
        if len(models) != len(names):
            raise ValueError(
                f"models and names must have the same length "
                f"(got {len(models)} models and {len(names)} names)"
            )
        self._models = models
        self._names = names
        self._store_input = store_input

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable label for the whole cascade."""
        return "cascade(" + ", ".join(self._names) + ")"

    @property
    def store_input(self) -> bool:
        """Whether the last input frame is stored in every result."""
        return self._store_input

    @store_input.setter
    def store_input(self, value: bool) -> None:
        self._store_input = value

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        input_data: Any,
        *,
        confidence: float = 0.5,
        classes: list[str] | None = None,
        **kwargs: Any,
    ) -> CascadePredictionResult:
        """Run all models on *input_data* independently and collect results.

        Each model receives the original *input_data* — predictions do **not**
        chain (the output of model *N* is not fed into model *N+1*).

        Args:
            input_data: Frame accepted by the underlying models (PIL Image,
                numpy array, file path, …).
            confidence: Confidence threshold forwarded to every model.
            classes: Optional class-name filter forwarded to every model.

        Returns:
            :class:`CascadePredictionResult` keyed by each model's display name.
        """
        results: dict[str, PredictionResult] = {}
        for name, model in zip(self._names, self._models):
            pred = model.predict(
                input_data,
                confidence=confidence,
                classes=classes,
                **kwargs,
            )
            results[name] = pred
            logger.debug("[cascade] %s: %d detection(s)", name, len(pred))

        return CascadePredictionResult(
            results=results,
            image_size=_extract_image_size(input_data),
            input_image=input_data if self._store_input else None,
        )

    def __repr__(self) -> str:
        return (
            f"CascadeModel(names={self._names!r}, store_input={self._store_input!r})"
        )
