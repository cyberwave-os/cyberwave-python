"""Cloud-hosted model wrapper: same ``.predict()`` API as :class:`LoadedModel`.

This is the small piece that makes ``cw.models.load("acme/models/sam-3.1")``
work identically to ``cw.models.load("yolov8n")``. Both return something
with ``.predict(image, **kwargs) -> PredictionResult`` — the edge case
uses a local runtime, the cloud case funnels through the Playground
``/mlmodels/{uuid}/run`` endpoint.

Why this lives under ``cyberwave.models`` instead of a separate module
=======================================================================

``cw.models.playground`` is the full-fidelity cloud surface for direct
inference — it exposes every playground knob (``structured_task``, ``frames``,
``depth_base64``, ``camera_intrinsics``, async polling, annotated-image export).

``CloudLoadedModel`` is the thin adapter that lets cloud models show up
in the same code snippet as edge models::

    import cyberwave as cw
    model = cw.models.load("acme/models/gemini-robotics-er")
    result = model.predict("scene.jpg", prompt="cups", structured_task="detect_points")

It's a ~100-LoC class — not a second client. All HTTP and schema work
happens inside ``PlaygroundClient``; this file only does the translation
between the edge ``.predict(x, **kw)`` shape and the cloud
``playground(...).run(**kw)`` shape.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

from cyberwave.models.types import (
    BoundingBox,
    CustomResult,
    Detection,
    DetectionResult,
    ImageResult,
    JsonResult,
    PredictionResult,
    QueuedPredictionResult,
    TextResult,
)

if TYPE_CHECKING:
    from cyberwave.models.playground import PlaygroundClient
    from cyberwave.rest import MLModelRunResultSchema, MLModelRunQueuedSchema, MLModelSchema

logger = logging.getLogger(__name__)


class CloudLoadedModel:
    """A cloud-hosted model loaded via :meth:`ModelManager.load`.

    Mirrors the public surface of :class:`cyberwave.models.LoadedModel`
    (``name``, ``runtime``, ``device``, ``predict``) so calling code can
    stay agnostic of whether the model lives on an edge node or the
    Playground. Exposes the underlying run result on
    ``.last_result`` for callers that need async polling or raw JSON.
    """

    def __init__(self, *, summary: MLModelSchema, client: PlaygroundClient) -> None:
        self._summary = summary
        self._client = client
        self._last_result: MLModelRunResultSchema | MLModelRunQueuedSchema | None = None

    # ------------------------------------------------------------------
    # Public surface mirroring LoadedModel
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._summary.name or self._summary.slug or self._summary.uuid

    @property
    def runtime(self) -> str:
        # Matches the ``runtime`` property shape of LoadedModel for code
        # that does e.g. ``if model.runtime == "ultralytics": ...``. We
        # advertise the hosting provider so consumers can still route.
        return f"cloud:{self._summary.model_provider_name or 'custom'}"

    @property
    def device(self) -> str:
        # Cloud inference runs on managed hardware; the convention used
        # throughout the rest of the SDK for "not on this machine".
        return "cloud"

    @property
    def summary(self) -> MLModelSchema:
        """The :class:`~cyberwave.rest.MLModelSchema` returned by ``/mlmodels/by-slug``."""
        return self._summary

    @property
    def last_result(self) -> MLModelRunResultSchema | MLModelRunQueuedSchema | None:
        """Full run result from the most recent ``predict()`` call.

        Useful for async cloud-node workloads, raw provider text, or calling
        :meth:`PlaygroundHandle.save_annotated_image`.
        """
        return self._last_result

    def predict(
        self,
        input_data: Any | None = None,
        *,
        prompt: str | None = None,
        structured_task: str | None = None,
        confidence: float | None = None,  # noqa: ARG002 - accepted for API parity
        classes: list[str] | None = None,  # noqa: ARG002 - accepted for API parity
        twin_uuid: str | None = None,
        **kwargs: Any,
    ) -> PredictionResult:
        """Run cloud inference with the same signature as local ``predict``.

        Extra kwargs (``frames``, ``depth_base64``, ``camera_intrinsics``,
        ``camera_pose``, ``history``, ``params``) are forwarded verbatim
        to :meth:`PlaygroundClient.run` so power-user features stay
        reachable without dropping to ``cw.models.playground(...).run()``.

        ``confidence`` and ``classes`` are accepted for signature parity
        with :class:`LoadedModel`. Cloud providers do not share a universal
        confidence/classes filter contract though, so these kwargs are not
        forwarded. When supplied we emit a warning instead of silently acting
        like the edge runtime.
        """
        if confidence is not None:
            warnings.warn(
                "CloudLoadedModel.predict() ignores confidence=... for now. "
                "Use prompt / structured_task / params for provider-specific "
                "filtering instead.",
                RuntimeWarning,
                stacklevel=2,
            )
        if classes:
            warnings.warn(
                "CloudLoadedModel.predict() ignores classes=... for now. "
                "Use prompt / structured_task / params for provider-specific "
                "filtering instead.",
                RuntimeWarning,
                stacklevel=2,
            )
        result = self._client.run(
            self._summary,
            image=input_data,
            prompt=prompt,
            structured_task=structured_task,
            twin_uuid=twin_uuid,
            **kwargs,
        )
        self._last_result = result
        return _to_prediction_result(result, model_slug=self._summary.slug)

    def __repr__(self) -> str:
        return (
            f"CloudLoadedModel(slug={self._summary.slug!r}, "
            f"uuid={self._summary.uuid!r}, "
            f"runtime={self.runtime!r})"
        )


# ---------------------------------------------------------------------------
# Cloud → edge result translation
# ---------------------------------------------------------------------------


def _to_prediction_result(
    result: MLModelRunResultSchema | MLModelRunQueuedSchema,
    *,
    model_slug: str | None = None,
) -> PredictionResult:
    """Convert a cloud run result into the edge ``PredictionResult``.

    Every :class:`PredictionResult` flowing out of this adapter has the
    full provider payload reachable on ``.metadata["mlmodel_run_result"]``
    so *nothing* is lost — callers that need ``output_format``, ``raw``,
    or the async ``workload_uuid`` find them there.

    Output mapping by ``output_format``:

    * ``text`` → :class:`TextResult`
    * ``json`` → :class:`JsonResult`
    * ``image`` → :class:`ImageResult` (``data_url`` / ``image_url`` / base64)
    * ``boxes`` / ``points`` / ``masks`` → :class:`DetectionResult` (spatial)
    * ``trajectory``, ``plan_steps``, ``detections_3d``, ``grasps``,
      ``relations``, ``raw`` → :class:`CustomResult` preserving structure
    """
    from cyberwave.rest.models.ml_model_run_queued_schema import MLModelRunQueuedSchema as _Queued

    metadata: dict[str, Any] = {
        "mlmodel_run_result": result,
        "model_slug": model_slug,
    }
    is_queued = isinstance(result, _Queued)

    if is_queued:
        metadata["output_format"] = None
        metadata["workload_uuid"] = result.workload_uuid  # type: ignore[union-attr]
        metadata["poll_url"] = result.poll_url  # type: ignore[union-attr]
        return QueuedPredictionResult(raw=None, metadata=metadata)

    output = result.output  # type: ignore[union-attr]
    output_format: str | None = result.output_format  # type: ignore[union-attr]
    provider_raw: str | None = getattr(result, "raw", None)
    metadata["output_format"] = output_format

    return _model_output_for_cloud(
        output_format=output_format,
        output=output,
        provider_raw=provider_raw,
        raw=output,
        metadata=metadata,
    )


def _model_output_for_cloud(
    *,
    output_format: str | None,
    output: Any,
    provider_raw: str | None,
    raw: Any,
    metadata: dict[str, Any],
) -> PredictionResult:
    fmt = (output_format or "").lower()

    if fmt == "text":
        text = output if isinstance(output, str) else (provider_raw or "")
        return TextResult(text=text, raw=raw, metadata=metadata)

    if fmt == "json":
        if isinstance(output, dict | list):
            return JsonResult(data=output, raw=raw, metadata=metadata)
        if isinstance(provider_raw, str) and provider_raw:
            import json

            try:
                return JsonResult(
                    data=json.loads(provider_raw),
                    raw=raw,
                    metadata=metadata,
                )
            except json.JSONDecodeError:
                pass
        return JsonResult(data=output, raw=raw, metadata=metadata)

    if fmt == "image":
        image = ImageResult.from_output(output)
        image.raw = raw
        image.metadata = metadata
        return image

    if isinstance(output, list):
        detections = _detections_from_list(output, fmt)
        if detections:
            return DetectionResult(detections, raw=raw, metadata=metadata)

    if fmt in _STRUCTURED_OBJECT_FORMATS and output is not None:
        return CustomResult(data=output, label=fmt, raw=raw, metadata=metadata)

    if fmt == "raw" and output is not None:
        return CustomResult(data=output, label="raw", raw=raw, metadata=metadata)

    if isinstance(output, str):
        return TextResult(text=output, raw=raw, metadata=metadata)

    if output is not None:
        return CustomResult(data=output, label=fmt or "unknown", raw=raw, metadata=metadata)

    return DetectionResult(detections=[], raw=raw, metadata=metadata)


_STRUCTURED_OBJECT_FORMATS = frozenset(
    {
        "trajectory",
        "plan_steps",
        "detections_3d",
        "grasps",
        "relations",
    }
)


def _detections_from_list(output: list[Any], fmt: str) -> list[Detection]:
    if fmt == "boxes":
        detections = [_detection_from_box(item) for item in output if _is_dict(item)]
    elif fmt == "points":
        detections = [_detection_from_point(item) for item in output if _is_dict(item)]
    elif fmt == "masks":
        detections = [_detection_from_mask(item) for item in output if _is_dict(item)]
    else:
        return []
    return [d for d in detections if d is not None]  # type: ignore[misc]


def _is_dict(item: Any) -> bool:
    return isinstance(item, dict)


def _detection_from_box(item: dict[str, Any]) -> Detection | None:
    box = item.get("box_2d")
    if not isinstance(box, list | tuple) or len(box) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    # Clamp inverted coords defensively — providers occasionally swap
    # top/bottom in their own coordinate convention.
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    try:
        bbox = BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax)
    except ValueError:
        return None
    label = str(item.get("label") or item.get("name") or "")
    confidence = _coerce_confidence(item.get("score", item.get("confidence", 1.0)))
    return Detection(
        label=label,
        confidence=confidence,
        bbox=bbox,
        metadata={k: v for k, v in item.items() if k not in {"box_2d", "label", "score"}},
    )


def _detection_from_point(item: dict[str, Any]) -> Detection | None:
    point = item.get("point")
    if not isinstance(point, list | tuple) or len(point) != 2:
        return None
    try:
        y, x = float(point[0]), float(point[1])
    except (TypeError, ValueError):
        return None
    # Zero-area bbox at the point so edge overlay code still renders a
    # marker at ``(x, y)``; consumers that care about the exact point
    # pull it back out of ``mask`` (which carries the raw ``[y, x]``).
    bbox = BoundingBox(x1=x, y1=y, x2=x, y2=y)
    label = str(item.get("label") or "")
    confidence = _coerce_confidence(item.get("score", 1.0))
    return Detection(
        label=label,
        confidence=confidence,
        bbox=bbox,
        mask=point,
        metadata={k: v for k, v in item.items() if k not in {"point", "label", "score"}},
    )


def _detection_from_mask(item: dict[str, Any]) -> Detection | None:
    det = _detection_from_box(item)
    if det is None:
        return None
    mask = item.get("mask")
    if mask is not None:
        det.mask = mask
    return det


def _coerce_confidence(value: Any) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 1.0
    # ``Detection.__post_init__`` enforces [0, 1]; provider scores
    # occasionally exceed that (e.g. OpenVLA logits). Clip silently
    # rather than refusing to translate the output.
    if conf < 0.0:
        return 0.0
    if conf > 1.0:
        return 1.0
    return conf


__all__ = ["CloudLoadedModel"]
