"""Cloud Playground client for the Cyberwave ``/api/v1/mlmodels`` API.

Exposed as ``cw.models.playground`` on the :class:`~cyberwave.client.Cyberwave`
client — a callable that returns a :class:`PlaygroundHandle` bound to a specific
model.

Example::

    import cyberwave as cw

    client = cw.Cyberwave(api_key="...")
    handle = client.models.playground("acme/models/gemini-robotics-er")

    result = handle.run(
        image="scene.jpg",
        prompt="cups",
        structured_task="detect_points",
    )
    handle.save_annotated_image("scene.jpg", "out.png")

The handle resolves the slug / UUID lazily on the first :meth:`PlaygroundHandle.run`
call so that constructing a handle carries zero network cost.

Types used
----------
All request/response types come from the generated ``cyberwave.rest`` layer:

* :class:`~cyberwave.rest.MLModelSchema` — catalog entry
* :class:`~cyberwave.rest.MLModelRunSchema` — run payload
* :class:`~cyberwave.rest.MLModelRunResultSchema` — synchronous result (HTTP 200)
* :class:`~cyberwave.rest.MLModelRunQueuedSchema` — async result (HTTP 202)

Structured actions
------------------
The local catalog of playground structured-task ids is defined here as
:data:`STRUCTURED_ACTIONS` and :class:`StructuredAction`. The authoritative list
lives on the backend; use :meth:`PlaygroundHandle.actions` to access the local
mirror without an API call.
"""

from __future__ import annotations

import logging
import uuid as _uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cyberwave.image import ImageSource
    from cyberwave.rest import (
        MLModelRunQueuedSchema,
        MLModelRunResultSchema,
        MLModelSchema,
    )
    from cyberwave.rest.api.default_api import DefaultApi

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured-action catalog (local mirror)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuredAction:
    """Slim descriptor of a playground structured task.

    The authoritative list and prompt templates live on the backend.
    This local mirror exists for autocomplete and docstring help only.
    """

    id: str
    label: str
    description: str
    output_format: str
    requires_image: bool = True


STRUCTURED_ACTIONS: tuple[StructuredAction, ...] = (
    StructuredAction(
        id="free",
        label="Free prompt",
        description="Pass the user prompt through unchanged.",
        output_format="text",
        requires_image=False,
    ),
    StructuredAction(
        id="caption",
        label="Caption / describe",
        description="Describe the input image in one concise sentence.",
        output_format="text",
    ),
    StructuredAction(
        id="detect_points",
        label="Detect points",
        description="Per-object 2D points ([y, x] normalized to 0-1000).",
        output_format="points",
    ),
    StructuredAction(
        id="detect_boxes",
        label="Detect bounding boxes",
        description="Per-object axis-aligned boxes ([ymin, xmin, ymax, xmax] normalized to 0-1000).",
        output_format="boxes",
    ),
    StructuredAction(
        id="segment",
        label="Segment",
        description="Per-object segmentation masks (box + base64 PNG cut-out).",
        output_format="masks",
    ),
    StructuredAction(
        id="point_trajectory",
        label="Predict image-plane trajectory",
        description="Ordered 2D waypoints matching the detect_points shape.",
        output_format="trajectory",
    ),
    StructuredAction(
        id="plan_steps",
        label="Plan steps",
        description="Decompose a goal into an ordered list of sub-tasks.",
        output_format="plan_steps",
        requires_image=False,
    ),
    StructuredAction(
        id="detect_objects_3d",
        label="Detect objects in 3D",
        description="Metric-3D detections (label + pose + size in the camera frame).",
        output_format="detections_3d",
    ),
    StructuredAction(
        id="predict_grasps",
        label="Predict grasps",
        description="Ranked parallel-jaw grasps (pose + width + score).",
        output_format="grasps",
    ),
    StructuredAction(
        id="detect_relations",
        label="Detect spatial relations (scene graph)",
        description="Scene graph of visible objects as subject / predicate / object triples.",
        output_format="relations",
    ),
)

_ACTIONS_BY_ID: dict[str, StructuredAction] = {a.id: a for a in STRUCTURED_ACTIONS}


def _get_action(task_id: str) -> StructuredAction | None:
    return _ACTIONS_BY_ID.get(task_id)


def _list_actions() -> list[StructuredAction]:
    return list(STRUCTURED_ACTIONS)


# ---------------------------------------------------------------------------
# PlaygroundHandle
# ---------------------------------------------------------------------------


class PlaygroundHandle:
    """A handle bound to a single model for cloud playground inference.

    Constructed via :meth:`PlaygroundClient.__call__` — do not instantiate
    directly::

        handle = cw.models.playground("acme/models/gemini-robotics-er")
        result = handle.run(image="scene.jpg", prompt="cups")

    Resolution (slug/UUID → ``MLModelSchema``) is lazy: the first call to
    :meth:`run` or :meth:`resolve` performs the lookup and caches the result
    on the handle. Passing an ``MLModelSchema`` directly skips the round-trip.
    """

    def __init__(
        self,
        model_ref: str | MLModelSchema,
        api: DefaultApi,
    ) -> None:
        self._model_ref = model_ref
        self._api = api
        self._resolved: MLModelSchema | None = (
            model_ref if not isinstance(model_ref, str) else None
        )
        self._last_result: MLModelRunResultSchema | MLModelRunQueuedSchema | None = None

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self) -> MLModelSchema:
        """Resolve and return the :class:`~cyberwave.rest.MLModelSchema` entry.

        Cached after the first call — subsequent calls return the same object
        without a network round-trip.
        """
        if self._resolved is not None:
            return self._resolved
        ref = self._model_ref
        if not isinstance(ref, str) or not ref.strip():
            raise ValueError("model_ref must be a non-empty string or MLModelSchema")
        if _looks_like_uuid(ref):
            self._resolved = self._api.src_app_api_mlmodels_get_mlmodel(ref)
        else:
            self._resolved = self._api.src_app_api_mlmodels_get_mlmodel_by_slug(ref)
        return self._resolved

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def actions(self) -> list[StructuredAction]:
        """Return the local mirror of the playground structured-action catalog.

        No API call — uses the bundled catalog. For the live server-side catalog
        call ``cw.api.src_app_api_mlmodels_list_structured_actions()`` directly.
        """
        return _list_actions()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        prompt: str | None = None,
        image: ImageSource | None = None,
        image_url: str | None = None,
        image_base64: str | None = None,
        audio_base64: str | None = None,
        audio_url: str | None = None,
        structured_task: str | None = None,
        twin_uuid: str | None = None,
        params: dict[str, Any] | None = None,
        frames: list[dict[str, Any]] | None = None,
        depth_base64: str | None = None,
        camera_intrinsics: dict[str, Any] | None = None,
        camera_pose: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> MLModelRunResultSchema | MLModelRunQueuedSchema:
        """Execute a playground run against the bound model.

        Args:
            prompt: User-facing prompt.
            image: File path, bytes, file-like, or PIL image — encoded to
                base64 automatically.
            image_url: URL the backend can fetch (alternative to ``image``).
            image_base64: Already-encoded base64 payload (rarely needed).
            audio_base64: Base64-encoded audio bytes (raw PCM or WAV).
            audio_url: URL the backend can fetch (alternative to ``audio_base64``).
            structured_task: Hint for provider-specific prompt building.
                See :data:`STRUCTURED_ACTIONS` for valid ids.
            twin_uuid: Reserved for VLA playground runs.
            params: Extra provider-specific parameters forwarded verbatim.
            frames: Ordered list of extra image frames for temporal /
                multi-camera perception.
            depth_base64: Optional 16-bit PNG depth map aligned to the
                primary frame.
            camera_intrinsics: Pinhole intrinsics ``{fx, fy, cx, cy}``.
            camera_pose: Camera pose in world / twin frame.
            history: Ordered conversation / perception history.

        Returns:
            :class:`~cyberwave.rest.MLModelRunResultSchema` (HTTP 200) or
            :class:`~cyberwave.rest.MLModelRunQueuedSchema` (HTTP 202).

        Raises:
            ValueError: for invalid or missing inputs.
            :class:`~cyberwave.exceptions.CyberwaveAPIError`: for HTTP errors.
        """
        if sum(bool(x) for x in (image, image_url, image_base64)) > 1:
            raise ValueError("Pass at most one of image, image_url, image_base64.")

        if image is not None:
            from cyberwave.image import encode_image_base64
            image_base64 = encode_image_base64(image)

        has_primary = any(
            x is not None
            for x in (prompt, image_base64, image_url, audio_base64, audio_url, frames)
        )
        if not has_primary:
            raise ValueError(
                "Nothing to run: provide at least one of prompt, image, "
                "image_url, image_base64, audio_base64, audio_url, or frames."
            )

        entry = self.resolve()

        if (
            structured_task is not None
            and entry.allowed_structured_tasks
            and structured_task not in entry.allowed_structured_tasks
        ):
            raise ValueError(
                f"Model {entry.name!r} does not advertise structured_task="
                f"{structured_task!r}. Allowed: {entry.allowed_structured_tasks}"
            )

        from cyberwave.rest.models.ml_model_run_schema import MLModelRunSchema

        schema = MLModelRunSchema(
            prompt=prompt,
            image_base64=image_base64,
            image_url=image_url,
            audio_base64=audio_base64,
            audio_url=audio_url,
            structured_task=structured_task,
            twin_uuid=twin_uuid,
            params=params,
            depth_base64=depth_base64,
        )

        # frames / camera_intrinsics / camera_pose / history are typed
        # sub-schemas; pass through only when provided to avoid validation
        # errors on models that don't accept them.
        if frames is not None:
            schema.frames = frames  # type: ignore[assignment]
        if camera_intrinsics is not None:
            schema.camera_intrinsics = camera_intrinsics  # type: ignore[assignment]
        if camera_pose is not None:
            schema.camera_pose = camera_pose  # type: ignore[assignment]
        if history is not None:
            schema.history = history  # type: ignore[assignment]

        response = self._api.src_app_api_mlmodels_run_mlmodel_with_http_info(
            uuid=entry.uuid,
            ml_model_run_schema=schema,
        )
        self._last_result = response.data
        return self._last_result

    # ------------------------------------------------------------------
    # Convenience exporters (stateful — use last result)
    # ------------------------------------------------------------------

    def save_annotated_image(
        self,
        source: Any,
        path: str,
        *,
        render: bool = True,
        embed_metadata: bool = True,
    ) -> str:
        """Save ``source`` to ``path`` with overlays from the last :meth:`run` result.

        Args:
            source: The image the model was run against (path / bytes / PIL).
            path: Destination PNG path.
            render: When ``True`` draw overlays (requires spatial output format).
                Set to ``False`` to just embed metadata without drawing.
            embed_metadata: Store result JSON in the PNG ``tEXt`` chunk.

        Returns:
            The resolved path string.

        Raises:
            RuntimeError: when called before :meth:`run` or on a queued result.
        """
        from cyberwave.rest.models.ml_model_run_queued_schema import MLModelRunQueuedSchema

        if self._last_result is None:
            raise RuntimeError("Call run() before save_annotated_image().")
        if isinstance(self._last_result, MLModelRunQueuedSchema):
            raise RuntimeError(
                "Cannot annotate image for a queued result. "
                f"Poll {self._last_result.poll_url} until completion first."
            )

        result = self._last_result
        output_format: str | None = getattr(result, "output_format", None)

        if render and output_format not in {"points", "boxes", "masks"}:
            raise RuntimeError(
                f"output_format={output_format!r} is not a spatial output. "
                "Pass render=False to archive the result without drawing."
            )

        from cyberwave.image import save_annotated_image

        payload: dict[str, Any] = {
            "output_format": output_format,
            "output": getattr(result, "output", None),
            "raw": getattr(result, "raw", None),
            "status": getattr(result, "status", "completed"),
            "workload_uuid": None,
            "model_uuid": self._resolved.uuid if self._resolved else None,
        }
        extra: dict[str, Any] = {}
        if self._resolved and self._resolved.slug:
            extra["model_slug"] = self._resolved.slug

        result_path = save_annotated_image(
            source,
            payload,
            path,
            render=render,
            embed_metadata=embed_metadata,
            metadata_extra=extra,
        )
        return str(result_path)

    def __repr__(self) -> str:
        ref = (
            self._resolved.slug or self._resolved.uuid
            if self._resolved
            else repr(self._model_ref)
        )
        return f"PlaygroundHandle({ref})"


# ---------------------------------------------------------------------------
# PlaygroundClient  (assigned to ModelManager.playground)
# ---------------------------------------------------------------------------


class PlaygroundClient:
    """Factory assigned to :attr:`~cyberwave.models.ModelManager.playground`.

    Calling it returns a :class:`PlaygroundHandle` bound to the given model::

        handle = cw.models.playground("acme/models/gemini-robotics-er")

    Also satisfies the interface expected by
    :class:`~cyberwave.models.cloud.CloudLoadedModel` (``get``, ``run``,
    ``fetch_weights_url``) so ``ModelManager._mlmodels_client`` can point here
    without changes to ``cloud.py`` runtime logic.
    """

    def __init__(self, api: DefaultApi) -> None:
        self._api = api

    def __call__(
        self, model_ref: str | MLModelSchema
    ) -> PlaygroundHandle:
        """Return a :class:`PlaygroundHandle` bound to *model_ref*.

        No API call is made here — resolution is deferred to the first
        :meth:`PlaygroundHandle.run` or :meth:`PlaygroundHandle.resolve` call.
        """
        return PlaygroundHandle(model_ref=model_ref, api=self._api)

    # ------------------------------------------------------------------
    # CloudLoadedModel compatibility interface
    # ------------------------------------------------------------------

    def get(self, model_ref: str) -> MLModelSchema:
        """Resolve a catalog entry by slug or UUID.

        Used internally by :class:`~cyberwave.models.cloud.CloudLoadedModel`
        and :class:`~cyberwave.models.manager.ModelManager`.
        """
        return self(model_ref).resolve()

    def run(
        self,
        model: str | MLModelSchema,
        **kwargs: Any,
    ) -> MLModelRunResultSchema | MLModelRunQueuedSchema:
        """Run ``model`` via a transient :class:`PlaygroundHandle`.

        Thin convenience for ``CloudLoadedModel`` which calls
        ``self._client.run(self._summary, image=..., ...)``.
        """
        return self(model).run(**kwargs)

    def fetch_weights_url(self, model: str | MLModelSchema) -> dict[str, Any]:
        """Return the signed-URL payload for downloading model checkpoint weights.

        Raises:
            :class:`~cyberwave.exceptions.CyberwaveAPIError`: ``404`` when the
                backend does not host a checkpoint for this model.
        """
        if isinstance(model, str):
            entry = self.get(model)
        else:
            entry = model
        result = self._api.src_app_api_mlmodels_get_mlmodel_weights(entry.uuid)
        if isinstance(result, dict):
            return result
        # generated client may return the Pydantic model; normalise to dict
        return dict(result) if result is not None else {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _looks_like_uuid(value: str) -> bool:
    try:
        _uuid.UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


__all__ = [
    "PlaygroundClient",
    "PlaygroundHandle",
    "STRUCTURED_ACTIONS",
    "StructuredAction",
]
