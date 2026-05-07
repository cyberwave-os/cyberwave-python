"""High-level Python client for the Cyberwave ML Model Playground API.

Exposed as ``cw.mlmodels`` on the :class:`cyberwave.Cyberwave` client. See
the package docstring for usage examples.

Implementation notes
--------------------
The client routes all HTTP traffic through the auto-generated ``ApiClient``
used by the rest of the SDK, so authentication, base-URL, proxies, and TLS
verification are configured once on ``Cyberwave(api_key=...)``.

Endpoints used:

* ``GET  /api/v1/mlmodels/by-slug?slug=...``
* ``GET  /api/v1/mlmodels/{uuid}``
* ``POST /api/v1/mlmodels/{uuid}/run``

The ``/run`` endpoint returns ``200`` for synchronous providers and ``202``
with a ``workload_uuid`` for models that defer to a cloud-node worker (e.g.
Hunyuan3D im2mesh). Both paths resolve to an :class:`MLModelRunResult`
and :meth:`MLModelRunResult.is_queued` flags the async case.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from cyberwave.exceptions import CyberwaveAPIError
from cyberwave.image import encode_image_base64
from cyberwave.mlmodels.actions import (
    STRUCTURED_ACTIONS,
    StructuredAction,
    get_action,
    list_actions,
)
from cyberwave.mlmodels.types import MLModelRunResult, MLModelSummary

if TYPE_CHECKING:
    from cyberwave.image import ImageSource

logger = logging.getLogger(__name__)


class MLModelsClient:
    """Cloud playground client for ``/api/v1/mlmodels``.

    Usage::

        import cyberwave as cw

        client = cw.Cyberwave(api_key="...")
        model = client.mlmodels.get("acme/models/gemini-robotics-er")

        result = client.mlmodels.run(
            model,
            image="scene.jpg",
            prompt="cups",
            structured_task="detect_points",
        )
        result.save_annotated_image("scene.jpg", "scene.annotated.png")

    The first argument of :meth:`run` can be a slug, UUID, or a
    :class:`MLModelSummary` returned by :meth:`get`.
    """

    def __init__(self, api_client: Any) -> None:
        self._api_client = api_client

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @staticmethod
    def list_structured_actions() -> list[StructuredAction]:
        """Return the local mirror of the playground structured-action catalog.

        Matches the JSON payload served at
        ``GET /api/v1/mlmodels/structured-actions``.
        """
        return list_actions()

    @staticmethod
    def get_structured_action(task_id: str) -> StructuredAction | None:
        """Return the :class:`StructuredAction` with ``id == task_id``."""
        return get_action(task_id)

    def fetch_structured_actions_catalog(self) -> dict[str, Any]:
        """Fetch the live catalog from the backend.

        Use when you want to pick up actions added server-side since your
        SDK version was released. Falls back to the local mirror when the
        network call fails.
        """
        try:
            return self._call_json("GET", "/api/v1/mlmodels/structured-actions")
        except CyberwaveAPIError as e:
            logger.info(
                "Falling back to local structured-actions catalog (fetch failed: %s)",
                e,
            )
            return {
                "version": 1,
                "actions": [
                    {
                        "id": a.id,
                        "label": a.label,
                        "description": a.description,
                        "output_format": a.output_format,
                        "requires_image": a.requires_image,
                    }
                    for a in STRUCTURED_ACTIONS
                ],
            }

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def get(self, model_ref: str) -> MLModelSummary:
        """Resolve a catalog entry by slug (``ws/models/name``) or UUID."""
        if not isinstance(model_ref, str) or not model_ref.strip():
            raise ValueError("model_ref must be a non-empty string")

        if _looks_like_uuid(model_ref):
            path = f"/api/v1/mlmodels/{model_ref}"
            data = self._call_json("GET", path)
        else:
            data = self._call_json(
                "GET",
                "/api/v1/mlmodels/by-slug",
                query_params=[("slug", model_ref)],
            )
        return MLModelSummary.from_api(data)

    # ------------------------------------------------------------------
    # Weights / artifacts
    # ------------------------------------------------------------------

    def fetch_weights_url(
        self, model: str | MLModelSummary
    ) -> dict[str, Any]:
        """Return a signed-URL payload for downloading this model's checkpoint.

        Thin wrapper around ``GET /api/v1/mlmodels/{uuid}/weights``. Returns
        the backend response verbatim — typically::

            {
                "signed_url": "https://storage.googleapis.com/...",
                "expires_at": "2026-04-22T...+00:00",
                "checkpoint_path": "ml_models/<uuid>/checkpoint.tar",
            }

        Only works for checkpoints that the backend actually hosts (private
        fine-tunes uploaded to Cyberwave's bucket). Public / API-gated
        models (Gemini, HF-hosted ones we don't mirror) return ``404``
        here and should be used via :meth:`run` instead.

        Raises:
            CyberwaveAPIError: ``404`` when the backend does not host a
                checkpoint for this model, or any other HTTP error.
            ValueError: when ``model`` is neither a string nor a summary.
        """
        if isinstance(model, MLModelSummary):
            uuid = model.uuid
        elif isinstance(model, str):
            if not model.strip():
                raise ValueError("model must be a non-empty string")
            uuid = model if _looks_like_uuid(model) else self.get(model).uuid
        else:
            raise TypeError(
                f"model must be a string or MLModelSummary, got {type(model)!r}"
            )
        return self._call_json("GET", f"/api/v1/mlmodels/{uuid}/weights")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        model: str | MLModelSummary,
        *,
        prompt: str | None = None,
        image: "ImageSource | None" = None,
        image_url: str | None = None,
        image_base64: str | None = None,
        structured_task: str | None = None,
        twin_uuid: str | None = None,
        params: dict[str, Any] | None = None,
        # Extended perception envelope (all optional, additive — the backend
        # accepts the primary contract unchanged). See the backend
        # ``MLModelRunSchema`` docstring for the field-by-field contract.
        frames: list[dict[str, Any]] | None = None,
        depth_base64: str | None = None,
        camera_intrinsics: dict[str, Any] | None = None,
        camera_pose: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        resolve: bool = True,
    ) -> MLModelRunResult:
        """Execute a playground run against ``model``.

        Args:
            model: Either a :class:`MLModelSummary`, a slug, or a UUID. When
                a string is passed we call :meth:`get` to resolve it unless
                ``resolve=False`` and ``model`` is already a UUID.
            prompt: User-facing prompt. For spatial tasks the backend will
                rewrite this into a provider-specific grounding instruction
                (see :mod:`cyberwave.mlmodels.actions`).
            image: File path, bytes, file-like, or PIL image. Encoded to
                base64 automatically — the convenience that replaces the 4
                lines of ``base64.b64encode(open(...).read())`` boilerplate.
            image_url: Alternative to ``image``: a URL the backend can fetch.
            image_base64: Alternative to ``image``: an already-encoded
                payload. Rarely needed.
            structured_task: One of ``free`` | ``caption`` | ``detect_points``
                | ``detect_boxes`` | ``segment`` | ``point_trajectory`` |
                ``plan_steps`` | ``detect_objects_3d`` | ``predict_grasps`` |
                ``detect_relations``. See
                :data:`cyberwave.mlmodels.STRUCTURED_ACTIONS`.
            twin_uuid: Reserved for VLA playground runs.
            params: Extra provider-specific parameters forwarded verbatim.
            frames: Optional ordered list of extra image frames for
                temporal / multi-camera perception. Each frame is a dict
                with ``image_base64`` / ``image_url`` plus optional
                ``timestamp`` (ms) and ``camera_id`` (``"wrist"`` /
                ``"overhead"``). When ``image`` / ``image_url`` /
                ``image_base64`` is omitted, the first frame is promoted to
                the primary input.
            depth_base64: Optional 16-bit PNG depth map aligned to the
                primary frame, base64-encoded. Pair with
                ``camera_intrinsics`` to unlock metric-3D structured tasks.
            camera_intrinsics: Optional pinhole intrinsics
                ``{fx, fy, cx, cy, width?, height?}``.
            camera_pose: Optional camera pose in world / twin frame:
                ``{position: [x, y, z], quaternion: [w, x, y, z], frame_id?}``.
            history: Optional ordered conversation / perception history for
                agentic loops. Each turn is a dict with ``role`` / ``content``
                and optional ``structured_task`` / ``output`` from prior
                runs.
            resolve: When ``True`` (default) and ``model`` is a string, we
                first call ``GET /mlmodels/by-slug`` or ``GET
                /mlmodels/{uuid}`` so the return value carries the resolved
                slug. Set to ``False`` if you already have the UUID and want
                to save a round-trip.

        Returns:
            :class:`MLModelRunResult`. Call
            :meth:`MLModelRunResult.is_queued` to check for async workloads.

        Raises:
            ValueError: for invalid ``structured_task`` or missing inputs.
            CyberwaveAPIError: for HTTP errors.
        """
        # Client-side validation of the global structured-task catalog is
        # delegated to the backend: the authoritative list lives in
        # ``cyberwave-backend/src/lib/structured_actions.py`` and was
        # previously mirrored here. Per-model validation against
        # ``summary.allowed_structured_tasks`` still runs below to give a
        # fast, targeted error without a round-trip when a summary is
        # available.

        if sum(bool(x) for x in (image, image_url, image_base64)) > 1:
            raise ValueError(
                "Pass at most one of image, image_url, image_base64."
            )

        if image is not None and image_base64 is None:
            image_base64 = encode_image_base64(image)

        summary: MLModelSummary | None = None
        if isinstance(model, MLModelSummary):
            uuid = model.uuid
            summary = model
        elif isinstance(model, str):
            if resolve or not _looks_like_uuid(model):
                summary = self.get(model)
                uuid = summary.uuid
            else:
                uuid = model
        else:
            raise TypeError(
                f"model must be a string or MLModelSummary, got {type(model)!r}"
            )

        if (
            structured_task is not None
            and summary is not None
            and summary.allowed_structured_tasks
            and structured_task not in summary.allowed_structured_tasks
        ):
            raise ValueError(
                f"Model {summary.name!r} does not advertise structured_task="
                f"{structured_task!r}. Allowed tasks: "
                f"{summary.allowed_structured_tasks}"
            )

        body: dict[str, Any] = {}
        if prompt is not None:
            body["prompt"] = prompt
        if image_base64 is not None:
            body["image_base64"] = image_base64
        if image_url is not None:
            body["image_url"] = image_url
        if structured_task is not None:
            body["structured_task"] = structured_task
        if twin_uuid is not None:
            body["twin_uuid"] = twin_uuid
        if params is not None:
            body["params"] = params
        if frames is not None:
            body["frames"] = frames
        if depth_base64 is not None:
            body["depth_base64"] = depth_base64
        if camera_intrinsics is not None:
            body["camera_intrinsics"] = camera_intrinsics
        if camera_pose is not None:
            body["camera_pose"] = camera_pose
        if history is not None:
            body["history"] = history

        # The primary-contract check: at least one of prompt / image /
        # image_url / image_base64 / frames must be present. We intentionally
        # don't count the envelope extensions (depth / intrinsics / history)
        # as standalone inputs — they are always ancillary to one of the
        # primary signals.
        has_primary = any(
            k in body
            for k in ("prompt", "image_base64", "image_url", "frames")
        )
        if not has_primary:
            raise ValueError(
                "Nothing to run: provide at least one of prompt, image, "
                "image_url, image_base64, or frames."
            )

        status_code, data = self._call_json_with_status(
            "POST",
            f"/api/v1/mlmodels/{uuid}/run",
            body=body,
        )

        result = MLModelRunResult.from_api(data, status_code=status_code)
        result.model_uuid = uuid
        result.model_slug = summary.slug if summary else None
        result.structured_task = structured_task
        return result

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _call_json(
        self,
        method: str,
        path: str,
        *,
        query_params: list[tuple[str, str]] | None = None,
        body: Any = None,
    ) -> dict[str, Any]:
        _, data = self._call_json_with_status(
            method, path, query_params=query_params, body=body
        )
        return data

    def _call_json_with_status(
        self,
        method: str,
        path: str,
        *,
        query_params: list[tuple[str, str]] | None = None,
        body: Any = None,
    ) -> tuple[int, dict[str, Any]]:
        try:
            header_params: dict[str, str] = {}
            if body is not None:
                header_params["Content-Type"] = "application/json"
            else:
                body = None

            _param = self._api_client.param_serialize(
                method=method,
                resource_path=path,
                query_params=query_params or [],
                header_params=header_params,
                body=body,
                auth_settings=["CustomTokenAuthentication"],
            )
            response_data = self._api_client.call_api(*_param)
            response_data.read()
            status = int(getattr(response_data, "status", 0) or 0)

            if status >= 400:
                raise CyberwaveAPIError(
                    f"{method} {path} failed: HTTP {status}",
                    status_code=status,
                    response_data=_safe_json(response_data.data),
                )

            return status, _safe_json(response_data.data) or {}
        except CyberwaveAPIError:
            raise
        except Exception as e:
            status = getattr(e, "status", None)
            try:
                status_int = int(status) if status is not None else None
            except (TypeError, ValueError):
                status_int = None
            raise CyberwaveAPIError(
                f"{method} {path} failed: {e}",
                status_code=status_int,
            ) from e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _looks_like_uuid(value: str) -> bool:
    """Cheap UUID heuristic (SDK-internal; accepts any UUID version)."""
    import uuid as _uuid

    try:
        _uuid.UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


def _safe_json(data: Any) -> dict[str, Any] | None:
    """Best-effort JSON decode of a urllib3 response body."""
    if data is None:
        return None
    if isinstance(data, (bytes, bytearray)):
        try:
            return json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None
    if isinstance(data, dict):
        return data
    return None
