"""Model manager — loading, caching, runtime detection, and catalog CRUD.

Exposed as ``cw.models`` on the ``Cyberwave`` client.  Three surfaces:

* **Runtime** — ``cw.models.load(id)`` searches the local model cache
  (populated by Edge Core), detects the appropriate runtime backend, and
  returns a :class:`~cyberwave.models.loaded_model.LoadedModel` with a
  stable ``.predict()`` API.  Cloud slugs are transparently routed to the
  Playground and a :class:`~cyberwave.models.cloud.CloudLoadedModel` is
  returned instead.

* **Catalog** — ``cw.models.list()``, ``.get()``, ``.delete()`` mirror the
  REST ``/api/v1/mlmodels`` endpoints so you can browse available models and
  pick one to pass to :meth:`load`::

      for m in cw.models.list(deployment="edge"):
          print(m.slug, m.model_external_id)
      model = cw.models.load(m)          # pass entry directly — no field inspection needed
      pred  = model.predict(frame)

* **Playground** — ``cw.models.playground("slug")`` returns a
  :class:`~cyberwave.models.playground.PlaygroundHandle` for cloud inference::

      result = cw.models.playground("acme/models/gemini-robotics-er").run(
          image="scene.jpg", prompt="cups", structured_task="detect_points",
      )
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from cyberwave.exceptions import CyberwaveAPIError, CyberwaveModelIntegrityError
from cyberwave.models.cloud import CloudLoadedModel
from cyberwave.models.loaded_model import LoadedModel
from cyberwave.models.runtimes import get_runtime

if TYPE_CHECKING:
    from cyberwave.models.cascade import CascadeModel
    from cyberwave.rest import MLModelSchema

logger = logging.getLogger(__name__)

MODEL_METADATA_FILENAME = "metadata.json"

DEFAULT_MODEL_DIR = "/app/models"
FALLBACK_MODEL_DIR = os.path.expanduser("~/.cyberwave/models")

_DOWNLOAD_CHUNK_SIZE = 1 << 20  # 1 MiB — matches the edge-core streaming buffer.
_DOWNLOAD_TIMEOUT_SECS = 300.0

# ---------------------------------------------------------------------------
# Filter shorthand helpers for list()
# ---------------------------------------------------------------------------

# Maps shorthand string → predicate over MLModelSchema (duck-typed via getattr
# so this file has zero import dependency on the generated REST schema).
_FILTER_PREDICATES: dict[str, Callable[[Any], bool]] = {
    "edge": lambda m: bool(getattr(m, "is_edge_compatible", False)),
    "cloud": lambda m: bool(getattr(m, "is_cloud_compatible", False)),
    "hybrid": lambda m: getattr(m, "deployment", "") == "hybrid",
    "image": lambda m: bool(getattr(m, "can_take_image_as_input", False)),
    "image_input": lambda m: bool(getattr(m, "can_take_image_as_input", False)),
    "video": lambda m: bool(getattr(m, "can_take_video_as_input", False)),
    "video_input": lambda m: bool(getattr(m, "can_take_video_as_input", False)),
    "text": lambda m: bool(getattr(m, "can_take_text_as_input", False)),
    "text_input": lambda m: bool(getattr(m, "can_take_text_as_input", False)),
    "audio": lambda m: bool(getattr(m, "can_take_audio_as_input", False)),
    "audio_input": lambda m: bool(getattr(m, "can_take_audio_as_input", False)),
    "action": lambda m: bool(getattr(m, "can_take_action_as_input", False)),
    "action_input": lambda m: bool(getattr(m, "can_take_action_as_input", False)),
    "trainable": lambda m: bool(getattr(m, "is_trainable", False)),
    "public": lambda m: getattr(m, "visibility", "") == "public",
}


def _apply_filter_shorthands(
    models: list[Any],
    filters: list[str],
) -> list[Any]:
    """Apply shorthand filter strings to a list of MLModelSchema objects.

    Known shorthands (see :data:`_FILTER_PREDICATES`) map directly to schema
    boolean / string fields.  Unknown strings are treated as **tag checks** —
    the entry must have at least one tag that contains the filter string
    (case-insensitive).  Multiple filters are ANDed together.
    """
    predicates: list[Callable[[Any], bool]] = []
    for f in filters:
        lower = f.lower()
        if lower in _FILTER_PREDICATES:
            predicates.append(_FILTER_PREDICATES[lower])
        else:
            # Unknown → tag substring check (capture loop var explicitly)
            predicates.append(
                lambda m, tag=lower: any(
                    tag in t.lower() for t in getattr(m, "tags", [])
                )
            )
    if not predicates:
        return models
    return [m for m in models if all(p(m) for p in predicates)]


class ModelManager:
    """Manages model loading, caching, and runtime selection."""

    def __init__(
        self,
        *,
        model_dir: str | None = None,
        default_device: str | None = None,
        data_bus: Any | None = None,
        api_client: Any | None = None,
    ) -> None:
        # ``api_client`` is the :class:`~cyberwave.rest.DefaultApi` instance.
        # When present:
        #   - catalog methods (list/get/delete) delegate via MLModelsResourceManager
        #   - ``self.playground`` (PlaygroundClient) is wired up for cloud inference
        #   - ``self._mlmodels_client`` is set to the same PlaygroundClient so
        #     CloudLoadedModel can call .get() / .run() / .fetch_weights_url()
        self._catalog: Any | None = None
        self._mlmodels_client: Any | None = None
        self.playground: Any | None = None
        if api_client is not None:
            from cyberwave.models.playground import PlaygroundClient
            from cyberwave.resources import MLModelsResourceManager

            self._catalog = MLModelsResourceManager(api_client)
            self.playground = PlaygroundClient(api_client)
            self._mlmodels_client = self.playground

        dir_from_env = os.environ.get("CYBERWAVE_MODEL_DIR")
        if model_dir:
            self._model_dir = Path(model_dir)
        elif dir_from_env:
            self._model_dir = Path(dir_from_env)
        elif Path(DEFAULT_MODEL_DIR).is_dir():
            self._model_dir = Path(DEFAULT_MODEL_DIR)
        else:
            # Fallback path under the user's home.  We deliberately do NOT
            # create the directory eagerly here: instantiating a
            # ``Cyberwave`` client (which creates a ``ModelManager``)
            # should not have file-system side effects.  Eager creation
            # interferes with ``cyberwave edge uninstall`` because the
            # CLI instantiates the SDK client *after* removing
            # ``~/.cyberwave`` to clean up backend edge registrations.
            #
            # The directory is created lazily in ``_resolve_model_path``
            # when Ultralytics actually needs a writable location for
            # auto-downloads.  We also deliberately do NOT create the
            # in-container default (``/app/models``) here — that mount
            # must come from Edge Core; creating an empty directory there
            # would mask a real "bind mount not configured" misconfig.
            self._model_dir = Path(FALLBACK_MODEL_DIR)

        self._default_device = default_device or os.environ.get(
            "CYBERWAVE_MODEL_DEVICE"
        )
        self._loaded: dict[str, LoadedModel | CloudLoadedModel] = {}
        self._data_bus_factory = data_bus if callable(data_bus) else lambda: data_bus
        self._resolved_data_bus: Any | None = None

    # ------------------------------------------------------------------
    # Catalog API  (REST /api/v1/mlmodels)
    # ------------------------------------------------------------------

    def _require_catalog(self) -> Any:
        if self._catalog is None:
            raise CyberwaveAPIError(
                "cw.models catalog operations require an API connection. "
                "Use 'cw = Cyberwave(api_key=...)' instead of constructing "
                "ModelManager directly without an api_client."
            )
        return self._catalog

    def run(self, model_id: str, **kwargs: Any) -> Any:
        """Run cloud model inference for automation / workflow workers.

        Alias used by generated workflow workers (``client.mlmodels.run``).
        Uses the product ``POST /mlmodels/{uuid}/run`` endpoint (credit-gated).
        """
        if self.playground is None:
            raise CyberwaveAPIError(
                "cw.models.run requires an API connection. "
                "Use 'cw = Cyberwave(api_key=...)' instead of constructing "
                "ModelManager directly without an api_client."
            )
        return self.playground(model_id).run(**kwargs, product=True)

    def list(
        self,
        *,
        filters: list[str] | None = None,
        deployment: str | None = None,
        edge_compatible: bool | None = None,
        model_external_id: str | None = None,
        supported_level: str | None = None,
        is_trainable: bool | None = None,
        catalog_seed_id: str | None = None,
    ) -> list[MLModelSchema]:
        """List ML model records visible to the authenticated user.

        Returns workspace-scoped models plus any public ones.

        Args:
            filters: Shorthand filter strings applied **client-side** after the
                server response.  Multiple strings are ANDed together.  Built-in
                shorthands:

                * capability: ``"image"`` / ``"image_input"``, ``"video"``,
                  ``"text"``, ``"audio"``, ``"action"``
                * deployment: ``"edge"``, ``"cloud"``, ``"hybrid"``
                * misc: ``"trainable"``, ``"public"``
                * unknown strings → tag substring check

                Example — edge models that accept image input::

                    edge_image_models = cw.models.list(filters=["edge", "image"])

            deployment: Server-side ``deployment`` filter (``"edge"``,
                ``"cloud"``, ``"hybrid"``).  Use ``filters=["edge"]`` for the
                equivalent client-side shorthand.
            model_external_id: Exact match on the external weight filename.

        Returns:
            List of :class:`~cyberwave.rest.MLModelSchema` objects.

        Example::

            for m in cw.models.list(deployment="edge"):
                print(m.slug, m.model_external_id)
        """
        results = self._require_catalog().list(
            deployment=deployment,
            edge_compatible=edge_compatible,
            model_external_id=model_external_id,
            supported_level=supported_level,
            is_trainable=is_trainable,
            catalog_seed_id=catalog_seed_id,
        )
        if filters:
            results = _apply_filter_shorthands(results, filters)
        return results

    def list_public(self, *, deployment: str | None = None) -> list[MLModelSchema]:
        """List public ML models (no workspace membership required)."""
        return self._require_catalog().list_public(deployment=deployment)

    def get(self, model_id: str) -> MLModelSchema:
        """Fetch a catalog record by slug (``ws/models/name``) or UUID."""
        return self._require_catalog().get(model_id)

    def get_by_uuid(self, uuid: str) -> MLModelSchema:
        """Fetch a catalog record by UUID."""
        return self._require_catalog().get_by_uuid(uuid)

    def get_by_slug(self, slug: str) -> MLModelSchema:
        """Fetch a catalog record by unified slug (``ws/models/name``)."""
        return self._require_catalog().get_by_slug(slug)

    def delete(self, uuid: str) -> dict[str, bool]:
        """Delete an ML model record by UUID.

        Returns ``{"success": True}`` on success.
        """
        return self._require_catalog().delete(uuid)

    def create(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Stub — not yet implemented.

        Use ``cw.api.src_app_api_mlmodels_create_mlmodel(...)`` directly,
        or wait for a typed wrapper to land in a future SDK release.
        """
        raise NotImplementedError(
            "cw.models.create() is not implemented yet. "
            "Call cw.api.src_app_api_mlmodels_create_mlmodel(ml_model_create_schema=...) directly."
        )

    def update(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Stub — not yet implemented.

        Use ``cw.api.src_app_api_mlmodels_update_mlmodel(...)`` directly,
        or wait for a typed wrapper to land in a future SDK release.
        """
        raise NotImplementedError(
            "cw.models.update() is not implemented yet. "
            "Call cw.api.src_app_api_mlmodels_update_mlmodel(uuid=..., ml_model_update_schema=...) directly."
        )

    # ------------------------------------------------------------------
    # Runtime API
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_to_model_id(model_entry: str | MLModelSchema) -> str:
        """Normalise a raw string or :class:`~cyberwave.rest.MLModelSchema` entry to a load-able ID.

        Resolution priority for catalog entries:

        1. ``sdk_load_id`` — explicitly set by the backend as the canonical
           local/edge load key (e.g. ``"yolo26n.pt"``).
        2. ``slug`` — unified cloud slug (``ws/models/name``), routes through
           the Playground.
        3. ``uuid`` — fallback; also cloud-routes via the UUID path.

        This lets callers pass a record straight out of ``cw.models.list()``
        without manually inspecting which field to use::

            for m in cw.models.list(deployment="edge"):
                model = cw.models.load(m)   # uses m.sdk_load_id
        """
        if isinstance(model_entry, str):
            return model_entry
        sdk_load_id = getattr(model_entry, "sdk_load_id", None)
        if sdk_load_id:
            return sdk_load_id
        slug = getattr(model_entry, "slug", None)
        if slug:
            return slug
        uuid = getattr(model_entry, "uuid", None)
        if uuid:
            return uuid
        raise TypeError(
            f"model_entry must be a str or MLModelSchema, got {type(model_entry).__name__!r}"
        )

    def _load_cascade(
        self,
        entries: list[str | MLModelSchema],
        *,
        runtime: str | None = None,
        device: str | None = None,
        download: bool = False,
        force_download: bool = False,
        download_url: str | None = None,
        store_input: bool = False,
        **kwargs: Any,
    ) -> CascadeModel:
        """Load multiple model entries and wrap them in a :class:`~cyberwave.models.cascade.CascadeModel`.

        Each entry is loaded with the shared kwargs (``runtime``, ``device``,
        etc.).  The ``store_input`` flag is forwarded to the cascade so that
        :meth:`~cyberwave.models.cascade.CascadePredictionResult.draw_on_top`
        can be called without an explicit image argument.
        """
        from cyberwave.models.cascade import CascadeModel

        if not entries:
            raise ValueError("Cannot create a cascade from an empty list.")

        models: list[LoadedModel | CloudLoadedModel] = []
        names: list[str] = []
        for entry in entries:
            # Prefer the human-readable catalog name; fall back to the load ID.
            name: str = getattr(entry, "name", None) or (
                entry if isinstance(entry, str) else repr(entry)
            )
            loaded = self.load(
                entry,
                runtime=runtime,
                device=device,
                download=download,
                force_download=force_download,
                download_url=download_url,
                **kwargs,
            )
            models.append(loaded)
            names.append(name)

        return CascadeModel(models, names, store_input=store_input)

    def load(
        self,
        model_id: str | MLModelSchema | list[str | MLModelSchema],
        *,
        runtime: str | None = None,
        device: str | None = None,
        download: bool = False,
        force_download: bool = False,
        download_url: str | None = None,
        store_input: bool = False,
        **kwargs: Any,
    ) -> LoadedModel | CloudLoadedModel | CascadeModel:
        """Load a model (or a cascade of models) by catalog ID or catalog entry.

        ``model_id`` can be:

        * A **string** — local weight filename (``"yolo26n.pt"``) or a
          Cyberwave catalog slug / UUID.
        * An :class:`~cyberwave.rest.MLModelSchema` record returned by
          ``cw.models.list()`` or ``cw.models.get()``.  The right load key
          is resolved automatically (``sdk_load_id`` → ``slug`` → ``uuid``),
          so you can pass the entry directly without inspecting its fields::

              for m in cw.models.list(deployment="edge"):
                  model = cw.models.load(m)
                  pred  = model.predict(frame)

        * A **list** of strings or :class:`~cyberwave.rest.MLModelSchema`
          records.  In this case a :class:`~cyberwave.models.cascade.CascadeModel`
          is returned.  Every model in the list receives the same input
          independently when :meth:`~cyberwave.models.cascade.CascadeModel.predict`
          is called, and results are collected into a
          :class:`~cyberwave.models.cascade.CascadePredictionResult` keyed by
          model display name::

              edge_models = cw.models.list(filters=["edge", "image"])
              cascade = cw.models.load(
                  [edge_models[0], edge_models[1]],
                  store_input=True,
              )
              pred = cascade.predict(image)
              print(pred[edge_models[0].name])   # PredictionResult for first model
              output_image = pred.draw_on_top()  # overlay all predictions

        Returns a cached instance on repeated calls with the same
        arguments. The runtime is auto-detected from the model ID when
        not specified.

        Args:
            store_input: When loading a **list** (cascade), store the input
                frame inside each :class:`~cyberwave.models.cascade.CascadePredictionResult`
                so :meth:`~cyberwave.models.cascade.CascadePredictionResult.draw_on_top`
                can be called without an explicit image argument.  Ignored for
                single-model loads.  Defaults to ``False``.

        Cloud routing
        -------------
        When the resolved ID looks like a Cyberwave catalog slug
        (``workspace/models/name``) or a UUID — and when this manager
        was constructed with a cloud client attached (the default when
        accessed via ``cw.models``) — the call is routed to the
        Playground and a :class:`CloudLoadedModel` is returned. This
        is the glue that makes the cloud snippet homogeneous with the
        edge snippet::

            cw.models.load("yolov8n").predict(frame)                # edge
            cw.models.load("acme/models/sam-3.1").predict("a.jpg")  # cloud

        Pass ``runtime="edge"`` to force local resolution when a cloud
        client is attached and you really want a local-file lookup for
        a slug-shaped identifier (rare).

        Local caching for Cyberwave-hosted weights
        ------------------------------------------
        Pass ``download=True`` to opt into on-demand weight download
        for cloud slugs. The SDK calls ``GET /mlmodels/{uuid}/weights``,
        streams the signed URL into ``~/.cyberwave/models/<slug>/``,
        verifies ``checksum_sha256`` (when the catalog advertises one),
        and then loads the result with the local runtime — exactly the
        same code path as an edge-core ``cw.models.load("yolov8n")``
        call. Repeated loads reuse the cached file; pass
        ``force_download=True`` to refresh it.

        When the local cache already contains weights for a slug the
        call implicitly takes the local path even without
        ``download=True`` — this is the "always use the local snippet"
        promise in the Playground docs: once you've downloaded a model
        once, every future ``cw.models.load(slug)`` call in the same
        environment loads the cached file.
        """
        # Cascade path — list of entries
        if isinstance(model_id, list):
            return self._load_cascade(
                model_id,
                runtime=runtime,
                device=device,
                download=download,
                force_download=force_download,
                download_url=download_url,
                store_input=store_input,
                **kwargs,
            )

        model_id = self._resolve_to_model_id(model_id)
        looks_cloud = self._looks_like_cloud_ref(model_id)
        wants_local_download = download or force_download

        if runtime != "edge" and looks_cloud and not wants_local_download:
            # Cache-first: if we've already downloaded this slug, prefer
            # the on-disk copy over a cloud round-trip. This is what
            # makes ``cw.models.load(slug)`` keep working offline after
            # the first ``download=True`` call.
            cached_path = self._find_cached_download(model_id)
            if cached_path is not None:
                logger.debug(
                    "Using cached local weights for %r at %s",
                    model_id,
                    cached_path,
                )
                return self._load_from_cached_download(
                    model_id=model_id,
                    model_path=cached_path,
                    runtime=runtime,
                    device=device,
                    **kwargs,
                )
            cloud = self._load_cloud(model_id)
            if cloud is not None:
                return cloud

        if wants_local_download and looks_cloud:
            return self._load_with_download(
                model_id=model_id,
                runtime=runtime,
                device=device,
                force_download=force_download,
                **kwargs,
            )

        effective_device = device or self._default_device or self._detect_device()
        resolved_runtime = runtime or self._detect_runtime(model_id)

        cache_key = f"{model_id}:{resolved_runtime}:{effective_device}"
        cached = self._loaded.get(cache_key)
        if isinstance(cached, LoadedModel):
            return cached

        rt = get_runtime(resolved_runtime)
        model_path = self._resolve_model_path(
            model_id,
            resolved_runtime,
            download_url=download_url,
        )

        self._verify_model_checksum(model_id, model_path)

        if not rt.supports_predict:
            logger.warning(
                "Runtime '%s' can load models but predict() is not yet "
                "implemented — inference calls will raise NotImplementedError",
                resolved_runtime,
            )

        logger.info(
            "Loading model '%s' with runtime '%s' on device '%s'",
            model_id,
            resolved_runtime,
            effective_device,
        )
        handle = rt.load(str(model_path), device=effective_device, **kwargs)

        loaded = LoadedModel(
            name=model_id,
            runtime=rt,
            model_handle=handle,
            device=effective_device,
            model_path=str(model_path),
            data_bus=self._resolve_data_bus(),
        )
        self._loaded[cache_key] = loaded
        return loaded

    def load_from_file(
        self,
        path: str,
        *,
        runtime: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> LoadedModel:
        """Load a model directly from a file path."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        resolved_runtime = runtime or self._detect_runtime_from_extension(p.suffix)
        rt = get_runtime(resolved_runtime)
        effective_device = device or self._default_device or self._detect_device()

        handle = rt.load(str(p), device=effective_device, **kwargs)
        return LoadedModel(
            name=p.stem,
            runtime=rt,
            model_handle=handle,
            device=effective_device,
            model_path=str(p),
            data_bus=self._resolve_data_bus(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_model_path(
        self,
        model_id: str,
        runtime: str,
        *,
        download_url: str | None = None,
    ) -> Path:
        """Resolve a catalog model ID to a local file path.

        Self-heal contract: when a staging directory at
        ``self._model_dir / model_id`` exists but contains no recognized
        weight file, it is treated as a poison-pill orphan left by a
        previously-failed Edge Core download (see
        ``cyberwave-edge-core/cyberwave_edge_core/model_manager.py``,
        ``_download_runtime_managed`` / ``_download_model`` create the
        directory before the network call). Returning that directory
        path here would forward a directory into ``torch.load`` and
        crash with ``IsADirectoryError`` on every worker start, so we
        prune it (best-effort) and let the Ultralytics auto-download
        convenience branch — or a clear ``FileNotFoundError`` for
        non-Ultralytics runtimes — take over.
        """
        exact = self._model_dir / model_id
        if exact.is_file():
            return exact

        for ext in self._runtime_extensions(runtime):
            candidate = self._model_dir / f"{model_id}{ext}"
            if candidate.exists():
                return candidate

        model_subdir = self._model_dir / model_id
        staged_dir_was_empty = False
        if model_subdir.is_dir():
            # faster-whisper stores a HuggingFace-style tree, not a single weight file.
            if runtime == "faster_whisper":
                return model_subdir
            for ext in self._runtime_extensions(runtime):
                for f in sorted(model_subdir.iterdir()):
                    if f.suffix == ext:
                        return f

            # No recognized weight file inside the staging directory.
            # Two distinct sub-cases, both of which previously fell
            # through to "return the directory path" and crashed
            # ``torch.load`` with ``IsADirectoryError`` on every worker
            # start.
            if self._is_empty_staging_dir(model_subdir):
                # Cruft-only orphan from a previously failed download
                # (mkdir runs before the network fetch on the Edge Core
                # side; an aborted fetch leaves the directory in place).
                # Prune so the fallback paths below can run on a clean
                # slate.
                logger.warning(
                    "Pruning orphan model staging directory at %s "
                    "(no recognized weight file inside — likely left "
                    "behind by a previous failed download).",
                    model_subdir,
                )
                shutil.rmtree(model_subdir, ignore_errors=True)
                staged_dir_was_empty = True
            else:
                # Operator-staged content (a README, a half-staged weight
                # with an unexpected extension, a sub-directory). The
                # "human always wins" invariant says we do **not** touch
                # the directory; raise an actionable error so the
                # operator can finish staging.
                stray = sorted(p.name for p in model_subdir.iterdir())
                raise FileNotFoundError(
                    f"Model '{model_id}' staging directory at "
                    f"{model_subdir} contains files but no recognized "
                    f"weight file for runtime '{runtime}' (expected one "
                    f"of: {self._runtime_extensions(runtime)}). Found: "
                    f"{stray}. Add the missing weight file or remove "
                    f"the directory to let Edge Core re-download."
                )

        # Ultralytics convenience: let the library auto-download from hub.
        # Place the file inside _model_dir so it lands on a writable mount
        # (the worker container's CWD is typically not writable).
        if runtime == "ultralytics":
            self._model_dir.mkdir(parents=True, exist_ok=True)
            return self._model_dir / model_id

        # faster-whisper downloads CTranslate2 weights into download_root.
        if runtime == "faster_whisper":
            self._model_dir.mkdir(parents=True, exist_ok=True)
            cache_root = self._model_dir / model_id
            cache_root.mkdir(parents=True, exist_ok=True)
            return cache_root

        if download_url is not None and download_url.strip():
            logger.info(
                "Downloading public weights for %r into %s",
                model_id,
                exact,
            )
            self._stream_download_to(download_url.strip(), exact)
            return exact

        if staged_dir_was_empty:
            raise FileNotFoundError(
                f"Model '{model_id}' staging directory at {model_subdir} "
                f"was empty and has been pruned. The previous Edge Core "
                f"download likely failed mid-way; restart the edge agent "
                f"to retry, or pre-stage weights manually."
            )

        raise FileNotFoundError(
            f"Model '{model_id}' not found in {self._model_dir}. "
            f"Ensure edge core has downloaded the model weights, "
            f"or use load_from_file()."
        )

    @staticmethod
    def _is_empty_staging_dir(model_subdir: Path) -> bool:
        """Return ``True`` iff *model_subdir* contains only orphan-cruft.

        Conservative definition of "orphan": the directory contains no
        recognizable weight file (handled by the caller before this
        helper runs) **and** every remaining entry is either the
        metadata sidecar (``MODEL_METADATA_FILENAME``) or a partial
        streaming download (``.dl_*.part`` — matches the temp-file
        prefix used by both the SDK's :meth:`_stream_download_to` and
        Edge Core's ``_stream_download``).

        We deliberately keep this strict: any other file in the
        directory (a half-written weight file with an unexpected
        extension, an operator's README, a sidecar config) blocks the
        prune, so a human always wins over the self-heal.
        """
        try:
            entries = list(model_subdir.iterdir())
        except OSError:
            return False
        for entry in entries:
            if not entry.is_file():
                return False
            name = entry.name
            if name == MODEL_METADATA_FILENAME:
                continue
            if name.startswith(".dl_") and name.endswith(".part"):
                continue
            return False
        return True

    # ------------------------------------------------------------------
    # Cloud routing
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_cloud_ref(model_id: str) -> bool:
        """Return True when ``model_id`` is a Cyberwave cloud reference.

        Conservative heuristic: ``{ws}/models/{name}`` (the canonical
        unified slug) and bare UUIDs. We intentionally do *not* claim
        HF-style ``{org}/{name}`` strings here to avoid hijacking what
        might be a future local-catalog namespace — the backend
        ``/mlmodels/by-slug`` call still accepts those if seeded, but
        the caller has to opt in via ``client.models.playground(...).resolve()``
        explicitly.
        """
        if not isinstance(model_id, str) or not model_id:
            return False
        if "/models/" in model_id:
            return True
        # UUID heuristic (any version).
        import uuid as _uuid

        try:
            _uuid.UUID(model_id)
            return True
        except (ValueError, AttributeError):
            return False

    # ------------------------------------------------------------------
    # Download-and-cache (Option 3)
    # ------------------------------------------------------------------

    @staticmethod
    def _slug_to_dirname(slug_or_id: str) -> str:
        """Map a Cyberwave slug / UUID to a filesystem-safe directory name.

        We intentionally keep the mapping human-readable — the goal is
        for ``ls ~/.cyberwave/models`` to be self-describing so a user
        can ``rm -rf`` a single model's cache without grepping UUIDs.

        Defence-in-depth: slugs come from the backend, but we still
        harden against path traversal in case a future backend bug
        lets ``..`` segments through. Any sequence of consecutive
        dots collapses to a single underscore, and leading dots are
        stripped entirely so the result can never resolve outside
        ``_model_dir``.
        """
        # Collapse the slug separator into a human-readable ``__`` so the
        # cache layout is ``~/.cyberwave/models/workspace__models__name/``.
        safe = slug_or_id.replace("/", "__").replace(os.sep, "__")
        # Character whitelist.
        safe = "".join(c for c in safe if c.isalnum() or c in "-_.")
        # Collapse multi-dot runs (``..``, ``...``) to a single ``_`` so
        # ``../evil`` can't survive the mapping.
        import re as _re

        safe = _re.sub(r"\.{2,}", "_", safe)
        # Strip leading dots just in case (``.hidden`` → ``hidden``); a
        # hidden dir would work, but it would confuse ``ls`` and block
        # ``rm -rf`` completions for users.
        safe = safe.lstrip(".") or "_"
        return safe

    def _download_cache_dir(self, model_id: str) -> Path:
        return self._model_dir / self._slug_to_dirname(model_id)

    def _find_cached_download(self, model_id: str) -> Path | None:
        """Return the path of a previously-downloaded weight file, or ``None``.

        Picks the first non-metadata file in the slug's cache dir.
        Empty / missing dirs return ``None``. We do not verify checksum
        here — that runs inside ``load()`` via
        :meth:`_verify_model_checksum`.
        """
        cache_dir = self._download_cache_dir(model_id)
        if not cache_dir.is_dir():
            return None
        for candidate in sorted(cache_dir.iterdir()):
            if candidate.name == MODEL_METADATA_FILENAME:
                continue
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        return None

    def _load_with_download(
        self,
        *,
        model_id: str,
        runtime: str | None,
        device: str | None,
        force_download: bool,
        **kwargs: Any,
    ) -> LoadedModel:
        """Download the checkpoint (if missing) and load it locally."""
        if self._mlmodels_client is None:
            raise RuntimeError(
                f"Cannot download {model_id!r}: no cloud client is attached. "
                f"Instantiate Cyberwave() and access .models — "
                f"bare ModelManager() has no way to reach the Playground."
            )

        summary = self._mlmodels_client.get(model_id)

        existing = self._find_cached_download(model_id)
        if existing is not None and not force_download:
            logger.info(
                "Using cached weights for %r at %s (pass force_download=True to refresh)",
                model_id,
                existing,
            )
            model_path = existing
        else:
            model_path = self._download_checkpoint(model_id, summary)

        return self._load_from_cached_download(
            model_id=model_id,
            model_path=model_path,
            runtime=runtime,
            device=device,
            **kwargs,
        )

    def _load_from_cached_download(
        self,
        *,
        model_id: str,
        model_path: Path,
        runtime: str | None,
        device: str | None,
        **kwargs: Any,
    ) -> LoadedModel:
        """Common tail for download-path and cache-hit: run the local runtime."""
        effective_device = device or self._default_device or self._detect_device()
        resolved_runtime = runtime or self._detect_runtime_from_extension(
            model_path.suffix
        )

        cache_key = f"{model_id}:{resolved_runtime}:{effective_device}"
        cached = self._loaded.get(cache_key)
        if isinstance(cached, LoadedModel):
            return cached

        rt = get_runtime(resolved_runtime)
        self._verify_model_checksum(model_id, model_path)

        logger.info(
            "Loading downloaded model %r with runtime %r on device %r (path=%s)",
            model_id,
            resolved_runtime,
            effective_device,
            model_path,
        )
        handle = rt.load(str(model_path), device=effective_device, **kwargs)
        loaded = LoadedModel(
            name=model_id,
            runtime=rt,
            model_handle=handle,
            device=effective_device,
            model_path=str(model_path),
            data_bus=self._resolve_data_bus(),
        )
        self._loaded[cache_key] = loaded
        return loaded

    def _download_checkpoint(self, model_id: str, summary: Any) -> Path:
        """Fetch the signed URL, stream-download, verify checksum, return path.

        The download happens in two steps so the retry story matches
        the edge-core implementation:

        1. ``GET /mlmodels/{uuid}/weights`` → signed URL (+ metadata).
        2. ``GET <signed_url>`` → binary payload, streamed via a temp
           file in the destination directory to avoid leaving a
           partially-written file on interrupt, then renamed into place.
        """
        try:
            payload = self._mlmodels_client.fetch_weights_url(summary)
        except CyberwaveAPIError as exc:
            if exc.status_code == 404:
                raise RuntimeError(
                    f"Cyberwave does not host checkpoint weights for {model_id!r}. "
                    f"Only private fine-tunes uploaded to your workspace can be "
                    f"downloaded — for public / API-gated models (Gemini, "
                    f"upstream Hugging Face, etc.) use "
                    f"cw.models.load({model_id!r}) without download=True to run "
                    f"via the Playground API."
                ) from exc
            raise

        signed_url = payload.get("signed_url") or payload.get("url")
        if not isinstance(signed_url, str) or not signed_url.strip():
            raise RuntimeError(
                f"Backend returned no signed_url for {model_id!r}: {payload!r}"
            )

        filename = self._filename_for_download(payload, summary)
        cache_dir = self._download_cache_dir(model_id)
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest = cache_dir / filename

        logger.info(
            "Downloading weights for %r into %s (expires_at=%s)",
            model_id,
            dest,
            payload.get("expires_at"),
        )
        self._stream_download_to(signed_url.strip(), dest)

        expected_sha = (summary.metadata or {}).get("checksum_sha256")
        if expected_sha:
            actual = _sha256_file(dest)
            if actual != expected_sha:
                # Don't leave a corrupt artefact on disk — future loads
                # would pick it up via the cache-first path.
                try:
                    dest.unlink()
                except OSError:
                    pass
                raise CyberwaveModelIntegrityError(
                    f"Downloaded weights for {model_id!r} failed checksum "
                    f"verification (expected {expected_sha[:16]}…, got "
                    f"{actual[:16]}…). Corrupt file removed."
                )
            # Persist the checksum sidecar so a subsequent cache-hit
            # load can re-verify without another cloud call.
            try:
                with open(cache_dir / MODEL_METADATA_FILENAME, "w") as fh:
                    json.dump(
                        {"checksum_sha256": expected_sha, "model_id": model_id},
                        fh,
                    )
            except OSError:
                logger.debug(
                    "Could not write %s metadata sidecar — non-fatal",
                    cache_dir,
                    exc_info=True,
                )

        return dest

    @staticmethod
    def _filename_for_download(payload: dict[str, Any], summary: Any) -> str:
        """Pick a reasonable filename for the downloaded checkpoint.

        Priority:

        1. An explicit ``filename`` field in the backend payload (future-
           proofing — the endpoint can start returning this without an
           SDK bump).
        2. The basename of ``checkpoint_path`` in the payload (that's
           what our backend actually stores).
        3. ``summary.metadata['filename']`` as a safety net for seed
           entries that encode the expected artefact name.
        4. ``checkpoint.tar`` as a final fallback — matches the backend's
           ``Content-Disposition`` default.
        """
        explicit = payload.get("filename")
        if isinstance(explicit, str) and explicit.strip():
            return Path(explicit.strip()).name

        checkpoint_path = payload.get("checkpoint_path")
        if isinstance(checkpoint_path, str) and checkpoint_path.strip():
            name = Path(checkpoint_path.strip()).name
            if name:
                return name

        meta_name = (getattr(summary, "metadata", None) or {}).get("filename")
        if isinstance(meta_name, str) and meta_name.strip():
            return Path(meta_name.strip()).name

        return "checkpoint.tar"

    @staticmethod
    def _stream_download_to(url: str, dest: Path) -> None:
        """Stream ``url`` to ``dest`` via a sibling tempfile, then rename.

        Mirrors ``cyberwave_edge_core.model_manager._stream_download``:
        downloading to ``tmp`` in the same directory as ``dest`` means
        the ``os.replace`` at the end is atomic on POSIX, so an
        interrupted download never leaves a half-written file that
        ``_find_cached_download`` would later mistake for a valid
        cache hit.

        We deliberately do not retry here — the signed URL has a TTL,
        and if it has expired the caller should re-enter
        :meth:`_download_checkpoint` to mint a fresh one.
        """
        from urllib.request import urlopen

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            dir=dest.parent, prefix=".dl_", suffix=".part"
        )
        tmp_path = Path(tmp_path_str)
        try:
            with (
                os.fdopen(tmp_fd, "wb") as fh,
                urlopen(  # noqa: S310
                    url,
                    timeout=_DOWNLOAD_TIMEOUT_SECS,
                ) as resp,
            ):
                while True:
                    chunk = resp.read(_DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    fh.write(chunk)
            tmp_path.replace(dest)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    def _load_cloud(self, model_id: str) -> CloudLoadedModel | None:
        """Resolve ``model_id`` via the Playground client, if attached.

        Returns ``None`` when no cloud client is wired up (e.g. someone
        instantiated a bare ``ModelManager()`` outside the
        ``Cyberwave`` client). The caller then falls back to local-file
        resolution, which will raise ``FileNotFoundError`` with a
        clearer message than a generic ``RuntimeError`` from here.
        """
        if self._mlmodels_client is None:
            logger.debug(
                "ModelManager.load(%r) looks like a cloud slug but no "
                "mlmodels_client is attached; falling back to local resolution.",
                model_id,
            )
            return None
        cache_key = f"cloud:{model_id}"
        cached = self._loaded.get(cache_key)
        if isinstance(cached, CloudLoadedModel):
            return cached
        summary = self._mlmodels_client.get(model_id)
        cloud = CloudLoadedModel(summary=summary, client=self._mlmodels_client)
        # Store under the cloud cache key so repeated loads of the same
        # slug return the same instance — parity with the local cache.
        self._loaded[cache_key] = cloud  # type: ignore[assignment]
        return cloud

    @staticmethod
    def _detect_runtime(model_id: str) -> str:
        """Heuristic: detect runtime from model ID or file extension.

        Catalog convention: a ``-onnx`` suffix (e.g. ``yolov8n-pose-onnx``)
        flags the entry as an ONNX export of an upstream model and routes
        to the ``onnxruntime`` backend, even when the stem matches a
        framework-specific keyword like ``yolo``.
        """
        lower = model_id.lower()

        # Catalog suffix takes precedence over framework-keyword heuristics
        # so e.g. ``yolov8n-pose-onnx`` resolves to onnxruntime, not ultralytics.
        if lower.endswith("-onnx") or lower.endswith(".onnx"):
            return "onnxruntime"
        # Hailo HEFs: explicit ``.hef`` extension, or catalog slug hints
        # (``_h8`` / ``_h8l`` / ``_hailo``) that the seed file uses to
        # distinguish hardware variants when the slug otherwise looks
        # like a generic ``yolov8s``.
        if lower.endswith(".hef") or any(
            tag in lower for tag in ("_h8l", "_h8", "_hailo", "-hailo")
        ):
            return "hailo"
        if any(k in lower for k in ("yolo", "yolov5", "yolov8", "yolov11")):
            return "ultralytics"
        if any(k in lower for k in ("background-subtraction", "haar", "cascade")):
            return "opencv"
        if lower.endswith((".gguf", ".bin")) and "whisper" in lower:
            return "whisper_cpp"
        if "faster-whisper" in lower or lower.startswith("systran/faster-whisper"):
            return "faster_whisper"
        if lower.endswith(".tflite"):
            return "tflite"
        if lower.endswith((".engine", ".trt")):
            return "tensorrt"
        if lower.endswith(".pth"):
            return "torch"

        raise ValueError(
            f"Cannot auto-detect runtime for model '{model_id}'. "
            f"Pass runtime= explicitly, e.g.: "
            f"cw.models.load('{model_id}', runtime='ultralytics')"
        )

    @staticmethod
    def _detect_runtime_from_extension(ext: str) -> str:
        """Map a file extension to a runtime name."""
        mapping: dict[str, str] = {
            ".pt": "ultralytics",
            ".onnx": "onnxruntime",
            ".tflite": "tflite",
            ".xml": "opencv",
            ".engine": "tensorrt",
            ".trt": "tensorrt",
            ".pth": "torch",
            ".hef": "hailo",
        }
        return mapping.get(ext.lower(), "ultralytics")

    @staticmethod
    def _runtime_extensions(runtime: str) -> list[str]:
        mapping: dict[str, list[str]] = {
            "ultralytics": [".pt", ".onnx", ".engine"],
            "onnxruntime": [".onnx"],
            "opencv": [".xml", ".caffemodel"],
            "tflite": [".tflite"],
            "torch": [".pt", ".pth"],
            "tensorrt": [".engine", ".trt"],
            "whisper_cpp": [".gguf", ".bin"],
            "faster_whisper": [],
            "hailo": [".hef"],
        }
        return mapping.get(runtime, [".pt"])

    @staticmethod
    def _verify_model_checksum(model_id: str, model_path: Path) -> None:
        """Validate model file against the metadata sidecar checksum if available."""
        metadata_path = model_path.parent / MODEL_METADATA_FILENAME
        if not metadata_path.exists():
            return
        try:
            with open(metadata_path) as fh:
                meta = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return

        expected = meta.get("checksum_sha256")
        if not expected:
            return

        actual = _sha256_file(model_path)
        if actual != expected:
            try:
                model_path.unlink()
            except OSError:
                pass
            raise CyberwaveModelIntegrityError(
                f"Model '{model_id}' failed checksum verification "
                f"(expected {expected[:16]}…, got {actual[:16]}…). "
                f"Corrupt file removed — it will be re-downloaded on next startup."
            )

    def _resolve_data_bus(self) -> Any | None:
        if self._resolved_data_bus is None and self._data_bus_factory is not None:
            try:
                self._resolved_data_bus = self._data_bus_factory()
            except Exception:
                logger.debug(
                    "Data bus not available for detection publishing", exc_info=True
                )
        return self._resolved_data_bus

    @staticmethod
    def _detect_device() -> str:
        return "cuda:0" if _cuda_is_usable() else "cpu"


_CUDA_PROBE_CACHE: bool | None = None


def _safe_call(fn: Any) -> Any:
    """Call ``fn()`` and return its result, or ``None`` if it raises.

    Used when collecting diagnostic fields for a CUDA probe failure: some
    accessors (e.g. ``torch.backends.cudnn.version()``) themselves raise on
    broken setups, and we don't want one bad accessor to null out the rest.
    """
    try:
        return fn()
    except Exception:
        return None


def _cuda_is_usable() -> bool:
    """Return True iff CUDA is present *and* cuDNN can actually run a conv2d.

    Auto-device detection used to return ``cuda:0`` whenever
    ``torch.cuda.is_available()`` was True, but on some hosts the CUDA runtime
    reports a usable GPU while cuDNN has no engine that supports it (common on
    Pascal / sm_61 with modern cuDNN 9 wheels, and on brand-new archs not yet
    baked into the installed torch build). Those setups crash at the first
    ``F.conv2d`` call with ``GET was unable to find an engine to execute this
    computation``.

    This helper runs a tiny conv2d on ``cuda:0`` once per module-load; on
    failure it logs the GPU / cuDNN / arch list and falls back to CPU so
    workers stay up. Set ``CYBERWAVE_MODEL_DEVICE=cuda:0`` to bypass this
    probe when you're sure CUDA is fine (e.g. the probe itself hit a
    transient OOM).
    """
    global _CUDA_PROBE_CACHE
    if _CUDA_PROBE_CACHE is not None:
        return _CUDA_PROBE_CACHE

    try:
        import torch
    except ImportError:
        _CUDA_PROBE_CACHE = False
        return False

    try:
        if not torch.cuda.is_available():
            _CUDA_PROBE_CACHE = False
            return False
    except Exception:
        _CUDA_PROBE_CACHE = False
        return False

    try:
        dev = torch.device("cuda:0")
        x = torch.zeros(1, 3, 8, 8, device=dev)
        w = torch.zeros(1, 3, 3, 3, device=dev)
        y = torch.nn.functional.conv2d(x, w)
        # Force the graph to flush and a D2H copy, so asynchronous
        # kernel-launch failures (e.g. cudaErrorNoKernelImageForDevice)
        # surface HERE rather than at the caller's first real predict().
        y.cpu()
        torch.cuda.synchronize()
    except Exception as exc:
        name = _safe_call(lambda: torch.cuda.get_device_name(0))
        cap = _safe_call(lambda: torch.cuda.get_device_capability(0))
        archs = _safe_call(torch.cuda.get_arch_list)
        cudnn_ver = _safe_call(torch.backends.cudnn.version)
        first_line = str(exc).splitlines()[0] if str(exc) else ""
        logger.warning(
            "CUDA device detected (%s, compute capability %s) but cuDNN "
            "cannot execute a conv2d probe — falling back to CPU. "
            "torch cuDNN=%s, build archs=%s. Error: %s: %s. "
            "Set CYBERWAVE_MODEL_DEVICE=cuda:0 to bypass this probe.",
            name,
            cap,
            cudnn_ver,
            archs,
            type(exc).__name__,
            first_line,
        )
        _CUDA_PROBE_CACHE = False
        return False

    _CUDA_PROBE_CACHE = True
    return True


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
