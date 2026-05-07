"""Model manager — loading, caching, and runtime/device detection.

Exposed as ``cw.models`` on the ``Cyberwave`` client.  Searches the
local model cache (populated by Edge Core), detects the appropriate
runtime backend, and returns a ``LoadedModel`` with a stable
``.predict()`` API.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from cyberwave.exceptions import CyberwaveAPIError, CyberwaveModelIntegrityError
from cyberwave.models.cloud import CloudLoadedModel
from cyberwave.models.loaded_model import LoadedModel
from cyberwave.models.runtimes import get_runtime

logger = logging.getLogger(__name__)

MODEL_METADATA_FILENAME = "metadata.json"

DEFAULT_MODEL_DIR = "/app/models"
FALLBACK_MODEL_DIR = os.path.expanduser("~/.cyberwave/models")

_DOWNLOAD_CHUNK_SIZE = 1 << 20  # 1 MiB — matches the edge-core streaming buffer.
_DOWNLOAD_TIMEOUT_SECS = 300.0


class ModelManager:
    """Manages model loading, caching, and runtime selection."""

    def __init__(
        self,
        *,
        model_dir: str | None = None,
        default_device: str | None = None,
        data_bus: Any | None = None,
        mlmodels_client: Any | None = None,
    ) -> None:
        # ``mlmodels_client`` is the :class:`MLModelsClient` owned by the
        # parent :class:`Cyberwave` client. When present, ``load()`` can
        # route cloud-slug references (``ws/models/name`` / UUID) to the
        # Playground API instead of the local file cache, so snippets like
        # ``cw.models.load("acme/models/sam-3.1").predict(image)`` work
        # identically to ``cw.models.load("yolov8n").predict(frame)``.
        self._mlmodels_client = mlmodels_client

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
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        model_id: str,
        *,
        runtime: str | None = None,
        device: str | None = None,
        download: bool = False,
        force_download: bool = False,
        **kwargs: Any,
    ) -> LoadedModel | CloudLoadedModel:
        """Load a model by catalog ID.

        Returns a cached instance on repeated calls with the same
        arguments. The runtime is auto-detected from the model ID when
        not specified.

        Cloud routing
        -------------
        When ``model_id`` looks like a Cyberwave catalog slug
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
        model_path = self._resolve_model_path(model_id, resolved_runtime)

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

    def _resolve_model_path(self, model_id: str, runtime: str) -> Path:
        """Resolve a catalog model ID to a local file path."""
        exact = self._model_dir / model_id
        if exact.is_file():
            return exact

        for ext in self._runtime_extensions(runtime):
            candidate = self._model_dir / f"{model_id}{ext}"
            if candidate.exists():
                return candidate

        model_subdir = self._model_dir / model_id
        if model_subdir.is_dir():
            for ext in self._runtime_extensions(runtime):
                for f in sorted(model_subdir.iterdir()):
                    if f.suffix == ext:
                        return f

        # Ultralytics convenience: let the library auto-download from hub.
        # Place the file inside _model_dir so it lands on a writable mount
        # (the worker container's CWD is typically not writable).
        if runtime == "ultralytics":
            self._model_dir.mkdir(parents=True, exist_ok=True)
            return self._model_dir / model_id

        raise FileNotFoundError(
            f"Model '{model_id}' not found in {self._model_dir}. "
            f"Ensure edge core has downloaded the model weights, "
            f"or use load_from_file()."
        )

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
        the caller has to opt in via ``client.mlmodels.get(...)``
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
        resolved_runtime = runtime or self._detect_runtime_from_extension(model_path.suffix)

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

    def _download_checkpoint(
        self, model_id: str, summary: Any
    ) -> Path:
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
    def _filename_for_download(
        payload: dict[str, Any], summary: Any
    ) -> str:
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
        try:
            import httpx  # noqa: PLC0415 — keep httpx as an opt-in import
        except ImportError as exc:
            raise RuntimeError(
                "download=True requires the 'httpx' package. Install it with "
                "pip install httpx (or pip install cyberwave[ml])."
            ) from exc

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            dir=dest.parent, prefix=".dl_", suffix=".part"
        )
        tmp_path = Path(tmp_path_str)
        try:
            with (
                os.fdopen(tmp_fd, "wb") as fh,
                httpx.stream(
                    "GET", url, timeout=_DOWNLOAD_TIMEOUT_SECS, follow_redirects=True
                ) as resp,
            ):
                resp.raise_for_status()
                for chunk in resp.iter_bytes(chunk_size=_DOWNLOAD_CHUNK_SIZE):
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
        if any(k in lower for k in ("yolo", "yolov5", "yolov8", "yolov11")):
            return "ultralytics"
        if any(k in lower for k in ("background-subtraction", "haar", "cascade")):
            return "opencv"
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
                logger.debug("Data bus not available for detection publishing", exc_info=True)
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
