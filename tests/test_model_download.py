"""Tests for ``cw.models.load(slug, download=True)`` — Option 3.

These tests cover the cache-and-download path that lets users run
Cyberwave-hosted cloud models locally without manually juggling the
backend's ``/mlmodels/{uuid}/weights`` signed-URL dance. The key
invariant: once a model is on disk, every subsequent
``cw.models.load(slug)`` call resolves to the cached file — even
without ``download=True``. That's the "always use the local snippet"
promise in the Playground docs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from cyberwave.exceptions import CyberwaveAPIError, CyberwaveModelIntegrityError
from cyberwave.mlmodels.types import MLModelSummary
from cyberwave.models.cloud import CloudLoadedModel
from cyberwave.models.loaded_model import LoadedModel
from cyberwave.models.manager import MODEL_METADATA_FILENAME, ModelManager
from cyberwave.models.runtimes.base import ModelRuntime
from cyberwave.models.types import PredictionResult


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeRuntime(ModelRuntime):
    """Minimal runtime so :meth:`ModelManager.load` can return a LoadedModel."""

    name = "fake"

    def is_available(self) -> bool:
        return True

    def load(self, model_path, *, device=None, **kwargs):
        return {"path": model_path, "device": device}

    def predict(self, model_handle, input_data, *, confidence=0.5, classes=None, **kwargs):
        return PredictionResult()


@pytest.fixture(autouse=True)
def _register_fake_runtime():
    """Register ``_FakeRuntime`` under every runtime name the detector picks.

    The download-tests use fake file extensions (``.pt`` / ``.onnx`` /
    ``.tar``), each of which maps to a real runtime name via
    :meth:`ModelManager._detect_runtime_from_extension`. We register the
    fake under all of them so ``runtime.load`` is a no-op deterministic
    call regardless of the artefact the backend hands us.
    """
    from cyberwave.models.runtimes import _RUNTIME_REGISTRY, register_runtime

    # Register as "fake" + aliases used by extension-based detection.
    class _FakeUltralytics(_FakeRuntime):
        name = "ultralytics"

    class _FakeOnnx(_FakeRuntime):
        name = "onnxruntime"

    class _FakeTorch(_FakeRuntime):
        name = "torch"

    previous = {
        k: _RUNTIME_REGISTRY.get(k)
        for k in ("fake", "ultralytics", "onnxruntime", "torch")
    }
    register_runtime(_FakeRuntime)
    register_runtime(_FakeUltralytics)
    register_runtime(_FakeOnnx)
    register_runtime(_FakeTorch)
    yield
    for name, prev in previous.items():
        if prev is None:
            _RUNTIME_REGISTRY.pop(name, None)
        else:
            _RUNTIME_REGISTRY[name] = prev


class _FakeMLModelsClient:
    """In-memory stand-in for :class:`cyberwave.mlmodels.MLModelsClient`.

    Records calls to ``get()`` and ``fetch_weights_url()``; the latter
    can be configured to raise a ``CyberwaveAPIError`` (e.g. a 404) to
    exercise error-handling branches in the download path.
    """

    def __init__(
        self,
        summary: MLModelSummary,
        *,
        weights_payload: dict[str, Any] | None = None,
        weights_error: Exception | None = None,
    ) -> None:
        self._summary = summary
        self._weights_payload = weights_payload or {
            "signed_url": "https://signed.example.com/checkpoint.pt",
            "expires_at": "2026-04-22T00:00:00+00:00",
            "checkpoint_path": "ml_models/abc/checkpoint.pt",
        }
        self._weights_error = weights_error
        self.get_calls: list[str] = []
        self.fetch_weights_calls: list[str | MLModelSummary] = []

    def get(self, model_ref: str) -> MLModelSummary:
        self.get_calls.append(model_ref)
        return self._summary

    def fetch_weights_url(self, model) -> dict[str, Any]:
        self.fetch_weights_calls.append(model)
        if self._weights_error is not None:
            raise self._weights_error
        return dict(self._weights_payload)


def _summary(
    *,
    slug: str = "acme/models/sam-3.1",
    uuid_str: str | None = None,
    checksum: str | None = None,
) -> MLModelSummary:
    metadata: dict[str, Any] = {}
    if checksum:
        metadata["checksum_sha256"] = checksum
    data = {
        "uuid": uuid_str or str(uuid4()),
        "slug": slug,
        "name": "SAM 3.1",
        "model_external_id": "sam-3.1",
        "model_provider_name": "custom",
        "output_format": "json",
        "deployment": "cloud",
        "can_take_image_as_input": True,
        "can_take_text_as_input": True,
        "playground_kind": "vlm-spatial-reasoner",
        "allowed_structured_tasks": ["segment", "free"],
        "execution_surfaces": ["playground", "edge"],
        "metadata": metadata,
        "tags": ["segmentation"],
    }
    return MLModelSummary.from_api(data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSlugToDirname:
    """The cache layout is user-facing (``ls ~/.cyberwave/models``) so it
    has explicit test coverage: users copy-paste ``rm -rf`` from these
    paths to clear single-model caches.
    """

    def test_slashes_collapse_to_double_underscore(self) -> None:
        assert (
            ModelManager._slug_to_dirname("acme/models/sam-3.1")
            == "acme__models__sam-3.1"
        )

    def test_strips_path_traversal_characters(self) -> None:
        # No ``..`` or leading slash should survive the mapping.
        result = ModelManager._slug_to_dirname("../evil/models/x")
        assert "/" not in result
        assert ".." not in result

    def test_uuid_passes_through(self) -> None:
        uid = str(uuid4())
        assert ModelManager._slug_to_dirname(uid) == uid


class TestDownloadHappyPath:
    def test_download_writes_to_cache_and_returns_loaded_model(
        self, tmp_path, monkeypatch
    ) -> None:
        summary = _summary()
        client = _FakeMLModelsClient(summary)
        mgr = ModelManager(mlmodels_client=client, model_dir=str(tmp_path))

        # Stub the download so we never hit the network: just write a
        # small file to the target path.
        def fake_stream(url: str, dest: Path) -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"\x00\x01\x02\x03")

        monkeypatch.setattr(ModelManager, "_stream_download_to", staticmethod(fake_stream))

        loaded = mgr.load("acme/models/sam-3.1", download=True)

        assert isinstance(loaded, LoadedModel)
        # The summary was resolved exactly once, and we asked for weights once.
        assert client.get_calls == ["acme/models/sam-3.1"]
        assert len(client.fetch_weights_calls) == 1
        # The file actually landed in the expected cache layout.
        expected_path = tmp_path / "acme__models__sam-3.1" / "checkpoint.pt"
        assert expected_path.exists()
        assert expected_path.read_bytes() == b"\x00\x01\x02\x03"
        # Runtime detection picked ``.pt`` → ultralytics.
        assert loaded.runtime == "ultralytics"

    def test_subsequent_load_without_download_reuses_cache(
        self, tmp_path, monkeypatch
    ) -> None:
        """The "always use the local snippet" promise: once a model is
        downloaded, ``cw.models.load(slug)`` with no flags must never
        call the Playground API or re-download.
        """
        summary = _summary()
        client = _FakeMLModelsClient(summary)
        mgr = ModelManager(mlmodels_client=client, model_dir=str(tmp_path))

        # Pre-populate the cache as if a prior ``download=True`` had
        # already written the artefact.
        cache_dir = tmp_path / "acme__models__sam-3.1"
        cache_dir.mkdir(parents=True)
        (cache_dir / "checkpoint.pt").write_bytes(b"cached-bytes")

        # Fail loudly if the manager tries to stream anything or hit
        # the cloud on what must be a cache-hit call.
        monkeypatch.setattr(
            ModelManager,
            "_stream_download_to",
            staticmethod(
                lambda url, dest: pytest.fail("Unexpected download on cache-hit")
            ),
        )

        loaded = mgr.load("acme/models/sam-3.1")

        assert isinstance(loaded, LoadedModel), (
            "Cached download must resolve to a local LoadedModel, not CloudLoadedModel"
        )
        # Cache-hit path must not have called fetch_weights_url at all.
        assert client.fetch_weights_calls == []
        # We also don't need to resolve a summary for a pure cache hit.
        assert client.get_calls == []

    def test_download_true_then_plain_load_returns_same_instance(
        self, tmp_path, monkeypatch
    ) -> None:
        """The two entry points should share the ``_loaded`` cache — a
        ``download=True`` call followed by a plain ``load()`` must
        return the same :class:`LoadedModel` instance."""
        summary = _summary()
        client = _FakeMLModelsClient(summary)
        mgr = ModelManager(mlmodels_client=client, model_dir=str(tmp_path))

        def fake_stream(url: str, dest: Path) -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x")

        monkeypatch.setattr(ModelManager, "_stream_download_to", staticmethod(fake_stream))

        first = mgr.load("acme/models/sam-3.1", download=True)
        second = mgr.load("acme/models/sam-3.1")

        assert first is second


class TestDownloadErrors:
    def test_backend_404_raises_clear_error(self, tmp_path, monkeypatch) -> None:
        summary = _summary()
        client = _FakeMLModelsClient(
            summary,
            weights_error=CyberwaveAPIError(
                "GET /mlmodels/abc/weights failed: HTTP 404", status_code=404
            ),
        )
        mgr = ModelManager(mlmodels_client=client, model_dir=str(tmp_path))

        # Hard failure: never touch the stream helper if the backend 404s.
        monkeypatch.setattr(
            ModelManager,
            "_stream_download_to",
            staticmethod(
                lambda url, dest: pytest.fail("Stream must not run on 404")
            ),
        )

        with pytest.raises(RuntimeError, match="does not host checkpoint weights"):
            mgr.load("acme/models/sam-3.1", download=True)

    def test_missing_cloud_client_raises_runtime_error(self, tmp_path) -> None:
        """A bare ``ModelManager()`` has no way to reach the API — we
        should surface a direct error rather than silently 404-ing."""
        mgr = ModelManager(model_dir=str(tmp_path))  # no mlmodels_client
        with pytest.raises(RuntimeError, match="no cloud client is attached"):
            mgr.load("acme/models/sam-3.1", download=True)

    def test_checksum_mismatch_removes_corrupt_file(
        self, tmp_path, monkeypatch
    ) -> None:
        expected = hashlib.sha256(b"good-bytes").hexdigest()
        summary = _summary(checksum=expected)
        client = _FakeMLModelsClient(summary)
        mgr = ModelManager(mlmodels_client=client, model_dir=str(tmp_path))

        # "Download" the wrong bytes so verification fails.
        def fake_stream(url: str, dest: Path) -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"tampered-bytes")

        monkeypatch.setattr(ModelManager, "_stream_download_to", staticmethod(fake_stream))

        with pytest.raises(CyberwaveModelIntegrityError, match="failed checksum"):
            mgr.load("acme/models/sam-3.1", download=True)

        # Corrupt artefact must be removed so a retry doesn't silently
        # re-use the tampered file via the cache-first path.
        cache_dir = tmp_path / "acme__models__sam-3.1"
        artefact = cache_dir / "checkpoint.pt"
        assert not artefact.exists()

    def test_checksum_match_persists_sidecar_metadata(
        self, tmp_path, monkeypatch
    ) -> None:
        good_bytes = b"good-bytes"
        checksum = hashlib.sha256(good_bytes).hexdigest()
        summary = _summary(checksum=checksum)
        client = _FakeMLModelsClient(summary)
        mgr = ModelManager(mlmodels_client=client, model_dir=str(tmp_path))

        monkeypatch.setattr(
            ModelManager,
            "_stream_download_to",
            staticmethod(
                lambda url, dest: (
                    dest.parent.mkdir(parents=True, exist_ok=True),
                    dest.write_bytes(good_bytes),
                )
            ),
        )

        mgr.load("acme/models/sam-3.1", download=True)

        metadata_path = (
            tmp_path / "acme__models__sam-3.1" / MODEL_METADATA_FILENAME
        )
        assert metadata_path.exists()
        meta = json.loads(metadata_path.read_text())
        assert meta["checksum_sha256"] == checksum

    def test_missing_signed_url_raises(self, tmp_path, monkeypatch) -> None:
        summary = _summary()
        client = _FakeMLModelsClient(
            summary, weights_payload={"expires_at": "2026-04-22T00:00:00+00:00"}
        )
        mgr = ModelManager(mlmodels_client=client, model_dir=str(tmp_path))

        with pytest.raises(RuntimeError, match="no signed_url"):
            mgr.load("acme/models/sam-3.1", download=True)


class TestForceDownload:
    def test_force_download_refreshes_cached_artefact(
        self, tmp_path, monkeypatch
    ) -> None:
        summary = _summary()
        client = _FakeMLModelsClient(summary)
        mgr = ModelManager(mlmodels_client=client, model_dir=str(tmp_path))

        cache_dir = tmp_path / "acme__models__sam-3.1"
        cache_dir.mkdir(parents=True)
        (cache_dir / "checkpoint.pt").write_bytes(b"stale")

        calls: list[str] = []

        def fake_stream(url: str, dest: Path) -> None:
            calls.append(url)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"fresh")

        monkeypatch.setattr(ModelManager, "_stream_download_to", staticmethod(fake_stream))

        mgr.load("acme/models/sam-3.1", force_download=True)

        assert calls, "force_download=True must re-fetch even on cache hit"
        assert (cache_dir / "checkpoint.pt").read_bytes() == b"fresh"


class TestFilenameInference:
    def test_prefers_explicit_filename_field(self) -> None:
        name = ModelManager._filename_for_download(
            {"filename": "adapter_model.safetensors"}, summary=None
        )
        assert name == "adapter_model.safetensors"

    def test_strips_directories_from_explicit_filename(self) -> None:
        """Defence in depth: a stray ``/`` in a backend payload must not
        let the SDK write outside the cache dir."""
        name = ModelManager._filename_for_download(
            {"filename": "../../etc/passwd"}, summary=None
        )
        # Only the basename survives.
        assert name == "passwd"

    def test_falls_back_to_checkpoint_path_basename(self) -> None:
        name = ModelManager._filename_for_download(
            {"checkpoint_path": "ml_models/uuid/weights.pt"}, summary=None
        )
        assert name == "weights.pt"

    def test_uses_summary_metadata_filename_as_tertiary(self) -> None:
        summary = MLModelSummary.from_api(
            {
                "uuid": str(uuid4()),
                "slug": None,
                "name": "x",
                "model_external_id": "x",
                "model_provider_name": "custom",
                "output_format": None,
                "deployment": "cloud",
                "can_take_image_as_input": True,
                "can_take_text_as_input": True,
                "metadata": {"filename": "policy.pt"},
            }
        )
        name = ModelManager._filename_for_download({}, summary=summary)
        assert name == "policy.pt"

    def test_defaults_to_checkpoint_tar_when_nothing_is_known(self) -> None:
        name = ModelManager._filename_for_download({}, summary=None)
        assert name == "checkpoint.tar"


class TestCacheFirstDispatch:
    """The user-facing promise: ``cw.models.load(slug)`` works in air-gap
    mode once the model is on disk. These tests pin that behavior so a
    future refactor of the dispatch code can't silently regress it.
    """

    def test_cache_hit_trumps_cloud_routing(self, tmp_path) -> None:
        summary = _summary()
        client = _FakeMLModelsClient(summary)
        mgr = ModelManager(mlmodels_client=client, model_dir=str(tmp_path))

        (tmp_path / "acme__models__sam-3.1").mkdir(parents=True)
        (tmp_path / "acme__models__sam-3.1" / "checkpoint.pt").write_bytes(b"x")

        loaded = mgr.load("acme/models/sam-3.1")

        assert isinstance(loaded, LoadedModel)
        assert not isinstance(loaded, CloudLoadedModel)

    def test_empty_cache_dir_still_routes_to_cloud(self, tmp_path) -> None:
        """A leftover empty dir must not fool the cache-hit check."""
        summary = _summary()
        client = _FakeMLModelsClient(summary)
        mgr = ModelManager(mlmodels_client=client, model_dir=str(tmp_path))

        (tmp_path / "acme__models__sam-3.1").mkdir(parents=True)

        loaded = mgr.load("acme/models/sam-3.1")

        assert isinstance(loaded, CloudLoadedModel)


# ---------------------------------------------------------------------------
# MLModelsClient.fetch_weights_url — thin but important
# ---------------------------------------------------------------------------


class TestFetchWeightsUrl:
    def test_calls_weights_endpoint_with_uuid(self) -> None:
        from cyberwave.mlmodels.client import MLModelsClient

        captured: dict[str, Any] = {}

        class _FakeApiClient:
            def param_serialize(self, **kwargs):
                captured.update(kwargs)
                return ("serialized",)

            def call_api(self, *args):
                class _Resp:
                    status = 200
                    data = json.dumps(
                        {
                            "signed_url": "https://signed.example.com/x.pt",
                            "expires_at": "2026-04-22T00:00:00+00:00",
                            "checkpoint_path": "ml_models/abc/x.pt",
                        }
                    ).encode("utf-8")

                    def read(self):
                        return None

                return _Resp()

        client = MLModelsClient(_FakeApiClient())
        uid = str(uuid4())

        payload = client.fetch_weights_url(uid)

        assert captured["resource_path"] == f"/api/v1/mlmodels/{uid}/weights"
        assert captured["method"] == "GET"
        assert payload["signed_url"].startswith("https://")

    def test_raises_on_blank_string(self) -> None:
        from cyberwave.mlmodels.client import MLModelsClient

        client = MLModelsClient(api_client=object())
        with pytest.raises(ValueError, match="non-empty string"):
            client.fetch_weights_url("")

    def test_raises_on_wrong_type(self) -> None:
        from cyberwave.mlmodels.client import MLModelsClient

        client = MLModelsClient(api_client=object())
        with pytest.raises(TypeError, match="string or MLModelSummary"):
            client.fetch_weights_url(123)  # type: ignore[arg-type]
