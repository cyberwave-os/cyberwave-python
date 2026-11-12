"""The manager downloads the catalog entries a model ``requires``.

Outside Edge Core (a notebook running an exported workflow) nothing has cached a
companion file such as YOLOE-26's ``mobileclip2_b.ts``, so the manager downloads the
catalog's pinned copy itself, verifies it, and caches it where Edge Core would. It does
this on the first load of a bare id (what generated workflows emit) and when a slug or
UUID is loaded with ``download=True``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cyberwave.exceptions import CyberwaveModelIntegrityError
from cyberwave.models import companions
from cyberwave.models.manager import ModelManager

ID = "mobileclip2_b.ts"
MODEL = "yoloe-26s-seg.pt"
PAYLOAD = b"text encoder"
SHA = hashlib.sha256(PAYLOAD).hexdigest()
URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts"


class _Catalog:
    """``MLModelsResourceManager``: its generated API exact-matches ``model_external_id``."""

    def __init__(self, entries: dict[str, Any], error: Exception | None = None) -> None:
        self.entries = entries
        self.error = error
        self.list_calls: list[str | None] = []
        self.timeouts: list[float | None] = []
        self.api = SimpleNamespace(src_app_api_mlmodels_list_mlmodels=self._list)

    def _list(
        self,
        *,
        model_external_id: str | None = None,
        _request_timeout: Any = None,
        **_: Any,
    ) -> list[Any]:
        self.list_calls.append(model_external_id)
        self.timeouts.append(_request_timeout)
        if self.error is not None:
            raise self.error
        entry = self.entries.get(model_external_id)  # type: ignore[arg-type]
        return [entry] if entry is not None else []


def _companion(**metadata: Any) -> Any:
    return SimpleNamespace(
        metadata={
            "download_url": URL,
            "checksum_sha256": SHA,
            "filename": ID,
            **metadata,
        }
    )


def _model(requires: Any = (ID,)) -> Any:
    return SimpleNamespace(metadata={"requires": list(requires)} if requires else {})


@pytest.fixture(autouse=True)
def _isolated_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(companions, "_extra_roots", [])


class _Streams(list):
    """The urls the fake download step was asked for; ``expected`` holds the checksums."""

    expected: list[str | None]


@pytest.fixture
def streamed(monkeypatch: pytest.MonkeyPatch) -> _Streams:
    urls = _Streams()
    urls.expected = []

    def fake_stream(url: str, dest: Path, expected_sha256: str | None = None) -> None:
        urls.append(url)
        urls.expected.append(expected_sha256)
        if expected_sha256 and expected_sha256 != SHA:
            # What the real step does: nothing is left behind.
            raise CyberwaveModelIntegrityError("failed checksum verification")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(PAYLOAD)

    monkeypatch.setattr(ModelManager, "_stream_download_to", staticmethod(fake_stream))
    return urls


def _manager(
    model_dir: Path, catalog: _Catalog | None, client: Any = None
) -> ModelManager:
    manager = ModelManager(model_dir=str(model_dir))
    manager._catalog = catalog
    manager._mlmodels_client = client
    return manager


def _fetch(
    tmp_path: Path, model: Any, entries: dict[str, Any] | None = None
) -> _Catalog:
    catalog = _Catalog(entries if entries is not None else {ID: _companion()})
    _manager(tmp_path, catalog)._fetch_companions(MODEL, model)
    return catalog


def test_it_downloads_a_required_file_into_the_edge_core_layout(
    tmp_path: Path, streamed: _Streams
) -> None:
    catalog = _fetch(tmp_path, _model())

    assert (tmp_path / ID / ID).read_bytes() == PAYLOAD
    assert streamed == [URL]
    assert streamed.expected == [SHA]  # checked while it is written
    assert catalog.list_calls == [ID]  # by external id: a slug lookup would 404


def test_the_runtime_can_then_stage_it_from_the_manager_dir(
    tmp_path: Path, streamed: _Streams, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CYBERWAVE_MODELS_DIR", raising=False)
    monkeypatch.delenv("CYBERWAVE_MODEL_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    target = tmp_path / "work"
    target.mkdir()
    _fetch(tmp_path / "models", _model())

    assert companions.stage_companion_assets(target) == [target / ID]


def test_the_catalog_lookup_has_a_time_limit(
    tmp_path: Path, streamed: _Streams
) -> None:
    catalog = _fetch(tmp_path, _model())

    assert catalog.timeouts and all(t is not None and t > 0 for t in catalog.timeouts)


@pytest.mark.parametrize(
    "leftover",
    [".dl_abc123.part", ".dl_k3j9x2ab"],
    ids=["sdk-name", "old-edge-core-name"],
)
def test_a_download_killed_midway_is_not_taken_for_a_cached_companion(
    tmp_path: Path, streamed: _Streams, leftover: str
) -> None:
    (tmp_path / ID).mkdir()
    (tmp_path / ID / leftover).write_bytes(b"half")  # left by a hard kill

    catalog = _fetch(tmp_path, _model())

    assert streamed == [URL]  # fetched anyway
    assert (tmp_path / ID / ID).read_bytes() == PAYLOAD
    assert catalog.list_calls == [ID]


@pytest.mark.parametrize("leftover", [".dl_abc123.part", ".dl_k3j9x2ab"])
def test_a_leftover_temp_file_is_not_the_cached_weights_either(
    tmp_path: Path, leftover: str
) -> None:
    (tmp_path / MODEL).mkdir()
    (tmp_path / MODEL / leftover).write_bytes(b"half")

    assert _manager(tmp_path, None)._find_cached_download(MODEL) is None


def test_a_file_already_cached_is_not_downloaded_again(
    tmp_path: Path, streamed: _Streams
) -> None:
    (tmp_path / ID).mkdir()
    (tmp_path / ID / ID).write_bytes(PAYLOAD)

    catalog = _fetch(tmp_path, _model())

    assert streamed == []
    assert catalog.list_calls == []  # no catalog request, no re-hash


def test_a_corrupt_download_is_removed_and_nothing_is_raised(
    tmp_path: Path, streamed: _Streams
) -> None:
    _fetch(tmp_path, _model(), {ID: _companion(checksum_sha256="0" * 64)})

    assert not (tmp_path / ID / ID).exists()


@pytest.mark.parametrize(
    "metadata",
    [{"checksum_sha256": None}, {"download_url": URL.replace("https", "http")}],
    ids=["no-checksum", "plain-http"],
)
def test_an_entry_without_a_pinned_https_url_is_skipped(
    tmp_path: Path, streamed: _Streams, metadata: dict[str, Any]
) -> None:
    _fetch(tmp_path, _model(), {ID: _companion(**metadata)})

    assert streamed == []


def test_a_path_like_id_cannot_leave_the_model_dir(
    tmp_path: Path, streamed: _Streams
) -> None:
    models = tmp_path / "models"
    _fetch(models, _model(["../../escape"]), {"../../escape": _companion()})

    assert not (tmp_path / "escape").exists()
    assert [p for p in tmp_path.rglob(ID) if models not in p.parents] == []


def test_a_model_that_requires_nothing_makes_no_lookup(
    tmp_path: Path, streamed: _Streams
) -> None:
    catalog = _fetch(tmp_path, _model(None))

    assert catalog.list_calls == []


def test_load_with_download_fetches_the_companions_of_the_model(
    tmp_path: Path, streamed: _Streams, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = SimpleNamespace(get=lambda ref: _model())  # a slug resolves to its entry
    manager = _manager(tmp_path, _Catalog({ID: _companion()}), client)
    weights = tmp_path / "yoloe" / "checkpoint.pt"
    monkeypatch.setattr(manager, "_download_checkpoint", lambda *_: weights)
    monkeypatch.setattr(manager, "_load_from_cached_download", lambda **_: "loaded")

    assert (
        manager._load_with_download(
            model_id="acme/models/yoloe",
            runtime=None,
            device=None,
            force_download=False,
        )
        == "loaded"
    )
    assert (tmp_path / ID / ID).read_bytes() == PAYLOAD


def test_the_first_load_of_a_bare_id_fetches_what_the_catalog_says_it_requires(
    tmp_path: Path, streamed: _Streams
) -> None:
    """``cw.models.load("yoloe-26s-seg.pt")``: what generated workflows emit."""
    catalog = _Catalog({MODEL: _model(), ID: _companion()})
    manager = _manager(tmp_path, catalog)

    path = manager._resolve_model_path(MODEL, "ultralytics")

    assert path == tmp_path / MODEL  # Ultralytics downloads the weights there
    assert (tmp_path / ID / ID).read_bytes() == PAYLOAD
    assert catalog.list_calls == [MODEL, ID]


def test_weights_already_on_disk_make_no_catalog_request(
    tmp_path: Path, streamed: _Streams
) -> None:
    (tmp_path / MODEL).write_bytes(b"weights")
    catalog = _Catalog({MODEL: _model(), ID: _companion()})

    _manager(tmp_path, catalog)._resolve_model_path(MODEL, "ultralytics")

    assert catalog.list_calls == []


@pytest.mark.parametrize(
    "catalog",
    [_Catalog({}, error=ConnectionError("offline")), None],
    ids=["offline", "no-api-connection"],
)
def test_a_catalog_that_cannot_be_asked_does_not_fail_the_load(
    tmp_path: Path, streamed: _Streams, catalog: _Catalog | None
) -> None:
    path = _manager(tmp_path, catalog)._resolve_model_path(MODEL, "ultralytics")

    assert path == tmp_path / MODEL
    assert streamed == []
