"""Companion files the model cache holds are placed where a runtime looks for them."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from cyberwave.models.companions import COMPANION_FILES, stage_companion_assets

ID = "mobileclip2_b.ts"
assert ID in COMPANION_FILES
FILENAME = ID


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setenv("CYBERWAVE_MODELS_DIR", str(root))
    monkeypatch.delenv("CYBERWAVE_MODEL_DIR", raising=False)
    # Keep /app/models and ~/.cyberwave out of the picture.
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    return root


@pytest.fixture
def target(tmp_path: Path) -> Path:
    directory = tmp_path / "yoloe-26s-seg.pt"
    directory.mkdir()
    return directory


def _cache(models_dir: Path, content: bytes = b"weights") -> Path:
    cached = models_dir / ID / FILENAME
    cached.parent.mkdir(parents=True)
    cached.write_bytes(content)
    return cached


def _refuse(*_args: object, **_kwargs: object) -> None:
    raise OSError(30, "Read-only file system")


def test_it_hard_links_the_cached_file_into_the_target(
    models_dir: Path, target: Path
) -> None:
    cached = _cache(models_dir)

    assert stage_companion_assets(target) == [target / FILENAME]

    placed = target / FILENAME
    assert placed.read_bytes() == b"weights"
    assert os.path.samefile(placed, cached)  # one file on disk, not two
    assert not placed.is_symlink()  # a link to a container path dangles on the host


def test_it_leaves_a_file_that_is_already_there(models_dir: Path, target: Path) -> None:
    _cache(models_dir)
    (target / FILENAME).write_bytes(b"mine")

    assert stage_companion_assets(target) == []
    assert (target / FILENAME).read_bytes() == b"mine"


def test_it_does_nothing_when_the_file_was_not_cached(
    models_dir: Path, target: Path
) -> None:
    assert stage_companion_assets(target) == []
    assert list(target.iterdir()) == []


def test_it_copies_when_the_cache_is_on_another_filesystem(
    models_dir: Path, target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cache(models_dir)
    monkeypatch.setattr(os, "link", _refuse)

    assert stage_companion_assets(target) == [target / FILENAME]
    assert (target / FILENAME).read_bytes() == b"weights"


def test_it_replaces_a_link_left_dangling(
    models_dir: Path, target: Path, tmp_path: Path
) -> None:
    _cache(models_dir)
    (target / FILENAME).symlink_to(tmp_path / "gone")

    assert stage_companion_assets(target) == [target / FILENAME]
    assert (target / FILENAME).read_bytes() == b"weights"


def test_a_failure_to_place_it_is_not_raised(
    models_dir: Path, target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cache(models_dir)
    monkeypatch.setattr(os, "link", _refuse)
    monkeypatch.setattr(shutil, "copy2", _refuse)

    assert stage_companion_assets(target) == []


def test_the_cache_is_found_in_the_first_root_that_has_it(
    tmp_path: Path, target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    for root, content in ((first, b"one"), (second, b"two")):
        (root / ID).mkdir(parents=True)
        (root / ID / FILENAME).write_bytes(content)
    monkeypatch.setenv("CYBERWAVE_MODELS_DIR", str(first))
    monkeypatch.setenv("CYBERWAVE_MODEL_DIR", str(second))

    stage_companion_assets(target)

    assert (target / FILENAME).read_bytes() == b"one"


def _replace(cached: Path, content: bytes) -> None:
    """Put a new file at the cache path, as a re-download does (a new file, same name)."""
    replacement = cached.with_suffix(".new")
    replacement.write_bytes(content)
    os.replace(replacement, cached)


def test_a_placed_link_follows_the_cache_file_when_it_is_replaced(
    models_dir: Path, target: Path
) -> None:
    cached = _cache(models_dir, b"v1")
    stage_companion_assets(target)
    _replace(cached, b"v2")  # the placed link still points at v1
    # No two files are told apart by the clock here: that is the next test's job.
    before = (target / FILENAME).stat().st_mtime_ns
    os.utime(cached, ns=(before, before))

    assert stage_companion_assets(target) == [target / FILENAME]

    assert (target / FILENAME).read_bytes() == b"v2"
    assert os.path.samefile(target / FILENAME, cached)


def test_a_replacement_the_same_size_and_age_is_still_noticed(
    models_dir: Path, target: Path
) -> None:
    """A new file of the same size, written within one clock tick of the old one (so the
    same modification time on a coarse clock), is a different file all the same."""
    cached = _cache(models_dir, b"v1")
    stage_companion_assets(target)
    before = cached.stat().st_mtime_ns
    _replace(cached, b"v2")
    os.utime(cached, ns=(before, before))  # as if the clock had not ticked
    assert cached.stat().st_size == 2 and cached.stat().st_mtime_ns == before

    assert stage_companion_assets(target) == [target / FILENAME]

    assert (target / FILENAME).read_bytes() == b"v2"


def test_every_model_gets_the_new_file_on_its_own_next_prompt(
    models_dir: Path, tmp_path: Path
) -> None:
    cached = _cache(models_dir, b"v1")
    folders = [tmp_path / "model-a", tmp_path / "model-b"]
    for folder in folders:
        folder.mkdir()
        stage_companion_assets(folder)
    _replace(cached, b"v22")

    for folder in folders:
        assert stage_companion_assets(folder) == [folder / FILENAME]
        assert (folder / FILENAME).read_bytes() == b"v22"


def test_a_placed_copy_is_refreshed_too_but_not_copied_again_while_current(
    models_dir: Path, target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached = _cache(models_dir, b"v1")
    monkeypatch.setattr(os, "link", _refuse)  # another filesystem: a copy

    stage_companion_assets(target)
    assert stage_companion_assets(target) == []  # unchanged cache: nothing to do
    _replace(cached, b"version two")

    assert stage_companion_assets(target) == [target / FILENAME]
    assert (target / FILENAME).read_bytes() == b"version two"


def test_a_file_this_module_did_not_place_is_never_replaced(
    models_dir: Path, target: Path
) -> None:
    cached = _cache(models_dir, b"v1")
    (target / FILENAME).write_bytes(b"downloaded by the runtime")
    _replace(cached, b"v2")

    assert stage_companion_assets(target) == []

    assert (target / FILENAME).read_bytes() == b"downloaded by the runtime"


def test_a_link_to_the_current_cache_file_is_left_alone(
    models_dir: Path, target: Path
) -> None:
    _cache(models_dir)
    stage_companion_assets(target)

    assert stage_companion_assets(target) == []
