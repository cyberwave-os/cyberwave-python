"""Companion files: weights a runtime loads by bare filename, not through the catalog.

Ultralytics asks for YOLOE-26's text encoder ``mobileclip2_b.ts`` by name, in the
current directory first. The catalog lists it as an entry the YOLOE-26 entries
``require``, so Edge Core downloads and caches it as ``<models dir>/<id>/<filename>``.
:func:`stage_companion_assets` puts it where the runtime looks: a hard link, so there
is no second copy and, unlike a symlink to a container path, nothing dangles when the
cache is read from the host.

Outside Edge Core (a notebook running an exported workflow) nothing has cached the
file, so ``ModelManager`` downloads what the catalog entry ``requires`` itself, on the
first load of a model whose weights are not on disk -- same pinned URL, same sha256
check, same cache layout.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger("cyberwave.models.companions")

#: Files a runtime looks for by bare name. Each is both the catalog id (as in
#: ``metadata["requires"]``) and the filename, so the cache holds ``<name>/<name>``.
COMPANION_FILES: tuple[str, ...] = ("mobileclip2_b.ts",)


#: Cache roots a :class:`~cyberwave.models.manager.ModelManager` downloaded into, for
#: runtimes that stage files later without a reference to the manager.
_extra_roots: list[Path] = []


def add_model_root(root: Path) -> None:
    """Remember a cache directory the SDK itself downloaded companions into."""
    if root not in _extra_roots:
        _extra_roots.append(root)


def _model_roots() -> list[Path]:
    """Directories the model cache can be in, in the Ultralytics runtime's order."""
    roots: list[Path] = list(_extra_roots)
    for var in ("CYBERWAVE_MODELS_DIR", "CYBERWAVE_MODEL_DIR"):
        value = os.environ.get(var)
        if value:
            roots.append(Path(value))
    roots.extend([Path("/app/models"), Path.home() / ".cyberwave" / "models"])
    return roots


def _cached_file(model_id: str, filename: str) -> Path | None:
    for root in _model_roots():
        candidate = root / model_id / filename
        if candidate.is_file():
            return candidate
    return None


def _marker(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.cw-placed")


def _fingerprint(cached: Path) -> str:
    """Identify one version of a cache file: it is replaced by a new file, not rewritten.

    The inode is what always changes then; size and modification time alone can match
    when two versions are the same size and written within one clock tick.
    """
    stat = cached.stat()
    return f"{stat.st_ino} {stat.st_size} {stat.st_mtime_ns}"


def _place(source: Path, destination: Path) -> None:
    """Link (or copy) *source* to *destination*, and note which cache file it was."""
    # Unlink first so a stale link is replaced, not overwritten in place: that would
    # rewrite the old file every other model's link still points at.
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:  # another filesystem: a copy is the only way
        shutil.copy2(source, destination)
    _marker(destination).write_text(_fingerprint(source))


def _is_stale(destination: Path, cached: Path) -> bool:
    """Whether this module placed *destination* and the cache file has changed since.

    Only what this module placed carries a marker, so a file the runtime downloaded
    itself, or anyone staged by hand, is never replaced.
    """
    try:
        return _marker(destination).read_text() != _fingerprint(cached)
    except OSError:  # no marker (not ours), or the cache file is gone
        return False


def stage_companion_assets(target_dir: Path) -> list[Path]:
    """Link the cached companion files into *target_dir*; return the ones it placed.

    A file already there is left alone, unless this module placed it and the cache file
    has been replaced since: that one is placed again from the current file. A file
    that is not cached is skipped. Nothing here raises: a runtime that does not find
    its file downloads it itself.
    """
    placed: list[Path] = []
    for filename in COMPANION_FILES:
        destination = target_dir / filename
        source = _cached_file(filename, filename)
        if source is None:
            continue
        if destination.exists() and not _is_stale(destination, source):
            continue
        try:
            _place(source, destination)
        except OSError as exc:
            logger.warning(
                "Could not place %s in %s (%s); the runtime will download it itself",
                filename,
                target_dir,
                exc,
            )
            continue
        placed.append(destination)
        logger.info("Placed %s from the model cache in %s", filename, target_dir)
    return placed
