"""Tests for :meth:`cyberwave.models.playground.PlaygroundHandle.save_annotated_image`.

The SDK offers two documented ways to archive a run result:

* ``cw.save_annotated_image(source, result, path)`` — the module-level form,
  which reflects over the whole result object.
* ``handle.save_annotated_image(source, path)`` — the method form, which
  builds its own metadata payload.

They must embed the same provenance. The method form used to assemble that
payload purely from locally-resolved state, so it silently dropped
``structured_task`` (which has no local equivalent) and reported the model the
caller *addressed* rather than the one the server actually ran.

These use ``render=False`` with a PNG source, which is the pure-Python
tEXt-chunk path — no Pillow, so the metadata contract is covered even where
the optional rendering extra is not installed.
"""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyberwave.image import read_annotated_metadata, save_annotated_image
from cyberwave.models.playground import PlaygroundHandle

# A minimal valid 1x1 PNG.
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


@pytest.fixture
def scene_png(tmp_path: Path) -> Path:
    path = tmp_path / "scene.png"
    path.write_bytes(_TINY_PNG)
    return path


def _handle_with_result(result: object, resolved: object | None) -> PlaygroundHandle:
    """Build a handle parked on ``result`` without touching the network."""
    handle = PlaygroundHandle(model_ref="acme/models/gemini-robotics-er", api=None)
    handle._resolved = resolved  # type: ignore[assignment]
    handle._last_result = result  # type: ignore[assignment]
    return handle


def test_embeds_server_reported_provenance(scene_png: Path, tmp_path: Path) -> None:
    """``structured_task`` reaches the PNG, and server identity wins."""
    result = SimpleNamespace(
        output_format="points",
        output=[{"point": [500, 500], "label": "cup"}],
        raw="[{...}]",
        status="completed",
        # The server ran a different model than the one addressed — a cascade
        # resolved to it — so these must win over ``_resolved``.
        model_uuid="server-uuid",
        model_slug="acme/models/served-by",
        structured_task="detect_points",
    )
    handle = _handle_with_result(
        result, SimpleNamespace(uuid="addressed-uuid", slug="acme/models/addressed")
    )

    out = tmp_path / "annotated.png"
    handle.save_annotated_image(scene_png, str(out), render=False)

    meta = read_annotated_metadata(out)
    assert meta is not None
    assert meta["structured_task"] == "detect_points"
    assert meta["model_uuid"] == "server-uuid"
    assert meta["model_slug"] == "acme/models/served-by"


def test_falls_back_to_locally_resolved_identity(
    scene_png: Path, tmp_path: Path
) -> None:
    """An older backend omits provenance; the resolved entry still identifies it."""
    result = SimpleNamespace(
        output_format="points",
        output=[{"point": [10, 20], "label": "cup"}],
        raw=None,
        status="completed",
    )
    handle = _handle_with_result(
        result, SimpleNamespace(uuid="addressed-uuid", slug="acme/models/addressed")
    )

    out = tmp_path / "annotated.png"
    handle.save_annotated_image(scene_png, str(out), render=False)

    meta = read_annotated_metadata(out)
    assert meta is not None
    assert meta["model_uuid"] == "addressed-uuid"
    assert meta["model_slug"] == "acme/models/addressed"
    assert meta["structured_task"] is None


def test_matches_the_module_level_helper(scene_png: Path, tmp_path: Path) -> None:
    """Both documented entry points must agree on what they embed.

    This is the regression the fix exists for: the two forms previously
    disagreed about ``structured_task`` for the very same result.
    """
    result = SimpleNamespace(
        output_format="points",
        output=[{"point": [1, 2], "label": "a"}],
        raw=None,
        status="completed",
        model_uuid="u",
        model_slug="s",
        structured_task="detect_points",
    )

    via_handle = tmp_path / "handle.png"
    _handle_with_result(result, None).save_annotated_image(
        scene_png, str(via_handle), render=False
    )

    via_module = tmp_path / "module.png"
    save_annotated_image(scene_png, result, via_module, render=False)

    handle_meta = read_annotated_metadata(via_handle)
    module_meta = read_annotated_metadata(via_module)
    assert handle_meta is not None and module_meta is not None
    for key in ("model_uuid", "model_slug", "structured_task", "output_format"):
        assert handle_meta[key] == module_meta[key], key
