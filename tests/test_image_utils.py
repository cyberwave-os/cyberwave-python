"""Tests for :mod:`cyberwave.image` — base64 + annotated image helpers."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest


pytest.importorskip("PIL", reason="cyberwave.image rendering needs Pillow")

from PIL import Image

from cyberwave.image import (
    decode_image_base64,
    encode_image_base64,
    read_annotated_metadata,
    save_annotated_image,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_png_bytes() -> bytes:
    """Return a small 32x32 solid-colour PNG as raw bytes."""
    img = Image.new("RGB", (32, 32), (200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def scene_png(tmp_path: Path, tiny_png_bytes: bytes) -> Path:
    path = tmp_path / "scene.png"
    path.write_bytes(tiny_png_bytes)
    return path


# ---------------------------------------------------------------------------
# Base64 helpers
# ---------------------------------------------------------------------------


class TestEncodeImageBase64:
    def test_from_path(self, scene_png: Path, tiny_png_bytes: bytes) -> None:
        encoded = encode_image_base64(scene_png)
        assert isinstance(encoded, str)
        assert base64.b64decode(encoded) == tiny_png_bytes

    def test_from_bytes(self, tiny_png_bytes: bytes) -> None:
        encoded = encode_image_base64(tiny_png_bytes)
        assert base64.b64decode(encoded) == tiny_png_bytes

    def test_from_file_like(self, tiny_png_bytes: bytes) -> None:
        encoded = encode_image_base64(io.BytesIO(tiny_png_bytes))
        assert base64.b64decode(encoded) == tiny_png_bytes

    def test_from_pil_image(self) -> None:
        img = Image.new("RGB", (8, 8), (0, 128, 255))
        encoded = encode_image_base64(img)
        assert isinstance(encoded, str)
        roundtrip = Image.open(io.BytesIO(base64.b64decode(encoded)))
        assert roundtrip.size == (8, 8)

    def test_include_data_url(self, tiny_png_bytes: bytes) -> None:
        encoded = encode_image_base64(tiny_png_bytes, include_data_url=True)
        assert encoded.startswith("data:image/png;base64,")

    def test_rejects_unsupported_type(self) -> None:
        with pytest.raises(TypeError):
            encode_image_base64(12345)  # type: ignore[arg-type]


class TestDecodeImageBase64:
    def test_roundtrip(self, tiny_png_bytes: bytes) -> None:
        encoded = base64.b64encode(tiny_png_bytes).decode("ascii")
        assert decode_image_base64(encoded) == tiny_png_bytes

    def test_strips_data_url_prefix(self, tiny_png_bytes: bytes) -> None:
        b64 = base64.b64encode(tiny_png_bytes).decode("ascii")
        url = f"data:image/png;base64,{b64}"
        assert decode_image_base64(url) == tiny_png_bytes

    def test_rejects_non_base64(self) -> None:
        with pytest.raises(ValueError):
            decode_image_base64("!!!not base64!!!")


# ---------------------------------------------------------------------------
# save_annotated_image
# ---------------------------------------------------------------------------


class TestSaveAnnotatedImage:
    def test_points_overlay_writes_png_with_metadata(
        self, scene_png: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "annotated.png"
        run_result = {
            "output_format": "points",
            "output": [
                {"point": [500, 500], "label": "cup"},
                {"point": [250, 750], "label": "mug"},
            ],
            "raw": "[{...}]",
            "status": "completed",
        }
        save_annotated_image(scene_png, run_result, out)
        assert out.exists()
        # Must still be a valid PNG.
        with Image.open(out) as img:
            assert img.format == "PNG"
            assert img.size == (32, 32)

        meta = read_annotated_metadata(out)
        assert meta is not None
        assert meta["output_format"] == "points"
        assert len(meta["output"]) == 2
        assert meta["output"][0]["label"] == "cup"
        # Preserved from the original run dict.
        assert meta["raw"] == "[{...}]"

    def test_boxes_overlay(
        self, scene_png: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "boxes.png"
        save_annotated_image(
            scene_png,
            {
                "output_format": "boxes",
                "output": [
                    {"box_2d": [100, 100, 900, 900], "label": "object"},
                ],
            },
            out,
        )
        assert out.exists()
        meta = read_annotated_metadata(out)
        assert meta is not None
        assert meta["output_format"] == "boxes"

    def test_masks_overlay_accepts_raw_and_data_url_masks(
        self, scene_png: Path, tmp_path: Path, tiny_png_bytes: bytes
    ) -> None:
        mask_b64 = base64.b64encode(tiny_png_bytes).decode("ascii")
        out = tmp_path / "masks.png"
        save_annotated_image(
            scene_png,
            {
                "output_format": "masks",
                "output": [
                    {
                        "box_2d": [0, 0, 1000, 500],
                        "mask": mask_b64,
                        "label": "left",
                    },
                    {
                        "box_2d": [0, 500, 1000, 1000],
                        "mask": f"data:image/png;base64,{mask_b64}",
                        "label": "right",
                    },
                ],
            },
            out,
        )
        assert out.exists()
        meta = read_annotated_metadata(out)
        assert meta is not None and len(meta["output"]) == 2

    def test_embed_metadata_false_writes_no_text_chunk(
        self, scene_png: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "no_meta.png"
        save_annotated_image(
            scene_png,
            {"output_format": "points", "output": [{"point": [500, 500]}]},
            out,
            embed_metadata=False,
        )
        assert read_annotated_metadata(out) is None

    def test_accepts_bare_list_with_explicit_output_format(
        self, scene_png: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "bare.png"
        save_annotated_image(
            scene_png,
            [{"point": [500, 500], "label": "x"}],
            out,
            output_format="points",
        )
        assert out.exists()

    def test_rejects_unsupported_output_format(
        self, scene_png: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError):
            save_annotated_image(
                scene_png,
                {"output_format": "mesh", "output": {}},
                tmp_path / "out.png",
            )

    def test_metadata_extra_merged(
        self, scene_png: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "extra.png"
        save_annotated_image(
            scene_png,
            {"output_format": "points", "output": []},
            out,
            metadata_extra={"prompt": "cups", "model_slug": "acme/models/gem-er"},
        )
        meta = read_annotated_metadata(out)
        assert meta is not None
        assert meta["prompt"] == "cups"
        assert meta["model_slug"] == "acme/models/gem-er"


class TestSaveAnnotatedImageRenderFalse:
    """Cheap path: ``render=False`` skips the overlay pipeline.

    The goal is to verify that (1) no drawing happens, (2) metadata still
    lands in the PNG, (3) non-renderable ``output_format`` values (text,
    free) are accepted, and (4) for PNG sources the pixel bytes are
    preserved verbatim (Pillow-free injection path).
    """

    def test_preserves_pixels_and_embeds_metadata_for_png_source(
        self, scene_png: Path, tmp_path: Path, tiny_png_bytes: bytes
    ) -> None:
        out = tmp_path / "cheap.png"
        run_result = {
            "output_format": "points",
            "output": [{"point": [500, 500], "label": "cup"}],
            "status": "completed",
        }
        save_annotated_image(scene_png, run_result, out, render=False)

        # Metadata present.
        meta = read_annotated_metadata(out)
        assert meta is not None
        assert meta["output"] == [{"point": [500, 500], "label": "cup"}]
        assert meta["status"] == "completed"

        # Pixel bytes identical to the original (we only spliced a tEXt
        # chunk, no decode/re-encode).
        with Image.open(scene_png) as a, Image.open(out) as b:
            assert a.size == b.size
            assert list(a.convert("RGB").getdata()) == list(
                b.convert("RGB").getdata()
            )

    def test_accepts_text_output_format_without_drawing(
        self, scene_png: Path, tmp_path: Path
    ) -> None:
        # ``text`` / ``free`` are not renderable but MUST be archivable via
        # render=False, because that's the whole point of the fast path.
        out = tmp_path / "caption.png"
        save_annotated_image(
            scene_png,
            {"output_format": "text", "output": "A red square on a white background."},
            out,
            render=False,
        )
        meta = read_annotated_metadata(out)
        assert meta is not None
        assert meta["output"] == "A red square on a white background."
        assert meta["output_format"] == "text"

    def test_render_false_works_without_pillow(
        self, scene_png: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PNG source + render=False must not hit ``_require_pillow``.

        This is the pipeline/low-power guarantee: no image decode, no draw,
        no re-encode. We fail the Pillow importer to make sure we never
        touch it on this path.
        """
        import cyberwave.image as img_mod

        def _boom(*_a, **_kw):
            raise AssertionError("render=False must not call _require_pillow")

        monkeypatch.setattr(img_mod, "_require_pillow", _boom)
        out = tmp_path / "no_pillow.png"
        save_annotated_image(
            scene_png,
            {"output_format": "points", "output": []},
            out,
            render=False,
        )
        meta = read_annotated_metadata(out)
        assert meta is not None

    def test_render_false_idempotent_replaces_existing_chunk(
        self, scene_png: Path, tmp_path: Path
    ) -> None:
        """Repeated save_annotated_image calls must not accumulate chunks."""
        out = tmp_path / "chained.png"
        save_annotated_image(
            scene_png,
            {"output_format": "points", "output": [{"point": [0, 0], "label": "a"}]},
            out,
            render=False,
        )
        size_after_first = out.stat().st_size

        # Re-annotate using the previous output as input.
        save_annotated_image(
            out,
            {"output_format": "points", "output": [{"point": [0, 0], "label": "b"}]},
            out,
            render=False,
        )
        size_after_second = out.stat().st_size

        # Same keyword → previous chunk replaced, not appended. File size
        # should stay stable-ish (allow small slack for payload delta).
        assert abs(size_after_second - size_after_first) < 256
        meta = read_annotated_metadata(out)
        assert meta is not None
        assert meta["output"][0]["label"] == "b"

    def test_render_false_no_metadata_is_plain_copy(
        self, scene_png: Path, tmp_path: Path, tiny_png_bytes: bytes
    ) -> None:
        out = tmp_path / "copy.png"
        save_annotated_image(
            scene_png,
            {"output_format": "text", "output": "irrelevant"},
            out,
            render=False,
            embed_metadata=False,
        )
        # No tEXt chunk written → byte-for-byte identical to the source.
        assert out.read_bytes() == tiny_png_bytes
