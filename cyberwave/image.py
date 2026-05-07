"""Image utilities for the Cyberwave SDK.

This module collects the small-but-tedious image bookkeeping that users
otherwise have to reimplement in every snippet:

* :func:`encode_image_base64` / :func:`decode_image_base64` turn paths, raw
  bytes, file-like objects, and PIL images into the ``image_base64`` payload
  expected by ``POST /api/v1/mlmodels/{uuid}/run`` and vice versa.
* :func:`save_annotated_image` bakes the structured output of a playground
  run (points, bounding boxes, segmentation masks) directly onto the source
  image and optionally embeds the raw JSON as a PNG ``tEXt`` chunk so
  downstream consumers (email attachments, audit logs, review tools) can
  recover the exact model output from the image alone.
* :func:`read_annotated_metadata` reads the embedded JSON back.

Pillow is used for rasterisation but is imported lazily and treated as an
optional dependency — the lightweight base64 helpers work without it.

Install with::

    pip install cyberwave[image]   # adds Pillow
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
import struct
import zlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Union

__all__ = [
    "ImageSource",
    "decode_image_base64",
    "encode_image_base64",
    "read_annotated_metadata",
    "save_annotated_image",
]


#: Types accepted by :func:`encode_image_base64` and :func:`save_annotated_image`.
#:
#: - ``str`` / :class:`pathlib.Path`: file path on disk.
#: - :class:`bytes`: already-loaded raw image bytes.
#: - file-like object with ``.read()``: e.g. an open file or ``io.BytesIO``.
#: - PIL ``Image.Image``: converted to PNG bytes before encoding.
ImageSource = Union[str, "os.PathLike[str]", bytes, bytearray, memoryview, Any]


# ---------------------------------------------------------------------------
# Base64 helpers
# ---------------------------------------------------------------------------


def _read_source_bytes(source: ImageSource) -> bytes:
    """Normalize any supported image source to raw bytes."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)

    if isinstance(source, (str, os.PathLike)):
        path = Path(os.fspath(source))
        return path.read_bytes()

    # PIL Image: duck-typed to avoid a hard dependency on Pillow.
    if hasattr(source, "save") and hasattr(source, "mode"):
        buf = io.BytesIO()
        source.save(buf, format="PNG")
        return buf.getvalue()

    # File-like / stream.
    read = getattr(source, "read", None)
    if callable(read):
        data = read()
        if isinstance(data, str):
            return data.encode("utf-8")
        return bytes(data)

    raise TypeError(
        f"Unsupported image source: {type(source)!r}. "
        f"Expected path, bytes, file-like object, or PIL.Image."
    )


def encode_image_base64(source: ImageSource, *, include_data_url: bool = False) -> str:
    """Return a base64-encoded image payload.

    Accepts file paths, raw bytes, file-like objects, or PIL images — the
    inputs you normally juggle manually when assembling a
    ``POST /mlmodels/{uuid}/run`` body.

    Args:
        source: Input image, see :data:`ImageSource`.
        include_data_url: When true, prefix the result with a
            ``data:image/png;base64,`` header so it can be embedded directly in
            HTML / markdown. The Cyberwave backend strips this header before
            decoding, so both variants are valid ``image_base64`` payloads.

    Example::

        from cyberwave.image import encode_image_base64
        payload = encode_image_base64("scene.jpg")

    """
    raw = _read_source_bytes(source)
    encoded = base64.b64encode(raw).decode("ascii")
    if include_data_url:
        return f"data:image/png;base64,{encoded}"
    return encoded


def decode_image_base64(payload: str) -> bytes:
    """Return the raw bytes for an image returned as a base64 string.

    Accepts both the bare base64 payload and a full ``data:image/...;base64,``
    URL (commonly found in segmentation ``mask`` fields).
    """
    if not isinstance(payload, str):
        raise TypeError(f"Expected str, got {type(payload)!r}")
    text = payload.strip()
    if text.startswith("data:") and "," in text:
        _, text = text.split(",", 1)
    try:
        return base64.b64decode(text, validate=True)
    except binascii.Error as e:
        raise ValueError(f"Invalid base64 image payload: {e}") from e


# ---------------------------------------------------------------------------
# Annotated image export
# ---------------------------------------------------------------------------


#: PNG ``tEXt`` chunk keyword used to embed the playground run metadata.
#:
#: ``tEXt`` keywords are limited to 1-79 ISO-8859-1 characters per the PNG
#: spec; ``cyberwave.run`` fits comfortably and is namespaced enough to avoid
#: collisions with other tooling.
_METADATA_KEY = "cyberwave.run"

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _is_png_bytes(data: bytes) -> bool:
    return len(data) >= 8 and data[:8] == _PNG_SIGNATURE


def _build_text_chunk(keyword: str, text: str) -> bytes:
    """Return a serialized PNG ``tEXt`` chunk for ``keyword``/``text``.

    Follows the PNG 1.2 spec: 4-byte big-endian length, 4-byte type, data,
    4-byte CRC-32 of type+data. Used for the Pillow-free fast path so we can
    inject metadata into an already-PNG source without re-encoding the pixels
    (and without paying the Pillow import cost when ``render=False``).
    """
    # ``tEXt`` keyword/text are Latin-1 with a single null separator, per spec.
    if len(keyword) < 1 or len(keyword) > 79:
        raise ValueError("PNG tEXt keyword must be 1-79 characters")
    body = keyword.encode("latin-1") + b"\x00" + text.encode("latin-1", errors="replace")
    chunk_type = b"tEXt"
    length = struct.pack(">I", len(body))
    crc = struct.pack(">I", zlib.crc32(chunk_type + body) & 0xFFFFFFFF)
    return length + chunk_type + body + crc


def _inject_png_text_chunk(
    png_bytes: bytes, keyword: str, text: str
) -> bytes:
    """Return a copy of ``png_bytes`` with a ``tEXt`` chunk inserted.

    Any existing ``tEXt`` chunk with the same keyword is replaced so repeated
    calls do not accumulate duplicate metadata.
    """
    if not _is_png_bytes(png_bytes):
        raise ValueError("Not a PNG payload (signature mismatch)")

    out = bytearray(png_bytes[:8])
    pos = 8
    new_chunk = _build_text_chunk(keyword, text)
    keyword_latin = keyword.encode("latin-1") + b"\x00"
    inserted = False

    while pos < len(png_bytes):
        if pos + 8 > len(png_bytes):
            # Truncated file — append remainder and bail.
            out.extend(png_bytes[pos:])
            break
        (chunk_len,) = struct.unpack(">I", png_bytes[pos : pos + 4])
        chunk_type = png_bytes[pos + 4 : pos + 8]
        chunk_end = pos + 8 + chunk_len + 4  # +4 for CRC
        chunk_data = png_bytes[pos + 8 : pos + 8 + chunk_len]

        # Skip any existing tEXt with the same keyword (replace semantics).
        if chunk_type == b"tEXt" and chunk_data.startswith(keyword_latin):
            pos = chunk_end
            continue

        # Insert our chunk right before IEND so it sits in a stable location.
        if chunk_type == b"IEND" and not inserted:
            out.extend(new_chunk)
            inserted = True

        out.extend(png_bytes[pos:chunk_end])
        pos = chunk_end

    if not inserted:
        # Malformed PNG without IEND — append anyway so metadata isn't lost.
        out.extend(new_chunk)

    return bytes(out)


def _require_pillow() -> Any:
    """Return the :mod:`PIL` module or raise a helpful error."""
    try:
        import PIL  # noqa: F401
        from PIL import Image, ImageDraw, PngImagePlugin  # noqa: F401

        return Image, ImageDraw, PngImagePlugin
    except ImportError as e:
        raise RuntimeError(
            "save_annotated_image requires Pillow. Install with: "
            "pip install 'cyberwave[image]' or pip install Pillow"
        ) from e


def _normalize_output(
    output: Any, output_format: str | None
) -> tuple[str, list[dict[str, Any]]]:
    """Normalize a playground run output into (format, items).

    The backend ``POST /mlmodels/{uuid}/run`` endpoint returns either a
    ``MLModelRunCompleted`` dict or the equivalent SDK dataclass. We also
    accept a bare list (assumed to be the ``output`` field already) as a
    convenience for users parsing provider responses manually.
    """
    # Prefer the dict shape emitted by the backend / SDK.
    if isinstance(output, Mapping):
        fmt = str(output.get("output_format") or output_format or "").strip()
        payload = output.get("output")
    else:
        # Dataclass with attributes?
        fmt_attr = getattr(output, "output_format", None)
        if fmt_attr is not None or hasattr(output, "output"):
            fmt = str(fmt_attr or output_format or "").strip()
            payload = getattr(output, "output", None)
        else:
            fmt = str(output_format or "").strip()
            payload = output

    if not fmt:
        raise ValueError(
            "Cannot determine output_format. Pass a run result dict/dataclass "
            "or provide output_format='points' | 'boxes' | 'masks' explicitly."
        )

    if fmt not in {"points", "boxes", "masks"}:
        raise ValueError(
            f"Unsupported output_format '{fmt}'. save_annotated_image renders "
            f"points, boxes, and masks."
        )

    if payload is None:
        items: list[dict[str, Any]] = []
    elif isinstance(payload, Mapping):
        # e.g. {"points": [...]} — unwrap a single list key.
        values = list(payload.values())
        items = values[0] if len(values) == 1 and isinstance(values[0], list) else [payload]
    elif isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
        items = [dict(item) if isinstance(item, Mapping) else item for item in payload]
    else:
        raise TypeError(
            f"Unexpected output payload type {type(payload).__name__}; expected list/dict."
        )

    return fmt, items


def _draw_points(draw: Any, points: list[dict[str, Any]], w: int, h: int) -> None:
    for p in points:
        y, x = p.get("point", (0, 0))
        cx = (float(x) / 1000.0) * w
        cy = (float(y) / 1000.0) * h
        r = max(4, int(min(w, h) * 0.008))
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=(244, 63, 94, 220),
            outline=(255, 255, 255, 255),
            width=2,
        )
        label = p.get("label")
        if label:
            draw.text(
                (cx + r + 4, cy - r),
                str(label),
                fill=(255, 255, 255, 255),
            )


def _draw_boxes(draw: Any, boxes: list[dict[str, Any]], w: int, h: int) -> None:
    for b in boxes:
        ymin, xmin, ymax, xmax = b.get("box_2d", (0, 0, 0, 0))
        x0 = (float(xmin) / 1000.0) * w
        y0 = (float(ymin) / 1000.0) * h
        x1 = (float(xmax) / 1000.0) * w
        y1 = (float(ymax) / 1000.0) * h
        draw.rectangle([x0, y0, x1, y1], outline=(59, 130, 246, 230), width=3)
        label = b.get("label")
        if label:
            draw.text((x0 + 4, y0 + 4), str(label), fill=(255, 255, 255, 255))


def _draw_masks(canvas: Any, masks: list[dict[str, Any]], w: int, h: int) -> None:
    # Paletted tints matching the frontend `MaskOverlay`.
    palette = [
        (244, 63, 94, 140),
        (59, 130, 246, 140),
        (16, 185, 129, 140),
        (234, 179, 8, 140),
        (168, 85, 247, 140),
        (249, 115, 22, 140),
    ]
    from PIL import Image, ImageDraw

    for idx, m in enumerate(masks):
        ymin, xmin, ymax, xmax = m.get("box_2d", (0, 0, 0, 0))
        x0 = int((float(xmin) / 1000.0) * w)
        y0 = int((float(ymin) / 1000.0) * h)
        x1 = int((float(xmax) / 1000.0) * w)
        y1 = int((float(ymax) / 1000.0) * h)
        box_w = max(1, x1 - x0)
        box_h = max(1, y1 - y0)

        mask_payload = m.get("mask")
        if not mask_payload:
            continue

        mask_bytes = decode_image_base64(mask_payload)
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L").resize(
            (box_w, box_h)
        )

        color = palette[idx % len(palette)]
        tint = Image.new("RGBA", (box_w, box_h), color)
        # Gate the tint by the mask's luminance so only the object region is
        # tinted; the rest stays transparent.
        tint.putalpha(mask_img)
        canvas.alpha_composite(tint, dest=(x0, y0))

        overlay_draw = ImageDraw.Draw(canvas)
        overlay_draw.rectangle(
            [x0, y0, x1, y1],
            outline=(color[0], color[1], color[2], 255),
            width=2,
        )
        label = m.get("label")
        if label:
            overlay_draw.text(
                (x0 + 6, y0 + 6),
                str(label),
                fill=(255, 255, 255, 255),
            )


def _build_metadata_blob(
    output: Any,
    *,
    normalized_fmt: str | None,
    normalized_items: list[dict[str, Any]] | None,
    metadata_extra: Mapping[str, Any] | None,
) -> str:
    """Return the JSON string embedded in the ``cyberwave.run`` tEXt chunk."""
    raw_output = output
    if (
        not isinstance(raw_output, (dict, list, str, int, float, bool))
        and raw_output is not None
    ):
        raw_output = getattr(raw_output, "__dict__", str(raw_output))

    meta: dict[str, Any] = {"sdk": "cyberwave-python"}
    if normalized_fmt is not None:
        meta["output_format"] = normalized_fmt
    if normalized_items is not None:
        meta["output"] = normalized_items

    if isinstance(raw_output, dict):
        # Propagate the bits that identify the run without duplicating ``output``.
        for key in (
            "raw",
            "status",
            "workload_uuid",
            "model_uuid",
            "model_slug",
            "output_format",
        ):
            if key in raw_output and key not in meta:
                meta[key] = raw_output[key]
        # If caller skipped normalization, embed whatever ``output`` they passed.
        if normalized_items is None and "output" in raw_output:
            meta["output"] = raw_output["output"]

    if metadata_extra:
        meta.update(dict(metadata_extra))

    return json.dumps(meta, ensure_ascii=True)


def save_annotated_image(
    source: ImageSource,
    output: Any,
    path: str | os.PathLike[str],
    *,
    render: bool = True,
    output_format: str | None = None,
    embed_metadata: bool = True,
    metadata_extra: Mapping[str, Any] | None = None,
) -> Path:
    """Save the source image with optional overlays + embedded run metadata.

    Supports the three structured outputs emitted by
    ``POST /mlmodels/{uuid}/run``:

    * ``points`` — ``[{"point": [y, x], "label": "..."}]`` (0-1000 scale).
    * ``boxes`` — ``[{"box_2d": [ymin, xmin, ymax, xmax], "label": "..."}]``.
    * ``masks`` — ``[{"box_2d": [...], "mask": "<base64 PNG>", "label": "..."}]``.

    When ``embed_metadata=True`` (the default) the output PNG carries a
    ``tEXt`` chunk with the raw model response so the image is
    self-describing — you can email it, archive it, and later recover the
    points / boxes / masks via :func:`read_annotated_metadata` with no
    companion JSON file.

    **Rendering is opt-out.** Set ``render=False`` to skip the overlay
    pipeline entirely when you only need the embedded metadata (e.g. an audit
    log that will be rendered on demand, or a free-prompt / caption result
    with no 2D geometry to draw). Benefits of ``render=False``:

    * No Pillow import when the source is already a PNG — pure-Python
      ``tEXt`` chunk injection preserves the original bytes and avoids a
      decode/re-encode pass.
    * Works for ``output_format`` values that aren't visually renderable
      (``text``, ``free``).
    * Significantly faster, lower memory, lower CO2. Use it in pipelines.

    Args:
        source: Original image (path / bytes / PIL.Image).
        output: Result of ``cw.mlmodels.run(...)`` (dict or dataclass) OR a
            bare list of points/boxes/masks with ``output_format`` passed
            explicitly.
        path: Destination file path (``.png``).
        render: When ``True`` (default) draw points/boxes/masks onto the
            image. When ``False`` just copy the source (no Pillow needed for
            PNG inputs) and embed the metadata.
        output_format: Required only when ``output`` is a bare list AND
            ``render=True``.
        embed_metadata: Whether to write the raw output as a PNG ``tEXt``
            chunk keyed ``cyberwave.run``. When ``render=False`` and this is
            also ``False`` the call is a no-op except for the file copy.
        metadata_extra: Optional extra fields merged into the embedded
            metadata (e.g. model UUID, prompt, timestamp).

    Returns:
        The :class:`pathlib.Path` the image was written to.

    Raises:
        RuntimeError: if Pillow is required but not installed.
        ValueError: if ``output`` / ``output_format`` are incompatible.

    Examples::

        # Full render (drawn overlays + embedded JSON).
        cw.save_annotated_image("scene.jpg", result, "scene.annotated.png")

        # Cheap path: just archive the source with embedded metadata.
        cw.save_annotated_image("scene.png", result, "scene.archived.png",
                                render=False)
    """
    out_path = Path(os.fspath(path))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ----- Fast path: no rasterization requested. ---------------------------
    if not render:
        img_bytes = _read_source_bytes(source)
        metadata_blob = (
            _build_metadata_blob(
                output,
                normalized_fmt=None,
                normalized_items=None,
                metadata_extra=metadata_extra,
            )
            if embed_metadata
            else None
        )

        # Zero-Pillow path for PNG sources: just splice a tEXt chunk.
        if _is_png_bytes(img_bytes):
            payload = (
                _inject_png_text_chunk(img_bytes, _METADATA_KEY, metadata_blob)
                if metadata_blob is not None
                else img_bytes
            )
            out_path.write_bytes(payload)
            return out_path

        # Non-PNG source: need Pillow to convert to PNG (so the tEXt chunk has
        # a valid carrier). Still skips the overlay draw, which is the
        # expensive part for large images with many detections.
        Image, _ImageDraw, PngImagePlugin = _require_pillow()
        base_img = Image.open(io.BytesIO(img_bytes))
        pnginfo = None
        if metadata_blob is not None:
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text(_METADATA_KEY, metadata_blob)
        base_img.convert("RGB").save(out_path, format="PNG", pnginfo=pnginfo)
        return out_path

    # ----- Full path: render overlays. --------------------------------------
    Image, ImageDraw, PngImagePlugin = _require_pillow()

    fmt, items = _normalize_output(output, output_format)

    img_bytes = _read_source_bytes(source)
    base_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    w, h = base_img.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if fmt == "points":
        _draw_points(ImageDraw.Draw(overlay), items, w, h)
    elif fmt == "boxes":
        _draw_boxes(ImageDraw.Draw(overlay), items, w, h)
    elif fmt == "masks":
        _draw_masks(overlay, items, w, h)

    annotated = Image.alpha_composite(base_img, overlay)

    pnginfo = None
    if embed_metadata:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text(
            _METADATA_KEY,
            _build_metadata_blob(
                output,
                normalized_fmt=fmt,
                normalized_items=items,
                metadata_extra=metadata_extra,
            ),
        )

    annotated.convert("RGB").save(out_path, format="PNG", pnginfo=pnginfo)
    return out_path


def _extract_png_text_chunk(png_bytes: bytes, keyword: str) -> str | None:
    """Return the ``tEXt`` payload for ``keyword`` in a raw PNG, or ``None``.

    Pure-Python scan so :func:`read_annotated_metadata` works without Pillow
    — mirrors the write-side :func:`_inject_png_text_chunk`.
    """
    if not _is_png_bytes(png_bytes):
        return None

    keyword_latin = keyword.encode("latin-1") + b"\x00"
    pos = 8
    while pos + 8 <= len(png_bytes):
        (chunk_len,) = struct.unpack(">I", png_bytes[pos : pos + 4])
        chunk_type = png_bytes[pos + 4 : pos + 8]
        data_start = pos + 8
        data_end = data_start + chunk_len
        if data_end + 4 > len(png_bytes):
            return None  # Truncated.
        if chunk_type == b"tEXt":
            chunk_data = png_bytes[data_start:data_end]
            if chunk_data.startswith(keyword_latin):
                return chunk_data[len(keyword_latin) :].decode(
                    "latin-1", errors="replace"
                )
        if chunk_type == b"IEND":
            return None
        pos = data_end + 4  # skip CRC
    return None


def read_annotated_metadata(
    path: str | os.PathLike[str],
) -> dict[str, Any] | None:
    """Return the metadata embedded by :func:`save_annotated_image`.

    Returns ``None`` when the PNG does not carry a ``cyberwave.run``
    tEXt chunk (e.g. a plain image the user forwarded through a tool that
    stripped the metadata).

    Pillow is not required for PNG inputs — the pure-Python scanner handles
    the common case so the full ``save → read`` round-trip stays zero-dep.
    For non-PNG inputs (JPEG etc.) we fall back to Pillow to parse container
    metadata.
    """
    p = Path(os.fspath(path))
    raw = p.read_bytes()

    # Fast path: PNG with a tEXt chunk we wrote ourselves.
    if _is_png_bytes(raw):
        payload = _extract_png_text_chunk(raw, _METADATA_KEY)
    else:
        Image, _, _ = _require_pillow()
        with Image.open(io.BytesIO(raw)) as img:
            text = getattr(img, "text", None) or {}
            info = getattr(img, "info", {}) or {}
        payload = text.get(_METADATA_KEY) or info.get(_METADATA_KEY)

    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {"_raw": payload}
