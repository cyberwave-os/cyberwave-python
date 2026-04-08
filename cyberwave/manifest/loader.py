"""YAML loading and key normalisation for cyberwave.yml manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from .schema import KNOWN_MANIFEST_FIELDS, ManifestSchema

CONFIG_FILE_NAME = "cyberwave.yml"

_MANIFEST_KEYS = ("cyberwave", "cyberwave-cloud-node")


def _extract_manifest_data(raw: dict) -> tuple[dict, str | None]:
    """Return ``(inner_dict, used_key)`` regardless of which top-level key was used.

    Tries known wrapper keys first.  Falls back to flat format only when
    every top-level key looks like a manifest field — prevents accidentally
    loading an unrelated YAML file (e.g. ``docker-compose.yml``) as a
    manifest.
    """
    for key in _MANIFEST_KEYS:
        if key in raw:
            data = raw[key]
            return (data if isinstance(data, dict) else {}), key
    if raw.keys() <= KNOWN_MANIFEST_FIELDS:
        return raw, None
    raise ValueError(
        "No 'cyberwave:' or 'cyberwave-cloud-node:' key found in the file, "
        "and the top-level keys do not match the manifest schema. "
        "Wrap your manifest under a 'cyberwave:' key."
    )


def from_dict(data: dict) -> ManifestSchema:
    """Parse and validate a manifest from a raw dictionary.

    Raises:
        pydantic.ValidationError: if the manifest fails schema validation.
        ValueError: if the wrapper key is missing and keys don't match.
    """
    manifest_data, _used_key = _extract_manifest_data(data)
    return ManifestSchema.model_validate(manifest_data)


def from_file(path: Optional[Path] = None) -> ManifestSchema:
    """Load, parse, and validate a manifest from a YAML file.

    Raises:
        FileNotFoundError: if the file does not exist.
        yaml.YAMLError: if the YAML is malformed.
        pydantic.ValidationError: if the manifest fails schema validation.
    """
    if path is None:
        path = Path.cwd() / CONFIG_FILE_NAME

    if not path.exists():
        readme_path = path.parent / "README.md"
        if readme_path.exists():
            text = readme_path.read_text()
            parts = text.split("---", 2)
            if len(parts) >= 3:
                raw = yaml.safe_load(parts[1]) or {}
                return from_dict(raw)
        raise FileNotFoundError(f"Manifest file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    return from_dict(raw)
