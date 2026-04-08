"""Structured manifest validation with field-level diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml as _yaml
from pydantic import ValidationError as PydanticValidationError

from .loader import _extract_manifest_data, from_dict
from .schema import KNOWN_MANIFEST_FIELDS, ManifestSchema


@dataclass
class ManifestFieldError:
    """A single validation error scoped to a manifest field."""

    field_path: str
    message: str
    value: object = None


@dataclass
class ManifestValidationResult:
    """Outcome of :func:`validate_manifest`."""

    valid: bool
    manifest: Optional[ManifestSchema]
    errors: list[ManifestFieldError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def format_errors(self) -> str:
        if not self.errors:
            return ""
        lines = ["Manifest validation failed:"]
        for e in self.errors:
            line = f"  \u2022 {e.field_path}: {e.message}"
            if e.value is not None:
                line += f"  (got: {e.value!r})"
            lines.append(line)
        if self.warnings:
            lines.append("")
            for w in self.warnings:
                lines.append(f"  \u26a0  {w}")
        return "\n".join(lines)


def _check_legacy_key(raw: dict) -> list[str]:
    """Return a warning list if the old ``cyberwave-cloud-node:`` key is detected."""
    if "cyberwave-cloud-node" in raw and "cyberwave" not in raw:
        return [
            "Key 'cyberwave-cloud-node:' detected. "
            "Consider migrating to 'cyberwave:' for full feature support."
        ]
    return []


def validate_manifest(
    path: Optional[Path] = None,
    data: Optional[dict] = None,
    *,
    lenient: bool = False,
) -> ManifestValidationResult:
    """Validate a manifest from a file path or raw dict.

    Args:
        path: Path to ``cyberwave.yml``. Defaults to ``./cyberwave.yml``.
        data: Raw dict (overrides *path*).
        lenient: If ``True``, unknown fields are demoted to warnings
            instead of errors.

    Returns:
        A :class:`ManifestValidationResult`.
    """
    warnings: list[str] = []
    try:
        if data is not None:
            warnings = _check_legacy_key(data)
            raw = data
        else:
            if path is None:
                path = Path.cwd() / "cyberwave.yml"
            with open(path, "r") as f:
                raw = _yaml.safe_load(f) or {}
            warnings = _check_legacy_key(raw)

        if lenient:
            inner, _used_key = _extract_manifest_data(raw)
            unknown = set(inner.keys()) - KNOWN_MANIFEST_FIELDS
            for u in sorted(unknown):
                warnings.append(f"Unknown field ignored (--lenient mode): '{u}'")
            known_only = {k: v for k, v in inner.items() if k in KNOWN_MANIFEST_FIELDS}
            manifest = ManifestSchema.model_validate(known_only)
        else:
            manifest = from_dict(raw)

        return ManifestValidationResult(valid=True, manifest=manifest, warnings=warnings)

    except PydanticValidationError as exc:
        errors = [
            ManifestFieldError(
                field_path=".".join(str(loc) for loc in err["loc"]),
                message=err["msg"],
                value=err.get("input"),
            )
            for err in exc.errors()
        ]
        return ManifestValidationResult(
            valid=False, manifest=None, errors=errors, warnings=warnings
        )

    except FileNotFoundError as exc:
        return ManifestValidationResult(
            valid=False,
            manifest=None,
            errors=[ManifestFieldError(field_path="path", message=str(exc))],
            warnings=[],
        )

    except _yaml.YAMLError as exc:
        return ManifestValidationResult(
            valid=False,
            manifest=None,
            errors=[ManifestFieldError(field_path="yaml", message=f"Invalid YAML: {exc}")],
            warnings=[],
        )

    except ValueError as exc:
        return ManifestValidationResult(
            valid=False,
            manifest=None,
            errors=[ManifestFieldError(field_path="structure", message=str(exc))],
            warnings=[],
        )
