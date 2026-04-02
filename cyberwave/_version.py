"""Version helpers for the Cyberwave Python SDK."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as metadata_version

STATIC_VERSION = "0.4.0"


def get_version() -> str:
    """Prefer installed package metadata with a source fallback."""
    try:
        return metadata_version("cyberwave")
    except PackageNotFoundError:
        return STATIC_VERSION
    except Exception:
        return STATIC_VERSION
