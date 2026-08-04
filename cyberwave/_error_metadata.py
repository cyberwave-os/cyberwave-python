"""Shared helper for the two-tier alert pattern: high-level description for
the UI, redacted exception text in metadata.technical_detail for drill-down.
"""

from __future__ import annotations

import re

_SECRET_PATTERN = re.compile(
    r"([A-Za-z0-9_]*(?:SECRET|TOKEN|API[_-]?KEY|PASSWORD|CREDENTIAL)[A-Za-z0-9_]*=)\S+",
    re.IGNORECASE,
)
_QUERY_STRING_PATTERN = re.compile(r"(https?://[^\s?]+)\?[^\s]*")

_DEFAULT_LIMIT = 500


def redact(text: str) -> str:
    """Best-effort strip of common secret shapes from raw exception/log text."""
    text = _SECRET_PATTERN.sub(r"\1<redacted>", text)
    text = _QUERY_STRING_PATTERN.sub(r"\1?<redacted>", text)
    return text


def format_error_metadata(
    error: BaseException, *, limit: int = _DEFAULT_LIMIT
) -> tuple[str, str]:
    """Return (error_code, technical_detail) for Alert.metadata."""
    error_code = type(error).__name__
    message = redact(str(error))[:limit]
    return error_code, f"{error_code}: {message}"
