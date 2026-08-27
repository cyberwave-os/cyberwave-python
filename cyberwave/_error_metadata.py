"""Secret-shape redaction for text and headers.

:func:`mask_sensitive_headers` runs at attachment, before headers are stapled
onto an exception, so consumers that never call ``__str__`` are covered too.
:func:`redact` runs at render on formatted text, for the two-tier alert pattern
(description for the UI, redacted detail in ``metadata.technical_detail``); it
is defence in depth only — text redaction cannot reach an attribute.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

_SECRET_PATTERN = re.compile(
    r"([A-Za-z0-9_]*(?:SECRET|TOKEN|API[_-]?KEY|PASSWORD|CREDENTIAL)[A-Za-z0-9_]*=)\S+",
    re.IGNORECASE,
)
_QUERY_STRING_PATTERN = re.compile(r"(https?://[^\s?]+)\?[^\s]*")
# Matches the width the friendly 401 handler already showed.
_TOKEN_PREVIEW_CHARS = 8

# RFC 6750 charset, so the match stops at the closing quote of a dict repr.
# Length-bounded at one over the preview width, and tied to it so the two cannot
# drift: unbounded, this rewrote an already-masked `Bearer cw_live_…` back to
# `<redacted>`, undoing the preview mask_header_value deliberately keeps, and ate
# the short word after any English "bearer" in the arbitrary driver text
# edge-core pipes through redact(). IGNORECASE stays — a bare lowercase
# `bearer <token>` in free text reaches no other rule.
_BEARER_PATTERN = re.compile(
    rf"(Bearer\s+)[A-Za-z0-9._~+/-]{{{_TOKEN_PREVIEW_CHARS + 1},}}=*",
    re.IGNORECASE,
)

_DEFAULT_LIMIT = 500

SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        "x-auth-token",
        "cookie",
    }
)

# A header dict rendered into exception text by code the SDK does not own, so
# mask_sensitive_headers never saw it. Takes the whole quoted value, keeping a
# recognised scheme: a cookie jar has secrets after its first separator, so a
# rule that stops at one would leave the rest.
_HEADER_DICT_PATTERN = re.compile(
    r"(['\"](?:" + "|".join(sorted(SENSITIVE_HEADERS)) + r")['\"]\s*:\s*)"
    r"(['\"])(Bearer\s+|Token\s+|Basic\s+)?[^'\"]*\2",
    re.IGNORECASE,
)

# Only an opaque token is safe to preview: Basic is base64 of user:pass, so 8
# characters decode to 6 real bytes of it.
_PREVIEWABLE_SCHEMES = frozenset({"bearer", "token"})

# Safe to echo: naming the scheme aids debugging and leaks nothing. Anything
# else before the first space is part of the credential, not a scheme name.
_KNOWN_SCHEMES = frozenset({"bearer", "token", "basic", "digest", "negotiate", "ntlm"})


def redact(text: str) -> str:
    """Best-effort strip of common secret shapes from raw exception/log text."""
    # First: the narrower rules below can plant a <redacted> inside a value and
    # leave the rest of it standing.
    text = _HEADER_DICT_PATTERN.sub(r"\1\2\3<redacted>\2", text)
    text = _SECRET_PATTERN.sub(r"\1<redacted>", text)
    text = _BEARER_PATTERN.sub(r"\1<redacted>", text)
    text = _QUERY_STRING_PATTERN.sub(r"\1?<redacted>", text)
    return text


def mask_header_value(value: str, *, header: str = "") -> str:
    """``"Bearer cw_live_abcdef..."`` -> ``"Bearer cw_live…"``.

    Replaced outright rather than previewed: a credential too short for a
    prefix to be safe, one carried under a scheme whose payload is structured
    (``Basic``), and a cookie jar — it leads with a value, not a scheme, so
    splitting on the first space would re-emit the first cookie verbatim.
    """
    if header.lower() == "cookie":
        return "…"
    scheme, sep, credential = value.partition(" ")
    if not sep:
        scheme, credential = "", value
    elif scheme.lower() not in _PREVIEWABLE_SCHEMES:
        # Only echo a real scheme. A value that merely contains a space would
        # otherwise emit everything before it verbatim.
        return f"{scheme} …" if scheme.lower() in _KNOWN_SCHEMES else "…"
    credential = credential.strip()
    preview = (
        credential[:_TOKEN_PREVIEW_CHARS]
        if len(credential) > _TOKEN_PREVIEW_CHARS
        else ""
    )
    return f"{scheme} {preview}…".lstrip() if scheme else f"{preview}…"


def mask_sensitive_headers(headers: Mapping[str, object] | None) -> dict[str, object]:
    """Copy ``headers`` with every credential-bearing value masked.

    Returns a plain dict, so it outlives the buffer it was copied from.
    """
    if not headers:
        return {}
    masked: dict[str, object] = {}
    for key, value in headers.items():
        # str(b"Authorization") is "b'Authorization'", which matches nothing.
        name = (
            key.decode("latin-1", "replace")
            if isinstance(key, (bytes, bytearray))
            else str(key)
        )
        if name.lower() not in SENSITIVE_HEADERS:
            masked[key] = value
        elif isinstance(value, str):
            masked[key] = mask_header_value(value, header=name)
        elif isinstance(value, (bytes, bytearray)):
            # latin-1 round-trips arbitrary bytes; this runs on an error path,
            # so a decode failure would replace the caller's real exception.
            masked[key] = mask_header_value(
                bytes(value).decode("latin-1", "replace"), header=name
            )
        else:
            # Fail closed: an unexpected type here is still a credential.
            masked[key] = "…"
    return masked


def format_error_metadata(
    error: BaseException, *, limit: int = _DEFAULT_LIMIT
) -> tuple[str, str]:
    """Return (error_code, technical_detail) for Alert.metadata."""
    error_code = type(error).__name__
    message = redact(str(error))[:limit]
    return error_code, f"{error_code}: {message}"
