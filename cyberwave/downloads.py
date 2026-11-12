"""One step for fetching a file over HTTP(S): stream it to a temp file, check it, rename it.

The SDK's model manager and Edge Core's model cache both download files this way, so the
temp-file naming, the redirect and credential handling, the proxy lookup and the
integrity check are written once. What to retry, and which source to try next, stay with
the caller: :func:`download_file` makes one attempt and says why it failed.

This is an internal module of Cyberwave's own packages (the SDK's model manager and
``cyberwave-edge-core``), not part of the SDK's documented public API: it may change
between releases, and third-party code should not import it.

Built on ``urllib3`` rather than ``urllib``: on a redirect to another host it drops the
``Authorization`` header instead of forwarding it, so a caller can send the Cyberwave
token to the backend without it following a redirect to a storage bucket.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import getproxies, proxy_bypass

import urllib3

from cyberwave.exceptions import CyberwaveModelIntegrityError

__all__ = [
    "DownloadAuthError",
    "DownloadError",
    "download_file",
    "is_partial_download",
]

#: How long a connection may sit without receiving data before the attempt fails.
_DEFAULT_TIMEOUT_SECS = 300.0
_CONNECT_TIMEOUT_SECS = 30.0
_CHUNK_SIZE = 1 << 20  # 1 MiB
_MAX_REDIRECTS = 5

#: A download in progress is a ``.dl_<random>.part`` file next to its destination. One
#: that is still there after a crash is not the file: see :func:`is_partial_download`.
_TEMP_PREFIX = ".dl_"
_TEMP_SUFFIX = ".part"


class DownloadError(RuntimeError):
    """A download failed. ``status`` is the HTTP status when the server answered."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class DownloadAuthError(DownloadError):
    """The server refused the request (401 or 403): trying again will not change that."""


def is_partial_download(name: str) -> bool:
    """Whether a file name is the temp file of a download that did not finish.

    Matches on the prefix alone, so it also catches the ``.dl_*`` files Edge Core wrote
    before it used this module.
    """
    return name.startswith(_TEMP_PREFIX)


def _redact(url: str) -> str:
    """The URL without its query string: signed URLs carry their token there."""
    return url.split("?", 1)[0]


def _pool_for(url: str) -> urllib3.PoolManager:
    """A pool for *url*, going through the proxy the environment names for it, if any.

    ``urllib3`` does not read ``HTTP_PROXY``/``HTTPS_PROXY``/``NO_PROXY`` on its own, and
    the ``urllib`` code this replaces did, so a device behind a proxy keeps working.
    """
    parts = urlsplit(url)
    proxy = getproxies().get(parts.scheme)
    if not proxy or proxy_bypass(parts.hostname or ""):
        return urllib3.PoolManager()
    userinfo = urlsplit(proxy)
    headers = None
    if userinfo.username:  # urllib3 does not turn proxy-URL credentials into a header
        headers = urllib3.make_headers(
            proxy_basic_auth=(
                f"{unquote(userinfo.username)}:{unquote(userinfo.password or '')}"
            )
        )
    return urllib3.ProxyManager(proxy, proxy_headers=headers)


def _open(
    url: str, headers: dict[str, str] | None, timeout: float
) -> urllib3.BaseHTTPResponse:
    """Send the GET and return the streaming response, or raise :class:`DownloadError`."""
    label = _redact(url)
    # Only redirects are followed here: every other failure goes back to the caller, who
    # owns the retry policy.
    retries = urllib3.Retry(
        total=None, connect=0, read=0, status=0, other=0, redirect=_MAX_REDIRECTS
    )
    try:
        response = _pool_for(url).request(
            "GET",
            url,
            headers=headers,
            preload_content=False,
            retries=retries,
            timeout=urllib3.Timeout(
                connect=min(timeout, _CONNECT_TIMEOUT_SECS), read=timeout
            ),
        )
    except urllib3.exceptions.HTTPError as exc:
        # str(exc) of a retry-exhausted error repeats the request URL, query string
        # included (a signed URL's token); its reason is the cause alone.
        reason = getattr(exc, "reason", None) or exc
        raise DownloadError(f"Could not reach {label}: {reason}") from exc
    if not 200 <= response.status < 300:
        # An error, or a 3xx that was not followed (a 300, a 304, a redirect without a
        # Location): none of them is the file. The body is not wanted either way.
        response.close()
        error = DownloadAuthError if response.status in (401, 403) else DownloadError
        raise error(f"{label} answered HTTP {response.status}", status=response.status)
    return response


def download_file(
    url: str,
    dest: Path,
    *,
    headers: dict[str, str] | None = None,
    expected_sha256: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT_SECS,
) -> str:
    """Download *url* to *dest* and return the SHA-256 of what was written.

    The body goes to a temp file next to *dest* and is renamed into place only when it
    is complete (and, with *expected_sha256*, matches), so *dest* is never half written;
    whatever happens, no temp file is left behind. One attempt: nothing is retried.

    Raises:
        DownloadAuthError: the server answered 401 or 403.
        DownloadError: any other failure to get the whole body (no connection, another
            HTTP error status, a timeout, a connection cut off midway).
        CyberwaveModelIntegrityError: *expected_sha256* was given and the body differs.
        OSError: the destination could not be written (a full disk, for one).
    """
    dest = Path(dest)
    response = _open(url, headers, timeout)  # first: a refusal creates no temp file
    digest = hashlib.sha256()
    tmp: Path | None = None
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=dest.parent, prefix=_TEMP_PREFIX, suffix=_TEMP_SUFFIX
        )
        tmp = Path(tmp_name)
        with os.fdopen(fd, "wb") as out:
            try:
                for chunk in response.stream(_CHUNK_SIZE):
                    out.write(chunk)
                    digest.update(chunk)
            except urllib3.exceptions.HTTPError as exc:
                raise DownloadError(
                    f"Download of {_redact(url)} was cut off: {exc}"
                ) from exc
        actual = digest.hexdigest()
        if expected_sha256 and actual != expected_sha256.lower():
            raise CyberwaveModelIntegrityError(
                f"{dest.name} failed checksum verification (expected "
                f"{expected_sha256[:16]}…, got {actual[:16]}…). Corrupt file removed."
            )
        os.replace(tmp, dest)
        return actual
    except BaseException:  # also an interrupt: never leave the temp file behind
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        raise
    finally:
        response.release_conn()
