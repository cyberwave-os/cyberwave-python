"""``cyberwave.downloads.download_file`` against real local HTTP servers."""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from cyberwave.downloads import (
    DownloadAuthError,
    DownloadError,
    download_file,
    is_partial_download,
)
from cyberwave.exceptions import CyberwaveModelIntegrityError

BODY = b"model weights " * 4096
SHA = hashlib.sha256(BODY).hexdigest()

_PROXY_VARS = (
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
)


class _Server:
    """A local HTTP server. ``handler(request)`` returns ``(status, headers, body)``."""

    def __init__(self, handler: Callable[[BaseHTTPRequestHandler], tuple]) -> None:
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                outer.requests.append(
                    {
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                    }
                )
                status, headers, body = handler(self)
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                if "Content-Length" not in headers:
                    self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
                if headers.get("X-Cut") == "1":
                    self.close_connection = True

            def log_message(self, *args: Any) -> None:
                pass

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_port
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def url(self, path: str = "/file") -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def serve() -> Iterator[Callable[..., _Server]]:
    servers: list[_Server] = []

    def make(handler: Callable[[BaseHTTPRequestHandler], tuple]) -> _Server:
        servers.append(_Server(handler))
        return servers[-1]

    yield make
    for server in servers:
        server.close()


@pytest.fixture(autouse=True)
def _no_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _PROXY_VARS:
        monkeypatch.delenv(name, raising=False)


def _ok(_request: BaseHTTPRequestHandler) -> tuple:
    return 200, {}, BODY


def _leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name != "model.bin")


def test_it_writes_the_body_and_returns_its_sha256(tmp_path: Path, serve: Any) -> None:
    server = serve(_ok)
    dest = tmp_path / "model.bin"

    digest = download_file(server.url(), dest, expected_sha256=SHA.upper())

    assert dest.read_bytes() == BODY
    assert digest == SHA
    assert _leftovers(tmp_path) == []


def test_it_creates_missing_parent_folders(tmp_path: Path, serve: Any) -> None:
    server = serve(_ok)
    dest = tmp_path / "a" / "b" / "model.bin"

    download_file(server.url(), dest)

    assert dest.read_bytes() == BODY


def test_a_wrong_checksum_leaves_nothing_behind(tmp_path: Path, serve: Any) -> None:
    server = serve(_ok)
    dest = tmp_path / "model.bin"
    dest.write_bytes(b"the old file")

    with pytest.raises(CyberwaveModelIntegrityError):
        download_file(server.url(), dest, expected_sha256="0" * 64)

    assert dest.read_bytes() == b"the old file"  # untouched
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, DownloadAuthError),
        (403, DownloadAuthError),
        (404, DownloadError),
        (500, DownloadError),
    ],
)
def test_an_error_status_is_reported_with_its_code(
    tmp_path: Path, serve: Any, status: int, error: type[DownloadError]
) -> None:
    server = serve(lambda _r: (status, {}, b"no"))
    dest = tmp_path / "model.bin"

    with pytest.raises(error) as caught:
        download_file(server.url(), dest)

    assert caught.value.status == status
    assert type(caught.value) is error
    assert len(server.requests) == 1  # one attempt: retrying is the caller's policy
    assert not dest.exists()
    assert _leftovers(tmp_path) == []


def test_an_unreachable_server_is_a_download_error(tmp_path: Path) -> None:
    with pytest.raises(DownloadError):
        download_file("http://127.0.0.1:1/file", tmp_path / "model.bin")

    assert _leftovers(tmp_path) == []


def test_a_connection_cut_off_midway_is_a_download_error(
    tmp_path: Path, serve: Any
) -> None:
    server = serve(
        lambda _r: (
            200,
            {"Content-Length": str(len(BODY)), "X-Cut": "1"},
            BODY[: len(BODY) // 3],
        )
    )
    dest = tmp_path / "model.bin"
    dest.write_bytes(b"the old file")

    with pytest.raises(DownloadError, match="cut off"):
        download_file(server.url(), dest)

    assert dest.read_bytes() == b"the old file"
    assert _leftovers(tmp_path) == []


def test_it_follows_a_redirect(tmp_path: Path, serve: Any) -> None:
    target = serve(_ok)
    origin = serve(lambda _r: (302, {"Location": target.url("/real")}, b""))

    download_file(origin.url("/start"), tmp_path / "model.bin")

    assert (tmp_path / "model.bin").read_bytes() == BODY
    assert target.requests[0]["path"] == "/real"


@pytest.mark.parametrize("status", [300, 301, 304])
def test_an_answer_that_is_not_a_download_is_rejected(
    tmp_path: Path, serve: Any, status: int
) -> None:
    """A 3xx that was not followed (no ``Location``, or a 300/304) is not the file."""
    server = serve(lambda _r: (status, {}, b"<html>not the model</html>"))
    dest = tmp_path / "model.bin"

    with pytest.raises(DownloadError) as caught:
        download_file(server.url(), dest)

    assert caught.value.status == status
    assert not dest.exists()
    assert _leftovers(tmp_path) == []


def _slow(_request: BaseHTTPRequestHandler) -> tuple:
    time.sleep(0.8)
    return 200, {}, BODY


@pytest.mark.parametrize("failure", ["refused", "redirect-loop", "no-answer"])
def test_the_signed_query_string_never_reaches_the_error_message(
    tmp_path: Path, serve: Any, failure: str
) -> None:
    """Signed URLs carry their token in the query string: it must not be repeated in
    an error that gets logged."""
    query = "?X-Goog-Signature=SECRET123"
    kwargs: dict[str, Any] = {}
    if failure == "refused":
        url = f"http://127.0.0.1:1/f.pt{query}"
    elif failure == "redirect-loop":
        server = serve(lambda _r: (302, {"Location": f"/again{query}"}, b""))
        url = server.url(f"/again{query}")
    else:
        url = serve(_slow).url(f"/f.pt{query}")
        kwargs["timeout"] = 0.3

    with pytest.raises(DownloadError) as caught:
        download_file(url, tmp_path / "model.bin", **kwargs)

    assert "SECRET123" not in str(caught.value)
    assert "127.0.0.1" in str(caught.value)  # still says where, just not with the token


def test_a_redirect_loop_is_a_download_error(tmp_path: Path, serve: Any) -> None:
    server = serve(lambda _r: (302, {"Location": "/again"}, b""))

    with pytest.raises(DownloadError):
        download_file(server.url("/again"), tmp_path / "model.bin")


def test_the_token_is_not_forwarded_to_another_host(tmp_path: Path, serve: Any) -> None:
    """The Cyberwave token goes to the backend, never on to where it redirects."""
    target = serve(_ok)
    origin = serve(lambda _r: (302, {"Location": target.url("/real")}, b""))

    download_file(
        origin.url("/start"),
        tmp_path / "model.bin",
        headers={"Authorization": "Bearer SECRET"},
    )

    assert origin.requests[0]["headers"]["authorization"] == "Bearer SECRET"
    assert "authorization" not in target.requests[0]["headers"]


def test_a_failed_download_keeps_the_destination_folder_clean(
    tmp_path: Path, serve: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even an interrupt (not an ``Exception``) removes the temp file."""
    server = serve(_ok)
    dest = tmp_path / "model.bin"

    def interrupted(*_args: Any, **_kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("cyberwave.downloads.os.replace", interrupted)

    with pytest.raises(KeyboardInterrupt):
        download_file(server.url(), dest)

    assert not dest.exists()
    assert _leftovers(tmp_path) == []


def test_a_leftover_temp_file_is_recognised_as_a_partial_download() -> None:
    assert is_partial_download(".dl_k3j9x2ab")  # what Edge Core used to leave
    assert is_partial_download(".dl_k3j9x2ab.part")  # what this module leaves
    assert not is_partial_download("mobileclip2_b.ts")
    assert not is_partial_download("model.bin")


def test_the_environment_proxy_is_used(
    tmp_path: Path, serve: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy = serve(_ok)
    monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{proxy.port}")

    download_file("http://files.example.invalid/model.bin", tmp_path / "model.bin")

    assert (tmp_path / "model.bin").read_bytes() == BODY
    # A proxy is sent the absolute URL of the file it should fetch.
    assert proxy.requests[0]["path"] == "http://files.example.invalid/model.bin"


def test_proxy_credentials_in_the_url_are_sent_as_basic_auth(
    tmp_path: Path, serve: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy = serve(_ok)
    monkeypatch.setenv("http_proxy", f"http://user:p%40ss@127.0.0.1:{proxy.port}")

    download_file("http://files.example.invalid/model.bin", tmp_path / "model.bin")

    assert proxy.requests[0]["headers"]["proxy-authorization"] == "Basic dXNlcjpwQHNz"


def test_no_proxy_is_honoured(
    tmp_path: Path, serve: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = serve(_ok)
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")  # nothing listens here
    monkeypatch.setenv("no_proxy", "127.0.0.1")

    download_file(server.url(), tmp_path / "model.bin")

    assert (tmp_path / "model.bin").read_bytes() == BODY
