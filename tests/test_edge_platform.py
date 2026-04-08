"""Tests for cyberwave.edge.platform — shared USB/IP detection logic."""

import subprocess
from unittest.mock import MagicMock

from cyberwave.edge import platform as edge_platform


def test_is_port_listening_returns_false_on_connect_refused(monkeypatch):
    import socket

    original_socket = socket.socket

    class FakeSocket:
        def __init__(self, *a, **kw):
            pass

        def settimeout(self, t):
            pass

        def connect_ex(self, addr):
            return 1  # connection refused

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(edge_platform.socket, "socket", FakeSocket)
    assert edge_platform.is_port_listening(3240) is False


def test_is_port_listening_returns_true_on_connect_success(monkeypatch):
    class FakeSocket:
        def __init__(self, *a, **kw):
            pass

        def settimeout(self, t):
            pass

        def connect_ex(self, addr):
            return 0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(edge_platform.socket, "socket", FakeSocket)
    assert edge_platform.is_port_listening(3240) is True


def test_is_usbip_server_running_false_on_non_darwin(monkeypatch):
    monkeypatch.setattr(edge_platform.platform, "system", lambda: "Linux")
    assert edge_platform.is_usbip_server_running() is False


def test_is_usbip_server_running_true_when_launchd_has_pid(monkeypatch):
    monkeypatch.setattr(edge_platform.platform, "system", lambda: "Darwin")
    mock_result = MagicMock(returncode=0, stdout="12345\t0\tcom.cyberwave.usbip\n")
    monkeypatch.setattr(edge_platform.subprocess, "run", lambda *a, **kw: mock_result)
    assert edge_platform.is_usbip_server_running() is True


def test_is_usbip_server_running_false_when_launchd_shows_no_pid(monkeypatch):
    monkeypatch.setattr(edge_platform.platform, "system", lambda: "Darwin")
    mock_result = MagicMock(returncode=0, stdout="-\t78\tcom.cyberwave.usbip\n")
    monkeypatch.setattr(edge_platform.subprocess, "run", lambda *a, **kw: mock_result)
    monkeypatch.setattr(edge_platform, "is_port_listening", lambda port: False)
    assert edge_platform.is_usbip_server_running() is False


def test_is_usbip_server_running_true_via_port_fallback(monkeypatch):
    monkeypatch.setattr(edge_platform.platform, "system", lambda: "Darwin")
    mock_result = MagicMock(returncode=1)
    monkeypatch.setattr(edge_platform.subprocess, "run", lambda *a, **kw: mock_result)
    monkeypatch.setattr(edge_platform, "is_port_listening", lambda port: True)
    assert edge_platform.is_usbip_server_running() is True


def test_is_usbip_server_running_false_on_nonzero_exit_and_no_port(monkeypatch):
    monkeypatch.setattr(edge_platform.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        edge_platform.subprocess,
        "run",
        lambda *a, **kw: MagicMock(returncode=1),
    )
    monkeypatch.setattr(edge_platform, "is_port_listening", lambda port: False)
    assert edge_platform.is_usbip_server_running() is False


def test_is_usbip_server_running_false_on_file_not_found(monkeypatch):
    monkeypatch.setattr(edge_platform.platform, "system", lambda: "Darwin")

    def raise_fnf(*a, **kw):
        raise FileNotFoundError("launchctl")

    monkeypatch.setattr(edge_platform.subprocess, "run", raise_fnf)
    monkeypatch.setattr(edge_platform, "is_port_listening", lambda port: False)
    assert edge_platform.is_usbip_server_running() is False


def test_is_usbip_server_running_false_on_timeout(monkeypatch):
    monkeypatch.setattr(edge_platform.platform, "system", lambda: "Darwin")

    def raise_timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="launchctl", timeout=5)

    monkeypatch.setattr(edge_platform.subprocess, "run", raise_timeout)
    monkeypatch.setattr(edge_platform, "is_port_listening", lambda port: False)
    assert edge_platform.is_usbip_server_running() is False
