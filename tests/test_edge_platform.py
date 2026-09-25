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
    # Port not listening yet, but launchd reports a live PID → running.
    monkeypatch.setattr(edge_platform.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(edge_platform, "is_port_listening", lambda port: False)
    mock_result = MagicMock(returncode=0, stdout='{\n\t"PID" = 12345;\n};\n')
    monkeypatch.setattr(edge_platform.subprocess, "run", lambda *a, **kw: mock_result)
    assert edge_platform.is_usbip_server_running() is True


def test_is_usbip_server_running_false_when_launchd_shows_no_pid(monkeypatch):
    monkeypatch.setattr(edge_platform.platform, "system", lambda: "Darwin")
    mock_result = MagicMock(returncode=0, stdout='{\n\t"LastExitStatus" = 78;\n};\n')
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


# ``launchctl list <label>`` (with a label argument) prints a property-list
# *dict*, NOT the tabular ``PID  Status  Label`` form that argument-less
# ``launchctl list`` emits. The two tests below feed the real dict output.

# Real output captured from a Mac where the job is registered but the server
# crashed on start (missing binary → wrapper exits 78 → LastExitStatus 19968).
# There is no ``"PID"`` key, so the job is NOT running.
_LAUNCHCTL_DICT_DEAD = """{
\t"StandardOutPath" = "/Users/x/.cyberwave/usbip.log";
\t"LimitLoadToSessionType" = "Aqua";
\t"Label" = "com.cyberwave.usbip";
\t"OnDemand" = false;
\t"LastExitStatus" = 19968;
\t"Program" = "/Users/x/.cyberwave/usbip_wrapper.sh";
\t"ProgramArguments" = (
\t\t"/Users/x/.cyberwave/usbip_wrapper.sh";
\t);
};
"""

# A running job additionally carries a live ``"PID"`` key.
_LAUNCHCTL_DICT_RUNNING = """{
\t"StandardOutPath" = "/Users/x/.cyberwave/usbip.log";
\t"Label" = "com.cyberwave.usbip";
\t"OnDemand" = false;
\t"LastExitStatus" = 0;
\t"PID" = 4242;
\t"Program" = "/Users/x/.cyberwave/usbip_wrapper.sh";
};
"""


def test_is_usbip_server_running_false_when_registered_but_crashed(monkeypatch):
    """Registered-but-dead launchd job (no PID, only LastExitStatus) → not running.

    Regression: the previous parser treated ``launchctl list <label>`` as a
    tabular ``PID Status Label`` table, so a dict line like
    ``"LastExitStatus" = 19968;`` matched ``len(parts) >= 3`` and wrongly
    reported the crashed server as running. This trapped ``cyberwave edge
    install`` ("already running", skip rebuild) and made edge-core force a
    broken USB/IP mode.
    """
    monkeypatch.setattr(edge_platform.platform, "system", lambda: "Darwin")
    mock_result = MagicMock(returncode=0, stdout=_LAUNCHCTL_DICT_DEAD)
    monkeypatch.setattr(edge_platform.subprocess, "run", lambda *a, **kw: mock_result)
    monkeypatch.setattr(edge_platform, "is_port_listening", lambda port: False)
    assert edge_platform.is_usbip_server_running() is False


def test_is_usbip_server_running_true_when_launchd_dict_has_live_pid(monkeypatch):
    monkeypatch.setattr(edge_platform.platform, "system", lambda: "Darwin")
    mock_result = MagicMock(returncode=0, stdout=_LAUNCHCTL_DICT_RUNNING)
    monkeypatch.setattr(edge_platform.subprocess, "run", lambda *a, **kw: mock_result)
    monkeypatch.setattr(edge_platform, "is_port_listening", lambda port: False)
    assert edge_platform.is_usbip_server_running() is True
