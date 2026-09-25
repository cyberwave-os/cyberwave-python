"""Host platform detection utilities for Cyberwave edge components.

Shared by the CLI (install-time) and edge-core (container launch-time)
so they agree on how to detect macOS USB/IP server state.
"""

import platform
import socket
import subprocess

USBIP_LAUNCHD_LABEL = "com.cyberwave.usbip"
USBIP_PORT = 3240


def is_port_listening(port: int, host: str = "127.0.0.1", timeout: float = 1) -> bool:
    """Return True if something is accepting TCP connections on *host*:*port*."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, port)) == 0
    except OSError:
        return False


def _launchd_job_has_live_pid(stdout: str) -> bool:
    """Return True when ``launchctl list <label>`` output shows a live PID.

    ``launchctl list <label>`` (with a label argument) prints a property-list
    *dict*, e.g.::

        {
            "Label" = "com.cyberwave.usbip";
            "LastExitStatus" = 19968;
            "PID" = 4242;          # present ONLY while the job is running
        };

    A job that is merely registered — or crash-looping under ``KeepAlive`` —
    has no ``"PID"`` key (only ``"LastExitStatus"``), so we treat the job as
    running only when a numeric ``"PID"`` entry is present.

    Note: this is NOT the tabular ``PID  Status  Label`` format that
    argument-less ``launchctl list`` emits; parsing this dict as a table was
    the original bug (every dict line looked like a running row).
    """
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith('"PID"'):
            continue
        _, _, rhs = stripped.partition("=")
        value = rhs.strip().rstrip(";").strip()
        if value.isdigit():
            return True
    return False


def is_usbip_server_running() -> bool:
    """Check whether the USB/IP host server is actually usable (macOS only).

    Returns False immediately on non-Darwin platforms.

    The USB/IP port accepting connections is the authoritative signal — a
    launchd job that is only *registered* (or crash-looping) cannot serve
    devices — so we probe the port first. As a secondary positive signal
    (covers the brief window before the listener is up), we also honour a
    launchd job that reports a live PID.
    """
    if platform.system() != "Darwin":
        return False

    if is_port_listening(USBIP_PORT):
        return True

    try:
        result = subprocess.run(
            ["launchctl", "list", USBIP_LAUNCHD_LABEL],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False

    return result.returncode == 0 and _launchd_job_has_live_pid(result.stdout)
