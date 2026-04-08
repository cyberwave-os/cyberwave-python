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


def is_usbip_server_running() -> bool:
    """Check whether the USB/IP host server is reachable (macOS only).

    Returns False immediately on non-Darwin platforms.
    Checks launchd first, then falls back to probing the USB/IP port.
    """
    if platform.system() != "Darwin":
        return False

    try:
        result = subprocess.run(
            ["launchctl", "list", USBIP_LAUNCHD_LABEL],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                parts = line.strip().split()
                if len(parts) >= 3 and parts[0] not in ("-", "PID"):
                    return True
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass

    return is_port_listening(USBIP_PORT)
