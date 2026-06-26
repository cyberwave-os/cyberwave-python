"""Shared utilities for :mod:`cyberwave.driver` (Python SDK edge drivers)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path


def get_sdk_version() -> str | None:
    """Return the installed ``cyberwave`` package version.

    Returns:
        Version string (e.g. ``"0.3.20"``) or ``None`` if the package is not installed.
    """
    try:
        from cyberwave import __version__

        return __version__
    except Exception:
        return None


async def check_device_reachable_async(
    ip: str, port: int = 9991, timeout: float = 3.0
) -> bool:
    """Probe a TCP port to verify a device is reachable.

    Attempts to open a TCP connection to ``ip:port`` within ``timeout``
    seconds.  Returns ``True`` if the connection succeeds, ``False`` on any
    network error or timeout.

    Args:
        ip: IP address or hostname of the device.
        port: TCP port to probe (default: ``9991``, the Go2 WebRTC port).
        timeout: Maximum seconds to wait for the connection (default: ``3.0``).
    """
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(ip, port), timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return True
    except (OSError, asyncio.TimeoutError):
        return False


def load_driver_manifest(anchor: str | Path) -> dict:
    """Load ``driver.manifest.json`` by walking up from *anchor*.

    Pass ``__file__`` of any module inside the driver package as *anchor*.
    The search walks up the directory tree until it finds
    ``driver.manifest.json`` or reaches the filesystem root.

    Args:
        anchor: Path to a file or directory inside the driver package
                (typically ``__file__`` of the driver module).

    Returns:
        Parsed manifest as a dict.

    Raises:
        FileNotFoundError: If no ``driver.manifest.json`` is found.
    """
    current = Path(anchor).resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        candidate = directory / "driver.manifest.json"
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"driver.manifest.json not found in any parent directory of {anchor}"
    )
