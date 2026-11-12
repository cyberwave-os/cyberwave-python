"""Host SocketCAN discovery and bring-up shared by CAN-attached drivers.

Arms on a USB-CAN adapter (gs_usb and similar) need the interface selected,
set to the right bitrate and brought up before the vendor ROS node can open
it. Every such driver used to carry a copy of the vendor's ``can_activate.sh``
port; this is the one copy. Pure ``ip``/``ethtool`` subprocess calls -- no ROS,
no driver state -- so it runs during ``connect_to_device`` before anything
else exists, and it is testable with a patched ``subprocess.run``.

``sudo`` is prefixed only when not already root, because edge containers
usually run as root while bench setups do not.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess

logger = logging.getLogger(__name__)


def list_can_interfaces(*, timeout_s: float = 5.0) -> list[str]:
    """Return CAN interface names from ``ip link show type can`` (empty if none)."""
    try:
        result = subprocess.run(
            ["ip", "-json", "link", "show", "type", "can"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0 or not (result.stdout or "").strip():
        return _parse_can_interfaces_text(result.stdout or result.stderr or "")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return _parse_can_interfaces_text(result.stdout)

    if not isinstance(payload, list):
        return []

    names: list[str] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        ifname = entry.get("ifname")
        if isinstance(ifname, str) and ifname:
            names.append(ifname)
    return names


def _parse_can_interfaces_text(output: str) -> list[str]:
    names: list[str] = []
    for line in output.splitlines():
        match = re.match(r"^\d+:\s+([^:@]+)", line.strip())
        if match:
            names.append(match.group(1))
    return names


class CanSetupError(RuntimeError):
    """Raised when a CAN interface cannot be selected/activated for bring-up."""


def _maybe_sudo() -> list[str]:
    """Prefix CAN commands with sudo only when not already root."""
    try:
        return [] if os.geteuid() == 0 else ["sudo"]
    except AttributeError:  # non-POSIX host
        return []


def _run(cmd: list[str], *, timeout_s: float = 10.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_s, check=False
    )


def _iface_bus_info(iface: str) -> str | None:
    """Return ``ethtool -i`` bus-info for a CAN interface (None if unavailable)."""
    try:
        result = _run([*_maybe_sudo(), "ethtool", "-i", iface])
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    for line in (result.stdout or "").splitlines():
        if line.strip().startswith("bus-info:"):
            return line.split(":", 1)[1].strip() or None
    return None


def describe_can_ports() -> dict[str, str]:
    """Log and return ``{canX: bus-info}`` (port of ``find_all_can_port.sh``).

    Best-effort diagnostic — never raises.
    """
    mapping: dict[str, str] = {}
    for iface in list_can_interfaces():
        bus = _iface_bus_info(iface)
        if bus:
            mapping[iface] = bus
            logger.info("CAN interface %s is on USB port %s", iface, bus)
        else:
            logger.info("CAN interface %s detected (bus-info unavailable)", iface)
    return mapping


def _link_state(iface: str) -> tuple[bool, int | None]:
    """Return ``(is_up, bitrate)`` for a CAN interface via ``ip -details link show``."""
    try:
        result = _run([*_maybe_sudo(), "ip", "-details", "link", "show", iface])
    except (OSError, subprocess.TimeoutExpired):
        return (False, None)
    if result.returncode != 0:
        return (False, None)
    out = result.stdout or ""
    flags = re.search(r"<([^>]*)>", out)
    is_up = bool(flags and re.search(r"\bUP\b", flags.group(1)))
    bitrate_match = re.search(r"bitrate (\d+)", out)
    bitrate = int(bitrate_match.group(1)) if bitrate_match else None
    return (is_up, bitrate)


def _select_interface(interface: str, usb_address: str | None) -> str:
    """Pick the source CAN interface to activate (port of can_activate.sh selection)."""
    ifaces = list_can_interfaces()
    if usb_address:
        for iface in ifaces:
            if _iface_bus_info(iface) == usb_address:
                return iface
        raise CanSetupError(
            f"No CAN interface at USB address {usb_address!r} (available: {ifaces})"
        )
    if interface in ifaces:
        return interface
    if len(ifaces) == 1:
        return ifaces[0]
    raise CanSetupError(
        f"Cannot select CAN interface {interface!r}: {len(ifaces)} present {ifaces}; "
        "pass the adapter's USB address (usb_address) to disambiguate"
    )


def bring_up_can_interface(
    interface: str, bitrate: int, usb_address: str | None = None
) -> None:
    """Activate a CAN interface at ``bitrate``, renaming it to ``interface``.

    Python port of the essentials of Agilex's ``can_activate.sh``. Idempotent: if
    the selected interface is already up at the target bitrate and correctly named,
    it is left untouched. Raises :class:`CanSetupError` on selection or command
    failure.
    """
    src = _select_interface(interface, usb_address)
    is_up, current_bitrate = _link_state(src)
    if is_up and current_bitrate == bitrate and src == interface:
        logger.info("CAN %s already up at %d bps — nothing to do", interface, bitrate)
        return

    sudo = _maybe_sudo()

    def ip(*args: str) -> None:
        cmd = [*sudo, "ip", *args]
        try:
            result = _run(cmd)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CanSetupError(f"`{' '.join(cmd)}` failed: {exc}") from exc
        if result.returncode != 0:
            raise CanSetupError(
                f"`{' '.join(cmd)}` failed (exit {result.returncode}): "
                f"{(result.stderr or result.stdout).strip()}"
            )

    if is_up:
        ip("link", "set", src, "down")
    ip("link", "set", src, "type", "can", "bitrate", str(bitrate))
    if src != interface:
        ip("link", "set", src, "name", interface)
    ip("link", "set", interface, "up")
    logger.info("CAN %s activated at %d bps (source iface %s)", interface, bitrate, src)
