"""Host-level system metric readers for Cyberwave edge devices.

This module centralises the parsing of ``/proc/meminfo`` and the Linux
``/sys/class/thermal`` sysfs so callers get consistent host memory and
CPU temperature readings across platforms.

Design notes:
- The dataclasses returned here are **raw** data carriers; they do not
  encode threshold/severity semantics.  Callers that care about
  thresholds wrap them with their own severity logic.
- The *dynamic* readers (:func:`read_host_memory`,
  :func:`read_host_cpu_temperature`) are Linux-only — they parse procfs
  and sysfs.  Callers should treat absence as "metric unknown", not as an
  error.
- The *facts* reader (:func:`read_host_facts`) is cross-platform: on
  macOS it falls back to ``sysctl`` for RAM, CPU model and (logical)
  CPU count, and to ``ifconfig`` for network interfaces.  Thermal source
  is left ``None`` on macOS since no live publisher samples it there.
- :func:`read_host_facts` also carries a *copy* of the memory and
  temperature gauges (see :class:`HostFacts`).  This is deliberate
  redundancy: the 5 s heartbeat that normally publishes them is stopped
  once drivers take over, and without the copy the dashboard loses those
  readings entirely for the rest of the session.
"""

from __future__ import annotations

import math
import os
import platform
import shutil
import socket
import struct
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Optional
from uuid import getnode


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Thermal zone ``type`` values that identify CPU sensors.  Anything matching
#: one of these (or whose ``type`` contains the substring ``"cpu"``) is
#: treated as a CPU thermal zone; everything else is a fallback candidate.
CPU_THERMAL_ZONE_TYPES = frozenset(
    {
        "x86_pkg_temp",
        "coretemp",
        "cpu-thermal",
        "cpu_thermal",
        "soc_thermal",
    }
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostMemoryInfo:
    """Snapshot of host memory usage parsed from ``/proc/meminfo``."""

    total_mb: float
    available_mb: float
    used_percent: float


@dataclass(frozen=True)
class HostCpuTemperature:
    """A CPU temperature reading from sysfs, in degrees Celsius."""

    celsius: float
    source: str


@dataclass(frozen=True)
class HostPowerDraw:
    """Instantaneous board power draw in milliwatts.

    ``source`` names the sysfs strategy that produced the value (e.g.
    ``"ina3221x:in_power0"`` for a Jetson VDD_IN rail vs
    ``"hwmon:power1_input x2"`` for a summed multi-rail reading) so
    operators can tell aggregate draw from a partial per-rail reading.
    """

    milliwatts: float
    source: str


@dataclass(frozen=True)
class NetworkInterfaceFacts:
    """A single non-loopback network interface (sysfs on Linux, ``ifconfig`` on macOS).

    ``ipv4_address`` is ``None`` when the interface has no IPv4 address
    bound (typically because it's down). Surfaced so a support engineer
    can find the current LAN-facing IP of a device without needing to
    scan the network — see :func:`read_network_interfaces`.

    ``mac_address`` stays populated on a down interface: it is a hardware
    property, and it is what a DHCP reservation keys on to give that
    interface a stable address in the first place.
    """

    name: str
    ipv4_address: Optional[str]
    mac_address: Optional[str]
    is_up: bool


@dataclass(frozen=True)
class HostFacts:
    """Host-level facts about the edge device, plus a coarse gauge copy.

    Most of these properties change rarely (RAM never, CPU model never,
    kernel only on upgrade) so they belong on the device's persistent
    identity record rather than on every ~5 s MQTT heartbeat.

    Some fields are exceptions and are marked as such below:
    ``network_interfaces`` (DHCP/WiFi changes), and the ``cpu_temp_c`` /
    ``memory_used_percent`` / ``memory_available_mb`` / ``power_mw`` gauges, which
    duplicate what the dynamic readers (:func:`read_host_memory`,
    :func:`read_host_cpu_temperature`) publish on the heartbeat.  The
    duplication exists because that heartbeat is not always running; see
    the field comment for the full rationale.

    Optional fields may be ``None`` when the underlying source is
    unavailable on the current platform.  Coverage matrix:

    - Linux: every field is populated when the corresponding
      procfs/sysfs source is readable.  ``cpu_count`` is the logical
      CPU count (one ``processor:`` entry per SMT thread).
      ``network_interfaces`` lists non-loopback interfaces discovered
      under ``/sys/class/net`` (see :func:`read_network_interfaces`).
      ``device_serial`` comes from ``/proc/cpuinfo`` (Raspberry Pi) or
      the device tree (Jetson).
    - macOS: ``memory_total_mb``, ``cpu_model`` and ``cpu_count``
      (logical, from ``hw.logicalcpu``) come from ``sysctl``, while
      ``network_interfaces`` comes from ``ifconfig``.
      ``thermal_source`` and ``cpu_temp_c`` are always ``None`` — there
      is no temperature reader on Darwin, and a "would-be source" string
      would be misleading.  ``memory_used_percent`` and
      ``memory_available_mb`` are also ``None``: ``sysctl`` gives the
      total but not the free/available split, which needs
      ``vm_stat``-style accounting we do not do yet.
      ``has_hardware_watchdog`` is always ``False``, and
      ``device_serial`` is always ``None``.
    - Other platforms: only ``platform``, ``kernel`` and
      ``has_hardware_watchdog`` are guaranteed; ``primary_mac_address``
      may be available from :func:`uuid.getnode`, and
      ``network_interfaces`` is ``()``.

    ``platform`` is always populated since :mod:`platform` works
    cross-platform.

    Software identity (``sdk_version``, ``edge_core_version``) reflects
    the version effectively in use by the calling process — in-process
    ``__version__`` first (which honors CI ``BUILD_VERSION`` stamps even
    when ``.dist-info`` is stripped, as in PyInstaller builds), falling
    back to :mod:`importlib.metadata`.  Each is ``None`` when the
    corresponding package is not loaded in the calling process.

    ``cli_version`` is intentionally **not** part of this schema: the
    CLI ships as a separate PyInstaller binary on production edges, so
    edge-core's Python process cannot observe its ``__version__`` and
    ``importlib.metadata`` does not see the standalone binary either.
    Surfacing it would require subprocess-probing ``cyberwave --version``
    from edge-core, which is out of scope for the host_facts uploader.
    """

    platform: str
    kernel: Optional[str]
    memory_total_mb: Optional[float]
    cpu_model: Optional[str]
    cpu_count: Optional[int]
    thermal_source: Optional[str]
    has_hardware_watchdog: bool
    # Software identity: which Cyberwave packages are running on this host.
    # Each is the version effectively in use by the calling process or
    # ``None`` when the package is not loaded.  This matters for rollout
    # tracking on the dashboard ("how many edges are still on the
    # previous SDK/edge-core release").
    sdk_version: Optional[str]
    edge_core_version: Optional[str]
    # Non-loopback network interfaces (Linux only). Unlike the fields
    # above, this one legitimately changes over the device's lifetime
    # (DHCP renewal, switching WiFi networks) -- it rides the same 30 s
    # keepalive cadence as everything else here so a changed IP is
    # reflected within one cycle. Defaults to `()` on platforms/kernels
    # where sysfs enumeration isn't available.
    network_interfaces: tuple["NetworkInterfaceFacts", ...] = ()
    # Hardware identity. Unlike ``network_interfaces`` these are single
    # values an operator copies somewhere else -- a DHCP reservation, an
    # RMA form, a support thread -- so they are surfaced independently of
    # per-interface enumeration and survive its absence.
    #
    # ``primary_mac_address`` is the MAC of the interface the kernel would
    # actually route out of (see :func:`read_primary_interface`), so it is
    # the one to put in a DHCP reservation and consumers need no selection
    # logic of their own. ``primary_interface_name`` names that interface.
    #
    # When nothing was enumerated it falls back to :func:`uuid.getnode`,
    # which is not tied to a reachable interface -- on macOS it routinely
    # returns an Apple-internal NIC that routes nothing. That case is
    # distinguishable: ``primary_interface_name`` is ``None``.
    # ``None`` when no hardware address is readable at all.
    primary_mac_address: Optional[str] = None
    # Name of the interface ``primary_mac_address`` was taken from, so a
    # consumer can render one authoritative ``name ip (mac)`` entry instead
    # of the whole inventory. ``None`` when the MAC came from the
    # ``uuid.getnode()`` last resort and belongs to no enumerated interface.
    primary_interface_name: Optional[str] = None
    # Board serial: Raspberry Pi via ``/proc/cpuinfo``, Jetson via the
    # device tree. ``None`` everywhere else, including macOS.
    device_serial: Optional[str] = None
    # Live gauges, duplicated from the dynamic readers. Not facts, and
    # not here for convenience: edge-core stops its bootstrap edge_health
    # publisher as soon as the first driver starts, and a containerised
    # driver cannot read the host's thermal sysfs or /proc/meminfo. So
    # after that handover these readings have no other route off the
    # device, and the dashboard would show a gap for the rest of the
    # session. They ride the 30 s keepalive instead of the 5 s heartbeat,
    # so treat them as a trend, not a sample. Consumers that may have
    # both should prefer whichever is fresher: the heartbeat's copy only
    # while heartbeats are still arriving, since a consumer holding the
    # last-received payload keeps those numbers frozen once they stop.
    cpu_temp_c: Optional[float] = None
    memory_used_percent: Optional[float] = None
    memory_available_mb: Optional[float] = None
    # Same handover rationale as the gauges above: a reading that vanishes
    # when the first driver starts reads as a fault. ``battery_wh`` is static
    # pack config, carried because it is only useful paired with ``power_mw``.
    power_mw: Optional[float] = None
    battery_wh: Optional[float] = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly dict, omitting keys whose source was unavailable.

        ``platform`` and ``has_hardware_watchdog`` are always present;
        every other key is included only when it is not ``None``.  This
        keeps ``Edge.metadata['host_facts']`` from accumulating ``null``
        sentinels on platforms that simply do not expose the source.
        """
        out: dict[str, object] = {
            "platform": self.platform,
            "has_hardware_watchdog": self.has_hardware_watchdog,
        }
        if self.kernel is not None:
            out["kernel"] = self.kernel
        if self.memory_total_mb is not None:
            out["memory_total_mb"] = self.memory_total_mb
        if self.cpu_model is not None:
            out["cpu_model"] = self.cpu_model
        if self.cpu_count is not None:
            out["cpu_count"] = self.cpu_count
        if self.thermal_source is not None:
            out["thermal_source"] = self.thermal_source
        if self.cpu_temp_c is not None:
            out["cpu_temp_c"] = self.cpu_temp_c
        if self.memory_used_percent is not None:
            out["memory_used_percent"] = self.memory_used_percent
        if self.memory_available_mb is not None:
            out["memory_available_mb"] = self.memory_available_mb
        if self.power_mw is not None:
            out["power_mw"] = self.power_mw
        if self.battery_wh is not None:
            out["battery_wh"] = self.battery_wh
        if self.sdk_version is not None:
            out["sdk_version"] = self.sdk_version
        if self.edge_core_version is not None:
            out["edge_core_version"] = self.edge_core_version
        if self.primary_mac_address is not None:
            out["primary_mac_address"] = self.primary_mac_address
        if self.primary_interface_name is not None:
            out["primary_interface_name"] = self.primary_interface_name
        if self.device_serial is not None:
            out["device_serial"] = self.device_serial
        if self.network_interfaces:
            out["network_interfaces"] = [
                {
                    "name": nic.name,
                    "ipv4_address": nic.ipv4_address,
                    "mac_address": nic.mac_address,
                    "is_up": nic.is_up,
                }
                for nic in self.network_interfaces
            ]
        return out


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def read_host_memory() -> Optional[HostMemoryInfo]:
    """Parse ``/proc/meminfo`` and return a memory snapshot.

    Returns ``None`` on non-Linux platforms, when ``/proc/meminfo`` is
    unreadable, or when ``MemTotal`` is missing/zero.  Falls back to
    ``MemFree + Buffers + Cached`` when ``MemAvailable`` is absent
    (kernels older than 3.14).
    """
    if platform.system() != "Linux":
        return None

    try:
        fields: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0].rstrip(":") in (
                    "MemTotal",
                    "MemAvailable",
                    "MemFree",
                    "Buffers",
                    "Cached",
                ):
                    fields[parts[0].rstrip(":")] = int(parts[1])
    except OSError:
        return None

    total_kb = fields.get("MemTotal", 0)
    if total_kb == 0:
        return None

    available_kb = fields.get("MemAvailable")
    if available_kb is None:
        available_kb = (
            fields.get("MemFree", 0)
            + fields.get("Buffers", 0)
            + fields.get("Cached", 0)
        )

    total_mb = total_kb / 1024.0
    available_mb = available_kb / 1024.0
    used_percent = (1.0 - available_mb / total_mb) * 100.0

    return HostMemoryInfo(
        total_mb=round(total_mb, 1),
        available_mb=round(available_mb, 1),
        used_percent=round(used_percent, 1),
    )


# ---------------------------------------------------------------------------
# CPU thermal
# ---------------------------------------------------------------------------


def discover_cpu_thermal_zones(
    thermal_base: Optional[Path] = None,
) -> list[Path]:
    """Return sysfs ``temp`` file paths for CPU-like thermal zones.

    Enumerates ``/sys/class/thermal/thermal_zone*`` (or the directory given
    by ``thermal_base``, mostly useful for testing) and prefers zones whose
    ``type`` matches :data:`CPU_THERMAL_ZONE_TYPES` or contains the
    substring ``"cpu"``.  When no CPU-typed zone is present, falls back to
    every readable zone so callers still get a temperature reading on
    non-standard kernels.

    Returns an empty list on non-Linux platforms or when no zones are
    discoverable.
    """
    if platform.system() != "Linux":
        return []

    base = thermal_base if thermal_base is not None else Path("/sys/class/thermal")
    if not base.exists():
        return []

    cpu_zones: list[Path] = []
    all_zones: list[Path] = []

    for zone in sorted(base.glob("thermal_zone*")):
        temp_file = zone / "temp"
        if not temp_file.exists():
            continue
        all_zones.append(temp_file)

        type_file = zone / "type"
        if not type_file.exists():
            continue
        try:
            zone_type = type_file.read_text().strip().lower()
        except OSError:
            continue
        if zone_type in CPU_THERMAL_ZONE_TYPES or "cpu" in zone_type:
            cpu_zones.append(temp_file)

    return cpu_zones if cpu_zones else all_zones


def read_thermal_zone_celsius(temp_path: Path) -> Optional[float]:
    """Read a single sysfs thermal zone temperature file (millidegrees → C)."""
    try:
        raw = temp_path.read_text().strip()
        return int(raw) / 1000.0
    except (OSError, ValueError):
        return None


def read_host_cpu_temperature(
    thermal_base: Optional[Path] = None,
) -> Optional[HostCpuTemperature]:
    """Return the hottest CPU thermal zone reading.

    Multi-core SoCs expose one zone per core/cluster, so we want the
    worst-case reading.  Returns ``None`` on non-Linux platforms or when
    no thermal zone is readable.
    """
    zones = discover_cpu_thermal_zones(thermal_base=thermal_base)
    if not zones:
        return None

    candidates: list[tuple[float, str]] = []
    for temp_file in zones:
        celsius = read_thermal_zone_celsius(temp_file)
        if celsius is None:
            continue
        zone_dir = temp_file.parent
        zone_type = ""
        type_file = zone_dir / "type"
        if type_file.exists():
            try:
                zone_type = type_file.read_text().strip().lower()
            except OSError:
                zone_type = ""
        source = zone_dir.name if not zone_type else f"{zone_dir.name}:{zone_type}"
        candidates.append((celsius, source))

    if not candidates:
        return None

    celsius, source = max(candidates, key=lambda item: item[0])
    return HostCpuTemperature(celsius=round(celsius, 1), source=source)


# ---------------------------------------------------------------------------
# Power draw
# ---------------------------------------------------------------------------

#: Ordered ``(glob, µW→mW divisor, source_label)`` strategies for
#: :func:`read_host_power_draw`.  The Jetson ``ina3221x`` VDD_IN rail
#: already reports milliwatts (divisor 1); the ``hwmon`` and
#: ``power_supply`` fallbacks report microwatts.  Ordering picks
#: total-board sensors over per-cell battery gauges.
_POWER_PROBE_STRATEGIES: tuple[tuple[str, float, str], ...] = (
    (
        "bus/i2c/drivers/ina3221x/*/iio:device*/in_power0_input",
        1.0,
        "ina3221x:in_power0",
    ),
    ("class/hwmon/hwmon*/power1_input", 1000.0, "hwmon:power1_input"),
    ("class/power_supply/BAT*/power_now", 1000.0, "power_supply:power_now"),
)


def read_host_power_draw(
    power_sysfs_base: Optional[Path] = None,
) -> Optional[HostPowerDraw]:
    """Return the instantaneous board power draw, or ``None`` if unavailable.

    Probes :data:`_POWER_PROBE_STRATEGIES` in order; the first strategy
    with at least one readable value wins.  Within a strategy, matching
    paths are summed so multi-rail hosts report total draw.
    ``power_sysfs_base`` is a test seam (defaults to ``/sys``).
    """
    if platform.system() != "Linux":
        return None

    base = power_sysfs_base if power_sysfs_base is not None else Path("/sys")
    if not base.exists():
        return None

    for glob_pattern, divisor, source_label in _POWER_PROBE_STRATEGIES:
        matches = sorted(base.glob(glob_pattern))
        if not matches:
            continue

        total_mw = 0.0
        readable_count = 0
        for path in matches:
            try:
                raw = path.read_text().strip()
                value = float(raw) / divisor
            except (OSError, ValueError):
                continue
            total_mw += value
            readable_count += 1

        if readable_count == 0:
            continue

        source = (
            source_label if readable_count == 1 else f"{source_label} x{readable_count}"
        )
        return HostPowerDraw(milliwatts=round(total_mw, 1), source=source)

    return None


# ---------------------------------------------------------------------------
# Static host facts
# ---------------------------------------------------------------------------

#: ``/dev/watchdog`` path consulted by :func:`read_host_facts`.  Exposed as a
#: module attribute so tests can monkeypatch it via :class:`pathlib.Path`.
HARDWARE_WATCHDOG_DEVICE = "/dev/watchdog"

#: Interface names never worth reporting -- loopback has no bearing on
#: reaching a device over the network.
_NETWORK_INTERFACE_SKIP = frozenset({"lo"})

#: Interface *name prefixes* to drop on Darwin.  A stock Mac lists ~30
#: interfaces where Linux lists two or three: VPN tunnels (``utun``),
#: AirDrop/AWDL, Thunderbolt bridges, and Apple-internal NICs (``anpi``)
#: that carry a real MAC but never route traffic anywhere.  None of them
#: help an operator reach the device, and enumerating them all would bury
#: the one or two physical NICs that matter.
#:
#: This covers only the families Apple names consistently.  The internal
#: ``anpi`` *peers* and the Thunderbolt *ports* are called ``en2``-``en9``
#: -- the same prefix as the real NICs -- so they cannot be excluded here;
#: :func:`_read_darwin_network_interfaces` drops those on ``media: none``
#: and bridge membership instead.
_DARWIN_VIRTUAL_INTERFACE_PREFIXES = (
    "anpi",
    "ap",
    "awdl",
    "bridge",
    "gif",
    "llw",
    "lo",
    "stf",
    "utun",
    "vlan",
    "vmenet",
)

#: Serial-number sources, in precedence order.  Raspberry Pi kernels put a
#: ``Serial`` line in ``/proc/cpuinfo``; Tegra (Jetson) kernels do not and
#: expose it through the device tree instead.
_CPUINFO_PATH = "/proc/cpuinfo"
_DEVICE_TREE_SERIAL_PATH = "/sys/firmware/devicetree/base/serial-number"

#: Serials some boards emit instead of omitting the field. Treated as absent.
_PLACEHOLDER_SERIALS = frozenset({"", "0", "0000000000000000"})

#: ``SIOCGIFADDR`` ioctl request number (Linux ``<linux/sockios.h>``), used
#: by :func:`_read_ipv4_address` to read an interface's IPv4 address
#: without shelling out or adding a third-party dependency.
_SIOCGIFADDR = 0x8915


def _read_ipv4_address(ifname: str) -> Optional[str]:
    """Return the IPv4 address bound to ``ifname``, or ``None``.

    ``None`` covers both "interface is down" and "interface has no IPv4
    address" -- callers shouldn't distinguish those cases, since either
    way there's nothing to SSH to.
    """
    try:
        import fcntl  # POSIX-only; deferred so importing this module never

        # fails on a non-POSIX platform that merely calls unrelated readers.
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            packed = fcntl.ioctl(
                sock.fileno(),
                _SIOCGIFADDR,
                struct.pack("256s", ifname[:15].encode("utf-8")),
            )
        finally:
            sock.close()
        return socket.inet_ntoa(packed[20:24])
    except (ImportError, OSError):
        return None


def _read_network_interface_facts(
    ifname: str, net_base: Path
) -> NetworkInterfaceFacts:
    """Build :class:`NetworkInterfaceFacts` for a single ``/sys/class/net`` entry."""
    mac_address: Optional[str] = None
    try:
        mac_address = (net_base / ifname / "address").read_text().strip() or None
    except OSError:
        mac_address = None

    is_up = False
    try:
        is_up = (net_base / ifname / "operstate").read_text().strip().lower() == "up"
    except OSError:
        is_up = False

    return NetworkInterfaceFacts(
        name=ifname,
        ipv4_address=_read_ipv4_address(ifname),
        mac_address=mac_address,
        is_up=is_up,
    )


def _ifconfig() -> Optional[str]:
    """Return raw ``ifconfig -a`` output on Darwin.  ``None`` on any error.

    Guarded with :func:`shutil.which` for the same reason as :func:`_sysctl`:
    a stripped-down environment must degrade, not raise.  This is the test
    seam for :func:`_read_darwin_network_interfaces`.
    """
    if not shutil.which("ifconfig"):
        return None
    try:
        out = subprocess.run(
            ["ifconfig", "-a"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout or None


def _read_darwin_network_interfaces() -> tuple[NetworkInterfaceFacts, ...]:
    """Parse ``ifconfig -a`` into the same shape the Linux sysfs reader returns.

    Darwin has no ``/sys/class/net``, and reaching link-layer addresses
    through ``getifaddrs`` would mean ``ctypes`` struct layouts that differ
    per architecture.  ``ifconfig`` is stable, always present on macOS, and
    already the thing an operator would run by hand.

    Only interfaces with an ``ether`` line are reported: an interface with
    no link layer (a VPN tunnel) is not something you can pin a DHCP
    reservation for, which is the reason this data is collected.

    Three signals in the output narrow ~30 interfaces down to the real NICs,
    all of them read from the text rather than guessed from names:

    - a name prefix, for the families Apple names consistently
      (:data:`_DARWIN_VIRTUAL_INTERFACE_PREFIXES`);
    - ``media: none``, meaning no physical layer at all.  That is what
      Apple's internal peers report (``en6``-``en9`` on an M-series Mac,
      sharing the ``anpi`` MAC block).  An *unplugged* real NIC reports
      ``media: autoselect (none)`` instead -- a media layer with no link --
      so it survives, which matters because its MAC is exactly what a DHCP
      reservation keys on;
    - membership of a ``bridge*``, which drops the Thunderbolt ports
      (``en2``-``en5``).  The bridge stanza enumerates its own members, so
      this needs no assumption about their names.
    """
    raw = _ifconfig()
    if not raw:
        return ()

    interfaces: list[NetworkInterfaceFacts] = []
    # Collected across the whole scan and applied at the end: the ``bridge0``
    # stanza that names its members appears *after* the members themselves.
    bridge_members: set[str] = set()
    name: Optional[str] = None
    mac: Optional[str] = None
    ipv4: Optional[str] = None
    # macOS reports link state two ways and they disagree: the ``RUNNING``
    # flag is pinned on for Apple's internal NICs, while ``status:``
    # tracks the actual link.  Prefer ``status:`` and fall back to the flag
    # for interface families that omit it.
    status: Optional[str] = None
    media: Optional[str] = None
    running_flag = False

    def flush() -> None:
        if name is None or mac is None:
            return
        if name.startswith(_DARWIN_VIRTUAL_INTERFACE_PREFIXES):
            return
        if media == "none":
            return
        is_up = status == "active" if status is not None else running_flag
        interfaces.append(
            NetworkInterfaceFacts(
                name=name, ipv4_address=ipv4, mac_address=mac, is_up=is_up
            )
        )

    for line in raw.splitlines():
        if line and not line[0].isspace():
            flush()
            header, _, rest = line.partition(":")
            name = header.strip() or None
            mac, ipv4, status, media = None, None, None, None
            running_flag = "RUNNING" in rest
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        # ``inet6`` must not match here -- it shares the ``inet`` prefix but
        # carries an address this schema has no field for.
        if fields[0] == "ether":
            mac = fields[1].lower()
        elif fields[0] == "member:":
            bridge_members.add(fields[1])
        elif fields[0] == "media:":
            media = fields[1]
        elif fields[0] == "inet" and ipv4 is None:
            ipv4 = fields[1]
        elif fields[0] == "status:":
            status = fields[1]

    flush()
    return tuple(nic for nic in interfaces if nic.name not in bridge_members)


def read_network_interfaces(
    net_base: Optional[Path] = None,
) -> tuple[NetworkInterfaceFacts, ...]:
    """Enumerate non-loopback network interfaces (Linux and macOS).

    Returns ``()`` on platforms with no reader, or when the underlying
    source is unreadable -- same "unavailable, not an error" degradation as
    every other reader in this module. Exists so a device's current
    LAN-facing IP(s) and MAC(s) can be surfaced without SSHing in or
    scanning the network first (the whole point being that neither may be
    possible).

    ``net_base`` is a test seam for the Linux path (defaults to
    ``/sys/class/net``); the macOS path is seamed on :func:`_ifconfig`.
    """
    system = platform.system()
    if system == "Darwin":
        return _read_darwin_network_interfaces()
    if system != "Linux":
        return ()

    base = net_base if net_base is not None else Path("/sys/class/net")
    if not base.exists():
        return ()

    try:
        ifnames = sorted(p.name for p in base.iterdir() if p.name not in _NETWORK_INTERFACE_SKIP)
    except OSError:
        return ()

    return tuple(_read_network_interface_facts(ifname, base) for ifname in ifnames)


#: Address the primary-interface probe "connects" to.  TEST-NET-1
#: (RFC 5737) is reserved for documentation and is never routed, and a UDP
#: ``connect()`` transmits nothing -- so this never touches the network.
_ROUTE_PROBE_ADDRESS = ("192.0.2.1", 9)


def _route_source_address() -> Optional[str]:
    """Return the source IPv4 the kernel would use to reach the internet.

    A UDP ``connect()`` performs no handshake and transmits nothing -- it
    only binds the socket, which makes the kernel run its routing decision
    and pick a source address.  Reading it back names the interface holding
    the default route without parsing a route table, shelling out, or adding
    a dependency, identically on Linux and Darwin.

    ``None`` when the host has no route at all (an edge on an isolated LAN,
    or one that booted before DHCP answered).  This is the test seam for
    :func:`read_primary_interface`.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(_ROUTE_PROBE_ADDRESS)
            address = sock.getsockname()[0]
        finally:
            sock.close()
    except OSError:
        return None
    return address if address and address != "0.0.0.0" else None


def _is_physical_interface(ifname: str, net_base: Optional[Path] = None) -> bool:
    """True when ``ifname`` is backed by real hardware (Linux).

    ``/sys/class/net/<if>/device`` is a symlink into the device tree that
    exists only for hardware-backed interfaces.  Bridges, ``veth`` pairs,
    bonds, ``dummy*`` and every kernel tunnel pseudo-device lack it, so this
    is the kernel's own answer to "is this a real NIC" -- which does not rot
    the way a list of name prefixes does as new interface families appear.

    Returns ``True`` on non-Linux platforms: Darwin has no sysfs, and its
    ``ifconfig`` reader already drops the pseudo-interfaces it can name.
    """
    if platform.system() != "Linux":
        return True
    base = net_base if net_base is not None else Path("/sys/class/net")
    return (base / ifname / "device").exists()


def read_primary_interface(
    interfaces: Optional[tuple[NetworkInterfaceFacts, ...]] = None,
    net_base: Optional[Path] = None,
) -> Optional[NetworkInterfaceFacts]:
    """Return the one interface this host actually reaches the network on.

    Asks the kernel instead of guessing.  A UDP ``connect()`` to
    :data:`_ROUTE_PROBE_ADDRESS` sends no packet -- it only makes the kernel
    resolve which source address it *would* use, i.e. the address on the
    interface holding the default route.  Matching that address back against
    the enumerated interfaces yields the NIC whose MAC a DHCP reservation
    has to key on.

    This is structural rather than heuristic, and that is the whole point:
    ``docker0``, a ``br-*`` compose bridge and a ``veth*`` peer *cannot*
    hold the default route, so no name blocklist is needed to keep the
    answer correct.  Picking "the first enumerated interface that is up"
    instead selects a container bridge on any edge running driver
    containers, because sysfs enumeration is alphabetical and ``br-`` and
    ``docker0`` sort before ``eth0``.

    Falls back to a physical interface that is up when the host has no
    usable route at all -- an edge on an isolated LAN, or one that booted
    before DHCP answered.  Returns ``None`` when nothing was enumerated.

    ``interfaces`` and ``net_base`` are test seams; production callers pass
    the tuple they already read so the probe costs no extra enumeration.
    """
    nics = (
        read_network_interfaces(net_base=net_base) if interfaces is None else interfaces
    )
    if not nics:
        return None

    source_ip = _route_source_address()
    if source_ip is not None:
        for nic in nics:
            if nic.ipv4_address == source_ip:
                return nic

    physical = [nic for nic in nics if _is_physical_interface(nic.name, net_base)]
    if not physical:
        return None
    return next((nic for nic in physical if nic.is_up), physical[0])


def read_primary_mac_address() -> Optional[str]:
    """Return a MAC from :func:`uuid.getnode` as ``aa:bb:cc:dd:ee:ff``, or ``None``.

    Last-resort fallback behind :func:`read_primary_interface`: this reads
    the same source the device fingerprint is derived from, so it is
    populated on platforms where interface enumeration finds nothing, but it
    is *not* necessarily a reachable interface -- ``getnode()`` takes
    whichever address the OS reports first, which on macOS is routinely an
    Apple-internal NIC. Prefer :func:`read_primary_interface`.

    Returns ``None`` when ``getnode()`` could not read any hardware address
    and fabricated a random node ID instead.  RFC 4122 has it set the
    multicast bit to flag exactly that, and uploading the value anyway
    would put a MAC-shaped string on the dashboard that matches no NIC in
    existence -- worse than showing nothing, because an operator would
    paste it into a DHCP reservation that can never match.
    """
    try:
        node = getnode()
    except Exception:
        return None
    if node >> 40 & 1:
        return None
    hex_digits = f"{node:012x}"
    return ":".join(hex_digits[i : i + 2] for i in range(0, 12, 2))


def read_device_serial(
    cpuinfo_path: Optional[Path] = None,
    device_tree_serial: Optional[Path] = None,
) -> Optional[str]:
    """Return the board serial number (Linux only), or ``None``.

    Raspberry Pi kernels expose it as a ``Serial`` line in
    ``/proc/cpuinfo``; Tegra (Jetson) kernels omit that line and carry it
    in the device tree instead, so both are consulted in that order.

    Both arguments are test seams; production code leaves them at their
    defaults.
    """
    if platform.system() != "Linux":
        return None

    cpuinfo = cpuinfo_path if cpuinfo_path is not None else Path(_CPUINFO_PATH)
    try:
        with open(cpuinfo) as f:
            for line in f:
                key, _, value = line.partition(":")
                if key.strip().lower() == "serial":
                    serial = value.strip().lower()
                    if serial not in _PLACEHOLDER_SERIALS:
                        return serial
                    break
    except OSError:
        pass

    dt_path = (
        device_tree_serial
        if device_tree_serial is not None
        else Path(_DEVICE_TREE_SERIAL_PATH)
    )
    try:
        # Device-tree string properties are NUL-terminated; the sentinel
        # would otherwise ride along into JSON and the dashboard.
        serial = dt_path.read_text(errors="replace").strip().strip("\x00").strip()
    except OSError:
        return None
    return serial if serial and serial not in _PLACEHOLDER_SERIALS else None


def _read_cpu_model_from_cpuinfo() -> tuple[Optional[str], Optional[int]]:
    """Parse ``/proc/cpuinfo`` and return ``(model_name, cpu_count)``.

    Linux-only.  Returns ``(None, None)`` on other platforms or when
    ``/proc/cpuinfo`` is unreadable.  ``model_name`` is the first ``model
    name``/``Model`` field encountered (x86 uses ``model name``, ARM uses
    ``Model``); when neither is present we fall back to the first
    ``Hardware`` line that older Raspberry Pi kernels emit.  ``cpu_count``
    is the number of ``processor:`` records, which matches the count of
    logical CPUs the kernel exposes.

    x86 ``model`` vs ARM ``Model``.  Both lowercase to ``model`` after
    :py:meth:`str.lower`, but they carry different things:

    - x86 ``/proc/cpuinfo`` emits ``model       : 158`` (the integer
      Intel/AMD CPU family identifier) *before* the human-readable
      ``model name : Intel(R) Core(TM) i7-9750H CPU``.  If we accepted
      ``model`` unconditionally, the integer wins because it comes
      first — and the dashboard's Host footer renders e.g. ``158 · sw
      watchdog`` for every x86 edge.  We saw this in production.
    - ARM ``/proc/cpuinfo`` emits ``Model       : Raspberry Pi 5 Model
      B Rev 1.0`` at the bottom of the file, which is the right
      source on Pis without a per-processor ``model name`` line.

    We therefore accept ``model`` only when the value is *not*
    all-digits.  ``isdigit()`` is enough — neither x86 family
    identifiers nor any meaningful ARM "Model" string are
    all-digits.  Mixed-content values like ``BCM2835`` or
    ``Raspberry Pi 5 Model B Rev 1.0`` still pass.
    """
    if platform.system() != "Linux":
        return None, None

    model_name: Optional[str] = None
    hardware_name: Optional[str] = None
    processor_count = 0

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, _, value = line.partition(":")
                key = key.strip().lower()
                value = value.strip()
                if not value:
                    continue
                if key == "processor":
                    processor_count += 1
                elif model_name is None and key == "model name":
                    model_name = value
                elif model_name is None and key == "model" and not value.isdigit():
                    model_name = value
                elif hardware_name is None and key == "hardware":
                    hardware_name = value
    except OSError:
        return None, None

    name = model_name or hardware_name
    count = processor_count if processor_count > 0 else None
    return name, count


def _sysctl(key: str) -> Optional[str]:
    """Read a single ``sysctl`` key on Darwin.  Returns ``None`` on any error.

    macOS ships ``sysctl`` in ``/usr/sbin`` which is always on the system
    ``PATH``; we still guard with :func:`shutil.which` so a stripped-down
    environment (CI container, ``nix`` shell, ...) cannot crash the reader.
    """
    if not shutil.which("sysctl"):
        return None
    try:
        out = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    value = out.stdout.strip()
    return value or None


def _read_package_version(distribution: str) -> Optional[str]:
    """Return the installed version of a distribution, or ``None``.

    Thin wrapper around :func:`importlib.metadata.version` that swallows
    :class:`PackageNotFoundError` so callers can probe optional packages
    without try/except boilerplate.  Useful when the SDK is imported by a
    process that did *not* install the CLI or edge-core companion
    packages — that is a legitimate state, not an error.
    """
    try:
        return _pkg_version(distribution)
    except PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover - defensive
        # ``importlib.metadata`` historically raised a few different
        # exception types across Python versions; catch broadly so a
        # malformed metadata file never crashes ``read_host_facts``.
        return None


def _read_module_version(module_name: str) -> Optional[str]:
    """Return ``module.__version__`` if the module is already in ``sys.modules``.

    Checks ``sys.modules`` rather than calling :func:`importlib.import_module`
    so probing a companion package that merely happens to be installed in
    the venv doesn't trigger its import side effects.
    """
    module = sys.modules.get(module_name)
    if module is None:
        return None
    version = getattr(module, "__version__", None)
    if isinstance(version, str) and version:
        return version
    return None


def _resolve_effective_version(module_name: str, distribution: str) -> Optional[str]:
    """In-process ``__version__`` first, then :mod:`importlib.metadata`.

    The first step is what surfaces CI ``BUILD_VERSION`` stamps and
    survives PyInstaller binaries that ship without ``.dist-info``.
    """
    version = _read_module_version(module_name)
    if version is not None:
        return version
    return _read_package_version(distribution)


def _read_software_versions() -> tuple[Optional[str], Optional[str]]:
    """Return ``(sdk_version, edge_core_version)`` effectively in use.

    Each entry resolves only when the corresponding package is imported
    in the calling process (or its distribution metadata is installed).

    CLI version is intentionally not reported: on production edges the
    CLI is a standalone PyInstaller binary, so neither ``sys.modules``
    nor ``importlib.metadata`` can observe it from edge-core's process.
    Surfacing it would require subprocess-probing ``cyberwave --version``,
    which the host_facts uploader deliberately avoids.
    """
    return (
        _resolve_effective_version("cyberwave", "cyberwave"),
        _resolve_effective_version("cyberwave_edge_core", "cyberwave-edge-core"),
    )


def _read_darwin_facts() -> tuple[Optional[float], Optional[str], Optional[int]]:
    """Read static facts on macOS via ``sysctl``.

    Returns ``(memory_total_mb, cpu_model, cpu_count)``.  We shell out to
    ``sysctl`` for the *static* slice of this information so the
    dashboard can render an Apple-Silicon edge with proper "Apple M2
    Pro" / "48 GB" labels instead of "unknown".

    ``memory_total_mb`` is rounded to one decimal place to match the
    Linux side (:func:`read_host_memory` rounds to ``round(total_mb,
    1)``); otherwise round-tripping through JSON would produce
    inconsistent precision across platforms.

    ``cpu_count`` is the *logical* CPU count: Linux's
    :func:`_read_cpu_model_from_cpuinfo` counts ``processor:`` records,
    which are per-logical-CPU (after SMT/hyperthreading expansion), so
    for semantic parity we read ``hw.logicalcpu`` and fall back to the
    legacy ``hw.ncpu`` alias.  Reading ``hw.physicalcpu`` would
    under-report core counts on x86 Macs with Hyper-Threading enabled.
    """
    if platform.system() != "Darwin":
        return None, None, None

    memory_total_mb: Optional[float] = None
    mem_raw = _sysctl("hw.memsize")
    if mem_raw is not None:
        try:
            memory_total_mb = round(int(mem_raw) / 1024.0 / 1024.0, 1)
        except (ValueError, TypeError):
            memory_total_mb = None

    cpu_model = _sysctl("machdep.cpu.brand_string")

    cpu_count: Optional[int] = None
    cpu_raw = _sysctl("hw.logicalcpu") or _sysctl("hw.ncpu")
    if cpu_raw is not None:
        try:
            cpu_count = int(cpu_raw)
        except (ValueError, TypeError):
            cpu_count = None

    return memory_total_mb, cpu_model, cpu_count


def read_host_facts(
    *,
    thermal_base: Optional[Path] = None,
    watchdog_device: Optional[Path] = None,
    net_base: Optional[Path] = None,
    power_sysfs_base: Optional[Path] = None,
) -> HostFacts:
    """Collect host facts for upload to the edge device's persistent
    identity record.

    Re-read on every tick of edge-core's host-facts keepalive (30 s), not
    once at startup — the gauge and network fields depend on that.  All
    readers degrade silently on missing sources: ``platform.platform()``
    always returns something, so :class:`HostFacts` is always
    constructible.

    Coverage:

    - Linux: memory total/used/available comes from ``/proc/meminfo``;
      CPU model/count from ``/proc/cpuinfo``; thermal source and the
      temperature reading from ``/sys/class/thermal/thermal_zone*``;
      hardware watchdog from ``/dev/watchdog``.
    - macOS (Darwin): memory total, CPU model and (logical) CPU count
      come from ``sysctl``.  ``thermal_source`` and ``cpu_temp_c`` are
      always ``None`` — there is no temperature reader on Darwin, and
      setting the source alone would falsely imply a live reading.
      ``memory_used_percent`` / ``memory_available_mb`` are ``None`` too
      (``sysctl`` gives the total, not the split).  ``power_mw`` is
      ``None``: :func:`read_host_power_draw` is sysfs-only.
      ``has_hardware_watchdog`` is always ``False``.
    - Other platforms: only ``platform`` and ``kernel`` are populated.

    ``thermal_base``, ``watchdog_device``, ``net_base`` and
    ``power_sysfs_base`` are test seams; production code should leave them
    at their defaults.
    """
    system = platform.system()

    memory_used_percent: Optional[float] = None
    memory_available_mb: Optional[float] = None

    if system == "Linux":
        memory = read_host_memory()
        memory_total_mb = memory.total_mb if memory is not None else None
        if memory is not None:
            # Same single /proc/meminfo read as the total above -- the
            # gauge copy costs no extra syscall.
            memory_used_percent = memory.used_percent
            memory_available_mb = memory.available_mb
        cpu_model, cpu_count = _read_cpu_model_from_cpuinfo()
    elif system == "Darwin":
        memory_total_mb, cpu_model, cpu_count = _read_darwin_facts()
    else:
        memory_total_mb, cpu_model, cpu_count = None, None, None

    # ``thermal_source`` is the identifier of the sysfs path the dynamic
    # publisher reads from -- semantically tied to an active temperature
    # reading.  We populate it only when a dynamic source actually
    # exists (currently Linux only); macOS could grow a ``macmon``-backed
    # reader in the future, at which point this branch can be extended
    # without changing the field's meaning.
    thermal_source: Optional[str] = None
    if system == "Linux":
        thermal_zones = discover_cpu_thermal_zones(thermal_base=thermal_base)
        if thermal_zones:
            zone_dir = thermal_zones[0].parent
            type_file = zone_dir / "type"
            zone_type = ""
            if type_file.exists():
                try:
                    zone_type = type_file.read_text().strip().lower()
                except OSError:
                    zone_type = ""
            thermal_source = (
                zone_dir.name if not zone_type else f"{zone_dir.name}:{zone_type}"
            )

    # Deliberately a separate read rather than reusing ``thermal_zones``
    # above: ``thermal_source`` names the *first* CPU zone, while the
    # reading we want is the *hottest* one across all of them. Collapsing
    # the two would silently change what ``thermal_source`` means.
    cpu_temp_c: Optional[float] = None
    if system == "Linux":
        cpu_temp = read_host_cpu_temperature(thermal_base=thermal_base)
        if cpu_temp is not None:
            cpu_temp_c = cpu_temp.celsius

    power_mw: Optional[float] = None
    if system == "Linux":
        power = read_host_power_draw(power_sysfs_base=power_sysfs_base)
        if power is not None:
            power_mw = power.milliwatts

    # Same env var edge-core reads for the heartbeat copy, rejected on the
    # same terms (it does the warning; this runs every ~30 s).  ``nan`` /
    # ``inf`` parse fine but serialise as bare JSON tokens that jsonb
    # refuses, failing the whole /discover POST -- and with it the
    # ``last_seen_at`` bump that keeps the edge out of "Offline".
    battery_wh: Optional[float] = None
    battery_wh_raw = os.environ.get("CYBERWAVE_BATTERY_WH", "").strip()
    if battery_wh_raw:
        try:
            parsed = float(battery_wh_raw)
        except ValueError:
            parsed = None
        battery_wh = parsed if parsed is not None and math.isfinite(parsed) else None

    wd_path = (
        watchdog_device if watchdog_device is not None else Path(HARDWARE_WATCHDOG_DEVICE)
    )
    has_hardware_watchdog = system == "Linux" and wd_path.exists()

    sdk_version, edge_core_version = _read_software_versions()
    network_interfaces = read_network_interfaces(net_base=net_base)

    # One MAC, resolved the same way on every platform: the interface the
    # kernel would actually route out of.  ``read_primary_mac_address()``
    # (``uuid.getnode()``) stays as the last resort for hosts where nothing
    # was enumerated -- it is the only MAC available there, but it is not
    # tied to a named interface, so ``primary_interface_name`` stays ``None``
    # and consumers can tell the two provenances apart.
    primary_interface = read_primary_interface(
        interfaces=network_interfaces, net_base=net_base
    )
    primary_mac_address = (
        primary_interface.mac_address if primary_interface is not None else None
    ) or read_primary_mac_address()
    primary_interface_name = (
        primary_interface.name
        if primary_interface is not None and primary_interface.mac_address
        else None
    )

    return HostFacts(
        platform=platform.platform(),
        kernel=platform.release() or None,
        memory_total_mb=memory_total_mb,
        cpu_model=cpu_model,
        cpu_count=cpu_count,
        thermal_source=thermal_source,
        has_hardware_watchdog=has_hardware_watchdog,
        sdk_version=sdk_version,
        edge_core_version=edge_core_version,
        network_interfaces=network_interfaces,
        primary_mac_address=primary_mac_address,
        primary_interface_name=primary_interface_name,
        device_serial=read_device_serial(),
        cpu_temp_c=cpu_temp_c,
        memory_used_percent=memory_used_percent,
        memory_available_mb=memory_available_mb,
        power_mw=power_mw,
        battery_wh=battery_wh,
    )


__all__ = [
    "CPU_THERMAL_ZONE_TYPES",
    "HARDWARE_WATCHDOG_DEVICE",
    "HostCpuTemperature",
    "HostFacts",
    "HostMemoryInfo",
    "HostPowerDraw",
    "NetworkInterfaceFacts",
    "discover_cpu_thermal_zones",
    "read_device_serial",
    "read_host_cpu_temperature",
    "read_host_facts",
    "read_host_memory",
    "read_host_power_draw",
    "read_network_interfaces",
    "read_primary_interface",
    "read_primary_mac_address",
    "read_thermal_zone_celsius",
]
