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
- The *static* reader (:func:`read_host_facts`) is cross-platform: on
  macOS it falls back to ``sysctl`` for RAM, CPU model and (logical)
  CPU count.  Thermal source is left ``None`` on macOS since no live
  publisher samples it there.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Optional


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
class HostFacts:
    """Static host-level facts about the edge device.

    These properties change rarely (RAM never, CPU model never, kernel only
    on upgrade) so they belong on the device's persistent identity record
    rather than on every ~5 s MQTT heartbeat.  The companion dynamic
    readers (:func:`read_host_memory`, :func:`read_host_cpu_temperature`)
    carry the values that actually move.

    Optional fields may be ``None`` when the underlying source is
    unavailable on the current platform.  Coverage matrix:

    - Linux: every field is populated when the corresponding
      procfs/sysfs source is readable.  ``cpu_count`` is the logical
      CPU count (one ``processor:`` entry per SMT thread).
    - macOS: ``memory_total_mb``, ``cpu_model`` and ``cpu_count``
      (logical, from ``hw.logicalcpu``) come from ``sysctl``.
      ``thermal_source`` is always ``None`` — there is no live
      temperature publisher on Darwin, and a "would-be source" string
      would be misleading.  ``has_hardware_watchdog`` is always
      ``False``.
    - Other platforms: only ``platform``, ``kernel`` and
      ``has_hardware_watchdog`` are guaranteed.

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
        if self.sdk_version is not None:
            out["sdk_version"] = self.sdk_version
        if self.edge_core_version is not None:
            out["edge_core_version"] = self.edge_core_version
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
# Static host facts
# ---------------------------------------------------------------------------

#: ``/dev/watchdog`` path consulted by :func:`read_host_facts`.  Exposed as a
#: module attribute so tests can monkeypatch it via :class:`pathlib.Path`.
HARDWARE_WATCHDOG_DEVICE = "/dev/watchdog"


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
) -> HostFacts:
    """Collect static host facts for upload to the edge device's persistent
    identity record.

    Designed to be called once at edge-core startup.  All readers degrade
    silently on missing sources: ``platform.platform()`` always returns
    something, so :class:`HostFacts` is always constructible.

    Coverage:

    - Linux: memory total comes from ``/proc/meminfo``; CPU model/count
      from ``/proc/cpuinfo``; thermal source from
      ``/sys/class/thermal/thermal_zone*``; hardware watchdog from
      ``/dev/watchdog``.
    - macOS (Darwin): memory total, CPU model and (logical) CPU count
      come from ``sysctl``.  ``thermal_source`` is always ``None`` —
      the heartbeat publisher does not currently sample temperature
      on macOS, and setting the field would falsely imply a live
      reading.  ``has_hardware_watchdog`` is always ``False``.
    - Other platforms: only ``platform`` and ``kernel`` are populated.

    ``thermal_base`` and ``watchdog_device`` are test seams; production
    code should leave them at their defaults.
    """
    system = platform.system()

    if system == "Linux":
        memory = read_host_memory()
        memory_total_mb = memory.total_mb if memory is not None else None
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

    wd_path = (
        watchdog_device if watchdog_device is not None else Path(HARDWARE_WATCHDOG_DEVICE)
    )
    has_hardware_watchdog = system == "Linux" and wd_path.exists()

    sdk_version, edge_core_version = _read_software_versions()

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
    )


__all__ = [
    "CPU_THERMAL_ZONE_TYPES",
    "HARDWARE_WATCHDOG_DEVICE",
    "HostCpuTemperature",
    "HostFacts",
    "HostMemoryInfo",
    "discover_cpu_thermal_zones",
    "read_host_cpu_temperature",
    "read_host_facts",
    "read_host_memory",
    "read_thermal_zone_celsius",
]
