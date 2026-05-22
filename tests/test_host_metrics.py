"""Unit tests for cyberwave.edge.host_metrics."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

import pytest

from cyberwave.edge.host_metrics import (
    CPU_THERMAL_ZONE_TYPES,
    HostCpuTemperature,
    HostFacts,
    HostMemoryInfo,
    discover_cpu_thermal_zones,
    read_host_cpu_temperature,
    read_host_facts,
    read_host_memory,
    read_thermal_zone_celsius,
)


# ---------------------------------------------------------------------------
# read_host_memory
# ---------------------------------------------------------------------------


class TestReadHostMemory:
    def test_returns_none_on_non_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        assert read_host_memory() is None

    def test_parses_meminfo(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        meminfo_path = tmp_path / "meminfo"
        meminfo_path.write_text(
            "MemTotal:        3906292 kB\n"
            "MemFree:          123456 kB\n"
            "MemAvailable:    1024000 kB\n"
            "Buffers:          102400 kB\n"
            "Cached:           512000 kB\n"
        )

        original_open = open

        def fake_open(path, *args, **kwargs):
            if str(path) == "/proc/meminfo":
                return original_open(str(meminfo_path), *args, **kwargs)
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)

        result = read_host_memory()
        assert isinstance(result, HostMemoryInfo)
        assert result.total_mb == pytest.approx(3906292 / 1024, abs=1)
        assert result.available_mb == pytest.approx(1024000 / 1024, abs=1)
        assert result.used_percent > 0

    def test_falls_back_to_free_plus_buffers_cached(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Older kernels (<3.14) lack ``MemAvailable``; we must approximate."""
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        meminfo_path = tmp_path / "meminfo"
        meminfo_path.write_text(
            "MemTotal:        4000000 kB\n"
            "MemFree:          500000 kB\n"
            "Buffers:          100000 kB\n"
            "Cached:           400000 kB\n"
        )

        original_open = open

        def fake_open(path, *args, **kwargs):
            if str(path) == "/proc/meminfo":
                return original_open(str(meminfo_path), *args, **kwargs)
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)

        result = read_host_memory()
        assert result is not None
        assert result.available_mb == pytest.approx(1000000 / 1024, abs=1)

    def test_returns_none_when_no_total(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        meminfo_path = tmp_path / "meminfo"
        meminfo_path.write_text("Buffers: 100 kB\n")

        original_open = open
        monkeypatch.setattr(
            "builtins.open",
            lambda path, *a, **kw: original_open(
                str(meminfo_path) if str(path) == "/proc/meminfo" else path, *a, **kw
            ),
        )
        assert read_host_memory() is None


# ---------------------------------------------------------------------------
# CPU thermal zone discovery
# ---------------------------------------------------------------------------


def _build_thermal_sysfs(root: Path, zones: list[tuple[str, int]]) -> Path:
    """Create a fake ``/sys/class/thermal`` tree under ``root``.

    Each ``(zone_type, millideg)`` pair becomes ``thermal_zone{i}``.
    """
    thermal_base = root / "thermal"
    thermal_base.mkdir(parents=True, exist_ok=True)
    for idx, (zone_type, millideg) in enumerate(zones):
        zone_dir = thermal_base / f"thermal_zone{idx}"
        zone_dir.mkdir()
        (zone_dir / "type").write_text(zone_type)
        (zone_dir / "temp").write_text(str(millideg))
    return thermal_base


class TestDiscoverCpuThermalZones:
    def test_returns_empty_on_non_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        assert discover_cpu_thermal_zones() == []

    def test_returns_empty_when_thermal_base_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        assert discover_cpu_thermal_zones(thermal_base=tmp_path / "missing") == []

    def test_prefers_cpu_typed_zones(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        base = _build_thermal_sysfs(
            tmp_path,
            [
                ("acpitz", 90_000),
                ("cpu-thermal", 55_000),
                ("gpu_thermal", 80_000),
                ("coretemp", 60_000),
            ],
        )
        zones = discover_cpu_thermal_zones(thermal_base=base)
        # Only cpu-thermal (zone1) and coretemp (zone3) should be selected.
        assert len(zones) == 2
        zone_names = {z.parent.name for z in zones}
        assert zone_names == {"thermal_zone1", "thermal_zone3"}

    def test_falls_back_to_all_zones_without_cpu_type(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        base = _build_thermal_sysfs(
            tmp_path,
            [
                ("acpitz", 50_000),
                ("pch_skylake", 65_000),
            ],
        )
        zones = discover_cpu_thermal_zones(thermal_base=base)
        assert len(zones) == 2

    def test_substring_cpu_match(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Zone types containing 'cpu' (e.g. 'cpu0_thermal') should match."""
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        base = _build_thermal_sysfs(
            tmp_path,
            [
                ("cpu0_thermal", 60_000),
                ("battery", 30_000),
            ],
        )
        zones = discover_cpu_thermal_zones(thermal_base=base)
        assert len(zones) == 1
        assert zones[0].parent.name == "thermal_zone0"


# ---------------------------------------------------------------------------
# Single-zone temperature read
# ---------------------------------------------------------------------------


class TestReadThermalZoneCelsius:
    def test_parses_millidegrees(self, tmp_path: Path) -> None:
        f = tmp_path / "temp"
        f.write_text("72500")
        assert read_thermal_zone_celsius(f) == pytest.approx(72.5)

    def test_returns_none_on_invalid(self, tmp_path: Path) -> None:
        f = tmp_path / "temp"
        f.write_text("not-a-number")
        assert read_thermal_zone_celsius(f) is None

    def test_returns_none_on_missing(self, tmp_path: Path) -> None:
        assert read_thermal_zone_celsius(tmp_path / "missing") is None


# ---------------------------------------------------------------------------
# read_host_cpu_temperature
# ---------------------------------------------------------------------------


class TestReadHostCpuTemperature:
    def test_returns_none_on_non_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        assert read_host_cpu_temperature() is None

    def test_picks_hottest_across_cpu_zones(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        base = _build_thermal_sysfs(
            tmp_path,
            [
                ("coretemp", 60_000),
                ("coretemp", 72_500),
                ("coretemp", 68_000),
            ],
        )
        result = read_host_cpu_temperature(thermal_base=base)
        assert isinstance(result, HostCpuTemperature)
        assert result.celsius == pytest.approx(72.5)
        assert "coretemp" in result.source

    def test_ignores_non_cpu_when_cpu_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        base = _build_thermal_sysfs(
            tmp_path,
            [
                ("acpitz", 90_000),
                ("cpu-thermal", 55_000),
            ],
        )
        result = read_host_cpu_temperature(thermal_base=base)
        assert result is not None
        assert result.celsius == pytest.approx(55.0)


def test_constants() -> None:
    assert "coretemp" in CPU_THERMAL_ZONE_TYPES
    assert "x86_pkg_temp" in CPU_THERMAL_ZONE_TYPES
    assert "cpu-thermal" in CPU_THERMAL_ZONE_TYPES


# ---------------------------------------------------------------------------
# read_host_facts
# ---------------------------------------------------------------------------


class TestHostFactsToDict:
    """Confirm that empty / unavailable sources are omitted from the JSON shape.

    The dashboard distinguishes "metric absent" from "metric is zero" by the
    *presence* of the key, so this is a real correctness invariant rather
    than a style preference.
    """

    def test_omits_none_optional_keys(self) -> None:
        facts = HostFacts(
            platform="Linux-x86_64",
            kernel=None,
            memory_total_mb=None,
            cpu_model=None,
            cpu_count=None,
            thermal_source=None,
            has_hardware_watchdog=False,
            sdk_version=None,
            cli_version=None,
            edge_core_version=None,
        )
        out = facts.to_dict()
        assert out == {
            "platform": "Linux-x86_64",
            "has_hardware_watchdog": False,
        }

    def test_keeps_populated_keys(self) -> None:
        facts = HostFacts(
            platform="Linux-aarch64",
            kernel="6.6.20",
            memory_total_mb=3906.0,
            cpu_model="Cortex-A72",
            cpu_count=4,
            thermal_source="thermal_zone0:cpu-thermal",
            has_hardware_watchdog=True,
            sdk_version="0.4.7",
            cli_version="0.12.4",
            edge_core_version="0.1.4",
        )
        out = facts.to_dict()
        assert out["platform"] == "Linux-aarch64"
        assert out["kernel"] == "6.6.20"
        assert out["memory_total_mb"] == 3906.0
        assert out["cpu_model"] == "Cortex-A72"
        assert out["cpu_count"] == 4
        assert out["thermal_source"] == "thermal_zone0:cpu-thermal"
        assert out["has_hardware_watchdog"] is True
        assert out["sdk_version"] == "0.4.7"
        assert out["cli_version"] == "0.12.4"
        assert out["edge_core_version"] == "0.1.4"

    def test_partial_versions_omits_unset_packages(self) -> None:
        """A standalone SDK install (no CLI, no edge-core) reports only
        ``sdk_version`` — the absent companions must not leak as ``None``
        into the JSON, since the dashboard distinguishes "missing key"
        from "key with null value"."""
        facts = HostFacts(
            platform="macOS-arm64",
            kernel=None,
            memory_total_mb=None,
            cpu_model=None,
            cpu_count=None,
            thermal_source=None,
            has_hardware_watchdog=False,
            sdk_version="0.4.7",
            cli_version=None,
            edge_core_version=None,
        )
        out = facts.to_dict()
        assert out["sdk_version"] == "0.4.7"
        assert "cli_version" not in out
        assert "edge_core_version" not in out


class TestSoftwareVersions:
    """Cover the ``importlib.metadata`` resolution path independently of
    whatever happens to be installed in the test venv."""

    def test_returns_three_values_when_all_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        canned = {
            "cyberwave": "1.2.3",
            "cyberwave-cli": "4.5.6",
            "cyberwave-edge-core": "7.8.9",
        }
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._pkg_version",
            lambda name: canned[name],
        )
        from cyberwave.edge.host_metrics import _read_software_versions

        assert _read_software_versions() == ("1.2.3", "4.5.6", "7.8.9")

    def test_missing_packages_resolve_to_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from importlib.metadata import PackageNotFoundError

        def fake(name: str) -> str:
            if name == "cyberwave":
                return "0.4.7"
            raise PackageNotFoundError(name)

        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._pkg_version", fake
        )
        from cyberwave.edge.host_metrics import _read_software_versions

        assert _read_software_versions() == ("0.4.7", None, None)

    def test_metadata_corruption_does_not_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A malformed ``METADATA`` file inside site-packages can raise
        non-``PackageNotFoundError`` exceptions on some Python builds; the
        defensive catch-all must keep the reader functional."""

        def fake(name: str) -> str:
            raise RuntimeError("malformed metadata for " + name)

        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._pkg_version", fake
        )
        from cyberwave.edge.host_metrics import _read_software_versions

        assert _read_software_versions() == (None, None, None)


class TestReadHostFacts:
    def test_always_constructible_on_unknown_platform(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Platforms outside {Linux, Darwin} still return a usable ``HostFacts``.

        ``platform.platform()`` is cross-platform so the dataclass is always
        constructible; everything else degrades to ``None``/``False``.  This
        guards the dot-access contract callers rely on.
        """
        monkeypatch.setattr(platform, "system", lambda: "OpenBSD")
        watchdog_path = tmp_path / "watchdog-absent"

        facts = read_host_facts(watchdog_device=watchdog_path)

        assert isinstance(facts, HostFacts)
        assert facts.platform  # always populated
        assert facts.has_hardware_watchdog is False
        assert facts.memory_total_mb is None
        assert facts.cpu_model is None
        assert facts.cpu_count is None
        assert facts.thermal_source is None

    def test_populates_from_fake_darwin_sources(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Darwin path: ``sysctl`` provides RAM/CPU; ``thermal_source``
        stays ``None`` because no live macOS publisher exists yet."""
        monkeypatch.setattr(platform, "system", lambda: "Darwin")

        # Hyper-Threaded x86 Mac scenario: physical=4, logical=8.  We must
        # report 8 (logical) to match Linux's ``processor:`` semantics.
        sysctl_values = {
            "hw.memsize": str(48 * 1024 * 1024 * 1024),  # 48 GiB
            "machdep.cpu.brand_string": "Intel Core i7-1068NG7",
            "hw.physicalcpu": "4",
            "hw.logicalcpu": "8",
            "hw.ncpu": "8",
        }

        def fake_which(name: str) -> str | None:
            if name == "sysctl":
                return "/usr/sbin/sysctl"
            return None

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            assert cmd[0] == "sysctl" and cmd[1] == "-n"
            value = sysctl_values.get(cmd[2], "")
            return subprocess.CompletedProcess(cmd, 0 if value else 1, stdout=value, stderr="")

        monkeypatch.setattr(
            "cyberwave.edge.host_metrics.shutil.which", fake_which
        )
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics.subprocess.run", fake_run
        )

        watchdog_dev = tmp_path / "watchdog-not-on-darwin"

        facts = read_host_facts(watchdog_device=watchdog_dev)

        assert facts.memory_total_mb == pytest.approx(48 * 1024.0, abs=0.1)
        assert facts.cpu_model == "Intel Core i7-1068NG7"
        # Logical count, NOT physical -- matches /proc/cpuinfo semantics.
        assert facts.cpu_count == 8
        # macOS never publishes a temperature today, so the static row
        # must not claim a thermal source.
        assert facts.thermal_source is None
        # Darwin never exposes /dev/watchdog even if the test seam file exists.
        assert facts.has_hardware_watchdog is False
        # SDK is installed in the test venv; the importlib.metadata reader
        # surfaces its real version, not ``None``.
        assert isinstance(facts.sdk_version, str) and facts.sdk_version

    def test_darwin_falls_back_to_hw_ncpu_when_logical_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Older Darwin builds expose only the legacy ``hw.ncpu`` alias."""
        monkeypatch.setattr(platform, "system", lambda: "Darwin")

        sysctl_values = {
            "hw.memsize": str(8 * 1024 * 1024 * 1024),
            "machdep.cpu.brand_string": "Apple M1",
            "hw.ncpu": "8",
        }

        def fake_which(name: str) -> str | None:
            return "/usr/sbin/sysctl" if name == "sysctl" else None

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            value = sysctl_values.get(cmd[2], "")
            return subprocess.CompletedProcess(cmd, 0 if value else 1, stdout=value, stderr="")

        monkeypatch.setattr(
            "cyberwave.edge.host_metrics.shutil.which", fake_which
        )
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics.subprocess.run", fake_run
        )

        facts = read_host_facts(watchdog_device=tmp_path / "absent")

        assert facts.cpu_count == 8

    def test_darwin_memory_is_rounded_to_one_decimal(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Match Linux ``read_host_memory``'s ``round(total_mb, 1)``
        precision so the JSON payload doesn't drift across platforms."""
        monkeypatch.setattr(platform, "system", lambda: "Darwin")

        # Odd byte count that produces a long-tailed float when divided.
        odd_bytes = 17_179_869_184 + 12_345  # ~16 GiB + change

        def fake_which(name: str) -> str | None:
            return "/usr/sbin/sysctl" if name == "sysctl" else None

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            if cmd[2] == "hw.memsize":
                return subprocess.CompletedProcess(cmd, 0, stdout=str(odd_bytes), stderr="")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

        monkeypatch.setattr(
            "cyberwave.edge.host_metrics.shutil.which", fake_which
        )
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics.subprocess.run", fake_run
        )

        facts = read_host_facts(watchdog_device=tmp_path / "absent")

        assert facts.memory_total_mb is not None
        # At most one decimal of fractional precision.
        assert facts.memory_total_mb == round(facts.memory_total_mb, 1)
        # And the value round-trips: same as round(odd_bytes / 1024^2, 1).
        assert facts.memory_total_mb == pytest.approx(
            round(odd_bytes / 1024.0 / 1024.0, 1), abs=1e-6
        )

    def test_darwin_handles_malformed_sysctl_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``sysctl`` returning garbage degrades the affected field to
        ``None`` rather than crashing the reader."""
        monkeypatch.setattr(platform, "system", lambda: "Darwin")

        def fake_which(name: str) -> str | None:
            return "/usr/sbin/sysctl" if name == "sysctl" else None

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            if cmd[2] == "hw.memsize":
                return subprocess.CompletedProcess(cmd, 0, stdout="not-an-int", stderr="")
            if cmd[2] == "machdep.cpu.brand_string":
                return subprocess.CompletedProcess(cmd, 0, stdout="Apple M3", stderr="")
            if cmd[2] in ("hw.logicalcpu", "hw.ncpu"):
                return subprocess.CompletedProcess(cmd, 0, stdout="garbage", stderr="")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

        monkeypatch.setattr(
            "cyberwave.edge.host_metrics.shutil.which", fake_which
        )
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics.subprocess.run", fake_run
        )

        facts = read_host_facts(watchdog_device=tmp_path / "absent")

        # Bad numeric inputs become ``None``; non-numeric cpu_model still survives.
        assert facts.memory_total_mb is None
        assert facts.cpu_model == "Apple M3"
        assert facts.cpu_count is None
        # to_dict() must omit the None fields entirely.
        d = facts.to_dict()
        assert "memory_total_mb" not in d
        assert "cpu_count" not in d
        assert d["cpu_model"] == "Apple M3"

    def test_populates_from_fake_linux_sources(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        # /proc/meminfo
        meminfo_path = tmp_path / "meminfo"
        meminfo_path.write_text(
            "MemTotal:        3906292 kB\n"
            "MemAvailable:    2000000 kB\n"
        )
        # /proc/cpuinfo
        cpuinfo_path = tmp_path / "cpuinfo"
        cpuinfo_path.write_text(
            "processor\t: 0\n"
            "model name\t: Cortex-A72\n"
            "\n"
            "processor\t: 1\n"
            "model name\t: Cortex-A72\n"
            "\n"
            "processor\t: 2\n"
            "model name\t: Cortex-A72\n"
            "\n"
            "processor\t: 3\n"
            "model name\t: Cortex-A72\n"
        )
        original_open = open

        def fake_open(path, *args, **kwargs):
            spath = str(path)
            if spath == "/proc/meminfo":
                return original_open(str(meminfo_path), *args, **kwargs)
            if spath == "/proc/cpuinfo":
                return original_open(str(cpuinfo_path), *args, **kwargs)
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)

        thermal_base = _build_thermal_sysfs(
            tmp_path, [("cpu-thermal", 55_000)]
        )
        watchdog_dev = tmp_path / "watchdog"
        watchdog_dev.write_text("")  # presence-only check

        facts = read_host_facts(
            thermal_base=thermal_base, watchdog_device=watchdog_dev
        )

        assert facts.memory_total_mb == pytest.approx(3906292 / 1024, abs=1)
        assert facts.cpu_model == "Cortex-A72"
        assert facts.cpu_count == 4
        assert facts.thermal_source is not None
        assert "cpu-thermal" in facts.thermal_source
        assert facts.has_hardware_watchdog is True

    def test_parses_arm_hardware_field_when_no_model_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Older Raspberry Pi kernels emit ``Hardware:`` instead of ``model name``."""
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        cpuinfo_path = tmp_path / "cpuinfo"
        cpuinfo_path.write_text(
            "processor\t: 0\n"
            "Hardware\t: BCM2835\n"
        )
        original_open = open

        def fake_open(path, *args, **kwargs):
            if str(path) == "/proc/cpuinfo":
                return original_open(str(cpuinfo_path), *args, **kwargs)
            if str(path) == "/proc/meminfo":
                raise FileNotFoundError()
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)
        watchdog_dev = tmp_path / "absent"

        facts = read_host_facts(
            thermal_base=tmp_path / "no-thermal", watchdog_device=watchdog_dev
        )
        assert facts.cpu_model == "BCM2835"
        assert facts.cpu_count == 1
