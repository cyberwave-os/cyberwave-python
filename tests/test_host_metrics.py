"""Unit tests for cyberwave.edge.host_metrics."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import pytest

from cyberwave.edge import host_metrics as host_metrics_module
from cyberwave.edge.host_metrics import (
    CPU_THERMAL_ZONE_TYPES,
    HostCpuTemperature,
    HostFacts,
    HostMemoryInfo,
    HostPowerDraw,
    NetworkInterfaceFacts,
    discover_cpu_thermal_zones,
    read_device_serial,
    read_host_cpu_temperature,
    read_host_facts,
    read_host_memory,
    read_host_power_draw,
    read_network_interfaces,
    read_primary_interface,
    read_primary_mac_address,
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
# read_host_power_draw
# ---------------------------------------------------------------------------


def _write_sysfs_file(path: Path, contents: str) -> None:
    """Materialise a fake sysfs file, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


class TestReadHostPowerDraw:
    def test_returns_none_on_non_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        assert read_host_power_draw() is None

    def test_returns_none_when_no_strategy_matches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        assert read_host_power_draw(power_sysfs_base=tmp_path) is None
        assert read_host_power_draw(power_sysfs_base=tmp_path / "missing") is None

    def test_reads_jetson_ina3221x_in_milliwatts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """INA3221 ``in_power0_input`` reports mW already (divisor=1)."""
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        _write_sysfs_file(
            tmp_path / "bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
            "8500",
        )
        result = read_host_power_draw(power_sysfs_base=tmp_path)
        assert isinstance(result, HostPowerDraw)
        assert result.milliwatts == pytest.approx(8500.0)
        assert result.source == "ina3221x:in_power0"

    def test_reads_hwmon_and_converts_microwatts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """hwmon and power_supply are in µW; must be divided by 1000."""
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        _write_sysfs_file(tmp_path / "class/hwmon/hwmon0/power1_input", "8500000")
        result = read_host_power_draw(power_sysfs_base=tmp_path)
        assert result is not None
        assert result.milliwatts == pytest.approx(8500.0)
        assert result.source == "hwmon:power1_input"

    def test_reads_power_supply_fallback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """power_supply is the last-resort strategy; label reflects it."""
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        _write_sysfs_file(tmp_path / "class/power_supply/BAT0/power_now", "12500000")
        result = read_host_power_draw(power_sysfs_base=tmp_path)
        assert result is not None
        assert result.milliwatts == pytest.approx(12500.0)
        assert result.source == "power_supply:power_now"

    def test_probe_order_ina3221_beats_hwmon(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Total-board sensors are preferred over generic fallbacks."""
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        _write_sysfs_file(
            tmp_path / "bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
            "8500",
        )
        _write_sysfs_file(tmp_path / "class/hwmon/hwmon0/power1_input", "999000000")
        result = read_host_power_draw(power_sysfs_base=tmp_path)
        assert result is not None
        assert result.source == "ina3221x:in_power0"
        assert result.milliwatts == pytest.approx(8500.0)

    def test_sums_multiple_hwmon_rails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Multi-rail SBCs: matches within one strategy are summed."""
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        _write_sysfs_file(tmp_path / "class/hwmon/hwmon0/power1_input", "3000000")
        _write_sysfs_file(tmp_path / "class/hwmon/hwmon1/power1_input", "5000000")
        result = read_host_power_draw(power_sysfs_base=tmp_path)
        assert result is not None
        assert result.milliwatts == pytest.approx(8000.0)
        assert result.source == "hwmon:power1_input x2"

    def test_falls_through_when_strategy_files_unreadable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Garbage in a matching path must fall through, not return None."""
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        _write_sysfs_file(
            tmp_path / "bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
            "not-a-number",
        )
        _write_sysfs_file(tmp_path / "class/hwmon/hwmon0/power1_input", "7500000")
        result = read_host_power_draw(power_sysfs_base=tmp_path)
        assert result is not None
        assert result.source == "hwmon:power1_input"
        assert result.milliwatts == pytest.approx(7500.0)


# ---------------------------------------------------------------------------
# read_network_interfaces
# ---------------------------------------------------------------------------


def _build_net_sysfs(
    root: Path, interfaces: dict[str, dict[str, str]]
) -> Path:
    """Create a fake ``/sys/class/net`` tree under ``root``.

    ``interfaces`` maps interface name to an optional ``{"address": ...,
    "operstate": ...}`` dict; missing keys simply aren't written, mirroring
    a real sysfs tree that may lack a file for a given driver.

    A truthy ``"device"`` key creates the ``device`` entry that real sysfs
    exposes as a symlink into the device tree for hardware-backed interfaces
    only -- the signal :func:`_is_physical_interface` reads.
    """
    net_base = root / "net"
    net_base.mkdir(parents=True, exist_ok=True)
    for name, files in interfaces.items():
        iface_dir = net_base / name
        iface_dir.mkdir()
        if "address" in files:
            (iface_dir / "address").write_text(files["address"])
        if "operstate" in files:
            (iface_dir / "operstate").write_text(files["operstate"])
        if files.get("device"):
            (iface_dir / "device").mkdir()
    return net_base


class TestReadNetworkInterfaces:
    def test_returns_empty_on_platforms_with_no_reader(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Linux reads sysfs and Darwin parses ``ifconfig``; nothing else is covered."""
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        assert read_network_interfaces() == ()

    def test_returns_empty_when_net_base_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        assert read_network_interfaces(net_base=tmp_path / "missing") == ()

    def test_skips_loopback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        base = _build_net_sysfs(
            tmp_path,
            {
                "lo": {"address": "00:00:00:00:00:00", "operstate": "unknown"},
                "eth0": {"address": "aa:bb:cc:dd:ee:ff", "operstate": "up"},
            },
        )
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._read_ipv4_address",
            lambda ifname: "192.168.1.42" if ifname == "eth0" else None,
        )
        result = read_network_interfaces(net_base=base)
        assert [nic.name for nic in result] == ["eth0"]

    def test_reads_mac_and_up_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        base = _build_net_sysfs(
            tmp_path,
            {
                "eth0": {"address": "aa:bb:cc:dd:ee:ff", "operstate": "up"},
                "wlan0": {"address": "11:22:33:44:55:66", "operstate": "down"},
            },
        )
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._read_ipv4_address",
            lambda ifname: {"eth0": "192.168.1.42"}.get(ifname),
        )
        result = {nic.name: nic for nic in read_network_interfaces(net_base=base)}

        assert result["eth0"] == NetworkInterfaceFacts(
            name="eth0",
            ipv4_address="192.168.1.42",
            mac_address="aa:bb:cc:dd:ee:ff",
            is_up=True,
        )
        # Down interface: no IPv4, ``is_up`` reflects operstate -- MAC is
        # still readable since it's a hardware property independent of link
        # state.
        assert result["wlan0"] == NetworkInterfaceFacts(
            name="wlan0",
            ipv4_address=None,
            mac_address="11:22:33:44:55:66",
            is_up=False,
        )

    def test_missing_address_or_operstate_files_degrade_gracefully(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A driver that doesn't expose ``address``/``operstate`` shouldn't crash the reader."""
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        base = _build_net_sysfs(tmp_path, {"eth0": {}})
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._read_ipv4_address", lambda ifname: None
        )
        result = read_network_interfaces(net_base=base)
        assert len(result) == 1
        assert result[0].mac_address is None
        assert result[0].is_up is False


#: Abridged ``ifconfig -a`` output covering every case the Darwin parser has to
#: get right: loopback, a tunnel with no link layer, an active NIC with both
#: addresses, an unplugged NIC with a MAC but no IP, and an Apple-internal NIC
#: that advertises ``RUNNING`` while its status says otherwise.
_DARWIN_IFCONFIG_SAMPLE = """\
lo0: flags=8049<UP,LOOPBACK,RUNNING,MULTICAST> mtu 16384
\tinet 127.0.0.1 netmask 0xff000000
utun0: flags=8051<UP,POINTOPOINT,RUNNING,MULTICAST> mtu 1380
\tinet6 fe80::1234%utun0 prefixlen 64 scopeid 0x10
anpi0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tether ae:74:6f:9e:d5:dd
\tstatus: inactive
en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tether 5c:e9:1e:6a:2b:11
\tinet6 fe80::abcd%en0 prefixlen 64 secured scopeid 0xb
\tinet 192.168.1.42 netmask 0xffffff00 broadcast 192.168.1.255
\tmedia: autoselect
\tstatus: active
en5: flags=8823<UP,BROADCAST,SMART,SIMPLEX,MULTICAST> mtu 1500
\tether aa:bb:cc:dd:ee:ff
\tstatus: inactive
"""


class TestReadNetworkInterfacesDarwin:
    """macOS enumeration via ``ifconfig``.

    Linux reads sysfs; Darwin has no ``/sys/class/net`` so the same facts
    come from parsing ``ifconfig -a``.  Without this the whole
    ``network_interfaces`` block was empty on every Mac edge (CYB-3232).
    """

    def test_parses_ifconfig_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._ifconfig",
            lambda: _DARWIN_IFCONFIG_SAMPLE,
        )
        result = {nic.name: nic for nic in read_network_interfaces()}

        assert result["en0"] == NetworkInterfaceFacts(
            name="en0",
            ipv4_address="192.168.1.42",
            mac_address="5c:e9:1e:6a:2b:11",
            is_up=True,
        )

    def test_keeps_a_nic_that_has_a_mac_but_no_address(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unplugged NIC is exactly when you need its MAC.

        Pinning a DHCP reservation is what gets the interface a *stable*
        address, so dropping MAC-only interfaces removes the one fact that
        would fix the problem.
        """
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._ifconfig",
            lambda: _DARWIN_IFCONFIG_SAMPLE,
        )
        result = {nic.name: nic for nic in read_network_interfaces()}

        assert result["en5"] == NetworkInterfaceFacts(
            name="en5",
            ipv4_address=None,
            mac_address="aa:bb:cc:dd:ee:ff",
            is_up=False,
        )

    def test_skips_loopback_tunnels_and_apple_internal_interfaces(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stock Mac lists ~30 interfaces; only the physical ones are reachable."""
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._ifconfig",
            lambda: _DARWIN_IFCONFIG_SAMPLE,
        )
        assert [nic.name for nic in read_network_interfaces()] == ["en0", "en5"]

    def test_status_beats_the_running_flag_for_link_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``RUNNING`` is not link state on Darwin.

        Apple's internal NICs advertise ``RUNNING`` permanently, so reading
        the flag alone reports every one of them as up.  ``status:`` is the
        authoritative signal and matches Linux's ``operstate``.
        """
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._ifconfig",
            lambda: "anpi0: flags=8863<UP,RUNNING,MULTICAST> mtu 1500\n"
            "\tether ae:74:6f:9e:d5:dd\n"
            "\tstatus: inactive\n"
            "en0: flags=8863<UP,RUNNING,MULTICAST> mtu 1500\n"
            "\tether 5c:e9:1e:6a:2b:11\n"
            "\tstatus: active\n",
        )
        # ``anpi0`` is filtered out entirely; ``en0`` is the observable case.
        result = {nic.name: nic for nic in read_network_interfaces()}
        assert result["en0"].is_up is True

    def test_returns_empty_when_ifconfig_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._ifconfig", lambda: None
        )
        assert read_network_interfaces() == ()


#: A Linux edge running driver containers, as ``/sys/class/net`` reports it.
#: ``br-*``/``docker0``/``veth*`` all carry real MACs, are ``up``, and sort
#: *ahead* of ``eth0`` -- so "first enumerated interface that is up" picks a
#: container bridge.  Only ``eth0`` and ``wlan0`` have a ``device`` entry.
_CONTAINER_HOST_INTERFACES = {
    "br-1a2b3c4d5e6f": {"address": "02:42:1f:aa:bb:cc", "operstate": "up"},
    "docker0": {"address": "02:42:3c:11:22:33", "operstate": "up"},
    "eth0": {"address": "dc:a6:32:11:22:33", "operstate": "up", "device": True},
    "lo": {"address": "00:00:00:00:00:00", "operstate": "unknown"},
    "veth9c2f0a1": {"address": "8a:1c:4d:7e:22:9b", "operstate": "up"},
    "wlan0": {"address": "dc:a6:32:44:55:66", "operstate": "down", "device": True},
}

_CONTAINER_HOST_IPV4 = {
    "br-1a2b3c4d5e6f": "172.18.0.1",
    "docker0": "172.17.0.1",
    "eth0": "192.168.1.42",
}


#: Abridged ``ifconfig -a`` from a real M-series Mac, covering the three
#: interface families that share the ``en`` prefix with the physical NICs and
#: so cannot be excluded by name: an internal ``anpi`` peer (``media: none``),
#: a Thunderbolt port (a ``member:`` of ``bridge0``), and an unplugged real
#: NIC (``media: autoselect (none)``) that has to survive both rules.
_DARWIN_IFCONFIG_EN_FAMILIES = """\
anpi0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tether ae:74:6f:9e:d5:dd
\tmedia: none
\tstatus: inactive
en6: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tether ae:74:6f:9e:d5:bd
\tmedia: none
\tstatus: inactive
en2: flags=8963<UP,BROADCAST,SMART,RUNNING,PROMISC,SIMPLEX,MULTICAST> mtu 1500
\tether 36:75:47:24:b3:40
\tmedia: autoselect <full-duplex>
\tstatus: inactive
en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tether 1c:1d:d3:eb:80:8c
\tinet 192.168.1.14 netmask 0xffffff00 broadcast 192.168.1.255
\tmedia: autoselect (1000baseT <full-duplex,flow-control>)
\tstatus: active
bridge0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tether 36:75:47:24:b3:40
\tmember: en2 flags=3<LEARNING,DISCOVER>
\tmedia: <unknown type>
\tstatus: inactive
en11: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tether 4c:c5:d9:8f:3f:50
\tmedia: autoselect (none)
\tstatus: inactive
"""


class TestReadNetworkInterfacesDarwinEnFamilies:
    """The ``en``-prefixed interfaces a name blocklist cannot separate.

    A stock Mac reports 11 interfaces after prefix filtering, of which nine
    are Apple-internal peers or Thunderbolt ports -- enough to bury the real
    NIC. They are distinguishable, just not by name (CYB-3232).
    """

    @pytest.fixture(autouse=True)
    def _darwin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._ifconfig",
            lambda: _DARWIN_IFCONFIG_EN_FAMILIES,
        )

    def test_drops_internal_peers_and_thunderbolt_ports(self) -> None:
        assert [nic.name for nic in read_network_interfaces()] == ["en0", "en11"]

    def test_drops_an_interface_with_no_media_layer(self) -> None:
        """``media: none`` means there is no physical layer at all.

        ``en6`` carries a real-looking MAC from the same block as ``anpi0``
        and asserts ``RUNNING``, so nothing else in the stanza gives it away.
        """
        assert "en6" not in {nic.name for nic in read_network_interfaces()}

    def test_drops_a_thunderbolt_port_named_by_its_bridge(self) -> None:
        """``en2`` is listed as a ``member:`` of ``bridge0``.

        The bridge stanza appears *after* its members in ``ifconfig`` output,
        so this only works if membership is applied after the whole scan.
        """
        assert "en2" not in {nic.name for nic in read_network_interfaces()}

    def test_keeps_an_unplugged_physical_nic(self) -> None:
        """``media: autoselect (none)`` is a media layer with no link.

        This is the case the whole field exists for: an unplugged port whose
        MAC you pin a DHCP reservation to so it gets a stable address. It
        must not be swept up with the internal peers.
        """
        result = {nic.name: nic for nic in read_network_interfaces()}
        assert result["en11"] == NetworkInterfaceFacts(
            name="en11",
            ipv4_address=None,
            mac_address="4c:c5:d9:8f:3f:50",
            is_up=False,
        )


class TestReadPrimaryInterface:
    """The single interface the host actually routes out of.

    Selecting by enumeration order and ``is_up`` picks ``br-*`` or
    ``docker0`` on any edge running driver containers, because sysfs
    enumeration is alphabetical and those sort ahead of ``eth0``. Asking the
    kernel which interface holds the default route cannot make that mistake:
    a bridge or ``veth`` peer never holds it (CYB-3232).
    """

    @pytest.fixture
    def container_host(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> Path:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._read_ipv4_address",
            lambda name: _CONTAINER_HOST_IPV4.get(name),
        )
        return _build_net_sysfs(tmp_path, _CONTAINER_HOST_INTERFACES)

    def test_picks_the_default_route_interface_not_the_first_up_one(
        self, monkeypatch: pytest.MonkeyPatch, container_host: Path
    ) -> None:
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._route_source_address",
            lambda: "192.168.1.42",
        )
        primary = read_primary_interface(net_base=container_host)
        assert primary is not None
        assert primary.name == "eth0"
        assert primary.mac_address == "dc:a6:32:11:22:33"

    def test_does_not_pick_a_container_bridge(
        self, monkeypatch: pytest.MonkeyPatch, container_host: Path
    ) -> None:
        """``02:42:…`` is Docker's own OUI -- an unmistakable wrong answer."""
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._route_source_address",
            lambda: "192.168.1.42",
        )
        primary = read_primary_interface(net_base=container_host)
        assert primary is not None
        assert not primary.name.startswith(("br-", "docker", "veth"))
        assert not primary.mac_address.startswith("02:42:")

    def test_falls_back_to_a_physical_interface_when_there_is_no_route(
        self, monkeypatch: pytest.MonkeyPatch, container_host: Path
    ) -> None:
        """An edge on an isolated LAN, or one that booted before DHCP.

        ``/sys/class/net/<if>/device`` exists only for hardware-backed
        interfaces, so it separates ``eth0``/``wlan0`` from the bridges and
        ``veth`` peers without a list of names to keep up to date.
        """
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._route_source_address", lambda: None
        )
        primary = read_primary_interface(net_base=container_host)
        assert primary is not None
        assert primary.name == "eth0"

    def test_fallback_prefers_an_up_physical_interface(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._read_ipv4_address", lambda name: None
        )
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._route_source_address", lambda: None
        )
        base = _build_net_sysfs(
            tmp_path,
            {
                "eth0": {"address": "aa:bb:cc:dd:ee:01", "operstate": "down", "device": True},
                "wlan0": {"address": "aa:bb:cc:dd:ee:02", "operstate": "up", "device": True},
            },
        )
        primary = read_primary_interface(net_base=base)
        assert primary is not None
        assert primary.name == "wlan0"

    def test_returns_none_when_nothing_was_enumerated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        assert read_primary_interface() is None

    def test_returns_none_when_no_interface_is_physical(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A container-only interface set is not something to hand an operator."""
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._read_ipv4_address", lambda name: None
        )
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._route_source_address", lambda: None
        )
        base = _build_net_sysfs(
            tmp_path, {"docker0": {"address": "02:42:3c:11:22:33", "operstate": "up"}}
        )
        assert read_primary_interface(net_base=base) is None

    def test_route_probe_never_transmits(self) -> None:
        """The probe address is reserved and the connect is UDP: no packets.

        Guards the property that makes calling this every 30 s acceptable --
        if the address or socket type ever changed, an edge would start
        emitting traffic to a documentation-only prefix.
        """
        assert host_metrics_module._ROUTE_PROBE_ADDRESS[0].startswith("192.0.2.")


class TestReadPrimaryMacAddress:
    """Cross-platform fallback MAC, from the same source as the fingerprint."""

    def test_formats_getnode_as_a_mac(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics.getnode", lambda: 0x5CE91E6A2B11
        )
        assert read_primary_mac_address() == "5c:e9:1e:6a:2b:11"

    def test_returns_none_when_getnode_fabricated_a_random_node_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``uuid.getnode()`` invents a random node ID when no NIC is readable.

        RFC 4122 has it set the multicast bit to mark the value as
        not-a-hardware-address.  Uploading it anyway would put a
        MAC-shaped string on the dashboard that matches no NIC on earth --
        worse than showing nothing, since the operator would paste it into
        a DHCP reservation that can never fire.
        """
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics.getnode", lambda: 0x010203040506
        )
        assert read_primary_mac_address() is None

    def test_returns_none_when_getnode_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom() -> int:
            raise OSError("no interfaces")

        monkeypatch.setattr("cyberwave.edge.host_metrics.getnode", boom)
        assert read_primary_mac_address() is None


class TestReadDeviceSerial:
    def test_returns_none_on_non_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        assert read_device_serial() is None

    def test_reads_raspberry_pi_serial_from_cpuinfo(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        cpuinfo = tmp_path / "cpuinfo"
        cpuinfo.write_text(
            "processor\t: 0\n"
            "Model\t\t: Raspberry Pi 5 Model B Rev 1.0\n"
            "Serial\t\t: 100000001234abcd\n"
        )
        assert (
            read_device_serial(cpuinfo_path=cpuinfo, device_tree_serial=tmp_path / "nope")
            == "100000001234abcd"
        )

    def test_falls_back_to_the_device_tree_on_jetson(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Tegra kernels carry no ``Serial`` line; the serial is in the device tree.

        The device-tree node is NUL-terminated, which has to be stripped or
        the value round-trips through JSON with a trailing ``\\x00``.
        """
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        cpuinfo = tmp_path / "cpuinfo"
        cpuinfo.write_text("processor\t: 0\nmodel name\t: ARMv8\n")
        dt_serial = tmp_path / "serial-number"
        dt_serial.write_bytes(b"1421921012345\x00")

        assert (
            read_device_serial(cpuinfo_path=cpuinfo, device_tree_serial=dt_serial)
            == "1421921012345"
        )

    def test_returns_none_when_no_source_exposes_a_serial(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        assert (
            read_device_serial(
                cpuinfo_path=tmp_path / "missing",
                device_tree_serial=tmp_path / "also-missing",
            )
            is None
        )

    def test_ignores_the_all_zero_placeholder_serial(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Some boards expose ``Serial : 0000000000000000`` rather than omitting it."""
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        cpuinfo = tmp_path / "cpuinfo"
        cpuinfo.write_text("Serial\t\t: 0000000000000000\n")
        assert (
            read_device_serial(cpuinfo_path=cpuinfo, device_tree_serial=tmp_path / "nope")
            is None
        )


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
        assert out["edge_core_version"] == "0.1.4"

    def test_partial_versions_omits_unset_packages(self) -> None:
        """A standalone SDK install (no edge-core) reports only
        ``sdk_version`` — the absent companion must not leak as ``None``
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
            edge_core_version=None,
        )
        out = facts.to_dict()
        assert out["sdk_version"] == "0.4.7"
        assert "edge_core_version" not in out
        # ``cli_version`` is intentionally not part of the host_facts
        # schema: edge-core (the sole producer) cannot observe the
        # standalone CLI binary on production edges.
        assert "cli_version" not in out

    def test_gauges_are_omitted_when_unread_but_kept_at_zero(self) -> None:
        """The gauges follow the same present-key contract as everything else.

        ``0.0`` is a legitimate reading (an idle percentage, a sub-zero-ish
        sensor) so it must survive into the JSON; only ``None`` is dropped.
        A truthiness check here would silently erase real data.
        """
        base_kwargs = dict(
            platform="Linux-aarch64",
            kernel=None,
            memory_total_mb=None,
            cpu_model=None,
            cpu_count=None,
            thermal_source=None,
            has_hardware_watchdog=False,
            sdk_version=None,
            edge_core_version=None,
        )

        unread = HostFacts(**base_kwargs).to_dict()
        assert "cpu_temp_c" not in unread
        assert "memory_used_percent" not in unread
        assert "memory_available_mb" not in unread

        at_zero = HostFacts(
            **base_kwargs,
            cpu_temp_c=0.0,
            memory_used_percent=0.0,
            memory_available_mb=0.0,
        ).to_dict()
        assert at_zero["cpu_temp_c"] == 0.0
        assert at_zero["memory_used_percent"] == 0.0
        assert at_zero["memory_available_mb"] == 0.0

    def test_network_interfaces_included_only_when_non_empty(self) -> None:
        base_kwargs = dict(
            platform="Linux-aarch64",
            kernel=None,
            memory_total_mb=None,
            cpu_model=None,
            cpu_count=None,
            thermal_source=None,
            has_hardware_watchdog=False,
            sdk_version=None,
            edge_core_version=None,
        )

        assert "network_interfaces" not in HostFacts(**base_kwargs).to_dict()

        facts = HostFacts(
            **base_kwargs,
            network_interfaces=(
                NetworkInterfaceFacts(
                    name="eth0",
                    ipv4_address="192.168.1.42",
                    mac_address="aa:bb:cc:dd:ee:ff",
                    is_up=True,
                ),
            ),
        )
        out = facts.to_dict()
        assert out["network_interfaces"] == [
            {
                "name": "eth0",
                "ipv4_address": "192.168.1.42",
                "mac_address": "aa:bb:cc:dd:ee:ff",
                "is_up": True,
            }
        ]

    def test_primary_mac_comes_from_the_default_route_interface(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``host_facts`` carries ONE MAC, and it is the routable one.

        The dashboard renders ``primary_mac_address`` with a copy button next
        to it, so it has to be the MAC a DHCP reservation keys on -- not
        whichever interface happened to sort first (CYB-3232).
        """
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._read_ipv4_address",
            lambda name: _CONTAINER_HOST_IPV4.get(name),
        )
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._route_source_address",
            lambda: "192.168.1.42",
        )
        base = _build_net_sysfs(tmp_path, _CONTAINER_HOST_INTERFACES)

        out = read_host_facts(net_base=base).to_dict()
        assert out["primary_mac_address"] == "dc:a6:32:11:22:33"
        assert out["primary_interface_name"] == "eth0"

    def test_primary_interface_name_absent_when_the_mac_came_from_getnode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The two provenances have to stay distinguishable on the wire.

        ``uuid.getnode()`` is not tied to a reachable interface, so a
        consumer must be able to tell "this is the routable MAC" from "this
        is the only MAC we could find at all".
        """
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics.getnode", lambda: 0x5CE91E6A2B11
        )
        base = _build_net_sysfs(tmp_path, {})

        out = read_host_facts(net_base=base).to_dict()
        assert out["primary_mac_address"] == "5c:e9:1e:6a:2b:11"
        assert "primary_interface_name" not in out

    def test_identity_fields_included_only_when_known(self) -> None:
        base_kwargs = dict(
            platform="Linux-aarch64",
            kernel=None,
            memory_total_mb=None,
            cpu_model=None,
            cpu_count=None,
            thermal_source=None,
            has_hardware_watchdog=False,
            sdk_version=None,
            edge_core_version=None,
        )

        bare = HostFacts(**base_kwargs).to_dict()
        assert "primary_mac_address" not in bare
        assert "device_serial" not in bare

        out = HostFacts(
            **base_kwargs,
            primary_mac_address="5c:e9:1e:6a:2b:11",
            device_serial="100000001234abcd",
        ).to_dict()
        assert out["primary_mac_address"] == "5c:e9:1e:6a:2b:11"
        assert out["device_serial"] == "100000001234abcd"


class TestSoftwareVersions:
    """Cover the layered version resolution (in-process ``__version__``
    first, then ``importlib.metadata``) independently of whatever happens
    to be installed in the test venv."""

    def test_prefers_in_process_dunder_version_over_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In-process ``__version__`` wins over ``importlib.metadata``.

        This is what surfaces CI ``BUILD_VERSION`` stamps and survives
        PyInstaller binaries that ship without ``.dist-info``.
        """
        module_versions = {
            "cyberwave": "1.2.3",
            "cyberwave_edge_core": "7.8.9",
        }
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._read_module_version",
            lambda name: module_versions.get(name),
        )
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._pkg_version",
            lambda name: "0.0.0",
        )
        from cyberwave.edge.host_metrics import _read_software_versions

        assert _read_software_versions() == ("1.2.3", "7.8.9")

    def test_falls_back_to_importlib_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from importlib.metadata import PackageNotFoundError

        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._read_module_version",
            lambda name: "0.4.7" if name == "cyberwave" else None,
        )

        def fake(name: str) -> str:
            raise PackageNotFoundError(name)

        monkeypatch.setattr("cyberwave.edge.host_metrics._pkg_version", fake)
        from cyberwave.edge.host_metrics import _read_software_versions

        assert _read_software_versions() == ("0.4.7", None)

    def test_metadata_corruption_does_not_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A malformed ``METADATA`` file inside site-packages can raise
        non-``PackageNotFoundError`` exceptions on some Python builds; the
        defensive catch-all must keep the reader functional."""

        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._read_module_version",
            lambda name: None,
        )

        def fake(name: str) -> str:
            raise RuntimeError("malformed metadata for " + name)

        monkeypatch.setattr("cyberwave.edge.host_metrics._pkg_version", fake)
        from cyberwave.edge.host_metrics import _read_software_versions

        assert _read_software_versions() == (None, None)


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
            return subprocess.CompletedProcess(
                cmd, 0 if value else 1, stdout=value, stderr=""
            )

        monkeypatch.setattr("cyberwave.edge.host_metrics.shutil.which", fake_which)
        monkeypatch.setattr("cyberwave.edge.host_metrics.subprocess.run", fake_run)

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
            return subprocess.CompletedProcess(
                cmd, 0 if value else 1, stdout=value, stderr=""
            )

        monkeypatch.setattr("cyberwave.edge.host_metrics.shutil.which", fake_which)
        monkeypatch.setattr("cyberwave.edge.host_metrics.subprocess.run", fake_run)

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
                return subprocess.CompletedProcess(
                    cmd, 0, stdout=str(odd_bytes), stderr=""
                )
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

        monkeypatch.setattr("cyberwave.edge.host_metrics.shutil.which", fake_which)
        monkeypatch.setattr("cyberwave.edge.host_metrics.subprocess.run", fake_run)

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
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="not-an-int", stderr=""
                )
            if cmd[2] == "machdep.cpu.brand_string":
                return subprocess.CompletedProcess(cmd, 0, stdout="Apple M3", stderr="")
            if cmd[2] in ("hw.logicalcpu", "hw.ncpu"):
                return subprocess.CompletedProcess(cmd, 0, stdout="garbage", stderr="")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

        monkeypatch.setattr("cyberwave.edge.host_metrics.shutil.which", fake_which)
        monkeypatch.setattr("cyberwave.edge.host_metrics.subprocess.run", fake_run)

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
            "MemTotal:        3906292 kB\nMemAvailable:    2000000 kB\n"
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

        thermal_base = _build_thermal_sysfs(tmp_path, [("cpu-thermal", 55_000)])
        watchdog_dev = tmp_path / "watchdog"
        watchdog_dev.write_text("")  # presence-only check
        net_base = _build_net_sysfs(
            tmp_path, {"eth0": {"address": "aa:bb:cc:dd:ee:ff", "operstate": "up"}}
        )
        monkeypatch.setattr(
            "cyberwave.edge.host_metrics._read_ipv4_address",
            lambda ifname: "192.168.1.42" if ifname == "eth0" else None,
        )

        facts = read_host_facts(
            thermal_base=thermal_base, watchdog_device=watchdog_dev, net_base=net_base
        )

        assert facts.memory_total_mb == pytest.approx(3906292 / 1024, abs=1)
        assert facts.cpu_model == "Cortex-A72"
        assert facts.cpu_count == 4
        assert facts.thermal_source is not None
        assert "cpu-thermal" in facts.thermal_source
        assert facts.has_hardware_watchdog is True
        # Gauge copy: read from the same sysfs/procfs sources as the
        # heartbeat's dynamic readers, so the dashboard keeps a reading
        # after the bootstrap publisher hands over to the drivers.
        assert facts.cpu_temp_c == pytest.approx(55.0)
        assert facts.memory_available_mb == pytest.approx(2000000 / 1024, abs=1)
        assert facts.memory_used_percent == pytest.approx(48.8, abs=0.2)
        assert facts.network_interfaces == (
            NetworkInterfaceFacts(
                name="eth0",
                ipv4_address="192.168.1.42",
                mac_address="aa:bb:cc:dd:ee:ff",
                is_up=True,
            ),
        )

    def test_gauge_copy_reports_hottest_zone_not_the_source_zone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``cpu_temp_c`` is the hottest zone; ``thermal_source`` names the first.

        These deliberately disagree. ``thermal_source`` is an identity
        field (which sysfs path the publisher reads) and its existing
        meaning must not shift, while a temperature reading is only useful
        as the worst case across cores. Zone 0 is the coolest here so a
        regression that collapsed the two would show 40, not 71.
        """
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        thermal_base = _build_thermal_sysfs(
            tmp_path,
            [("cpu-thermal", 40_000), ("coretemp", 71_000), ("coretemp", 66_000)],
        )

        facts = read_host_facts(
            thermal_base=thermal_base, watchdog_device=tmp_path / "absent"
        )

        assert facts.cpu_temp_c == pytest.approx(71.0)
        assert facts.thermal_source is not None
        assert facts.thermal_source.startswith("thermal_zone0")

    def test_gauge_copy_is_none_when_no_thermal_zone_is_readable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A host with no CPU thermal zone omits the reading rather than
        reporting 0 °C, which the dashboard would colour as healthy."""
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        empty_thermal = tmp_path / "thermal"
        empty_thermal.mkdir()

        facts = read_host_facts(
            thermal_base=empty_thermal, watchdog_device=tmp_path / "absent"
        )

        assert facts.cpu_temp_c is None
        assert facts.thermal_source is None
        assert "cpu_temp_c" not in facts.to_dict()

    def test_carries_power_draw_and_battery_pack(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Power rides the keepalive too, so the dashboard's host row keeps the
        same field set once drivers stop the heartbeat that used to carry it."""
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setenv("CYBERWAVE_BATTERY_WH", "86.7")
        _write_sysfs_file(tmp_path / "class/hwmon/hwmon2/power1_input", "12300000")

        facts = read_host_facts(
            thermal_base=tmp_path / "no-thermal",
            watchdog_device=tmp_path / "absent",
            power_sysfs_base=tmp_path,
        )

        assert facts.power_mw == pytest.approx(12300.0)
        assert facts.battery_wh == pytest.approx(86.7)
        assert facts.to_dict()["power_mw"] == pytest.approx(12300.0)
        assert facts.to_dict()["battery_wh"] == pytest.approx(86.7)

    def test_omits_power_and_battery_when_unreadable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No rail and an unparseable pack size drop out rather than reporting 0,
        which the dashboard would render as a board drawing no power."""
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setenv("CYBERWAVE_BATTERY_WH", "not-a-float")

        facts = read_host_facts(
            thermal_base=tmp_path / "no-thermal",
            watchdog_device=tmp_path / "absent",
            power_sysfs_base=tmp_path / "no-power",
        )

        assert facts.power_mw is None
        assert facts.battery_wh is None
        assert "power_mw" not in facts.to_dict()
        assert "battery_wh" not in facts.to_dict()

    @pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "1e400"])
    def test_omits_non_finite_battery_pack(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: str
    ) -> None:
        """``float()`` accepts these without raising, but they serialise as bare
        ``NaN``/``Infinity`` tokens that PostgreSQL's jsonb rejects -- which would
        fail the whole /discover POST, and with it the ``last_seen_at`` bump that
        keeps a healthy edge out of "Offline"."""
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setenv("CYBERWAVE_BATTERY_WH", raw)

        facts = read_host_facts(
            thermal_base=tmp_path / "no-thermal",
            watchdog_device=tmp_path / "absent",
            power_sysfs_base=tmp_path / "no-power",
        )

        assert facts.battery_wh is None
        assert "battery_wh" not in facts.to_dict()
        # The payload has to survive a strict encoder, not just Python's.
        json.dumps(facts.to_dict(), allow_nan=False)

    def test_parses_arm_hardware_field_when_no_model_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Older Raspberry Pi kernels emit ``Hardware:`` instead of ``model name``."""
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        cpuinfo_path = tmp_path / "cpuinfo"
        cpuinfo_path.write_text("processor\t: 0\nHardware\t: BCM2835\n")
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
