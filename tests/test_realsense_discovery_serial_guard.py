"""``RealSenseDiscovery.get_device_info`` must survive serial-less devices.

librealsense raises from ``get_info`` -- it does not return ``None`` -- when a
device cannot report a field. Serial-less entries are routine: a unit already
opened by another process, one in recovery/DFU mode, one caught mid-enumeration.

Before the guard, reading through such a device aborted the whole search, so a
correctly pinned camera became unopenable purely because another camera sorted
ahead of it in the enumeration. On a two-RealSense host that meant whichever
driver started second could never find its own camera, no matter how correct
its ``metadata.serial_number`` was.

No hardware needed: ``rs`` is stubbed at module level.
"""

from __future__ import annotations

import pytest

from cyberwave.sensor import config as rs_config


class _CameraInfo:
    """Stand-in for ``rs.camera_info`` -- identity-comparable field tokens."""

    name = "name"
    serial_number = "serial_number"
    firmware_version = "firmware_version"
    usb_type_descriptor = "usb_type_descriptor"
    product_line = "product_line"
    physical_port = "physical_port"


class _FakeSensor:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_info(self, field):
        return self._name

    def get_stream_profiles(self):
        return []


class _FakeDevice:
    """A librealsense device whose unreadable fields raise, as the real one does."""

    def __init__(self, fields: dict[str, str]) -> None:
        self._fields = fields

    def supports(self, field) -> bool:
        return field in self._fields

    def get_info(self, field) -> str:
        if field not in self._fields:
            raise RuntimeError(f"info {field} not supported by the device!")
        return self._fields[field]

    def query_sensors(self):
        return [_FakeSensor("Stereo Module")]


def _device(serial: str | None, busid: str | None = None) -> _FakeDevice:
    fields = {
        _CameraInfo.name: "Intel RealSense D435",
        _CameraInfo.firmware_version: "5.13.0.50",
    }
    if serial is not None:
        fields[_CameraInfo.serial_number] = serial
    if busid is not None:
        fields[_CameraInfo.physical_port] = (
            f"/sys/devices/.../usb{busid[0]}/{busid}/{busid}:1.0/video4linux/video0"
        )
    return _FakeDevice(fields)


class _FakeContext:
    def __init__(self, devices: list[_FakeDevice]) -> None:
        self._devices = devices

    def query_devices(self):
        return self._devices


@pytest.fixture
def fake_rs(monkeypatch):
    """Install a stub ``rs`` module and let discovery believe it is present."""

    def _install(devices: list[_FakeDevice]):
        stub = type("_RsStub", (), {})()
        stub.camera_info = _CameraInfo
        stub.context = lambda: _FakeContext(devices)
        monkeypatch.setattr(rs_config, "rs", stub, raising=False)
        monkeypatch.setattr(rs_config, "_has_realsense", True, raising=False)

    return _install


def test_serial_less_device_does_not_abort_the_search(fake_rs):
    """The regression: an unreadable device must be skipped, not fatal.

    Ordering is the whole point -- the serial-less device comes *first*, so a
    lookup that reads through it dies before reaching the requested camera.
    """
    fake_rs([_device(None), _device("213722070420")])

    info = rs_config.RealSenseDiscovery.get_device_info("213722070420")

    assert info is not None
    assert info.serial_number == "213722070420"


def test_requested_serial_still_found_when_listed_first(fake_rs):
    """Guarding must not disturb the ordinary path."""
    fake_rs([_device("213722070420"), _device("918512073127")])

    info = rs_config.RealSenseDiscovery.get_device_info("213722070420")

    assert info is not None
    assert info.serial_number == "213722070420"


def test_absent_serial_returns_none_rather_than_a_substitute(fake_rs):
    """A pin that matches nothing must not silently bind another camera.

    The driver turns this ``None`` into a hard "could not be opened" error; if
    it returned a device instead, the twin would stream different footage.
    """
    fake_rs([_device(None), _device("918512073127")])

    assert rs_config.RealSenseDiscovery.get_device_info("213722070420") is None


def test_unpinned_lookup_skips_serial_less_devices(fake_rs):
    """``serial_number=None`` picks the first *usable* device.

    Previously it took the first device outright, then raised while filling in
    ``RealSenseDeviceInfo.serial_number`` -- the same bug one line further on.
    """
    fake_rs([_device(None), _device("918512073127")])

    info = rs_config.RealSenseDiscovery.get_device_info()

    assert info is not None
    assert info.serial_number == "918512073127"


def test_all_devices_serial_less_returns_none(fake_rs):
    """No readable device is 'not found', not a crash."""
    fake_rs([_device(None), _device(None)])

    assert rs_config.RealSenseDiscovery.get_device_info() is None
    assert rs_config.RealSenseDiscovery.get_device_info("213722070420") is None


# ---------------------------------------------------------------------------
# USB descriptor serial alias
#
# ``RealSenseConfig.from_device`` calls ``get_device_info`` before the streamer
# ever runs its own alias, so discovery has to accept the USB serial too.
# Observed on a Pi 5: a twin pinned to USB serial 214523026669 crash-looped
# with "No RealSense device found" although that camera (librealsense serial
# 213722070420) was on the bus.
# ---------------------------------------------------------------------------


@pytest.fixture
def usb_serials(tmp_path, monkeypatch):
    """Point sysfs at *tmp_path* and write ``{busid: usb_serial}`` into it."""

    def _write(serials: dict[str, str]) -> None:
        for busid, serial in serials.items():
            (tmp_path / busid).mkdir()
            (tmp_path / busid / "serial").write_text(serial + "\n")
        monkeypatch.setattr(rs_config, "_SYS_USB_DEVICES", str(tmp_path))

    return _write


def test_usb_descriptor_serial_resolves_to_its_device(fake_rs, usb_serials):
    fake_rs(
        [_device("918512073127", busid="2-1"), _device("213722070420", busid="4-1")]
    )
    usb_serials({"2-1": "926223022293", "4-1": "214523026669"})

    info = rs_config.RealSenseDiscovery.get_device_info("214523026669")

    assert info is not None
    assert info.serial_number == "213722070420"


def test_librealsense_serial_wins_over_a_colliding_usb_serial(fake_rs, usb_serials):
    """An exact librealsense match is taken even if another device's USB serial
    happens to carry the same value."""
    fake_rs(
        [_device("918512073127", busid="2-1"), _device("213722070420", busid="4-1")]
    )
    usb_serials({"2-1": "213722070420", "4-1": "214523026669"})

    info = rs_config.RealSenseDiscovery.get_device_info("213722070420")

    assert info is not None
    assert info.serial_number == "213722070420"


def test_unknown_usb_serial_still_returns_none(fake_rs, usb_serials):
    fake_rs([_device("213722070420", busid="4-1")])
    usb_serials({"4-1": "214523026669"})

    assert rs_config.RealSenseDiscovery.get_device_info("999999999999") is None


def test_config_from_device_accepts_a_usb_serial(fake_rs, usb_serials, monkeypatch):
    """The call the driver actually makes: it must not raise for a USB serial."""
    fake_rs([_device("213722070420", busid="4-1")])
    usb_serials({"4-1": "214523026669"})
    monkeypatch.setattr(
        rs_config.RealSenseDeviceInfo,
        "get_color_resolutions",
        lambda self: [(640, 480)],
    )
    monkeypatch.setattr(
        rs_config.RealSenseDeviceInfo,
        "get_color_fps_options",
        lambda self, w, h, *a, **k: [30],
    )
    monkeypatch.setattr(
        rs_config.RealSenseDeviceInfo, "get_color_formats", lambda self, w, h: ["BGR8"]
    )

    config = rs_config.RealSenseConfig.from_device(
        serial_number="214523026669", enable_depth=False
    )

    # The requested serial is handed through; the streamer aliases it again
    # when it pins the pipeline.
    assert config.serial_number == "214523026669"
