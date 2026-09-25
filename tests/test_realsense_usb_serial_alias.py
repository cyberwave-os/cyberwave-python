"""A RealSense twin may be pinned by *either* of its two serials.

A D400-series camera reports two unrelated values:

* ``camera_info.serial_number`` -- what librealsense matches in
  ``config.enable_device``.
* the USB descriptor's ``iSerial`` -- what udev, ``v4l2-ctl`` and therefore the
  CLI's camera discovery see.

They are different numbers (observed on a Pi 5: librealsense ``213722070420``
is USB ``214523026669``). Every host-side tool that could populate a twin's
``metadata.serial_number`` can only see the USB one, so before this alias a
correctly-pinned depth twin failed with "serial not found" while the camera sat
right there on the bus.

``_usb_descriptor_serial`` is exercised directly; the enumeration guard is
covered alongside it because both live in the same loop.
"""

from __future__ import annotations

import pytest

# The helpers live in ``config`` (discovery needs them, and ``camera_rs``
# imports from there), so no pyav is required to exercise them.
from cyberwave.sensor import config as rs_config


class _CameraInfo:
    name = "name"
    serial_number = "serial_number"
    physical_port = "physical_port"


class _FakeDevice:
    def __init__(self, fields: dict[str, str]) -> None:
        self._fields = fields

    def supports(self, field) -> bool:
        return field in self._fields

    def get_info(self, field) -> str:
        if field not in self._fields:
            raise RuntimeError(f"info {field} not supported by the device!")
        return self._fields[field]


@pytest.fixture
def fake_rs(monkeypatch):
    stub = type("_RsStub", (), {})()
    stub.camera_info = _CameraInfo
    monkeypatch.setattr(rs_config, "rs", stub, raising=False)
    return stub


def _device(busid: str | None = "4-1", rs_serial: str | None = "213722070420"):
    fields: dict[str, str] = {_CameraInfo.name: "Intel RealSense D435"}
    if rs_serial is not None:
        fields[_CameraInfo.serial_number] = rs_serial
    if busid is not None:
        fields[_CameraInfo.physical_port] = (
            f"/sys/devices/platform/axi/1000120000.pcie/1f00200000.usb/xhci-hcd.1/"
            f"usb4/{busid}/{busid}:1.0/video4linux/video0"
        )
    return _FakeDevice(fields)


def test_usb_serial_is_read_from_sysfs(fake_rs, tmp_path, monkeypatch):
    """The bus id is extracted from ``physical_port`` and read out of sysfs."""
    (tmp_path / "4-1").mkdir()
    (tmp_path / "4-1" / "serial").write_text("214523026669\n")
    monkeypatch.setattr(rs_config, "_SYS_USB_DEVICES", str(tmp_path))

    assert rs_config._usb_descriptor_serial(_device()) == "214523026669"


def test_busid_parsing_stops_at_the_interface(fake_rs, tmp_path, monkeypatch):
    """``4-1:1.0`` is an interface, not a device -- the bus id is ``4-1``.

    Matching the interface instead would look up a sysfs path that never
    exists, silently disabling the alias.
    """
    (tmp_path / "4-1").mkdir()
    (tmp_path / "4-1" / "serial").write_text("214523026669\n")
    monkeypatch.setattr(rs_config, "_SYS_USB_DEVICES", str(tmp_path))

    assert rs_config._usb_descriptor_serial(_device(busid="4-1")) == "214523026669"


def test_multi_level_busid_is_handled(fake_rs, tmp_path, monkeypatch):
    """A camera behind a hub has a dotted bus id (``1-1.4``)."""
    (tmp_path / "1-1.4").mkdir()
    (tmp_path / "1-1.4" / "serial").write_text("A260203000509667\n")
    monkeypatch.setattr(rs_config, "_SYS_USB_DEVICES", str(tmp_path))

    assert rs_config._usb_descriptor_serial(_device(busid="1-1.4")) == "A260203000509667"


def test_missing_sysfs_entry_returns_none(fake_rs, tmp_path, monkeypatch):
    """Best-effort: an unreadable serial disables the alias, never raises."""
    monkeypatch.setattr(rs_config, "_SYS_USB_DEVICES", str(tmp_path))

    assert rs_config._usb_descriptor_serial(_device()) is None


def test_device_without_physical_port_returns_none(fake_rs, tmp_path, monkeypatch):
    """``get_info`` would raise here -- the helper must absorb it."""
    monkeypatch.setattr(rs_config, "_SYS_USB_DEVICES", str(tmp_path))

    assert rs_config._usb_descriptor_serial(_device(busid=None)) is None


def test_unparseable_port_returns_none(fake_rs, tmp_path, monkeypatch):
    """A port string in an unexpected shape must not raise."""
    monkeypatch.setattr(rs_config, "_SYS_USB_DEVICES", str(tmp_path))
    dev = _FakeDevice(
        {_CameraInfo.serial_number: "1", _CameraInfo.physical_port: "/dev/video0"}
    )

    assert rs_config._usb_descriptor_serial(dev) is None


def test_busid_parser_handles_each_shape():
    """Parse the bus id directly -- the alias silently dies if this is wrong.

    Regression: an earlier version used a regex that was mangled by shell
    escaping when shipped into a container, so it matched nothing and the
    alias quietly did not exist. Parsing by split has no escape layer to lose.
    """
    parse = rs_config._usb_busid_from_physical_port

    assert parse("/sys/devices/.../xhci-hcd.0/usb2/2-1/2-1:1.0/video4linux/video6") == "2-1"
    assert parse("/sys/devices/.../usb4/4-1/4-1:1.0/video4linux/video0") == "4-1"
    assert parse("/sys/devices/.../usb1/1-1.4/1-1.4:1.0/video4linux/video8") == "1-1.4"
    # ``usb`` must be followed by digits -- ``usbmisc`` is not a root hub.
    assert parse("/sys/devices/.../usbmisc/foo/bar") is None
    assert parse("/dev/video0") is None
    assert parse("") is None
