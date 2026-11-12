"""SocketCAN discovery and bring-up helpers (``cyberwave.driver.support.can``)."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from cyberwave.driver.support.can import (
    CanSetupError,
    _parse_can_interfaces_text,
    bring_up_can_interface,
    describe_can_ports,
    list_can_interfaces,
)


def test_parse_can_interfaces_text() -> None:
    output = "3: can0: <NOARP,UP,LOWER_UP,ECHO> mtu 16\n4: can1: <NOARP,ECHO> mtu 16\n"
    assert _parse_can_interfaces_text(output) == ["can0", "can1"]


def test_list_can_interfaces_json() -> None:
    payload = '[{"ifname":"can0","operstate":"UP"}]'
    with patch(
        "cyberwave.driver.support.can.subprocess.run",
        return_value=MagicMock(returncode=0, stdout=payload, stderr=""),
    ):
        assert list_can_interfaces() == ["can0"]


def test_list_can_interfaces_empty_on_cmd_failure() -> None:
    with patch(
        "cyberwave.driver.support.can.subprocess.run",
        return_value=MagicMock(returncode=1, stdout="", stderr=""),
    ):
        assert list_can_interfaces() == []


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


def test_bring_up_single_interface_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("os.geteuid", lambda: 0, raising=False)
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:4] == ["ip", "-details", "link", "show"]:
            return _completed(stdout="2: can0: <NOARP,ECHO> mtu 16")  # down
        return _completed()

    with (
        patch(
            "cyberwave.driver.support.can.list_can_interfaces", return_value=["can0"]
        ),
        patch("cyberwave.driver.support.can.subprocess.run", side_effect=fake_run),
    ):
        bring_up_can_interface("can0", 1_000_000)

    assert ["ip", "link", "set", "can0", "type", "can", "bitrate", "1000000"] in calls
    assert ["ip", "link", "set", "can0", "up"] in calls
    # root => no sudo prefix
    assert all(cmd[0] != "sudo" for cmd in calls)


def test_bring_up_idempotent_when_already_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("os.geteuid", lambda: 0, raising=False)
    show = (
        "3: can0: <NOARP,UP,LOWER_UP,ECHO> mtu 16 ... bitrate 1000000 sample-point ..."
    )

    def fake_run(cmd, **kwargs):
        if cmd[:4] == ["ip", "-details", "link", "show"]:
            return _completed(stdout=show)
        raise AssertionError(f"unexpected mutation: {cmd}")

    with (
        patch(
            "cyberwave.driver.support.can.list_can_interfaces", return_value=["can0"]
        ),
        patch("cyberwave.driver.support.can.subprocess.run", side_effect=fake_run),
    ):
        bring_up_can_interface("can0", 1_000_000)


def test_bring_up_selects_by_usb_address_and_renames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("os.geteuid", lambda: 0, raising=False)
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["ethtool", "-i"]:
            iface = cmd[2]
            bus = "1-2.2:1.0" if iface == "can5" else "1-1:1.0"
            return _completed(stdout=f"driver: gs_usb\nbus-info: {bus}\n")
        if cmd[:4] == ["ip", "-details", "link", "show"]:
            return _completed(stdout="9: can5: <NOARP,ECHO> mtu 16")  # down
        return _completed()

    with (
        patch(
            "cyberwave.driver.support.can.list_can_interfaces",
            return_value=["can0", "can5"],
        ),
        patch("cyberwave.driver.support.can.subprocess.run", side_effect=fake_run),
    ):
        bring_up_can_interface("can0", 1_000_000, usb_address="1-2.2:1.0")

    assert ["ip", "link", "set", "can5", "name", "can0"] in calls
    assert ["ip", "link", "set", "can0", "up"] in calls


def test_bring_up_ambiguous_without_usb_address_raises() -> None:
    with (
        patch(
            "cyberwave.driver.support.can.list_can_interfaces",
            return_value=["can0", "can1"],
        ),
        patch(
            "cyberwave.driver.support.can.subprocess.run", side_effect=AssertionError
        ),
        pytest.raises(CanSetupError),
    ):
        bring_up_can_interface("canX", 1_000_000)


def test_bring_up_usb_address_no_match_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("os.geteuid", lambda: 0, raising=False)

    def fake_run(cmd, **kwargs):
        return _completed(stdout="bus-info: 9-9:9.9\n")

    with (
        patch(
            "cyberwave.driver.support.can.list_can_interfaces", return_value=["can0"]
        ),
        patch("cyberwave.driver.support.can.subprocess.run", side_effect=fake_run),
        pytest.raises(CanSetupError),
    ):
        bring_up_can_interface("can0", 1_000_000, usb_address="1-2.2:1.0")


def test_bring_up_prepends_sudo_when_not_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("os.geteuid", lambda: 1000, raising=False)
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:5] == ["sudo", "ip", "-details", "link", "show"]:
            return _completed(stdout="2: can0: <NOARP,ECHO> mtu 16")
        return _completed()

    with (
        patch(
            "cyberwave.driver.support.can.list_can_interfaces", return_value=["can0"]
        ),
        patch("cyberwave.driver.support.can.subprocess.run", side_effect=fake_run),
    ):
        bring_up_can_interface("can0", 1_000_000)

    assert ["sudo", "ip", "link", "set", "can0", "up"] in calls


def test_describe_can_ports_never_raises_when_ethtool_missing() -> None:
    with (
        patch(
            "cyberwave.driver.support.can.list_can_interfaces", return_value=["can0"]
        ),
        patch(
            "cyberwave.driver.support.can.subprocess.run",
            side_effect=OSError("no ethtool"),
        ),
    ):
        assert describe_can_ports() == {}


def test_list_can_interfaces_empty_when_ip_is_missing() -> None:
    with patch(
        "cyberwave.driver.support.can.subprocess.run",
        side_effect=subprocess.TimeoutExpired("ip", 5.0),
    ):
        assert list_can_interfaces() == []


def test_bring_up_command_failure_raises_with_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("os.geteuid", lambda: 0, raising=False)

    def fake_run(cmd, **kwargs):
        if cmd[:4] == ["ip", "-details", "link", "show"]:
            return _completed(stdout="2: can0: <NOARP,ECHO> mtu 16")
        return _completed(returncode=2, stderr="RTNETLINK answers: busy")

    with (
        patch(
            "cyberwave.driver.support.can.list_can_interfaces", return_value=["can0"]
        ),
        patch("cyberwave.driver.support.can.subprocess.run", side_effect=fake_run),
        pytest.raises(CanSetupError, match="RTNETLINK answers: busy"),
    ):
        bring_up_can_interface("can0", 1_000_000)
