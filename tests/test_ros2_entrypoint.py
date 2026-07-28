"""ros2 entrypoint: rclpy stubbing, manifest resolution, write-manifest, exit codes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cyberwave.driver.ros2.entrypoint import (
    _resolve_manifest_path,
    run_driver_main,
    stub_rclpy_for_manifest_export,
)


def test_stub_rclpy_makes_rclpy_importable(monkeypatch):
    for mod in list(sys.modules):
        if mod == "rclpy" or mod.startswith("rclpy."):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    stub_rclpy_for_manifest_export()
    import rclpy  # noqa: F401
    from rclpy.lifecycle import LifecycleNode  # noqa: F401

    assert rclpy.ok()


def test_manifest_path_env_wins(tmp_path, monkeypatch):
    explicit = tmp_path / "custom.yaml"
    explicit.write_text("x: 1")
    monkeypatch.setenv("CW_DRIVER_MANIFEST", str(explicit))
    assert _resolve_manifest_path(anchor=None) == str(explicit)


def test_manifest_path_sibling_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("CW_DRIVER_MANIFEST", raising=False)
    anchor = tmp_path / "my_driver.py"
    anchor.write_text("")
    (tmp_path / "manifest.yaml").write_text("x: 1")
    assert _resolve_manifest_path(anchor=str(anchor)) == str(tmp_path / "manifest.yaml")


def test_manifest_path_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("CW_DRIVER_MANIFEST", raising=False)
    anchor = tmp_path / "my_driver.py"
    anchor.write_text("")
    assert _resolve_manifest_path(anchor=str(anchor)) is None


class _FakeDriver:
    DEFAULT_NODE_NAME = "fake_driver"
    written: list[Path] = []

    @classmethod
    def write_manifest(cls, out: Path) -> Path:
        cls.written.append(out)
        out.write_text("manifest: fake")
        return out


def test_write_manifest_subcommand(tmp_path):
    out = tmp_path / "m.yaml"
    _FakeDriver.written.clear()
    run_driver_main(_FakeDriver, argv=["prog", "write-manifest", str(out)])
    assert _FakeDriver.written == [out]
    assert out.read_text() == "manifest: fake"


class _ExplodingDriver:
    DEFAULT_NODE_NAME = "boom"

    def __init__(self, node_name, manifest_path):
        raise RuntimeError("cannot construct")


def test_run_failure_exits_1(tmp_path, monkeypatch):
    monkeypatch.delenv("CW_DRIVER_MANIFEST", raising=False)
    with pytest.raises(SystemExit) as exc:
        run_driver_main(_ExplodingDriver, anchor=str(tmp_path / "d.py"), argv=["prog"])
    assert exc.value.code == 1


class _InterruptedDriver:
    DEFAULT_NODE_NAME = "sleepy"

    def __init__(self, node_name, manifest_path):
        pass

    def run(self):
        raise KeyboardInterrupt


def test_keyboard_interrupt_exits_0(tmp_path, monkeypatch):
    monkeypatch.delenv("CW_DRIVER_MANIFEST", raising=False)
    with pytest.raises(SystemExit) as exc:
        run_driver_main(_InterruptedDriver, anchor=str(tmp_path / "d.py"), argv=["prog"])
    assert exc.value.code == 0
