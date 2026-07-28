"""Generic Docker/CLI entrypoint for ROS2 drivers.

Centralizes the boilerplate every driver's ``main.py`` used to hand-roll, so
each driver's entrypoint is::

    from cyberwave.driver.ros2.entrypoint import run_driver_main, stub_rclpy_for_manifest_export
    import sys

    if __name__ == "__main__":
        if len(sys.argv) > 1 and sys.argv[1] in ("write-manifest", "write_manifest"):
            stub_rclpy_for_manifest_export()
        from my_driver import MyDriver  # defer: driver modules import rclpy via BaseROS2Driver

        run_driver_main(MyDriver, gate_import="my_vendor_msgs", anchor=__file__)

Handles: ``write-manifest [out]`` (with rclpy stubbed so no ROS install is
needed), manifest path resolution (``CW_DRIVER_MANIFEST`` env > sibling
``manifest.yaml``), managed-launch ROS env bootstrap before vendor imports,
``CW_ROS2_NODE_NAME`` resolution, and standard exit codes (0 on interrupt,
1 on failure).
"""

from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

logger = logging.getLogger(__name__)


def stub_rclpy_for_manifest_export() -> None:
    """Allow ``write-manifest`` without a full ROS install."""
    if "rclpy" in sys.modules:
        return
    _rclpy = types.ModuleType("rclpy")
    _rclpy.ok = lambda: True
    _rclpy.init = lambda *a, **k: None
    _rclpy.spin_until_future_complete = lambda *a, **k: None
    sys.modules["rclpy"] = _rclpy
    sys.modules["rclpy.executors"] = types.ModuleType("rclpy.executors")
    sys.modules["rclpy.executors"].MultiThreadedExecutor = MagicMock
    sys.modules["rclpy.qos"] = types.ModuleType("rclpy.qos")
    sys.modules["rclpy.qos"].QoSProfile = MagicMock
    sys.modules["rclpy.qos"].qos_profile_sensor_data = MagicMock()
    sys.modules["rclpy.lifecycle"] = types.ModuleType("rclpy.lifecycle")
    sys.modules["rclpy.lifecycle"].LifecycleNode = type("LifecycleNode", (), {})
    sys.modules["rclpy.lifecycle"].State = MagicMock
    sys.modules["rclpy.lifecycle"].TransitionCallbackReturn = MagicMock
    sys.modules["rclpy.parameter"] = types.ModuleType("rclpy.parameter")
    sys.modules["rclpy.parameter"].Parameter = MagicMock
    sys.modules["lifecycle_msgs"] = types.ModuleType("lifecycle_msgs")
    sys.modules["lifecycle_msgs.msg"] = types.ModuleType("lifecycle_msgs.msg")
    sys.modules["lifecycle_msgs.msg"].Transition = MagicMock
    sys.modules["lifecycle_msgs.srv"] = types.ModuleType("lifecycle_msgs.srv")
    sys.modules["lifecycle_msgs.srv"].ChangeState = MagicMock
    sys.modules["rcl_interfaces"] = types.ModuleType("rcl_interfaces")
    sys.modules["rcl_interfaces.msg"] = types.ModuleType("rcl_interfaces.msg")
    sys.modules["rcl_interfaces.msg"].ParameterDescriptor = MagicMock
    sys.modules["rcl_interfaces.msg"].ParameterType = MagicMock
    sys.modules["rcl_interfaces.msg"].SetParametersResult = MagicMock
    sys.modules["std_msgs"] = types.ModuleType("std_msgs")
    sys.modules["std_msgs.msg"] = types.ModuleType("std_msgs.msg")
    sys.modules["std_msgs.msg"].String = MagicMock


def _resolve_manifest_path(*, anchor: str | None, filename: str = "manifest.yaml") -> str | None:
    """``CW_DRIVER_MANIFEST`` env > ``<anchor dir>/<filename>`` > None."""
    explicit = os.environ.get("CW_DRIVER_MANIFEST", "").strip()
    if explicit:
        return explicit
    if anchor is not None:
        sibling = Path(anchor).resolve().parent / filename
        if sibling.is_file():
            return str(sibling)
    return None


def _bootstrap_ros_runtime(driver_cls: Any, manifest_path: str | None, gate_import: str | None) -> None:
    from .manifest import resolve_node_manifest  # noqa: PLC0415 (after stub decision)
    from .ros_setup_env import (  # noqa: PLC0415
        bootstrap_ros_python_process,
        collect_ros_setup_scripts,
    )

    node_name = os.environ.get("CW_ROS2_NODE_NAME", driver_cls.DEFAULT_NODE_NAME)
    manifest = resolve_node_manifest(driver_cls, manifest_path, node_name=node_name)
    if manifest.managed_launch is not None:
        bootstrap_ros_python_process(
            collect_ros_setup_scripts(manifest.managed_launch),
            gate_import=gate_import,
        )


def run_driver_main(
    driver_cls: Any,
    *,
    gate_import: str | None = None,
    manifest_filename: str = "manifest.yaml",
    anchor: str | None = None,
    argv: list[str] | None = None,
) -> None:
    """Standard driver entrypoint. ``anchor`` is the caller's ``__file__``
    (locates the sibling manifest); ``argv`` defaults to ``sys.argv``."""
    logging.basicConfig(level=logging.INFO)
    args = list(sys.argv if argv is None else argv)

    if len(args) > 1 and args[1] in ("write-manifest", "write_manifest"):
        stub_rclpy_for_manifest_export()
        out = (
            Path(args[2])
            if len(args) > 2
            else Path(anchor or ".").resolve().parent / manifest_filename
        )
        path = driver_cls.write_manifest(out)
        print(f"Wrote {path}")
        return

    manifest_path = _resolve_manifest_path(anchor=anchor, filename=manifest_filename)
    if manifest_path:
        logger.info("Using driver manifest file: %s", manifest_path)
    else:
        logger.info(
            "No manifest file found; using %s.define_node_manifest() at runtime",
            driver_cls.__name__,
        )

    # Bootstrap failures must not prevent the runtime fallback to
    # driver_cls.define_node_manifest().
    try:
        _bootstrap_ros_runtime(driver_cls, manifest_path, gate_import)
    except Exception:
        logger.warning("ROS runtime bootstrap skipped", exc_info=True)

    try:
        node_name = os.environ.get("CW_ROS2_NODE_NAME", driver_cls.DEFAULT_NODE_NAME)
        driver = driver_cls(node_name, manifest_path)
        driver.run()
    except (KeyboardInterrupt, TimeoutError) as exc:
        logger.info("%s exiting: %s", driver_cls.__name__, exc)
        sys.exit(0)
    except Exception:
        logger.exception("%s failed", driver_cls.__name__)
        sys.exit(1)
