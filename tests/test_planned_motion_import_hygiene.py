"""The SDK ships no planner-framework code: a driver that has one plugs it in.

MoveIt and ``moveit_msgs`` exist only in ROS images, so a backend for them
lives in the driver package that owns that dependency and registers itself
through ``PlannedMotionMixin._motion_backend_factories``. These tests keep the
SDK honest about that, and keep ``import cyberwave.driver`` light.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import cyberwave

_PACKAGE = Path(cyberwave.__file__).resolve().parent

#: ``import moveit_msgs``, ``from moveit_msgs.msg import ...``,
#: ``from moveit.planning import ...``, ``import moveit_commander`` and the
#: dynamic ``importlib.import_module("moveit...")`` form.
_MOVEIT_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+moveit(?:_\w+)?(?:\.|\s|$)"
    r"|import_module\(\s*[\"']moveit",
    re.MULTILINE,
)


def test_no_module_under_cyberwave_imports_moveit() -> None:
    offenders = sorted(
        str(path.relative_to(_PACKAGE.parent))
        for path in _PACKAGE.rglob("*.py")
        if _MOVEIT_IMPORT.search(path.read_text(encoding="utf-8"))
    )
    assert offenders == []


def test_the_driver_package_names_no_moveit_message_types() -> None:
    driver = _PACKAGE / "driver"
    offenders = sorted(
        str(path.relative_to(_PACKAGE.parent))
        for path in driver.rglob("*.py")
        if "moveit_msgs" in path.read_text(encoding="utf-8")
    )
    assert offenders == []
    assert not (driver / "motion" / "moveit").exists()


def test_the_pattern_catches_the_import_forms() -> None:
    for line in (
        "import moveit_msgs",
        "from moveit_msgs.msg import CollisionObject",
        "    from moveit_msgs.action import MoveGroup",
        "from moveit.planning import MoveItPy",
        "import moveit_commander",
        'importlib.import_module("moveit_msgs.srv")',
    ):
        assert _MOVEIT_IMPORT.search(line), line
    for line in ("import moveitless", "# no moveit here", "from .motion import x"):
        assert not _MOVEIT_IMPORT.search(line), line


def test_importing_the_driver_package_loads_no_ros_or_planner_dependencies() -> None:
    probe = (
        "import sys\n"
        "import cyberwave.driver\n"
        "import cyberwave.driver.motion\n"
        "import cyberwave.driver.control.planned_motion\n"
        "heavy = ('rclpy', 'pinocchio', 'scipy', 'moveit_msgs')\n"
        "print(','.join(m for m in heavy if m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert result.stdout.strip() == ""
