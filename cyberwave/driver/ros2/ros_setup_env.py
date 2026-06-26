"""Apply colcon/ROS setup.bash environment to the current Python process."""

from __future__ import annotations

import importlib
import logging
import os
import shlex
import subprocess
import sys
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .manifest import ManifestManagedLaunch

logger = logging.getLogger(__name__)

# Vars that must match the sourced workspace for vendor msg imports.
_ROS_WORKSPACE_ENV_KEYS = frozenset(
    {
        "AMENT_PREFIX_PATH",
        "CMAKE_PREFIX_PATH",
        "COLCON_PREFIX_PATH",
        "LD_LIBRARY_PATH",
        "PATH",
        "PKG_CONFIG_PATH",
        "PYTHONPATH",
        "ROS_DISTRO",
        "ROS_PYTHON_VERSION",
        "ROS_VERSION",
    }
)


def collect_ros_setup_scripts(
    spec: ManifestManagedLaunch | None,
) -> list[str]:
    """Return ordered setup.bash paths (underlay then overlay).

    Paths come only from the manifest spec and the ``ROS_SETUP`` /
    ``ROS_SETUP_OVERLAY`` env vars — the SDK never guesses a ROS distro or a
    vendor workspace location. Drivers declare their own paths in the manifest.
    """
    scripts: list[str] = []

    def _append(path: str) -> None:
        p = path.strip()
        if p and p not in scripts:
            scripts.append(p)

    if spec is not None:
        _append(spec.ros_setup)
        _append(spec.ros_overlay)

    _append(os.environ.get("ROS_SETUP", ""))
    _append(os.environ.get("ROS_SETUP_OVERLAY", ""))

    return scripts


def apply_ros_setup_environment(
    scripts: list[str],
    *,
    log_prefix: str = "ros_setup_env",
) -> None:
    """Source *scripts* in bash and merge ROS workspace vars into ``os.environ``."""
    scripts = [s for s in scripts if Path(s).is_file()]
    if not scripts:
        logger.warning("%s: no setup.bash scripts found; vendor ROS msgs may fail to import", log_prefix)
        return

    snippet = " && ".join(f"source {shlex.quote(s)}" for s in scripts)
    cmd = f"{snippet} && env"
    try:
        result = subprocess.run(
            ["bash", "-lc", cmd],
            capture_output=True,
            text=True,
            check=True,
            timeout=30.0,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "%s: failed to source %s (%s); vendor ROS msgs may fail to import",
            log_prefix,
            scripts,
            exc,
        )
        return

    merged = 0
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key in _ROS_WORKSPACE_ENV_KEYS or key.startswith("ROS_"):
            os.environ[key] = value
            merged += 1

    extend_sys_path_from_ament()

    logger.info(
        "%s: sourced %d setup script(s), merged %d ROS env var(s): %s",
        log_prefix,
        len(scripts),
        merged,
        scripts,
    )


def extend_sys_path_from_ament() -> None:
    """Add ament install site-packages dirs to ``sys.path`` (same interpreter only)."""
    seen = set(sys.path)
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(":"):
        if not prefix:
            continue
        for pattern in (
            f"{prefix}/lib/python3*/site-packages",
            f"{prefix}/local/lib/python3*/dist-packages",
        ):
            for path in sorted(glob(pattern)):
                if path not in seen:
                    sys.path.insert(0, path)
                    seen.add(path)


def _ros_python_executable(scripts: list[str]) -> str:
    existing = [s for s in scripts if Path(s).is_file()]
    if not existing:
        return ""
    snippet = " && ".join(f"source {shlex.quote(s)}" for s in existing)
    result = subprocess.run(
        ["bash", "-lc", f"{snippet} && which python3"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30.0,
    )
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip().splitlines()[-1].strip()


def bootstrap_ros_python_process(
    scripts: list[str],
    *,
    gate_import: str = "",
    ready_flag: str = "CW_ROS_PYTHON_READY",
    argv: list[str] | None = None,
) -> None:
    """Source ROS workspaces and re-exec under the workspace ``python3`` when needed.

    Vendor ROS message packages are built against the system ROS Python. Running
    the entrypoint from a conda/venv interpreter often cannot import them even
    when ``PYTHONPATH`` is correct, so re-exec under the workspace interpreter.
    """
    if os.environ.get(ready_flag) == "1":
        return

    apply_ros_setup_environment(scripts)

    if gate_import:
        try:
            importlib.import_module(gate_import)
            os.environ[ready_flag] = "1"
            return
        except ImportError:
            pass

    ros_python = _ros_python_executable(scripts)
    current = str(Path(sys.executable).resolve())
    if not ros_python:
        logger.warning(
            "ros_setup_env: could not resolve ROS python3 after sourcing %s",
            scripts,
        )
        return

    ros_python_resolved = str(Path(ros_python).resolve())
    if ros_python_resolved == current:
        os.environ[ready_flag] = "1"
        return

    launch_argv = argv if argv is not None else sys.argv
    script = str(Path(launch_argv[0]).resolve())
    child_argv = [ros_python, script, *launch_argv[1:]]
    os.environ[ready_flag] = "1"
    logger.info(
        "ros_setup_env: re-exec under ROS python %s (was %s) for vendor imports",
        ros_python_resolved,
        current,
    )
    os.execve(ros_python, child_argv, os.environ)
