"""Spawn and supervise vendor ``ros2 launch`` subprocesses from manifest specs."""

from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import tempfile
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rclpy.node import Node

    from .manifest import ManifestManagedLaunch
from .ros_setup_env import collect_ros_setup_scripts

logger = logging.getLogger(__name__)

_DEFAULT_LOG_BASENAME = "cyberwave_managed_launch.log"


class ManagedRosLaunch:
    """Run a managed ``ros2 launch`` child process for a ROS 2 driver."""

    def __init__(
        self,
        spec: ManifestManagedLaunch,
        *,
        node: Node | None = None,
        launch_args: dict[str, Any] | None = None,
    ) -> None:
        self._spec = spec
        self._node = node
        self._launch_args = dict(spec.launch_args)
        if launch_args:
            self._launch_args.update(launch_args)
        self._proc: subprocess.Popen[str] | None = None
        self._log_handle: object | None = None
        self._log_path: str = ""

    def build_shell_command(self) -> str:
        setup = self._ros_setup_snippet()
        args = " ".join(
            f"{shlex.quote(str(k))}:={self._format_launch_value(v)}"
            for k, v in self._launch_args.items()
        )
        # An empty package means launch_file is a standalone path (or filename
        # on the launch search path): ``ros2 launch <file>`` rather than
        # ``ros2 launch <package> <file>``. Used by drivers that ship their own
        # launch file outside any ROS package (e.g. the namespaced Piper launch).
        if self._spec.package:
            launch = (
                f"ros2 launch {shlex.quote(self._spec.package)} "
                f"{shlex.quote(self._spec.launch_file)}"
            )
        else:
            launch = f"ros2 launch {shlex.quote(self._spec.launch_file)}"
        if args:
            launch = f"{launch} {args}"
        return f"{setup} && exec {launch}"

    def _ros_setup_snippet(self) -> str:
        """Source ROS underlay + optional overlay (conda venv does not affect child)."""
        scripts = collect_ros_setup_scripts(self._spec)
        return " && ".join(f"source {shlex.quote(s)}" for s in scripts)

    @staticmethod
    def _format_launch_value(v: object) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        return str(v)

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _resolve_log_path(self) -> str:
        if self._spec.log_file:
            return self._spec.log_file
        env_path = os.environ.get("CW_MANAGED_LAUNCH_LOG", "").strip()
        if env_path:
            return env_path
        return os.path.join(tempfile.gettempdir(), _DEFAULT_LOG_BASENAME)

    def _tail_log(self, *, max_bytes: int = 4000) -> str:
        if not self._log_path or not os.path.isfile(self._log_path):
            return ""
        try:
            with open(self._log_path, "rb") as f:
                data = f.read(max_bytes)
            return data.decode("utf-8", errors="replace")
        except OSError:
            return ""

    def start(self) -> None:
        if self.is_running:
            return
        cmd = self.build_shell_command()
        self._log_path = self._resolve_log_path()
        logger.info(
            "ManagedRosLaunch starting (log=%s): %s",
            self._log_path,
            cmd,
        )
        self._log_handle = open(self._log_path, "w", encoding="utf-8")
        child_env = os.environ.copy()
        domain_id = os.environ.get("ROS_DOMAIN_ID") or os.environ.get(
            "CW_ROS2_DOMAIN_ID"
        )
        if domain_id:
            child_env["ROS_DOMAIN_ID"] = domain_id
        self._proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            stdout=self._log_handle,  # type: ignore[arg-type]
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=child_env,
        )
        time.sleep(0.5)
        if self._proc.poll() is not None:
            code = self._proc.returncode
            if self._log_handle is not None:
                self._log_handle.flush()
            tail = self._tail_log()
            raise RuntimeError(
                f"Managed launch exited immediately (code={code}). "
                f"Check {self._log_path}. Last output:\n{tail}"
            )

    def _service_visible(self, name: str) -> bool:
        """Return True when *name* appears on the ROS graph (LifecycleNode-safe)."""
        node = self._node
        if node is None:
            return False
        candidates = {name, name.lstrip("/"), f"/{name.lstrip('/')}"}
        graph_getter = getattr(node, "get_service_names_and_types", None)
        if callable(graph_getter):
            try:
                for svc_name, _ in graph_getter():
                    if svc_name in candidates:
                        return True
            except Exception:
                logger.exception("get_service_names_and_types failed during readiness")
            return False
        waiter = getattr(node, "wait_for_service", None)
        if callable(waiter):
            try:
                return bool(waiter(name, timeout_sec=0.1))
            except Exception:
                pass
        return False

    def wait_ready(self) -> None:
        if self._spec.readiness.kind != "service":
            raise NotImplementedError("v1 supports readiness.kind=service only")
        if self._node is None:
            raise RuntimeError("wait_ready requires rclpy node for service probe")
        name = self._spec.readiness.name
        deadline = time.monotonic() + self._spec.readiness.timeout_s
        while time.monotonic() < deadline:
            if self._service_visible(name):
                logger.info("ManagedRosLaunch ready: service %s available", name)
                return
            if self._proc and self._proc.poll() is not None:
                tail = self._tail_log()
                raise RuntimeError(
                    f"Managed launch exited before {name} ready "
                    f"(code={self._proc.returncode}). Log: {self._log_path}\n{tail}"
                )
        tail = self._tail_log()
        raise TimeoutError(
            f"Service {name} not ready within {self._spec.readiness.timeout_s}s. "
            f"Log: {self._log_path}\n{tail}"
        )

    def stop(self, *, grace_s: float = 10.0) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            else:
                try:
                    proc.wait(timeout=grace_s)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(timeout=5)
        self._proc = None
        if self._log_handle is not None:
            try:
                self._log_handle.close()  # type: ignore[union-attr]
            except Exception:
                pass
            self._log_handle = None
