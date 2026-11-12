from __future__ import annotations

from unittest.mock import MagicMock, patch

from cyberwave.driver.ros2.manifest import ManifestManagedLaunch, ManifestReadiness
from cyberwave.driver.ros2.managed_launch import ManagedRosLaunch


def _spec() -> ManifestManagedLaunch:
    return ManifestManagedLaunch(
        package="piper",
        launch_file="start_single_piper.launch.py",
        launch_args={"can_port": "can0", "auto_enable": False},
        readiness=ManifestReadiness(
            kind="service", name="/enable_srv", timeout_s=30.0
        ),
        ros_setup="/fake/setup.bash",
    )


def test_build_launch_command() -> None:
    ml = ManagedRosLaunch(_spec())
    cmd = ml.build_shell_command()
    assert "source /fake/setup.bash" in cmd
    assert "ros2 launch piper start_single_piper.launch.py" in cmd
    assert "can_port:=can0" in cmd
    assert "auto_enable:=false" in cmd


def test_start_is_idempotent() -> None:
    ml = ManagedRosLaunch(_spec())
    with (
        patch("cyberwave.driver.ros2.managed_launch.subprocess.Popen") as popen,
        patch("builtins.open", MagicMock()),
        patch("cyberwave.driver.ros2.managed_launch.time.sleep"),
    ):
        popen.return_value = MagicMock(pid=123, poll=lambda: None)
        ml.start()
        first = ml._proc
        ml.start()
        assert ml._proc is first
        popen.assert_called_once()


def test_stop_sends_sigterm_to_process_group() -> None:
    ml = ManagedRosLaunch(_spec())
    proc = MagicMock(pid=999, poll=lambda: None)
    ml._proc = proc
    with (
        patch("cyberwave.driver.ros2.managed_launch.os.getpgid", return_value=999),
        patch("cyberwave.driver.ros2.managed_launch.os.killpg") as killpg,
    ):
        ml.stop(grace_s=0.01)
        killpg.assert_called()


def test_wait_ready_uses_service_graph() -> None:
    node = MagicMock()
    node.get_service_names_and_types.side_effect = [
        [],
        [("/enable_srv", ["piper_msgs/srv/Enable"])],
    ]
    ml = ManagedRosLaunch(_spec(), node=node)
    ml._proc = MagicMock(poll=lambda: None)
    ml.wait_ready()
    assert node.get_service_names_and_types.call_count >= 2


def _run_to_exit(
    monkeypatch, tmp_path, *, forward: bool
) -> tuple[ManagedRosLaunch, str]:
    log = tmp_path / "launch.log"
    monkeypatch.setenv("CW_MANAGED_LAUNCH_LOG", str(log))
    if forward:
        monkeypatch.setenv("CW_MANAGED_LAUNCH_LOG_TO_STDOUT", "true")
    else:
        monkeypatch.delenv("CW_MANAGED_LAUNCH_LOG_TO_STDOUT", raising=False)
    ml = ManagedRosLaunch(_spec())
    ml.build_shell_command = lambda: "echo vendor up; sleep 1; echo CAN feedback stale"
    ml.start()
    ml._proc.wait(timeout=5)
    ml.stop()
    return ml, log.read_text()


def test_launch_output_reaches_the_container_log_when_asked(
    monkeypatch, tmp_path, capsys
):
    """The vendor node's warnings were only in a file inside the container."""
    _, logged = _run_to_exit(monkeypatch, tmp_path, forward=True)
    out = capsys.readouterr().out
    assert "[managed_launch] vendor up\n" in out
    assert "[managed_launch] CAN feedback stale\n" in out
    # The file is still written: a failed start quotes its tail.
    assert logged == "vendor up\nCAN feedback stale\n"


def test_launch_output_stays_in_the_file_by_default(monkeypatch, tmp_path, capsys):
    _, logged = _run_to_exit(monkeypatch, tmp_path, forward=False)
    assert "[managed_launch]" not in capsys.readouterr().out
    assert logged == "vendor up\nCAN feedback stale\n"
