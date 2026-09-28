"""Tests for the worker startup status reporter (cyberwave.workers.status).

The contract is mostly about not getting in the way: never raise, never block
startup, go quiet when the status directory isn't mounted. The phase strings
are duplicated in edge-core and the backend, so the test pinning them is what
stands between a rename here and silent degradation there.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from cyberwave.workers.status import (
    DEFAULT_WORKER_STATUS_PATH,
    PHASE_BOOTING,
    PHASE_CONNECTING_MQTT,
    PHASE_LOADING_MODELS,
    PHASE_READY,
    PHASE_SUBSCRIBED,
    PHASE_WARMING_UP,
    WORKER_STATUS_PATH_ENV,
    WorkerStatusReporter,
    get_status_reporter,
    reset_status_reporter,
)


@pytest.fixture
def status_path(tmp_path: Path) -> Path:
    return tmp_path / "worker-status.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_publish_writes_the_phase(status_path: Path) -> None:
    reporter = WorkerStatusReporter(status_path)
    reporter.publish(PHASE_BOOTING)

    record = read(status_path)
    assert record["phase"] == PHASE_BOOTING
    assert record["version"] == 1
    assert record["seq"] == 1
    assert isinstance(record["boot_id"], str) and record["boot_id"]


def test_publish_carries_model_counts_for_the_progress_bar(status_path: Path) -> None:
    reporter = WorkerStatusReporter(status_path)
    reporter.publish(PHASE_WARMING_UP, models_total=5, models_ready=2, detail="yolo")

    record = read(status_path)
    assert record["models_total"] == 5
    assert record["models_ready"] == 2
    assert record["detail"] == "yolo"


def test_sequence_increments_so_a_reader_can_order_records(status_path: Path) -> None:
    reporter = WorkerStatusReporter(status_path)
    reporter.publish(PHASE_BOOTING)
    reporter.publish(PHASE_READY)

    assert read(status_path)["seq"] == 2
    assert read(status_path)["phase"] == PHASE_READY


def test_each_publish_replaces_the_file_atomically(status_path: Path) -> None:
    """No temp files left behind, and the target is always complete JSON."""
    reporter = WorkerStatusReporter(status_path)
    for phase in (PHASE_BOOTING, PHASE_CONNECTING_MQTT, PHASE_READY):
        reporter.publish(phase)
        read(status_path)  # parses => never observed half-written

    siblings = list(status_path.parent.iterdir())
    assert siblings == [status_path]


def test_disabled_when_the_status_directory_is_not_mounted(tmp_path: Path) -> None:
    """Creating the directory ourselves would produce a file only this
    container can see, leaving the supervisor waiting on a dead path."""
    reporter = WorkerStatusReporter(tmp_path / "absent" / "worker-status.json")

    assert reporter.enabled is False
    reporter.publish(PHASE_BOOTING)  # must not raise
    assert not (tmp_path / "absent").exists()


def test_an_unmounted_directory_is_reported_at_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Going quiet is the right behaviour; going quiet *quietly* is not.

    The supervisor's only fallback is "running means started", which is
    exactly the pre-existing behaviour this file exists to replace — so a
    misconfigured mount reproduces the old bug with nothing anywhere saying
    so. This log is the sole signal an operator gets.
    """
    with caplog.at_level(logging.WARNING, logger="cyberwave.workers.status"):
        WorkerStatusReporter(tmp_path / "absent" / "worker-status.json")

    assert any("not mounted" in r.message for r in caplog.records)


def test_a_failing_write_disables_the_reporter_instead_of_raising(
    status_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A worker must never fail to start because nobody can be told about it."""
    reporter = WorkerStatusReporter(status_path)

    def boom(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr("cyberwave.workers.status.tempfile.mkstemp", boom)
    with caplog.at_level(logging.WARNING, logger="cyberwave.workers.status"):
        reporter.publish(PHASE_BOOTING)

    assert reporter.enabled is False
    assert not status_path.exists()
    # Reporting is off for good after this, so the one warning is all there is.
    assert any("will not be reported" in r.message for r in caplog.records)


def test_path_comes_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(WORKER_STATUS_PATH_ENV, str(tmp_path / "custom.json"))
    reporter = WorkerStatusReporter()

    assert reporter.path == tmp_path / "custom.json"
    reporter.publish(PHASE_READY)
    assert read(tmp_path / "custom.json")["phase"] == PHASE_READY


def test_default_path_matches_the_edge_core_mount() -> None:
    """Edge-core bind-mounts this exact path; an unmirrored change here
    silently stops all startup reporting."""
    assert DEFAULT_WORKER_STATUS_PATH == "/run/cyberwave/worker-status.json"


def test_phase_names_are_the_strings_the_other_components_expect() -> None:
    """Mirrored in edge-core's ``WORKER_STARTUP_PHASES`` and the backend's
    ``_PHASE_MESSAGES``, so a rename must fail here rather than degrade there."""
    assert (
        PHASE_BOOTING,
        PHASE_CONNECTING_MQTT,
        PHASE_LOADING_MODELS,
        PHASE_SUBSCRIBED,
        PHASE_WARMING_UP,
        PHASE_READY,
    ) == (
        "booting",
        "connecting_mqtt",
        "loading_models",
        "subscribed",
        "warming_up",
        "ready",
    )


def test_singleton_shares_one_boot_id_across_call_sites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The client and the runtime both publish; one file, one sequence."""
    monkeypatch.setenv(WORKER_STATUS_PATH_ENV, str(tmp_path / "worker-status.json"))
    reset_status_reporter()
    try:
        first = get_status_reporter()
        first.publish(PHASE_BOOTING)
        second = get_status_reporter()
        second.publish(PHASE_READY)

        assert first is second
        assert read(tmp_path / "worker-status.json")["seq"] == 2
    finally:
        reset_status_reporter()
