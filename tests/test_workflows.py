"""
Unit tests for WorkflowManager, WorkflowRunManager, Workflow, and WorkflowRun.

These tests mock the private HTTP helpers so no network access is required.
"""

from unittest.mock import MagicMock, patch

import pytest

from cyberwave.exceptions import CyberwaveTimeoutError
from cyberwave.workflows import (
    Workflow,
    WorkflowManager,
    WorkflowRun,
    WorkflowRunManager,
)


def _make_client():
    client = MagicMock()
    client.config.workspace_id = "ws-uuid"
    return client


# ======================================================================
# WorkflowManager
# ======================================================================


class TestWorkflowManager:

    def test_list_returns_workflow_objects(self):
        client = _make_client()
        manager = WorkflowManager(client)
        raw = [
            {"uuid": "wf-1", "name": "Pick", "is_active": True},
            {"uuid": "wf-2", "name": "Place", "is_active": False},
        ]

        with patch("cyberwave.workflows._list_workflows", return_value=raw):
            workflows = manager.list()

        assert len(workflows) == 2
        assert isinstance(workflows[0], Workflow)
        assert workflows[0].uuid == "wf-1"
        assert workflows[0].name == "Pick"
        assert workflows[0].status == "active"
        assert workflows[1].name == "Place"
        assert workflows[1].status == "inactive"

    def test_get_returns_workflow(self):
        client = _make_client()
        manager = WorkflowManager(client)
        raw = {"uuid": "wf-1", "name": "Inspect", "is_active": True}

        with patch("cyberwave.workflows._get_workflow", return_value=raw) as mock_get:
            wf = manager.get("wf-1")

        mock_get.assert_called_once_with(client, "wf-1")
        assert isinstance(wf, Workflow)
        assert wf.name == "Inspect"

    def test_get_by_slug_returns_workflow(self):
        client = _make_client()
        manager = WorkflowManager(client)
        raw = [{"uuid": "wf-1", "name": "Inspect", "slug": "inspect"}]

        with patch("cyberwave.workflows._list_workflows", return_value=raw) as mock_list:
            wf = manager.get_by_slug("inspect", workspace_id="ws-1")

        mock_list.assert_called_once_with(client, workspace_id="ws-1", slug="inspect")
        assert isinstance(wf, Workflow)
        assert wf.slug == "inspect"

    def test_get_by_slug_returns_none_when_missing(self):
        client = _make_client()
        manager = WorkflowManager(client)

        with patch("cyberwave.workflows._list_workflows", return_value=[]):
            wf = manager.get_by_slug("missing", workspace_id="ws-1")

        assert wf is None

    def test_get_by_slug_full_unified_slug(self):
        """get_by_slug with full unified slug uses /by-slug endpoint."""
        client = _make_client()
        manager = WorkflowManager(client)
        raw = {"uuid": "wf-1", "name": "Inspect", "slug": "acme/workflows/inspect"}

        with patch("cyberwave.workflows._get_workflow_by_slug", return_value=raw):
            wf = manager.get_by_slug("acme/workflows/inspect")

        assert isinstance(wf, Workflow)
        assert wf.slug == "acme/workflows/inspect"

    def test_trigger_returns_workflow_run(self):
        client = _make_client()
        manager = WorkflowManager(client)
        raw = {"uuid": "run-1", "workflow_id": "wf-1", "status": "running", "inputs": {}}

        with patch(
            "cyberwave.workflows._trigger_workflow", return_value=raw
        ) as mock_trigger:
            run = manager.trigger("wf-1", inputs={"speed": 0.5})

        mock_trigger.assert_called_once_with(client, "wf-1", {"speed": 0.5})
        assert isinstance(run, WorkflowRun)
        assert run.status == "running"


# ======================================================================
# WorkflowRunManager
# ======================================================================


class TestWorkflowRunManager:

    def test_list_with_filters(self):
        client = _make_client()
        manager = WorkflowRunManager(client)
        raw = [{"uuid": "run-1", "status": "error"}]

        with patch(
            "cyberwave.workflows._list_workflow_runs", return_value=raw
        ) as mock_list:
            runs = manager.list(workflow_id="wf-1", status="error")

        mock_list.assert_called_once_with(
            client, "wf-1", "error", limit=None, offset=None, before=None
        )
        assert len(runs) == 1
        assert runs[0].status == "error"

    def test_list_forwards_paging(self):
        client = _make_client()
        manager = WorkflowRunManager(client)

        with patch(
            "cyberwave.workflows._list_workflow_runs", return_value=[]
        ) as mock_list:
            manager.list(limit=50, offset=100, before="run-9")

        mock_list.assert_called_once_with(
            client, None, None, limit=50, offset=100, before="run-9"
        )

    def test_iter_walks_every_page_with_the_cursor(self):
        client = _make_client()
        manager = WorkflowRunManager(client)
        pages = {
            None: [{"uuid": "run-1"}, {"uuid": "run-2"}],
            "run-2": [{"uuid": "run-3"}, {"uuid": "run-4"}],
            "run-4": [{"uuid": "run-5"}],
        }

        with patch(
            "cyberwave.workflows._list_workflow_runs",
            side_effect=lambda *_a, limit, offset, before: pages[before],
        ) as mock_list:
            runs = list(manager.iter(workflow_id="wf-1", page_size=2))

        assert [run.uuid for run in runs] == [
            "run-1", "run-2", "run-3", "run-4", "run-5"
        ]
        calls = mock_list.call_args_list
        assert [c.kwargs["before"] for c in calls] == [None, "run-2", "run-4"]
        assert all(c.kwargs["offset"] is None for c in calls)
        assert all(c.args[1] == "wf-1" for c in calls)

    def test_iter_stops_after_a_full_last_page(self):
        client = _make_client()
        manager = WorkflowRunManager(client)
        pages = {None: [{"uuid": "run-1"}, {"uuid": "run-2"}], "run-2": []}

        with patch(
            "cyberwave.workflows._list_workflow_runs",
            side_effect=lambda *_a, limit, offset, before: pages[before],
        ):
            runs = list(manager.iter(page_size=2))

        assert [run.uuid for run in runs] == ["run-1", "run-2"]

    def test_iter_keeps_its_place_while_runs_change(self):
        """Runs finishing and starting mid-walk must neither skip nor end it.

        With offset paging, run-1 leaving the ``running`` filter shifted run-3
        out of reach, and two new runs filling page two ended the walk early.
        """
        client = _make_client()
        manager = WorkflowRunManager(client)
        # Newest first, as the backend orders them.
        rows = [{"uuid": f"run-{i}", "status": "running"} for i in range(1, 6)]

        def fake_endpoint(_client, _wf, status, *, limit, offset, before):
            visible = [r for r in rows if r["status"] == status]
            if before is not None:
                position = next(i for i, r in enumerate(rows) if r["uuid"] == before)
                visible = [r for r in visible if rows.index(r) > position]
            return [dict(r) for r in visible[:limit]]

        with patch(
            "cyberwave.workflows._list_workflow_runs", side_effect=fake_endpoint
        ):
            walk = manager.iter(status="running", page_size=2)
            first_page = [next(walk).uuid, next(walk).uuid]
            rows[0]["status"] = "success"
            rows[:0] = [
                {"uuid": "run-new-1", "status": "running"},
                {"uuid": "run-new-0", "status": "running"},
            ]
            rest = [run.uuid for run in walk]

        assert first_page + rest == ["run-1", "run-2", "run-3", "run-4", "run-5"]

    def test_iter_stops_when_server_ignores_the_cursor(self):
        """A backend without cursors returns the same page for every request."""
        client = _make_client()
        manager = WorkflowRunManager(client)
        page = [{"uuid": "run-1"}, {"uuid": "run-2"}]

        with patch(
            "cyberwave.workflows._list_workflow_runs", return_value=page
        ) as mock_list:
            runs = list(manager.iter(page_size=2))

        assert [run.uuid for run in runs] == ["run-1", "run-2"]
        assert mock_list.call_count == 2


# ======================================================================
# Workflow object
# ======================================================================


class TestWorkflow:

    def test_trigger_delegates_to_manager(self):
        client = _make_client()
        run_mock = WorkflowRun(client, {"uuid": "run-1", "status": "running"})
        client.workflows = MagicMock()
        client.workflows.trigger.return_value = run_mock

        wf = Workflow(client, {"uuid": "wf-1", "is_active": True})
        run = wf.trigger(inputs={"x": 1})

        client.workflows.trigger.assert_called_once_with("wf-1", inputs={"x": 1})
        assert run.uuid == "run-1"

    def test_runs_delegates_to_run_manager(self):
        client = _make_client()
        client.workflow_runs = MagicMock()
        client.workflow_runs.list.return_value = []

        wf = Workflow(client, {"uuid": "wf-1"})
        result = wf.runs(status="success")

        client.workflow_runs.list.assert_called_once_with(
            workflow_id="wf-1",
            status="success",
            limit=None,
            offset=None,
            before=None,
        )
        assert result == []

    def test_runs_forwards_paging(self):
        client = _make_client()
        client.workflow_runs = MagicMock()
        client.workflow_runs.list.return_value = []

        Workflow(client, {"uuid": "wf-1"}).runs(limit=10, before="run-9")

        client.workflow_runs.list.assert_called_once_with(
            workflow_id="wf-1", status=None, limit=10, offset=None, before="run-9"
        )

    def test_iter_runs_delegates_to_run_manager(self):
        client = _make_client()
        client.workflow_runs = MagicMock()
        client.workflow_runs.iter.return_value = iter([])

        assert list(Workflow(client, {"uuid": "wf-1"}).iter_runs(page_size=50)) == []

        client.workflow_runs.iter.assert_called_once_with(
            workflow_id="wf-1", status=None, page_size=50
        )

    def test_status_derived_from_is_active(self):
        client = _make_client()
        assert Workflow(client, {"uuid": "1", "is_active": True}).status == "active"
        assert Workflow(client, {"uuid": "2", "is_active": False}).status == "inactive"

    def test_slug_property(self):
        client = _make_client()
        wf = Workflow(client, {"uuid": "wf-1", "slug": "image-to-asset"})
        assert wf.slug == "image-to-asset"

    def test_repr(self):
        client = _make_client()
        wf = Workflow(client, {"uuid": "wf-1", "name": "Pick", "is_active": True})
        assert "wf-1" in repr(wf)
        assert "Pick" in repr(wf)


# ======================================================================
# WorkflowRun object — properties
# ======================================================================


class TestWorkflowRunProperties:

    def test_properties_from_dict(self):
        client = _make_client()
        run = WorkflowRun(
            client,
            {
                "uuid": "run-1",
                "workflow_id": "wf-1",
                "status": "success",
                "inputs": {"speed": 0.5},
                "result": {"ok": True},
                "error": None,
                "started_at": "2026-01-01T00:00:00",
                "finished_at": "2026-01-01T00:00:05",
            },
        )
        assert run.uuid == "run-1"
        assert run.workflow_id == "wf-1"
        assert run.status == "success"
        assert run.inputs == {"speed": 0.5}
        assert run.result == {"ok": True}
        assert run.error is None
        assert run.completed_at == run.finished_at

    def test_is_terminal(self):
        client = _make_client()
        for status in ("success", "error", "canceled"):
            run = WorkflowRun(client, {"uuid": "r", "status": status})
            assert run.is_terminal, f"{status} should be terminal"

        for status in ("running", "waiting", "requested"):
            run = WorkflowRun(client, {"uuid": "r", "status": status})
            assert not run.is_terminal, f"{status} should not be terminal"

    def test_duration_with_datetime_objects(self):
        from datetime import datetime, timezone

        client = _make_client()
        start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
        run = WorkflowRun(
            client,
            {
                "uuid": "r",
                "started_at": start,
                "finished_at": end,
                "status": "success",
            },
        )
        assert run.duration == 10.0

    def test_duration_none_when_not_finished(self):
        client = _make_client()
        run = WorkflowRun(
            client,
            {"uuid": "r", "started_at": "2026-01-01T00:00:00", "finished_at": None},
        )
        assert run.duration is None


# ======================================================================
# WorkflowRun — refresh / wait / cancel
# ======================================================================


class TestWorkflowRunPolling:

    def test_refresh_updates_data(self):
        client = _make_client()
        run = WorkflowRun(client, {"uuid": "run-1", "status": "running"})

        updated = {"uuid": "run-1", "status": "success", "result": {"ok": True}}
        with patch("cyberwave.workflows._get_workflow_run", return_value=updated):
            run.refresh()

        assert run.status == "success"

    def test_wait_returns_on_terminal(self):
        client = _make_client()
        run = WorkflowRun(client, {"uuid": "run-1", "status": "running"})

        call_count = 0

        def fake_get(c, uuid):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                return {"uuid": "run-1", "status": "success"}
            return {"uuid": "run-1", "status": "running"}

        with patch("cyberwave.workflows._get_workflow_run", side_effect=fake_get):
            run.wait(timeout=10, poll_interval=0.01)

        assert run.status == "success"
        assert call_count == 2

    def test_wait_raises_on_timeout(self):
        client = _make_client()
        run = WorkflowRun(client, {"uuid": "run-1", "status": "running"})

        always_running = {"uuid": "run-1", "status": "running"}
        with patch(
            "cyberwave.workflows._get_workflow_run", return_value=always_running
        ):
            with pytest.raises(CyberwaveTimeoutError, match="did not complete"):
                run.wait(timeout=0.05, poll_interval=0.01)

    def test_wait_returns_immediately_when_already_terminal(self):
        client = _make_client()
        run = WorkflowRun(client, {"uuid": "run-1", "status": "success"})

        with patch("cyberwave.workflows._get_workflow_run") as mock_get:
            run.wait(timeout=5)

        mock_get.assert_not_called()

    def test_cancel_updates_data(self):
        client = _make_client()
        run = WorkflowRun(client, {"uuid": "run-1", "status": "running"})

        canceled = {"uuid": "run-1", "status": "canceled"}
        with patch(
            "cyberwave.workflows._cancel_workflow_run", return_value=canceled
        ) as mock_cancel:
            run.cancel()

        mock_cancel.assert_called_once_with(client, "run-1")
        assert run.status == "canceled"

    def test_wait_through_multiple_status_transitions(self):
        """Simulate running -> waiting -> success transition."""
        client = _make_client()
        run = WorkflowRun(client, {"uuid": "run-1", "status": "running"})

        statuses = iter(["running", "waiting", "waiting", "success"])

        def fake_get(c, uuid):
            return {"uuid": "run-1", "status": next(statuses)}

        with patch("cyberwave.workflows._get_workflow_run", side_effect=fake_get):
            run.wait(timeout=10, poll_interval=0.01)

        assert run.status == "success"


class TestTriggerParams:
    def test_params_are_forwarded_only_when_given(self):
        client = _make_client()
        manager = WorkflowManager(client)
        raw = {"uuid": "run-1", "workflow_id": "wf-1", "status": "running", "inputs": {}}
        params = {"route": {"waypoints": [[0, 0], [1, 2]]}, "speed": 2}

        with patch(
            "cyberwave.workflows._trigger_workflow", return_value=raw
        ) as mock_trigger:
            manager.trigger("wf-1", params=params)

        mock_trigger.assert_called_once_with(client, "wf-1", None, params=params)

    def test_workflow_trigger_forwards_params(self):
        client = _make_client()
        client.workflows = MagicMock()
        wf = Workflow(client, {"uuid": "wf-1", "is_active": True})

        wf.trigger(inputs={"x": 1}, params={"speed": 2})

        client.workflows.trigger.assert_called_once_with(
            "wf-1", inputs={"x": 1}, params={"speed": 2}
        )
