"""
WorkflowExecutionReporter and WorkflowExecutionManager.

Used by the generated worker source (``wf_*.py``) and by SDK consumers
that want to report mission-workflow progress back to Cyberwave.

Transport
---------
Reporters publish to MQTT topics under
``cyberwave/workflow/{workflow_uuid}/execution/...`` which the backend
consumer (``src/app/mqtt_consumer.py``) funnels into the
``workflow_execution_ingress`` service. MQTT is the only supported
transport — callers must be connected to a Cyberwave-backed MQTT broker
via ``client.mqtt`` before calling :meth:`WorkflowExecutionManager.start`.

Example::

    client = Cyberwave(api_key="...")

    reporter = client.workflow_executions.start(
        workflow_uuid="wf-uuid",
        source_type="edge",
    )
    try:
        for node_uuid in node_uuids:
            reporter.node_started(node_uuid)
            try:
                run_node(node_uuid)
                reporter.node_finished(node_uuid, output_data=[...])
            except Exception as exc:
                reporter.node_error(node_uuid, error=str(exc))
                raise
        reporter.finished(status="success")
    except Exception as exc:
        reporter.finished(status="error", error_message=str(exc))
        raise
"""

from __future__ import annotations

import json
import logging
import uuid as _uuid_mod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .exceptions import CyberwaveError

if TYPE_CHECKING:
    from .client import Cyberwave

logger = logging.getLogger(__name__)

_EXECUTION_TERMINAL_STATUSES = frozenset({"success", "error", "canceled"})
_NODE_TERMINAL_STATUSES = frozenset({"success", "error", "skipped"})


class WorkflowExecutionReporter:
    """Stateful MQTT reporter that owns a single execution.

    Created via :meth:`WorkflowExecutionManager.start`, which sends the
    initial ``started`` event synchronously. Subsequent publishes are
    best-effort: failures are logged but not re-raised so a transient
    broker hiccup doesn't poison the running workflow. Callers that
    need strict error propagation can construct the reporter with
    ``strict=True``.
    """

    def __init__(
        self,
        client: "Cyberwave",
        *,
        workflow_uuid: str,
        execution_uuid: str,
        source_type: Optional[str] = "edge",
        strict: bool = False,
    ) -> None:
        self._client = client
        self._workflow_uuid = workflow_uuid
        self._execution_uuid = execution_uuid
        self._source_type = source_type
        self._strict = strict
        self._finished = False

    @property
    def workflow_uuid(self) -> str:
        return self._workflow_uuid

    @property
    def execution_uuid(self) -> str:
        return self._execution_uuid

    # ------------------------------------------------------------------
    # Public reporter API
    # ------------------------------------------------------------------

    def node_started(
        self,
        node_uuid: str,
        *,
        input_data: Optional[List[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark a node as running."""
        self._send_node_event(
            node_uuid,
            status="running",
            input_data=input_data,
            metadata=metadata,
        )

    def node_finished(
        self,
        node_uuid: str,
        *,
        output_data: Optional[List[Any]] = None,
        execution_time_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark a node as having finished successfully."""
        self._send_node_event(
            node_uuid,
            status="success",
            output_data=output_data,
            execution_time_ms=execution_time_ms,
            metadata=metadata,
        )

    def node_error(
        self,
        node_uuid: str,
        *,
        error: str,
        execution_time_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark a node as failed."""
        self._send_node_event(
            node_uuid,
            status="error",
            error_message=str(error),
            execution_time_ms=execution_time_ms,
            metadata=metadata,
        )

    def node_skipped(
        self,
        node_uuid: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark a node as skipped (e.g. conditional branch not taken)."""
        self._send_node_event(
            node_uuid,
            status="skipped",
            metadata=metadata,
        )

    def finished(
        self,
        *,
        status: str = "success",
        error_message: Optional[str] = None,
    ) -> None:
        """Apply a terminal status to the execution.

        Idempotent: calling ``finished`` more than once is a no-op
        after the first call, matching the backend's terminal-wins
        behaviour.
        """
        if self._finished:
            return
        if status not in _EXECUTION_TERMINAL_STATUSES:
            raise ValueError(
                f"Invalid execution status {status!r}. "
                f"Expected one of {sorted(_EXECUTION_TERMINAL_STATUSES)}."
            )
        payload: Dict[str, Any] = {"status": status}
        if error_message is not None:
            payload["error_message"] = error_message
        if self._source_type:
            payload["source_type"] = self._source_type

        topic = (
            f"cyberwave/workflow/{self._workflow_uuid}/execution/"
            f"{self._execution_uuid}/finished"
        )
        self._publish(topic, payload, event="finished")
        self._finished = True

    # ------------------------------------------------------------------
    # Context manager sugar
    # ------------------------------------------------------------------

    def __enter__(self) -> "WorkflowExecutionReporter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._finished:
            return
        if exc is None:
            self.finished(status="success")
        else:
            self.finished(status="error", error_message=str(exc))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _send_node_event(
        self,
        node_uuid: str,
        *,
        status: str,
        input_data: Optional[List[Any]] = None,
        output_data: Optional[List[Any]] = None,
        error_message: Optional[str] = None,
        execution_time_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        valid = _NODE_TERMINAL_STATUSES | {"pending", "running"}
        if status not in valid:
            raise ValueError(
                f"Invalid node status {status!r}. Expected one of {sorted(valid)}."
            )
        payload: Dict[str, Any] = {"status": status}
        if input_data is not None:
            payload["input_data"] = list(input_data)
        if output_data is not None:
            payload["output_data"] = list(output_data)
        if error_message is not None:
            payload["error_message"] = error_message
        if execution_time_ms is not None:
            payload["execution_time_ms"] = int(execution_time_ms)
        if metadata:
            payload["metadata"] = dict(metadata)
        if self._source_type:
            payload["source_type"] = self._source_type

        topic = (
            f"cyberwave/workflow/{self._workflow_uuid}/execution/"
            f"{self._execution_uuid}/node/{node_uuid}"
        )
        self._publish(topic, payload, event=f"node:{status}")

    def _send_started(
        self,
        *,
        trigger_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send the initial ``started`` event.

        Called from :meth:`WorkflowExecutionManager.start` before the
        reporter is returned to the caller. Kept private so external
        callers can't re-send it and double-seed node rows. Always
        strict: node events downstream reference this execution UUID,
        so a silently-dropped ``started`` would make them 404.
        """
        payload: Dict[str, Any] = {"execution_uuid": self._execution_uuid}
        if trigger_data:
            payload["trigger_data"] = dict(trigger_data)
        if self._source_type:
            payload["source_type"] = self._source_type

        topic = f"cyberwave/workflow/{self._workflow_uuid}/execution/started"
        self._publish(topic, payload, event="started", force_strict=True)

    def _publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        *,
        event: str,
        force_strict: bool = False,
    ) -> None:
        mqtt = getattr(self._client, "mqtt", None)
        if mqtt is None:
            message = (
                f"workflow_executions: cannot publish {event}; "
                "client has no MQTT transport attached"
            )
            if self._strict or force_strict:
                raise CyberwaveError(message)
            logger.warning(message)
            return

        if not getattr(mqtt, "connected", False):
            message = (
                f"workflow_executions: cannot publish {event} to {topic}; "
                "MQTT client is not connected"
            )
            if self._strict or force_strict:
                raise CyberwaveError(message)
            logger.warning(message)
            return

        prefix = getattr(mqtt, "topic_prefix", "") or ""
        full_topic = f"{prefix}{topic}"
        try:
            mqtt.publish(full_topic, payload)
        except Exception as exc:  # noqa: BLE001
            message = (
                f"workflow_executions: MQTT {event} publish to "
                f"{full_topic} failed: {exc}"
            )
            if self._strict or force_strict:
                raise CyberwaveError(message) from exc
            logger.warning(message)


class WorkflowExecutionManager:
    """Manager accessed as ``client.workflow_executions``."""

    def __init__(self, client: "Cyberwave") -> None:
        self._client = client

    def start(
        self,
        *,
        workflow_uuid: str,
        execution_uuid: Optional[str] = None,
        trigger_data: Optional[Dict[str, Any]] = None,
        source_type: Optional[str] = "edge",
        strict: bool = False,
    ) -> WorkflowExecutionReporter:
        """Begin reporting a new workflow execution over MQTT.

        Args:
            workflow_uuid: UUID of the workflow being executed.
            execution_uuid: Optional client-minted execution UUID. When
                omitted, a uuid4 is generated so the caller can refer
                to the execution before receiving a response (the
                backend accepts this and treats repeats idempotently).
            trigger_data: Optional payload describing what kicked off
                the run; carried through into ``WorkflowExecution.trigger_data``.
            source_type: Tag stored on the execution/node metadata for
                lineage (``"edge"``, ``"sim"``, ``"tele"``, ``"edit"``).
            strict: When ``True``, every MQTT publish failure is raised.
                The initial ``started`` publish is always strict.

        Returns:
            A :class:`WorkflowExecutionReporter` that has already sent
            its ``started`` event.
        """
        resolved_uuid = execution_uuid or str(_uuid_mod.uuid4())
        reporter = WorkflowExecutionReporter(
            self._client,
            workflow_uuid=workflow_uuid,
            execution_uuid=resolved_uuid,
            source_type=source_type,
            strict=strict,
        )
        reporter._send_started(trigger_data=trigger_data)
        return reporter

    def list_captures(
        self,
        execution_uuid: str,
        *,
        workflow_node_uuid: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List navigation captures attached to a workflow execution.

        When ``workflow_node_uuid`` is provided, results are filtered client-side
        to the captures tagged by that node. Returned items include a stable
        ``attachment_uuid`` alias for the attachment ``uuid`` so generated
        workers can expose a flat node output shape.
        """
        api_client = self._client.api.api_client
        _param = api_client.param_serialize(
            method="GET",
            resource_path=f"/api/v1/workflows/executions/{execution_uuid}/captures",
            auth_settings=["CustomTokenAuthentication"],
        )
        response_data = api_client.call_api(*_param)
        response_data.read()
        raw_items = json.loads(response_data.data.decode("utf-8"))
        if not isinstance(raw_items, list):
            return []

        captures: List[Dict[str, Any]] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            metadata = raw_item.get("metadata")
            if workflow_node_uuid and (
                not isinstance(metadata, dict)
                or metadata.get("workflow_node_uuid") != workflow_node_uuid
            ):
                continue
            capture = dict(raw_item)
            if capture.get("attachment_uuid") is None and capture.get("uuid") is not None:
                capture["attachment_uuid"] = capture.get("uuid")
            captures.append(capture)
        return captures
