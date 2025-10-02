"""Mission planning helpers for the compact SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from .compact_api import (
    CompactTwin,
    _await_result,
    _get_client,
    _resolve_twin,
)


@dataclass
class MissionTaskSummary:
    """Structured summary of a planned mission task."""

    id: str
    title: str
    command: str
    payload: Dict[str, Any]
    twin_uuid: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MissionPlanResult:
    """Result returned from the mission planning endpoint."""

    prompt: str
    run_uuid: str
    executed: bool
    tasks: List[MissionTaskSummary]
    ignored_instructions: Optional[List[str]] = None
    results: Optional[List[Dict[str, Any]]] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    twin_map: Dict[str, CompactTwin] = field(default_factory=dict)

    def task_commands(self) -> List[str]:
        return [task.command for task in self.tasks]


class CompactAgent:
    """Mission planning helper that coordinates with backend workflow services."""

    def __init__(self, agent_uuid: Optional[str] = None, name: Optional[str] = None):
        self.agent_uuid = agent_uuid
        self.name = name or agent_uuid or "compact-agent"
        self._client = _get_client()
        self._last_run_uuid: Optional[str] = None

    def plan(
        self,
        prompt: str,
        twins: Sequence[Union[str, CompactTwin]],
        *,
        execute: bool = False,
    ) -> MissionPlanResult:
        twin_objects = self._prepare_twins(twins)
        if not twin_objects:
            raise ValueError("At least one twin is required to plan a mission")

        environment_ids = {
            self._resolve_environment_uuid(twin) for twin in twin_objects if self._resolve_environment_uuid(twin)
        }
        if len(environment_ids) != 1:
            raise ValueError("All twins must belong to the same environment for mission planning")
        environment_uuid = environment_ids.pop()

        run_uuid = self._start_run(environment_uuid, prompt)

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "twin_uuids": [self._require_uuid(twin) for twin in twin_objects],
            "dry_run": not execute,
        }
        if self.agent_uuid:
            payload["agent_uuid"] = self.agent_uuid

        response = _await_result(
            self._client._http.post(
                f"missions/runs/{run_uuid}/prompt-workflow",
                payload,
            )
        )

        self._last_run_uuid = run_uuid
        return self._to_plan_result(response, twin_objects)

    def execute(
        self,
        prompt: str,
        twins: Sequence[Union[str, CompactTwin]],
    ) -> MissionPlanResult:
        return self.plan(prompt, twins, execute=True)

    def run_log(
        self,
        *,
        since: Optional[float] = None,
        after: Optional[int] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        if not self._last_run_uuid:
            raise RuntimeError("No mission run available. Call plan() or execute() first.")

        params: Dict[str, Any] = {"limit": limit}
        if since is not None:
            params["since"] = since
        if after is not None:
            params["after"] = after

        return _await_result(
            self._client._http.get(
                f"missions/runs/{self._last_run_uuid}/log",
                params=params,
            )
        )

    def _prepare_twins(self, twins: Sequence[Union[str, CompactTwin]]) -> List[CompactTwin]:
        resolved: List[CompactTwin] = []
        for item in twins:
            twin = _resolve_twin(item) if not isinstance(item, CompactTwin) else item
            if twin is None:
                raise ValueError(f"Twin '{item}' not found. Use cw.twin() to create it first.")
            _await_result(twin._ensure_twin_exists())
            resolved.append(twin)
        return resolved

    def _resolve_environment_uuid(self, twin: CompactTwin) -> Optional[str]:
        if twin._environment_uuid:
            return twin._environment_uuid
        return twin.environment_id

    def _require_uuid(self, twin: CompactTwin) -> str:
        if not twin.uuid or twin.uuid.startswith("local-"):
            raise RuntimeError(
                f"Twin '{twin.name}' does not have a backend UUID. Configure authentication to enable mission planning."
            )
        return twin.uuid

    def _start_run(self, environment_uuid: str, prompt: str) -> str:
        run = _await_result(
            self._client.runs.start(
                environment_uuid=environment_uuid,
                mission_spec={
                    "description": prompt,
                    "created_via": "compact_agent",
                },
            )
        )
        run_uuid = run.get("uuid") or run.get("run_uuid") or run.get("id")
        if not run_uuid:
            raise RuntimeError("Mission run did not return a UUID")
        return run_uuid

    def _to_plan_result(
        self,
        response: Dict[str, Any],
        twins: Sequence[CompactTwin],
    ) -> MissionPlanResult:
        tasks_payload = response.get("tasks") or []
        tasks: List[MissionTaskSummary] = []
        for item in tasks_payload:
            tasks.append(
                MissionTaskSummary(
                    id=str(item.get("id") or ""),
                    title=str(item.get("title") or ""),
                    command=str(item.get("command") or ""),
                    payload=dict(item.get("payload") or {}),
                    twin_uuid=str(item.get("target_twin_uuid") or ""),
                    metadata=dict(item.get("metadata") or {}),
                )
            )

        twin_map = {self._require_uuid(twin): twin for twin in twins if twin.uuid}

        return MissionPlanResult(
            prompt=response.get("prompt") or "",
            run_uuid=response.get("run_uuid") or self._last_run_uuid or "",
            executed=bool(response.get("executed")),
            tasks=tasks,
            ignored_instructions=response.get("ignored_instructions"),
            results=response.get("results"),
            raw=dict(response),
            twin_map=twin_map,
        )


def agent(agent_uuid: Optional[str] = None, name: Optional[str] = None) -> CompactAgent:
    """Create a compact agent wrapper for mission planning."""

    return CompactAgent(agent_uuid=agent_uuid, name=name)


__all__ = [
    "agent",
    "CompactAgent",
    "MissionPlanResult",
    "MissionTaskSummary",
]
