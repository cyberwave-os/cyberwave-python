"""RL task scene-entity + strict task-spec SDK helpers.

This module exposes a small, REST-only client for the RL task endpoints
that are not yet covered by the auto-generated :mod:`cyberwave.rest`
surface. It deliberately stays thin: each method is a one-line wrapper
around the corresponding ``/api/v1/rl-tasks/...`` endpoint so the
public Python surface tracks the backend contract 1:1.

Typical use from a demo / setup script::

    from cyberwave import Cyberwave
    from cyberwave.rl_tasks import RLTaskClient

    cw = Cyberwave()
    rl = RLTaskClient(cw)

    rl.assign_articulation_entity(
        task_uuid,
        twin_uuid=robot_twin_uuid,
        name="robot",
        actuators=[{"type": "xml_position", "target_names_expr": [".*__openarm_left_joint[1-7]$"]}],
    )
    rl.assign_rigid_entity(task_uuid, twin_uuid=cube_twin_uuid, name="cube", base_type="free")
    rl.regenerate_scene_cfg(task_uuid)
    spec = rl.export_task_spec_python(task_uuid)

The strict task spec is the round-trip interchange::

    rl.import_task_spec_python(task_uuid, spec.content)
"""

from __future__ import annotations

import json as _json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import urllib3

if TYPE_CHECKING:
    from cyberwave.client import Cyberwave


@dataclass(frozen=True)
class TaskSpecExport:
    """Result of :meth:`RLTaskClient.export_task_spec_python`."""

    path: str
    content: str
    schema_version: int
    entity_count: int


class RLTaskClient:
    """REST helpers for RL task scene entities + strict task spec.

    The client borrows the base URL and API key from the supplied
    :class:`Cyberwave` instance so callers do not need to manage auth
    separately.
    """

    def __init__(self, client: Cyberwave, *, timeout: float = 60.0) -> None:
        self._base_url = client.config.base_url.rstrip("/")
        self._api_key = client.config.api_key
        self._timeout = timeout
        self._http = urllib3.PoolManager()

    # ------------------------------------------------------------------
    # Low-level HTTP helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _headers(self, *, json_body: bool = True) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _raise_for(self, resp: urllib3.BaseHTTPResponse, context: str) -> None:
        if resp.status >= 300:
            body = resp.data.decode("utf-8", errors="replace")
            raise RuntimeError(f"{context} failed {resp.status}: {body[:500]}")

    def _request(
        self, method: str, path: str, *, json: dict[str, Any] | None = None
    ) -> urllib3.BaseHTTPResponse:
        return self._http.request(
            method,
            self._url(path),
            json=json,
            headers=self._headers(json_body=json is not None),
            timeout=self._timeout,
        )

    def _get(self, path: str) -> Any:
        resp = self._request("GET", path)
        self._raise_for(resp, f"GET {path}")
        return resp.json()

    def _post(self, path: str, *, json: dict[str, Any]) -> Any:
        resp = self._request("POST", path, json=json)
        self._raise_for(resp, f"POST {path}")
        return resp.json()

    def _put(self, path: str, *, json: dict[str, Any]) -> Any:
        resp = self._request("PUT", path, json=json)
        self._raise_for(resp, f"PUT {path}")
        return resp.json()

    def _patch(self, path: str, *, json: dict[str, Any]) -> Any:
        resp = self._request("PATCH", path, json=json)
        self._raise_for(resp, f"PATCH {path}")
        return resp.json()

    def _delete(self, path: str) -> Any:
        resp = self._request("DELETE", path)
        self._raise_for(resp, f"DELETE {path}")
        return resp.json() if resp.data else {}

    # ------------------------------------------------------------------
    # Scene-entity CRUD
    # ------------------------------------------------------------------

    def list_scene_entities(self, task_uuid: str) -> list[dict[str, Any]]:
        """Return the RL task's current scene-entity rows."""
        return self._get(f"/api/v1/rl-tasks/{task_uuid}/scene-entities")

    def create_scene_entity(
        self, task_uuid: str, *, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a single scene entity from a fully-formed payload."""
        return self._post(f"/api/v1/rl-tasks/{task_uuid}/scene-entities", json=payload)

    def update_scene_entity(
        self, task_uuid: str, entity_uuid: str, *, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Patch a single scene-entity row."""
        return self._patch(
            f"/api/v1/rl-tasks/{task_uuid}/scene-entities/{entity_uuid}",
            json=payload,
        )

    def delete_scene_entity(self, task_uuid: str, entity_uuid: str) -> dict[str, Any]:
        """Delete a single scene-entity row."""
        return self._delete(
            f"/api/v1/rl-tasks/{task_uuid}/scene-entities/{entity_uuid}"
        )

    def replace_scene_entities(
        self, task_uuid: str, entities: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Replace the RL task's whole scene-entity set in one transaction."""
        return self._put(
            f"/api/v1/rl-tasks/{task_uuid}/scene-entities",
            json={"entities": entities},
        )

    def get_scene_entity_hints(self, task_uuid: str) -> dict[str, Any]:
        """Return the backend-derived hints for every twin in the task's env.

        Each hint exposes the values the backend will use to seed /
        reconcile a scene-entity row created against that twin:
        ``base_type``, ``entity_kind``, ``zero_root_pose``, initial
        pose + joint positions, and the controllable joint /
        actuator candidates pulled from the twin's
        ``universal_schema``. Use this to drive a composer UI or
        validate that an SDK payload aligns with what the codegen
        will actually emit before sending the create / replace call.
        """

        return self._get(f"/api/v1/rl-tasks/{task_uuid}/scene-entity-hints")

    # ------------------------------------------------------------------
    # Convenience helpers — used by demos so the canonical workflow is
    # discoverable from the SDK docs.
    # ------------------------------------------------------------------

    def assign_articulation_entity(
        self,
        task_uuid: str,
        *,
        twin_uuid: str,
        name: str,
        actuators: list[dict[str, Any]] | None = None,
        soft_joint_pos_limit_factor: float | None = None,
        initial_state: dict[str, Any] | None = None,
        selectors: dict[str, Any] | None = None,
        sensors: list[dict[str, Any]] | None = None,
        include_contacts: bool = False,
        base_type: str = "fixed",
        entity_cfg: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assign a twin as an mjlab articulation entity.

        ``actuators`` follows the strict schema: a list of dicts with
        ``type`` (one of ``xml_position``, ``xml_velocity``, ``xml_motor``,
        ``xml_muscle``) and ``target_names_expr`` (list of regex patterns).
        mjlab's ``Xml*ActuatorCfg`` wrappers bind to actuators that the
        MJCF already defines, so PD gains / effort limits / damping
        live on those XML elements. The only extra kwargs accepted at
        the actuator group level are ``armature`` and ``frictionloss``.

        ``entity_cfg`` is the reference to the mjlab ``EntityCfg`` (or
        subclass / factory) the codegen should emit. Defaults to
        ``{"module": "mjlab.entity", "symbol": "EntityCfg", "kind":
        "class"}``; pass ``{"module": "openarm_entity", "symbol":
        "OpenArmCfg"}`` to wire in a user-uploaded subclass.
        """

        articulation: dict[str, Any] = {}
        if actuators:
            articulation["actuators"] = actuators
        if soft_joint_pos_limit_factor is not None:
            articulation["soft_joint_pos_limit_factor"] = soft_joint_pos_limit_factor

        payload: dict[str, Any] = {
            "name": name,
            "twin_uuid": twin_uuid,
            "entity_kind": "articulation",
            "base_type": base_type,
            "include_actuators": bool(actuators),
            "include_contacts": include_contacts,
            "articulation": articulation,
        }
        if entity_cfg:
            payload["entity_cfg"] = entity_cfg
        if initial_state:
            payload["initial_state"] = initial_state
        if selectors:
            payload["selectors"] = selectors
        if sensors:
            payload["sensors"] = sensors
        return self.create_scene_entity(task_uuid, payload=payload)

    def assign_rigid_entity(
        self,
        task_uuid: str,
        *,
        twin_uuid: str,
        name: str,
        base_type: str = "fixed",
        initial_state: dict[str, Any] | None = None,
        selectors: dict[str, Any] | None = None,
        sensors: list[dict[str, Any]] | None = None,
        include_contacts: bool = False,
        entity_cfg: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assign a twin as an mjlab rigid-object entity.

        ``base_type="free"`` opts into a freejoint-driven respawn; the
        backend automatically sets ``zero_root_pose=False`` for free
        entities so the authored MJCF pose is kept intact.

        ``entity_cfg`` defaults to the vanilla
        ``mjlab.entity.EntityCfg``. Override it (and upload the source
        file via the source-files API) to drive a custom subclass.
        """

        payload: dict[str, Any] = {
            "name": name,
            "twin_uuid": twin_uuid,
            "entity_kind": "rigid_object",
            "base_type": base_type,
            "include_actuators": False,
            "include_contacts": include_contacts,
        }
        if entity_cfg:
            payload["entity_cfg"] = entity_cfg
        if initial_state:
            payload["initial_state"] = initial_state
        if selectors:
            payload["selectors"] = selectors
        if sensors:
            payload["sensors"] = sensors
        return self.create_scene_entity(task_uuid, payload=payload)

    # ------------------------------------------------------------------
    # SceneCfg regeneration
    # ------------------------------------------------------------------

    def regenerate_scene_cfg(
        self,
        task_uuid: str,
        *,
        scene_cfg_path: str = "scene_cfg.py",
        factory_name: str = "make_scene_cfg",
    ) -> dict[str, Any]:
        """Trigger a server-side regenerate of ``scene_cfg.py``.

        After scene-entity changes you should call this once so the
        Cyberwave-owned source tree reflects the new entities.
        """
        return self._post(
            f"/api/v1/rl-tasks/{task_uuid}/regenerate-scene-cfg",
            json={"scene_cfg_path": scene_cfg_path, "factory_name": factory_name},
        )

    # ------------------------------------------------------------------
    # Strict Python task spec (round-trip)
    # ------------------------------------------------------------------

    def export_task_spec_python(self, task_uuid: str) -> TaskSpecExport:
        """Return the canonical ``cyberwave_task_spec.py`` for a task."""

        payload = self._get(f"/api/v1/rl-tasks/{task_uuid}/task-spec.py")
        return TaskSpecExport(
            path=payload["path"],
            content=payload["content"],
            schema_version=int(payload["schema_version"]),
            entity_count=int(payload["entity_count"]),
        )

    def import_task_spec_python(
        self, task_uuid: str, source: str, *, validate_only: bool = False
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Apply a strict task-spec module to an RL task.

        With ``validate_only=True`` we hit the validate endpoint and
        return its diagnostic payload (``{"valid": bool, ...}``) instead
        of mutating the RL task.
        """

        if validate_only:
            return self._post(
                f"/api/v1/rl-tasks/{task_uuid}/task-spec/validate",
                json={"content": source, "validate_only": True},
            )
        return self._put(
            f"/api/v1/rl-tasks/{task_uuid}/task-spec.py",
            json={"content": source, "validate_only": False},
        )

    # ------------------------------------------------------------------
    # Runtime target + version pins
    # ------------------------------------------------------------------

    def set_runtime(
        self,
        task_uuid: str,
        *,
        runtime_target: str | None = None,
        runtime_accelerator: str | None = None,
        runtime_versions: dict[str, str] | None = None,
        policy_interface: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Patch the RL task's runtime metadata via the standard update endpoint.

        ``runtime_target`` is ``"cyberwave-rl"`` or
        ``"cyberwave-rl-experimental"``. ``runtime_accelerator`` is
        ``"cpu"`` or ``"gpu"`` — informational for the experimental
        runtime, drives container selection for dockerized runs.
        ``runtime_versions`` is a free-form ``{package: version}`` map;
        the UI normally exposes the single ``"runtime_bundle"`` pin.
        ``policy_interface`` describes the observation / action
        contract so inference workers can validate weights before
        running.
        """
        payload: dict[str, Any] = {}
        if runtime_target is not None:
            payload["runtime_target"] = runtime_target
        if runtime_accelerator is not None:
            payload["runtime_accelerator"] = runtime_accelerator
        if runtime_versions is not None:
            payload["runtime_versions"] = dict(runtime_versions)
        if policy_interface is not None:
            payload["policy_interface"] = dict(policy_interface)
        if not payload:
            return self._get(f"/api/v1/rl-tasks/{task_uuid}")
        return self._put(f"/api/v1/rl-tasks/{task_uuid}", json=payload)

    # ------------------------------------------------------------------
    # Duplicate (clone)
    # ------------------------------------------------------------------

    def clone_task(
        self,
        task_uuid: str,
        *,
        name: str | None = None,
        description: str | None = None,
        workspace_uuid: str | None = None,
        environment_uuid: str | None = None,
        project_uuid: str | None = None,
        visibility: str | None = None,
    ) -> dict[str, Any]:
        """Duplicate an RL task definition.

        Copies metadata, runtime contract, source files, and scene
        entities into a new task row. Checkpoints and inference runs
        are intentionally not cloned. The default destination is the
        source task's workspace; pass ``workspace_uuid`` to clone into
        another workspace you can write to.
        """

        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if workspace_uuid is not None:
            payload["workspace_uuid"] = workspace_uuid
        if environment_uuid is not None:
            payload["environment_uuid"] = environment_uuid
        if project_uuid is not None:
            payload["project_uuid"] = project_uuid
        if visibility is not None:
            payload["visibility"] = visibility
        return self._post(f"/api/v1/rl-tasks/{task_uuid}/clone", json=payload)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def list_checkpoints(self, task_uuid: str) -> list[dict[str, Any]]:
        """List trained policy checkpoints registered on an RL task."""
        return self._get(f"/api/v1/rl-tasks/{task_uuid}/checkpoints")

    def register_checkpoint(
        self,
        task_uuid: str,
        *,
        name: str,
        weights_url: str | None = None,
        attachment_uuid: str | None = None,
        description: str | None = None,
        runtime_target: str | None = None,
        runtime_accelerator: str | None = None,
        runtime_versions: dict[str, str] | None = None,
        policy_interface: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register a trained checkpoint.

        Exactly one of ``weights_url`` and ``attachment_uuid`` must be
        set. When the runtime / interface fields are omitted, the
        backend copies them from the parent RL task so the common case
        stays a one-liner.
        """
        payload: dict[str, Any] = {"name": name}
        if weights_url is not None:
            payload["weights_url"] = weights_url
        if attachment_uuid is not None:
            payload["attachment_uuid"] = attachment_uuid
        if description is not None:
            payload["description"] = description
        if runtime_target is not None:
            payload["runtime_target"] = runtime_target
        if runtime_accelerator is not None:
            payload["runtime_accelerator"] = runtime_accelerator
        if runtime_versions is not None:
            payload["runtime_versions"] = dict(runtime_versions)
        if policy_interface is not None:
            payload["policy_interface"] = dict(policy_interface)
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        return self._post(f"/api/v1/rl-tasks/{task_uuid}/checkpoints", json=payload)

    def delete_checkpoint(self, task_uuid: str, checkpoint_uuid: str) -> dict[str, Any]:
        return self._delete(
            f"/api/v1/rl-tasks/{task_uuid}/checkpoints/{checkpoint_uuid}"
        )

    def upload_checkpoint(
        self,
        task_uuid: str,
        path: str | os.PathLike[str],
        *,
        name: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Upload a trained policy file and register the checkpoint row.

        Convenience wrapper around the
        ``POST /api/v1/rl-tasks/{task_uuid}/checkpoints/upload`` endpoint.
        Handles the multipart upload so the typical "I just trained
        locally" path is a one-liner::

            rl.upload_checkpoint(
                task_uuid,
                "cyberwave-rl/logs/<task>/<run>/checkpoints/best_agent.pt",
                name="best_agent",
            )

        Returns the same ``RLTaskCheckpointSchema`` dict as
        :meth:`register_checkpoint` so callers can immediately read
        ``uuid`` / ``attachment_uuid`` for downstream play / inference
        commands.
        """

        source = Path(os.fspath(path))
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"Checkpoint path does not exist: {source}")

        data: dict[str, str] = {}
        if name is not None:
            data["name"] = str(name)
        if description is not None:
            data["description"] = str(description)
        if metadata is not None:
            data["metadata"] = _json.dumps(metadata)

        # urllib3 builds the multipart/form-data body from a single ``fields``
        # mapping. File parts are ``(filename, data, content_type)`` tuples and
        # require the bytes in memory (urllib3 does not stream a file handle the
        # way ``requests`` does). Do not set Content-Type here — urllib3 emits
        # the multipart boundary itself.
        fields: dict[str, Any] = dict(data)
        fields["file"] = (
            source.name,
            source.read_bytes(),
            "application/octet-stream",
        )
        resp = self._http.request(
            "POST",
            self._url(f"/api/v1/rl-tasks/{task_uuid}/checkpoints/upload"),
            fields=fields,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self._timeout,
        )
        self._raise_for(resp, f"POST /api/v1/rl-tasks/{task_uuid}/checkpoints/upload")
        return resp.json()

    def download_checkpoint(
        self,
        task_uuid: str,
        checkpoint_uuid: str,
        *,
        destination: str | os.PathLike[str] | None = None,
        chunk_size: int = 1024 * 1024,
    ) -> Path:
        """Download a registered checkpoint's weights to disk.

        Resolves attachment-backed and URL-backed checkpoints uniformly:

        * Attachment checkpoints redirect to a CDN-signed URL via the
          generic ``GET /api/v1/attachments/{uuid}/download`` route.
        * URL-backed checkpoints stream straight from ``weights_url``.

        Returns the destination :class:`~pathlib.Path`. When
        ``destination`` is ``None`` the file is written next to
        ``cwd / "<checkpoint_name>.pt"``.
        """

        ckpt = self._get(f"/api/v1/rl-tasks/{task_uuid}/checkpoints/{checkpoint_uuid}")

        attachment_uuid = ckpt.get("attachment_uuid")
        weights_url = (ckpt.get("weights_url") or "").strip()
        if attachment_uuid:
            url = self._url(f"/api/v1/attachments/{attachment_uuid}/download")
            headers = self._headers(json_body=False)
        elif weights_url:
            url = weights_url
            # External URLs do not accept our auth header; only attach
            # the bearer when the URL points at the same backend host.
            headers = (
                self._headers(json_body=False)
                if weights_url.startswith(self._base_url)
                else {}
            )
        else:
            raise RuntimeError(
                f"Checkpoint {checkpoint_uuid} has no attachment or weights_url"
            )

        if destination is None:
            filename = self._guess_checkpoint_filename(ckpt)
            target = Path.cwd() / filename
        else:
            target = Path(os.fspath(destination))
            if target.is_dir():
                target = target / self._guess_checkpoint_filename(ckpt)

        target.parent.mkdir(parents=True, exist_ok=True)
        # ``preload_content=False`` keeps the body unread so we can stream it to
        # disk in chunks; PoolManager follows redirects by default (the old
        # ``allow_redirects=True``).
        resp = self._http.request(
            "GET",
            url,
            headers=headers,
            preload_content=False,
            timeout=self._timeout,
        )
        try:
            self._raise_for(resp, f"GET {url}")
            with target.open("wb") as out:
                for chunk in resp.stream(chunk_size):
                    if chunk:
                        out.write(chunk)
        finally:
            resp.release_conn()
        return target

    @staticmethod
    def _guess_checkpoint_filename(checkpoint: dict[str, Any]) -> str:
        """Pick a sensible local filename for a downloaded checkpoint."""

        meta = checkpoint.get("metadata") or {}
        original = meta.get("original_filename")
        if isinstance(original, str) and original:
            return os.path.basename(original)
        name = (checkpoint.get("name") or "checkpoint").strip() or "checkpoint"
        return f"{name}.pt"

    @staticmethod
    def prepare_checkpoint_for_play(
        *,
        task_id: str,
        checkpoint_path: str | os.PathLike[str],
        logs_root: str | os.PathLike[str] = "logs",
        run_name: str = "uploaded",
        filename: str = "best_agent.pt",
    ) -> Path:
        """Stage a loose checkpoint into the ``logs/<task>/<run>/checkpoints`` layout.

        ``cyberwave-rl``'s ``view_task`` / ``play`` entrypoints walk the
        ``logs/<task>/<run>/checkpoints/<file>`` directory tree to pick
        up runs. After downloading a checkpoint into an arbitrary
        location, call this helper to drop it into the expected layout::

            staged = RLTaskClient.prepare_checkpoint_for_play(
                task_id="Cyberwave-Openarm-Sdc-Reachout",
                checkpoint_path="/tmp/best_agent.pt",
            )
            # staged == logs/<task_id>/uploaded/checkpoints/best_agent.pt

        The file is copied (not moved) so the caller can keep the
        original around. The function returns the staged path.
        """

        source = Path(os.fspath(checkpoint_path))
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"Checkpoint path does not exist: {source}")
        target_dir = Path(os.fspath(logs_root)) / task_id / run_name / "checkpoints"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / filename
        shutil.copyfile(source, target)
        return target

    # ------------------------------------------------------------------
    # Backend-dispatched inference (no MuJoCo loop)
    # ------------------------------------------------------------------

    def launch_inference(
        self,
        task_uuid: str,
        *,
        checkpoint_uuid: str,
        twin_uuid: str | None = None,
        runtime_target: str | None = None,
        runtime_accelerator: str | None = None,
        runtime_versions: dict[str, str] | None = None,
        max_steps: int | None = None,
        control_rate_hz: float | None = None,
        mode: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Dispatch a backend inference workload.

        The backend does *not* run MuJoCo; the worker bootstraps the
        requested runtime versions, loads the policy weights, and loops
        the env against Cyberwave joint state + MQTT commands.
        """
        payload: dict[str, Any] = {"checkpoint_uuid": checkpoint_uuid}
        if twin_uuid is not None:
            payload["twin_uuid"] = twin_uuid
        if runtime_target is not None:
            payload["runtime_target"] = runtime_target
        if runtime_accelerator is not None:
            payload["runtime_accelerator"] = runtime_accelerator
        if runtime_versions is not None:
            payload["runtime_versions"] = dict(runtime_versions)
        if max_steps is not None:
            payload["max_steps"] = int(max_steps)
        if control_rate_hz is not None:
            payload["control_rate_hz"] = float(control_rate_hz)
        if mode is not None:
            payload["mode"] = mode
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        return self._post(f"/api/v1/rl-tasks/{task_uuid}/inference", json=payload)

    def list_inference_runs(self, task_uuid: str) -> list[dict[str, Any]]:
        return self._get(f"/api/v1/rl-tasks/{task_uuid}/inference")

    # ------------------------------------------------------------------
    # Orchestration tabs (Actions / Observations / RL Config)
    # ------------------------------------------------------------------
    #
    # These mirror the editor's "Actions", "Observations", and "RL Config"
    # tabs 1:1 so a demo / setup script can author the same specs the UI
    # produces. ``get_orchestration_hints`` returns the scene-derived
    # metadata (articulated entities, actuator groups per entity,
    # actuated vs passive joint names, available sensors) the editor
    # uses to populate dropdowns and validate picker output.

    def get_orchestration_hints(self, task_uuid: str) -> dict[str, Any]:
        """Return the backend's orchestration hints for the editor tabs.

        Hint payload (subset):

        * ``articulation_entities`` — names of articulated scene entities.
        * ``entity_actuator_types`` — flat ``{entity: [actuator_type,...]}``.
        * ``entity_actuator_groups`` — ``{entity: [{type, target_names_expr}]}``.
        * ``entity_actuated_joint_names`` — concrete actuated joints per entity.
        * ``entity_passive_joint_names`` — joints exposed by the schema but
          not connected to any actuator group.
        * ``available_sensors`` — sensor rows wired in scene entities.

        Use this before constructing an action / observation spec to
        confirm the joint-picker output aligns with the backend's
        actuator-safe set.
        """

        return self._get(f"/api/v1/rl-tasks/{task_uuid}/orchestration-hints")

    def get_action_spec(self, task_uuid: str) -> dict[str, Any]:
        """Return the persisted ``ACTION_SPEC`` payload (Actions tab).

        The response is ``{"action_spec": {...}}`` — same shape the
        editor renders. An empty / unset spec returns ``{"action_spec":
        {}}`` rather than a 404.
        """

        return self._get(f"/api/v1/rl-tasks/{task_uuid}/actions")

    def set_action_spec(
        self, task_uuid: str, action_spec: dict[str, Any]
    ) -> dict[str, Any]:
        """Replace the persisted action spec.

        ``action_spec`` is the same dict shape produced by the editor:
        ``{"schema_version": 1, "actions": [...]}``. The backend
        validates and normalizes the payload (including actuator-safe
        joint checks) before persisting; the response carries the
        normalized form so the caller can mirror server defaults.
        """

        return self._put(
            f"/api/v1/rl-tasks/{task_uuid}/actions",
            json={"action_spec": dict(action_spec)},
        )

    def get_observation_spec(self, task_uuid: str) -> dict[str, Any]:
        """Return the persisted ``OBSERVATION_SPEC`` payload."""

        return self._get(f"/api/v1/rl-tasks/{task_uuid}/observations")

    def set_observation_spec(
        self, task_uuid: str, observation_spec: dict[str, Any]
    ) -> dict[str, Any]:
        """Replace the persisted observation spec.

        ``observation_spec`` mirrors the editor payload:
        ``{"schema_version": 1, "groups": {<group>: {"terms": [...]}}}``.
        The backend re-validates joint pickers against the same
        actuator-safe set used by the Actions tab.
        """

        return self._put(
            f"/api/v1/rl-tasks/{task_uuid}/observations",
            json={"observation_spec": dict(observation_spec)},
        )

    def get_rl_config_spec(self, task_uuid: str) -> dict[str, Any]:
        """Return the persisted ``RL_CONFIG_SPEC`` payload (RL Config tab)."""

        return self._get(f"/api/v1/rl-tasks/{task_uuid}/rl-config")

    def set_rl_config_spec(
        self, task_uuid: str, rl_config_spec: dict[str, Any]
    ) -> dict[str, Any]:
        """Replace the persisted RL config / network builder spec."""

        return self._put(
            f"/api/v1/rl-tasks/{task_uuid}/rl-config",
            json={"rl_config_spec": dict(rl_config_spec)},
        )


# ---------------------------------------------------------------------------
# Convenience builders for ``sensors=[...]`` payloads
# ---------------------------------------------------------------------------


def make_camera_sensor(
    name: str,
    *,
    source_twin_uuid: str,
    schema_sensor_names: list[str],
    data_types: list[str],
    bind_mjcf_camera: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fovy: float | None = None,
    min_depth: float | None = None,
    max_depth: float | None = None,
) -> dict[str, Any]:
    """Build a Cyberwave RL camera sensor payload.

    The composer's "Cameras" subsection produces the same shape. One
    row per physical device; RGB and depth members from the same device
    share a row, ``data_types`` chooses which modes mjlab will render.

    ``bind_mjcf_camera`` is the schema sensor name whose MJCF
    ``<camera>`` element mjlab binds to. When omitted, prefer the
    depth member if ``data_types`` includes ``"depth"``; otherwise
    fall back to the first listed schema sensor.

    ``min_depth`` / ``max_depth`` (metres) define the depth-camera
    normalization range used by both the simulator renderer and the
    controller observation reconstruction. When omitted the simulator
    falls back to its own defaults.
    """

    if bind_mjcf_camera is None:
        if "depth" in data_types and "depth_camera" in schema_sensor_names:
            bind_mjcf_camera = "depth_camera"
        elif schema_sensor_names:
            bind_mjcf_camera = schema_sensor_names[0]
        else:
            raise ValueError("make_camera_sensor: schema_sensor_names cannot be empty")
    payload: dict[str, Any] = {
        "type": "camera",
        "name": name,
        "source_twin_uuid": source_twin_uuid,
        "schema_sensor_names": list(schema_sensor_names),
        "data_types": list(data_types),
        "bind_mjcf_camera": bind_mjcf_camera,
    }
    if width is not None:
        payload["width"] = int(width)
    if height is not None:
        payload["height"] = int(height)
    if fovy is not None:
        payload["fovy"] = float(fovy)
    if min_depth is not None:
        payload["min_depth"] = float(min_depth)
    if max_depth is not None:
        payload["max_depth"] = float(max_depth)
    return payload


def make_contact_sensor(
    name: str,
    *,
    primary_mode: str,
    primary_pattern: str,
    secondary_entity: str | None = None,
    secondary_mode: str | None = None,
    secondary_pattern: str | None = None,
    primary_exclude: list[str] | None = None,
    secondary_exclude: list[str] | None = None,
    fields: list[str] | None = None,
    reduce: str = "maxforce",
    num_slots: int = 1,
    history_length: int = 0,
    track_air_time: bool = False,
    global_frame: bool = False,
) -> dict[str, Any]:
    """Build a Cyberwave RL contact sensor payload.

    The codegen turns this into one ``ContactSensorCfg(...)`` call.
    Pass ``secondary_pattern=None`` for an "any contact" sensor that
    counts hits against arbitrary bodies; pass ``secondary_entity`` to
    constrain matches to another scene entity by its Cyberwave name.
    """

    primary: dict[str, Any] = {
        "mode": primary_mode,
        "pattern": primary_pattern,
    }
    if primary_exclude:
        primary["exclude"] = list(primary_exclude)
    payload: dict[str, Any] = {
        "type": "contact",
        "name": name,
        "primary": primary,
        "fields": list(fields or ["found"]),
        "reduce": reduce,
        "num_slots": int(num_slots),
        "history_length": int(history_length),
    }
    if secondary_pattern is not None:
        secondary: dict[str, Any] = {
            "mode": secondary_mode or primary_mode,
            "pattern": secondary_pattern,
        }
        if secondary_entity:
            secondary["entity"] = secondary_entity
        if secondary_exclude:
            secondary["exclude"] = list(secondary_exclude)
        payload["secondary"] = secondary
    if track_air_time:
        payload["track_air_time"] = True
    if global_frame:
        payload["global_frame"] = True
    return payload


# ---------------------------------------------------------------------------
# Convenience builders for the orchestration-tab specs
# ---------------------------------------------------------------------------
#
# These helpers exist so demos / setup scripts can produce the same
# payloads the editor saves on a "Save" click. They do **not** call
# the backend; they only assemble the dict shape. Hand them to
# :meth:`RLTaskClient.set_action_spec` / ``set_observation_spec`` /
# ``set_rl_config_spec`` to persist.


def _coerce_action_scalar_or_map(
    value: float | dict[str, float] | None,
    *,
    field_name: str,
) -> float | dict[str, float] | None:
    """Coerce a scalar-or-map action parameter into the persisted shape.

    Mirrors the backend validator: ``None`` passes through (so the
    caller can decide whether to omit the field), numeric values are
    cast to ``float``, and dict values are validated key-by-key.
    """

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number or {{joint: number}} mapping")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, dict):
        out: dict[str, float] = {}
        for key, sub in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    f"{field_name} mapping keys must be non-empty joint names"
                )
            if isinstance(sub, bool) or not isinstance(sub, int | float):
                raise ValueError(f"{field_name}['{key}'] must be a number")
            out[key.strip()] = float(sub)
        return out
    raise ValueError(f"{field_name} must be a number or {{joint: number}} mapping")


def make_action_term(
    name: str,
    *,
    action_type: str,
    entity: str,
    target_names_expr: list[str],
    scale: float | dict[str, float] = 1.0,
    offset: str | float | dict[str, float] | None = None,
    use_default_offset: bool | None = None,
    baseline_delta: float | dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build one entry of ``ACTION_SPEC["actions"]``.

    Mirrors the editor's Actions tab: ``action_type`` is one of
    ``position_delta`` / ``position`` / ``velocity`` / ``effort``.
    ``target_names_expr`` lists the concrete joint names the policy
    drives (regex patterns are still accepted for legacy import).

    Numeric parameters accept either a uniform scalar applied to
    every selected joint or a ``{joint_name: value}`` mapping for
    per-joint authoring:

    * ``scale``: maps raw policy outputs to commanded values. For
      ``position_delta`` it is the per-step joint delta at
      ``|action| = 1``.
    * ``offset``: absolute-target bias for the non-delta action
      types. Legacy symbolic strings (``"default"`` / ``"zero"`` /
      ``"none"``) are also accepted and round-trip unchanged.
      Ignored at runtime for ``position_delta`` — use
      ``baseline_delta`` instead.
    * ``baseline_delta``: constant per-step delta added to the
      current joint position before the policy contribution
      (``target = q_now + baseline_delta + action * scale``). Only
      meaningful for ``position_delta`` actions.

    ``use_default_offset=True`` makes ``position`` / ``velocity``
    actions reference the joint home (``default_joint_pos`` /
    ``default_joint_vel``) instead of zero; the numeric ``offset`` is
    ignored at runtime in that mode.

    Optional fields are emitted only when explicitly set so the
    persisted payload stays minimal.
    """

    term: dict[str, Any] = {
        "name": name,
        "type": action_type,
        "entity": entity,
        "target_names_expr": list(target_names_expr),
    }
    coerced_scale = _coerce_action_scalar_or_map(scale, field_name="scale")
    if coerced_scale is not None:
        term["scale"] = coerced_scale
    if offset is not None:
        if isinstance(offset, str):
            term["offset"] = offset
        else:
            coerced_offset = _coerce_action_scalar_or_map(offset, field_name="offset")
            if coerced_offset is not None:
                term["offset"] = coerced_offset
    if use_default_offset is not None:
        term["use_default_offset"] = bool(use_default_offset)
    if baseline_delta is not None:
        if action_type != "position_delta":
            raise ValueError(
                "baseline_delta is only supported on position_delta actions"
            )
        coerced_baseline = _coerce_action_scalar_or_map(
            baseline_delta, field_name="baseline_delta"
        )
        if coerced_baseline is not None:
            term["baseline_delta"] = coerced_baseline
    return term


def make_action_spec(actions: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap a list of action terms into the persisted ``ACTION_SPEC`` shape."""

    return {"schema_version": 1, "actions": list(actions)}


def make_observation_term(
    name: str,
    *,
    term_type: str,
    entity: str | None = None,
    target_names_expr: list[str] | None = None,
    links: list[str] | None = None,
    source: str | None = None,
    components: list[str] | None = None,
    frame: str | None = None,
    sensor: str | None = None,
    data_type: str | None = None,
    module: str | None = None,
    symbol: str | None = None,
    kind: str | None = None,
    kwargs: dict[str, Any] | None = None,
    scale: float | None = None,
    noise_std: float | None = None,
    clip: list[float] | None = None,
) -> dict[str, Any]:
    """Build one entry of ``OBSERVATION_SPEC["groups"][...]["terms"]``.

    ``term_type`` is the observation type the editor exposes:

    * joint terms (``joint_position`` / ``joint_velocity`` /
      ``joint_effort``) require ``entity`` + ``target_names_expr``.
    * ``link_observation`` is the collapsed link term: requires
      ``entity`` + ``links`` + ``source`` (``"pose"`` or ``"twist"``)
      + ``components``. ``components`` allows ``x|y|z|quat`` under
      ``source="pose"`` and ``x|y|z|rx|ry|rz`` under ``source="twist"``;
      ``quat`` stays as a single 4-channel token.
    * ``previous_action`` carries no extra fields.
    * ``camera`` requires ``sensor`` + ``data_type``.
    * ``custom`` requires ``module`` + ``symbol`` + ``kind``.
    """

    term: dict[str, Any] = {"name": name, "type": term_type}
    if scale is not None:
        term["scale"] = float(scale)
    if noise_std is not None:
        term["noise_std"] = float(noise_std)
    if clip is not None:
        term["clip"] = list(clip)
    if entity is not None:
        term["entity"] = entity
    if target_names_expr is not None:
        term["target_names_expr"] = list(target_names_expr)
    if links is not None:
        term["links"] = list(links)
    if source is not None:
        term["source"] = source
    if components is not None:
        term["components"] = list(components)
    if frame is not None:
        term["frame"] = frame
    if sensor is not None:
        term["sensor"] = sensor
    if data_type is not None:
        term["data_type"] = data_type
    if module is not None:
        term["module"] = module
    if symbol is not None:
        term["symbol"] = symbol
    if kind is not None:
        term["kind"] = kind
    if kwargs:
        term["kwargs"] = dict(kwargs)
    return term


def make_observation_spec(
    groups: dict[str, list[dict[str, Any]]],
    *,
    concatenate_terms: dict[str, bool] | None = None,
    enable_corruption: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Wrap observation groups + terms into the persisted ``OBSERVATION_SPEC`` shape.

    ``groups`` maps group name (``"policy"`` / ``"critic"`` / ...) to a
    list of terms. ``concatenate_terms`` / ``enable_corruption`` map
    group name → flag and default to the backend's defaults
    (``True`` / ``False`` respectively) when omitted.
    """

    out_groups: dict[str, Any] = {}
    cat = concatenate_terms or {}
    corruption = enable_corruption or {}
    for group_name, terms in groups.items():
        out_groups[group_name] = {
            "terms": list(terms),
            "concatenate_terms": bool(cat.get(group_name, True)),
            "enable_corruption": bool(corruption.get(group_name, False)),
        }
    return {"schema_version": 1, "groups": out_groups}


def make_rl_config_spec(
    *,
    algorithm: str = "ppo",
    backend: str = "skrl",
    network: dict[str, Any] | None = None,
    trainer: dict[str, Any] | None = None,
    algorithm_config: dict[str, Any] | None = None,
    experiment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the persisted ``RL_CONFIG_SPEC`` payload.

    Mirrors the editor's RL Config tab. ``algorithm`` is the lowercase
    string the backend persists (``"ppo"`` / ``"sac"`` / ...).
    ``network`` is the structured network builder spec exposed under
    ``RL_CONFIG_SPEC["network"]`` (consumed at runtime by
    :func:`rl.network_builder.build_skrl_model_cfg`).
    """

    spec: dict[str, Any] = {
        "schema_version": 1,
        "backend": backend,
        "algorithm": algorithm,
    }
    if network is not None:
        spec["network"] = dict(network)
    if trainer is not None:
        spec["trainer"] = dict(trainer)
    if algorithm_config is not None:
        spec["algorithm_config"] = dict(algorithm_config)
    if experiment is not None:
        spec["experiment"] = dict(experiment)
    return spec


# ---------------------------------------------------------------------------
# Spec preset helpers
# ---------------------------------------------------------------------------
#
# These collapse the most common Actions / Observations / RL Config
# editor presets into one-liner constructors so setup scripts can stay
# declarative. They build on the ``make_*_term`` / ``make_*_spec``
# helpers above and never call the backend; pass the result to
# :meth:`RLTaskClient.set_action_spec` / ``set_observation_spec`` /
# ``set_rl_config_spec`` to persist.


def make_position_delta_action(
    name: str,
    *,
    entity: str,
    target_names_expr: list[str],
    scale: float | dict[str, float] = 0.04,
    baseline_delta: float | dict[str, float] | None = None,
) -> dict[str, Any]:
    """Editor preset: a single ``position_delta`` action term.

    Mirrors the "Joint delta-position" preset on the Actions tab,
    which is the recommended action shape for SDC-style manipulation
    demos (the demo's :data:`DEMO_ACTION_SCALE` is 0.04).

    ``scale`` and ``baseline_delta`` accept either a uniform scalar
    or a ``{joint_name: value}`` mapping for per-joint authoring.
    ``baseline_delta`` defaults to ``None`` (omitted) so existing
    demos round-trip unchanged; setting it persists a non-zero
    per-step bias added to the current joint position before the
    policy contribution.
    """

    return make_action_term(
        name,
        action_type="position_delta",
        entity=entity,
        target_names_expr=list(target_names_expr),
        scale=scale,
        offset="none",
        use_default_offset=False,
        baseline_delta=baseline_delta,
    )


def make_joint_observation(
    name: str,
    *,
    kind: str,
    entity: str,
    target_names_expr: list[str],
) -> dict[str, Any]:
    """Editor preset: a joint observation term.

    ``kind`` is one of ``"position"`` / ``"velocity"`` / ``"effort"``
    (matches the editor's three joint observation presets).
    """

    type_map = {
        "position": "joint_position",
        "velocity": "joint_velocity",
        "effort": "joint_effort",
    }
    if kind not in type_map:
        raise ValueError(
            f"make_joint_observation: kind must be one of {sorted(type_map)}; "
            f"got {kind!r}"
        )
    return make_observation_term(
        name,
        term_type=type_map[kind],
        entity=entity,
        target_names_expr=list(target_names_expr),
    )


def make_previous_action_observation(
    name: str = "last_action",
    *,
    action_name: str | None = None,
) -> dict[str, Any]:
    """Editor preset: a ``previous_action`` observation term."""

    return make_observation_term(name, term_type="previous_action") | (
        {"action_name": action_name} if action_name is not None else {}
    )


def make_camera_observation(
    name: str,
    *,
    sensor: str,
    data_type: str = "depth",
) -> dict[str, Any]:
    """Editor preset: a camera observation term (defaults to depth)."""

    return make_observation_term(
        name,
        term_type="camera",
        sensor=sensor,
        data_type=data_type,
    )


def make_custom_observation(
    name: str,
    *,
    module: str,
    symbol: str | None = None,
    kind: str = "function",
    kwargs: dict[str, Any] | None = None,
    shape: list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Editor preset: a ``custom`` observation term pointing at user code.

    ``symbol`` defaults to ``name`` because the demo convention names
    the local Python function after the term itself
    (``ee_to_cube_offset`` / ``cube_to_goal`` / ...).
    """

    term = make_observation_term(
        name,
        term_type="custom",
        module=module,
        symbol=symbol if symbol is not None else name,
        kind=kind,
        kwargs=kwargs,
    )
    if shape is not None:
        term["shape"] = list(shape)
    return term


def make_default_ppo_rlmodule_config(
    *,
    hidden_layers: list[int] | None = None,
    activation: str = "ELU",
    layer_norm: bool = False,
    initial_log_std: float = -0.5,
    min_log_std: float = -20.0,
    max_log_std: float = 2.0,
    learning_rate: float = 3e-4,
    rollouts: int = 16,
    learning_epochs: int = 4,
    mini_batches: int = 2,
    discount_factor: float = 0.99,
    lambda_: float = 0.95,
    entropy_loss_scale: float = 0.01,
    value_loss_scale: float = 0.5,
    ratio_clip: float = 0.2,
    value_clip: float = 0.2,
    grad_norm_clip: float = 0.5,
    clip_predicted_values: bool = True,
    kl_threshold: float = 0.0,
    timesteps: int = 100_000,
    write_interval: int | str = "auto",
    checkpoint_interval: int | str = "auto",
    experiment_directory: str | None = None,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """Editor preset: PPO + rlmodule shared-MLP network.

    Mirrors the editor's "PPO defaults" with the same hidden geometry
    the demo had baked into ``DEFAULT_NETWORK_SPEC`` before runtime
    spec consumption. Useful as the seed for any setup script that
    wants the same defaults the editor would.
    """

    layers = list(hidden_layers) if hidden_layers is not None else [256, 256, 128]
    network = {
        "library": "rlmodule",
        "topology": "shared",
        "architecture": "mlp",
        "blocks": {
            "shared": {
                "type": "mlp",
                "layers": [
                    {
                        "units": int(u),
                        "activation": activation,
                        "layer_norm": layer_norm,
                    }
                    for u in layers
                ],
                "layer_norm_position": "pre",
            }
        },
        "output": {
            "initial_log_std": initial_log_std,
            "min_log_std": min_log_std,
            "max_log_std": max_log_std,
        },
    }
    trainer: dict[str, Any] = {"timesteps": timesteps}
    if write_interval != "auto":
        trainer["write_interval"] = write_interval
    if checkpoint_interval != "auto":
        trainer["checkpoint_interval"] = checkpoint_interval

    algorithm_config: dict[str, Any] = {
        "learning_rate": learning_rate,
        "rollouts": rollouts,
        "learning_epochs": learning_epochs,
        "mini_batches": mini_batches,
        "discount_factor": discount_factor,
        "lambda_": lambda_,
        "entropy_loss_scale": entropy_loss_scale,
        "value_loss_scale": value_loss_scale,
        "ratio_clip": ratio_clip,
        "value_clip": value_clip,
        "grad_norm_clip": grad_norm_clip,
        "clip_predicted_values": clip_predicted_values,
        "kl_threshold": kl_threshold,
    }
    experiment: dict[str, Any] = {}
    if experiment_directory is not None:
        experiment["directory"] = experiment_directory
    if experiment_name is not None:
        experiment["name"] = experiment_name

    return make_rl_config_spec(
        algorithm="ppo",
        backend="skrl",
        network=network,
        trainer=trainer,
        algorithm_config=algorithm_config,
        experiment=experiment or None,
    )


__all__ = [
    "RLTaskClient",
    "TaskSpecExport",
    "make_action_spec",
    "make_action_term",
    "make_camera_observation",
    "make_camera_sensor",
    "make_contact_sensor",
    "make_custom_observation",
    "make_default_ppo_rlmodule_config",
    "make_joint_observation",
    "make_observation_spec",
    "make_observation_term",
    "make_position_delta_action",
    "make_previous_action_observation",
    "make_rl_config_spec",
]
