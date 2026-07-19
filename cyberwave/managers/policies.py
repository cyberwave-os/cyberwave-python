"""Controller policy resource manager (``cw.policies``)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

from ..exceptions import CyberwaveError
from ..twin._helpers import (
    _SDK_JOINT_INPUT_DEVICES,
    _pick_default_sdk_joint_policy_uuid,
    _policy_is_sdk_joint_teleop_candidate,
    _sdk_auto_attach_controller_enabled,
)

if TYPE_CHECKING:
    from ..client import Cyberwave
    from ..twin.base import Twin

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttachedPolicyInfo:
    """Summary of the controller policy attached to a twin."""

    policy_uuid: Optional[str]
    policy_name: Optional[str]
    controller_type: Optional[str]
    input_device: Optional[str]


class PolicyManager:
    """List and assign controller policies for twins."""

    def __init__(self, client: Cyberwave) -> None:
        self._client = client

    def list(self, *, twin: Optional[Twin] = None) -> List[Any]:
        """List controller policies, optionally scoped to a twin's asset/workspace."""
        api = getattr(getattr(self._client, "twins", None), "api", None)
        if api is None:
            raise CyberwaveError("Client does not expose a controller-policies API")
        asset_uuid = twin.asset_id if twin is not None else None
        workspace_uuid = twin._get_workspace_uuid() if twin is not None else None
        try:
            return api.src_app_api_controller_policies_list_controller_policies(
                asset_uuid=asset_uuid or None,
                workspace_uuid=workspace_uuid,
            )
        except Exception as e:
            raise CyberwaveError(f"Failed to list controller policies: {e}") from e

    def assign(self, twin: Twin, policy: Any) -> None:
        """Attach a controller policy to a twin (REST)."""
        twin.policy._apply_policy(policy)

    def unassign(self, twin: Twin) -> None:
        """Clear the twin's controller policy assignment."""
        twin._unassign_controller_policy()


class TwinPolicyHandle:
    """Controller policy attach/read and keyboard teleop entry point."""

    def __init__(self, twin: Twin) -> None:
        self._twin = twin

    def list(self) -> List[Any]:
        return self._twin.client.policies.list(twin=self._twin)

    def get(self) -> Any | None:
        """Fetch the attached policy object, or None."""
        attached = self.attached
        if not attached.policy_uuid:
            return None
        api = getattr(getattr(self._twin.client, "twins", None), "api", None)
        if api is None:
            return None
        try:
            return api.src_app_api_controller_policies_get_controller_policy(
                attached.policy_uuid
            )
        except Exception:
            return None

    def assign(self, policy: Any) -> None:
        self._apply_policy(policy)

    def unassign(self) -> None:
        """Clear this twin's controller policy assignment."""
        self._twin._unassign_controller_policy()

    @property
    def attached(self) -> AttachedPolicyInfo:
        policy_uuid: Optional[str] = None
        if hasattr(self._twin._data, "controller_policy_uuid"):
            raw = self._twin._data.controller_policy_uuid
            policy_uuid = str(raw) if raw else None
        elif isinstance(self._twin._data, dict):
            raw = self._twin._data.get("controller_policy_uuid")
            policy_uuid = str(raw) if raw else None

        from ..twin._helpers import _get_twin_metadata

        meta = _get_twin_metadata(self._twin._data)
        bindings = meta.get("controller_policy_bindings") or {}
        slice_meta = bindings.get(policy_uuid or "", {}) if policy_uuid else {}

        return AttachedPolicyInfo(
            policy_uuid=policy_uuid,
            policy_name=slice_meta.get("name") or meta.get("controller_policy_name"),
            controller_type=slice_meta.get("controller_type")
            or meta.get("controller_type"),
            input_device=slice_meta.get("input_device") or meta.get("input_device"),
        )

    def playground_actuations(self) -> frozenset[str]:
        """Actuation names the attached controller policy renders in the playground.

        Mirrors the frontend's ``PlaygroundLocomotionCommandDrivers``, which reads
        the same policy's ``metadata.keyboard_bindings`` to resolve a catalog
        command's velocity: an actuation counts here only if its binding carries a
        ``playground`` extension. Used by ``twin.commands.<name>()`` to allow
        commands that a specific asset's controller wires up for playground preview,
        without hardcoding the command list into the SDK. Cached per policy UUID on
        the twin so repeated preflight checks (e.g. before each ``commands.<name>()``
        call) don't refetch the policy over the network.
        """
        policy_uuid = self.attached.policy_uuid
        if not policy_uuid:
            return frozenset()

        cached = getattr(self._twin, "_playground_actuations_cache", None)
        if cached is not None and cached[0] == policy_uuid:
            return cached[1]

        policy = self.get()
        metadata = getattr(policy, "metadata", None) if policy is not None else None
        keyboard_bindings = metadata.get("keyboard_bindings") if isinstance(metadata, dict) else None
        actuations = frozenset(
            str(binding["actuation"])
            for binding in (keyboard_bindings or [])
            if isinstance(binding, dict) and binding.get("actuation") and binding.get("playground")
        )
        self._twin._playground_actuations_cache = (policy_uuid, actuations)
        return actuations

    def _has_attached_teleop_policy(self) -> bool:
        attached = self.attached
        return bool(
            attached.policy_uuid
            and str(attached.controller_type or "").lower() == "teleop"
        )

    def ensure_attached(self) -> None:
        """Require a teleop controller policy before motion MQTT commands."""
        if self._twin._controller_ensured:
            if not self._has_attached_teleop_policy():
                raise CyberwaveError(
                    "Cannot send motion commands without an attached teleop controller policy."
                )
            return

        had_teleop = self._has_attached_teleop_policy()
        newly_attached = False

        if not _sdk_auto_attach_controller_enabled():
            logger.debug(
                "Twin %s: auto-attach disabled; skipping controller assignment",
                self._twin.uuid,
            )
        elif getattr(getattr(self._twin.client, "twins", None), "api", None) is None:
            logger.debug(
                "Twin %s: client has no controller-policies API; skipping auto-attach",
                self._twin.uuid,
            )
        elif had_teleop:
            pass
        else:
            policies = self.list()
            policy = self._pick_controller_policy(policies)
            self._apply_policy(policy)
            newly_attached = True

        if not self._has_attached_teleop_policy():
            raise CyberwaveError(
                "Cannot send motion commands without an attached teleop controller policy."
            )

        if newly_attached:
            logger.warning(
                "Twin %s: controller policy was just attached. This command may have been "
                "lost while the robot was still setting up; retry in a few seconds.",
                self._twin.uuid,
            )

        from ..twin._helpers import _check_controller_ready_live

        runtime_mode = getattr(
            getattr(self._twin.client, "config", None), "runtime_mode", "live"
        )
        if runtime_mode == "live" and not _check_controller_ready_live():
            raise CyberwaveError("Robot controller is not ready for live joint commands.")

        self._twin._controller_ensured = True

    def keyboard(
        self,
        bindings: Any,
        *,
        step: float = 0.05,
        rate_hz: int = 20,
        fetch_initial: bool = True,
        verbose: bool = True,
    ) -> Any:
        from ..keyboard import KeyboardBindings, KeyboardTeleop

        payload = (
            bindings.build() if isinstance(bindings, KeyboardBindings) else bindings
        )
        return KeyboardTeleop(
            self._twin,
            payload,
            step=step,
            rate_hz=rate_hz,
            fetch_initial=fetch_initial,
            verbose=verbose,
        )

    def _pick_controller_policy(self, policies: List[Any]) -> Any:
        current = self.attached.policy_uuid
        if current:
            cur_policy = next((p for p in policies if str(p.uuid) == current), None)
            if cur_policy is None:
                logger.warning(
                    "Twin %s: assigned controller %r not visible in workspace policy list",
                    self._twin.uuid,
                    current,
                )
            elif str(getattr(cur_policy, "controller_type", "") or "").lower() == "teleop":
                return cur_policy

        candidates = [p for p in policies if _policy_is_sdk_joint_teleop_candidate(p)]
        if not candidates:
            raise CyberwaveError(
                "No controller policy suitable for SDK joint commands was found "
                f"(need a teleop policy with input_device in "
                f"{sorted(_SDK_JOINT_INPUT_DEVICES)!r})."
            )
        chosen_uuid = _pick_default_sdk_joint_policy_uuid(candidates)
        return next(p for p in candidates if str(p.uuid) == chosen_uuid)

    def _apply_policy(self, policy: Any) -> None:
        self._twin._apply_controller_policy(policy)
