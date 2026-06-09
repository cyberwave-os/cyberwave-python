"""Shared helpers for the ``cyberwave.twin`` package (PR0 split)."""

import asyncio
from copy import deepcopy
import json
import math
import os
import threading
import time
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Dict, Any, List, Callable, Type

from ..exceptions import CyberwaveError
from ..constants import SOURCE_TYPE_SIM, SOURCE_TYPE_SIM_TELE, SOURCE_TYPE_TELE


# Load capabilities cache for runtime class selection
_CAPABILITIES_CACHE: Optional[Dict[str, Any]] = None

logger = logging.getLogger(__name__)


def _run_coroutine_blocking(coro) -> None:  # type: ignore[no-untyped-def]
    """Run an async coroutine in a blocking fashion, compatible with running event loops.

    When called from an environment that already has a running event loop (e.g. Jupyter
    notebooks, Google Colab, IPython), ``asyncio.run()`` raises a ``RuntimeError``.
    In those cases we spin up a dedicated background thread with its own event loop so
    the coroutine can run to completion while the caller's thread blocks normally.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We're inside a running event loop (e.g. Jupyter/Colab).  Run the coroutine
        # in a separate OS thread that owns a fresh event loop.
        exc_holder: list = []

        def _run_in_thread() -> None:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                new_loop.run_until_complete(coro)
            except Exception as exc:
                exc_holder.append(exc)
            finally:
                new_loop.close()

        t = threading.Thread(target=_run_in_thread, daemon=True)
        t.start()
        t.join()
        if exc_holder:
            raise exc_holder[0]
    else:
        asyncio.run(coro)


def _normalize_locomotion_source_type(source_type: Optional[str]) -> Optional[str]:
    """Preserve legacy ``sim`` callers while publishing ``sim_tele`` commands."""
    if source_type == SOURCE_TYPE_SIM:
        return SOURCE_TYPE_SIM_TELE
    return source_type


# Twin-command MQTT messages that do not require an attached teleop policy.
_NON_MOTION_TWIN_COMMANDS = frozenset(
    {
        "battery_check",
        "lights_on",
        "lights_off",
        "get_status",
    }
)


def motion_outbound_requires_policy(command: str) -> bool:
    """True for joint updates and locomotion/flight (and similar) twin commands."""
    if command == "joint_update":
        return True
    return command not in _NON_MOTION_TWIN_COMMANDS


def _default_control_source_type(client: Any) -> str:
    """Default MQTT control source for outbound twin/joint commands."""
    config = getattr(client, "config", None)
    explicit = getattr(config, "source_type", None) if config else None
    if explicit == SOURCE_TYPE_SIM:
        return SOURCE_TYPE_SIM_TELE
    if explicit in {SOURCE_TYPE_SIM_TELE, SOURCE_TYPE_TELE}:
        return str(explicit)
    runtime_mode = getattr(config, "runtime_mode", "live") if config else "live"
    return SOURCE_TYPE_SIM_TELE if runtime_mode == "simulation" else SOURCE_TYPE_TELE


# Teleop policies with these ``metadata["input_device"]`` values are used when
# auto-attaching a controller for ``joints.set`` (see ``_ensure_controller_ready``).
_SDK_JOINT_INPUT_DEVICES: frozenset[str] = frozenset({"sdk", "keyboard"})


def _sdk_auto_attach_controller_enabled() -> bool:
    """When false (``CYBERWAVE_SDK_AUTO_ATTACH_CONTROLLER=0``), skip REST controller assignment."""
    raw = os.environ.get("CYBERWAVE_SDK_AUTO_ATTACH_CONTROLLER", "1")
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _policy_is_sdk_joint_teleop_candidate(policy: Any) -> bool:
    """Teleop policies suitable for SDK joint MQTT (sdk or keyboard input_device)."""
    ctype = getattr(policy, "controller_type", None) or ""
    if str(ctype).lower() != "teleop":
        return False
    meta = policy.metadata if isinstance(policy.metadata, dict) else {}
    device = meta.get("input_device")
    return isinstance(device, str) and device in _SDK_JOINT_INPUT_DEVICES


def _pick_default_sdk_joint_policy_uuid(candidates: List[Any]) -> str:
    """Prefer ``input_device`` sdk, then keyboard; else stable tie-break by policy uuid."""
    if not candidates:
        raise CyberwaveError("Internal error: empty SDK joint controller candidates")

    def _rank(pol: Any) -> tuple[int, str]:
        meta = pol.metadata if isinstance(pol.metadata, dict) else {}
        dev = meta.get("input_device")
        if dev == "sdk":
            return (0, str(pol.uuid))
        if dev == "keyboard":
            return (1, str(pol.uuid))
        return (2, str(pol.uuid))

    return str(sorted(candidates, key=_rank)[0].uuid)


def _check_controller_ready_live() -> bool:
    """Whether the robot edge path is ready to accept live teleop joint commands.

    TODO: Infer readiness from edge telemetry (e.g. joint_states / heartbeat flowing).
    """
    logger.debug(
        "Live controller readiness check is stubbed True; "
        "TODO: verify robot/edge connectivity before joint commands."
    )
    return True


def _get_twin_metadata(data: Any) -> dict:
    """Extract the twin's current metadata dict (returns a shallow copy)."""
    if hasattr(data, "metadata"):
        meta = data.metadata
    elif isinstance(data, dict):
        meta = data.get("metadata")
    else:
        meta = None
    return dict(meta) if isinstance(meta, dict) else {}


def _twin_display_name(twin: Any) -> str:
    """Human-readable twin label for operator alerts."""
    name = getattr(twin, "name", None)
    if name:
        return str(name)
    data = getattr(twin, "_data", None)
    if data is not None:
        if hasattr(data, "name") and data.name:
            return str(data.name)
        if isinstance(data, dict) and data.get("name"):
            return str(data["name"])
    return str(getattr(twin, "uuid", "twin"))


def _attached_controller_display_name(twin_data: Any) -> str:
    """Policy label from twin metadata before unassign clears it."""
    meta = _get_twin_metadata(twin_data)
    return str(
        meta.get("controller_policy_name")
        or meta.get("controller_policy_uuid")
        or "controller"
    )


def _alert_source_type_for_client(client: Any) -> str:
    """REST alert ``source_type`` for SDK-initiated controller lifecycle events."""
    from ..constants import SOURCE_TYPE_SIM, SOURCE_TYPE_SIM_TELE

    config = getattr(client, "config", None)
    explicit = getattr(config, "source_type", None) if config else None
    if explicit in {SOURCE_TYPE_SIM, SOURCE_TYPE_SIM_TELE}:
        return "simulation"
    runtime_mode = getattr(config, "runtime_mode", "live") if config else "live"
    return "simulation" if runtime_mode == "simulation" else "cloud"


def _emit_controller_policy_alert(
    twin: Any,
    *,
    action: Literal["assign", "unassign", "disconnect"],
    policy: Any | None = None,
) -> None:
    """Best-effort operator alert for controller assign/unassign/disconnect."""
    client = getattr(twin, "client", None)
    publish = getattr(client, "publish_alert", None)
    if not callable(publish):
        return

    twin_name = _twin_display_name(twin)
    twin_uuid = str(getattr(twin, "uuid", ""))
    if not twin_uuid:
        return

    policy_uuid: Optional[str] = None
    if action == "assign" and policy is not None:
        policy_name = str(getattr(policy, "name", "") or getattr(policy, "uuid", "controller"))
        policy_uuid = str(getattr(policy, "uuid", "") or "") or None
        title = "Assigning controller"
        description = f"Assigning {policy_name} to {twin_name}."
    elif action == "disconnect":
        policy_name = _attached_controller_display_name(getattr(twin, "_data", None))
        meta = _get_twin_metadata(getattr(twin, "_data", None))
        policy_uuid = meta.get("controller_policy_uuid")
        if policy_uuid is not None:
            policy_uuid = str(policy_uuid)
        title = "Disconnecting"
        description = (
            f"Disconnecting {twin_name} and removing controller {policy_name}."
        )
    else:
        policy_name = _attached_controller_display_name(getattr(twin, "_data", None))
        meta = _get_twin_metadata(getattr(twin, "_data", None))
        policy_uuid = meta.get("controller_policy_uuid")
        if policy_uuid is not None:
            policy_uuid = str(policy_uuid)
        title = "Removing controller"
        description = f"Removing controller {policy_name} from {twin_name}."

    metadata: Dict[str, Any] = {"action": action}
    if policy_uuid:
        metadata["controller_policy_uuid"] = policy_uuid
    if policy_name:
        metadata["controller_policy_name"] = policy_name

    try:
        publish(
            twin_uuid,
            title,
            description=description,
            alert_type="controller_state",
            severity="info",
            category="technical",
            source_type=_alert_source_type_for_client(client),
            force=True,
            metadata=metadata,
        )
    except Exception as exc:
        logger.debug(
            "Twin %s: controller %s alert failed (non-fatal): %s",
            twin_uuid[:8],
            action,
            exc,
        )


def _build_controller_assignment_metadata(twin_data: Any, policy: Any | None) -> dict:
    """Build metadata for assign (``policy`` set) or unassign (``policy`` is None).

    Mirrors the frontend's ``buildTwinControllerUpdatePayload`` logic so the UI
    can immediately reflect the assignment without a full page refresh.
    """
    base = _get_twin_metadata(twin_data)
    if policy is None:
        base["controller_policy_uuid"] = None
        base["controller_policy_name"] = None
        base["controller_type"] = None
        base["control_mode"] = None
        return base
    base["controller_policy_uuid"] = str(policy.uuid)
    base["controller_policy_name"] = str(getattr(policy, "name", "") or "")
    base["controller_type"] = str(getattr(policy, "controller_type", "") or "")
    base["control_mode"] = "joint_control"
    return base


def _load_capabilities_cache() -> Dict[str, Any]:
    """Load the capabilities cache from JSON file."""
    global _CAPABILITIES_CACHE
    if _CAPABILITIES_CACHE is None:
        cache_path = Path(__file__).resolve().parent.parent / "assets_capabilities.json"
        if cache_path.exists():
            with open(cache_path, "r") as f:
                _CAPABILITIES_CACHE = json.load(f)
        else:
            _CAPABILITIES_CACHE = {}
    return _CAPABILITIES_CACHE


def _get_asset_capabilities(registry_id: str) -> Dict[str, Any]:
    """Get capabilities for an asset by registry_id."""
    cache = _load_capabilities_cache()
    asset_data = cache.get(registry_id, {})
    return asset_data.get("capabilities", {})


def _decode_frame(jpeg_bytes: bytes, format: str) -> Any:
    """Decode JPEG bytes into the requested output format.

    Args:
        jpeg_bytes: Raw JPEG image bytes.
        format: ``"path"`` | ``"bytes"`` | ``"numpy"`` | ``"pil"``.

    Returns:
        Decoded frame in the chosen format.
    """
    if format == "bytes":
        return jpeg_bytes

    if format == "path":
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".jpg", prefix="cyberwave_")
        with os.fdopen(fd, "wb") as f:
            f.write(jpeg_bytes)
        return path

    if format == "numpy":
        try:
            import numpy as np
            import cv2  # type: ignore[import-untyped]
        except ImportError:
            raise CyberwaveError(
                "numpy and opencv-python are required for format='numpy'. "
                "Install with: pip install numpy opencv-python"
            )
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise CyberwaveError("Failed to decode JPEG frame")
        return frame

    if format == "pil":
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError:
            raise CyberwaveError(
                "Pillow is required for format='pil'. Install with: pip install Pillow"
            )
        import io

        return Image.open(io.BytesIO(jpeg_bytes))

    raise CyberwaveError(
        f"Unknown format '{format}'. Supported: 'path', 'bytes', 'numpy', 'pil'"
    )
