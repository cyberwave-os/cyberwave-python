"""manifest.py — YAML manifest loader for Cyberwave ROS 2 Python drivers.

The manifest schema is shared across Cyberwave ROS 2 driver implementations,
so the same manifest.yaml file works regardless of driver language.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

DRIVER_MANIFEST_FILE_NAME = "manifest.yaml"
_CW_DRIVER_ROOT_KEYS = frozenset(
    {"registry_id", "driver_family", "mqtt", "zenoh", "commands"}
)


class TopicRole(Enum):
    Publisher = "publisher"
    Subscription = "subscription"


class ServiceRole(Enum):
    Server = "server"
    Client = "client"


@dataclass
class ManifestParam:
    name: str
    type: str
    default_value: str = ""
    description: str = ""
    read_only: bool = False


@dataclass
class ManifestTopic:
    name: str
    role: TopicRole = TopicRole.Publisher
    msg_type: str = ""
    default_value: str = ""
    description: str = ""


@dataclass
class ManifestService:
    name: str
    role: ServiceRole = ServiceRole.Server
    srv_type: str = ""
    default_value: str = ""
    description: str = ""


@dataclass
class ManifestReadiness:
    kind: str = "service"  # v1: service only
    name: str = ""
    timeout_s: float = 60.0


@dataclass
class ManifestManagedLaunch:
    package: str
    launch_file: str
    launch_args: dict[str, str | int | float | bool] = field(default_factory=dict)
    readiness: ManifestReadiness = field(default_factory=ManifestReadiness)
    log_file: str = ""
    ros_setup: str = ""  # ROS underlay setup.bash (else env ROS_SETUP)
    ros_overlay: str = ""  # vendor workspace setup.bash (else env ROS_SETUP_OVERLAY)


@dataclass
class NodeManifest:
    node_name: str = ""
    description: str = ""
    params: list[ManifestParam] = field(default_factory=list)
    topics: list[ManifestTopic] = field(default_factory=list)
    services: list[ManifestService] = field(default_factory=list)
    managed_launch: ManifestManagedLaunch | None = None


def default_node_manifest(node_name: str = "cyberwave_ros2_driver") -> NodeManifest:
    """Built-in ROS node manifest when no ``manifest.yaml`` is provided."""
    return NodeManifest(
        node_name=node_name,
        description="Cyberwave ROS 2 driver (built-in default manifest)",
        params=[
            ManifestParam(
                name="asset_key",
                type="string",
                default_value="",
                description="Cyberwave asset key (CYBERWAVE_ASSET_KEY / CW_ROS2_ASSET_KEY)",
            ),
            ManifestParam(
                name="twin_uuid",
                type="string",
                default_value="",
                description="Cyberwave twin UUID (CYBERWAVE_TWIN_UUID / CW_ROS2_TWIN_UUID)",
            ),
            ManifestParam(
                name="tick_rate_hz",
                type="int",
                default_value="10",
                description="Lifecycle tick rate in Hz",
                read_only=True,
            ),
        ],
    )


def _parse_manifest_root(root: dict[str, Any]) -> NodeManifest:
    m = NodeManifest(
        node_name=root.get("node_name", ""),
        description=root.get("description", ""),
    )

    for item in root.get("parameters", []):
        m.params.append(
            ManifestParam(
                name=item["name"],
                type=item["type"],
                default_value=str(item.get("default", "")),
                description=item.get("description", ""),
                read_only=bool(item.get("read_only", False)),
            )
        )

    for item in root.get("topics", []):
        role_str = item.get("role", "publisher")
        if role_str == "publisher":
            role = TopicRole.Publisher
        elif role_str == "subscription":
            role = TopicRole.Subscription
        else:
            raise RuntimeError(f"Unknown topic role '{role_str}'")
        m.topics.append(
            ManifestTopic(
                name=item["name"],
                role=role,
                msg_type=item.get("msg_type", ""),
                default_value=str(item.get("default", "")),
                description=item.get("description", ""),
            )
        )

    for item in root.get("services", []):
        role_str = item.get("role", "server")
        if role_str == "server":
            role = ServiceRole.Server
        elif role_str == "client":
            role = ServiceRole.Client
        else:
            raise RuntimeError(f"Unknown service role '{role_str}'")
        m.services.append(
            ManifestService(
                name=item["name"],
                role=role,
                srv_type=item.get("srv_type", ""),
                default_value=str(item.get("default", "")),
                description=item.get("description", ""),
            )
        )

    ml = root.get("managed_launch")
    if isinstance(ml, dict):
        readiness_raw = ml.get("readiness") or {}
        m.managed_launch = ManifestManagedLaunch(
            package=str(ml["package"]),
            launch_file=str(ml["launch_file"]),
            launch_args={str(k): v for k, v in (ml.get("launch_args") or {}).items()},
            readiness=ManifestReadiness(
                kind=str(readiness_raw.get("kind", "service")),
                name=str(readiness_raw.get("name", "")),
                timeout_s=float(readiness_raw.get("timeout_s", 60)),
            ),
            log_file=str(ml.get("log_file", "")),
            ros_setup=str(ml.get("ros_setup", "")),
            ros_overlay=str(ml.get("ros_overlay", "")),
        )

    return m


def node_manifest_to_dict(manifest: NodeManifest) -> dict[str, Any]:
    """Serialize :class:`NodeManifest` to a YAML-compatible mapping."""
    root: dict[str, Any] = {
        "node_name": manifest.node_name,
        "description": manifest.description,
        "parameters": [
            {
                "name": p.name,
                "type": p.type,
                "default": p.default_value,
                "description": p.description,
                **({"read_only": True} if p.read_only else {}),
            }
            for p in manifest.params
        ],
        "topics": [
            {
                "name": t.name,
                "role": t.role.value,
                "msg_type": t.msg_type,
                "default": t.default_value,
                "description": t.description,
            }
            for t in manifest.topics
        ],
        "services": [
            {
                "name": s.name,
                "role": s.role.value,
                "srv_type": s.srv_type,
                "default": s.default_value,
                "description": s.description,
            }
            for s in manifest.services
        ],
    }
    if manifest.managed_launch is not None:
        ml = manifest.managed_launch
        launch_block: dict[str, Any] = {
            "package": ml.package,
            "launch_file": ml.launch_file,
            "launch_args": dict(ml.launch_args),
            "readiness": {
                "kind": ml.readiness.kind,
                "name": ml.readiness.name,
                "timeout_s": ml.readiness.timeout_s,
            },
        }
        if ml.log_file:
            launch_block["log_file"] = ml.log_file
        if ml.ros_setup:
            launch_block["ros_setup"] = ml.ros_setup
        if ml.ros_overlay:
            launch_block["ros_overlay"] = ml.ros_overlay
        root["managed_launch"] = launch_block
    return root


def merge_combined_driver_manifest(
    node_manifest: NodeManifest | dict[str, Any],
    cw_driver: dict[str, Any],
) -> dict[str, Any]:
    """Merge ROS node manifest fields with uncompiled ``cw-driver`` root keys."""
    node_dict = (
        node_manifest
        if isinstance(node_manifest, dict)
        else node_manifest_to_dict(node_manifest)
    )
    combined = dict(node_dict)
    for key in _CW_DRIVER_ROOT_KEYS:
        if key in cw_driver:
            combined[key] = cw_driver[key]
    return combined


def dump_combined_driver_manifest(
    manifest: dict[str, Any],
    path: Path | str,
    *,
    header_comment: str | None = None,
) -> Path:
    """Write a combined driver manifest (ROS + cw-driver) to disk."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(
        manifest,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    lines: list[str] = []
    if header_comment:
        for line in header_comment.strip().splitlines():
            lines.append(f"# {line}")
        lines.append("")
    lines.append(body.rstrip())
    lines.append("")
    file_path.write_text("\n".join(lines), encoding="utf-8")
    return file_path.resolve()


def resolve_node_manifest(
    driver_cls: type[Any],
    manifest_path: str | None,
    *,
    node_name: str,
) -> NodeManifest:
    """Load YAML manifest or fall back to ``define_node_manifest`` on *driver_cls*."""
    if manifest_path and os.path.isfile(manifest_path):
        return load_manifest(manifest_path, node_name=node_name)
    if manifest_path:
        logger.info(
            "ROS manifest %r not found; using %s.define_node_manifest()",
            manifest_path,
            driver_cls.__name__,
        )
    factory = getattr(driver_cls, "define_node_manifest", None)
    if callable(factory):
        return factory(node_name)
    return default_node_manifest(node_name)


def load_manifest(
    path: str | None,
    *,
    node_name: str = "cyberwave_ros2_driver",
) -> NodeManifest:
    """Load a NodeManifest from YAML, or built-in defaults if *path* is missing.

    When *path* is empty or the file does not exist, returns
    :func:`default_node_manifest` (logs at INFO). Parse errors on an
    existing file still raise :exc:`RuntimeError`.
    """
    if not path or not os.path.isfile(path):
        if path:
            logger.info(
                "ROS manifest %r not found; using built-in defaults for node %r",
                path,
                node_name,
            )
        return default_node_manifest(node_name)

    try:
        with open(path, encoding="utf-8") as f:
            root = yaml.safe_load(f)
    except OSError as e:
        if e.errno == 2:  # ENOENT
            logger.info(
                "ROS manifest %r not found; using built-in defaults for node %r",
                path,
                node_name,
            )
            return default_node_manifest(node_name)
        raise RuntimeError(f"Failed to load manifest from '{path}': {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load manifest from '{path}': {e}") from e

    if not isinstance(root, dict):
        raise RuntimeError(f"Manifest root must be a mapping, got {type(root).__name__}")

    # Support combined cw-driver.yml format: descend into ros2: section if present.
    if isinstance(root.get("ros2"), dict):
        root = root["ros2"]
    return _parse_manifest_root(root)
