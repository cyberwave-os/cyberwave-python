"""cw-driver.yml I/O helpers for the Python SDK (compile on backend only).

Drivers export uncompiled cw-driver root dicts via
:meth:`~cyberwave.driver.interface.registry_mixin.InterfaceRegistryMixin.cw_driver`.
:meth:`~cyberwave.twin.driver.TwinDriverHandle.set_schema` posts them to
``POST /api/v1/twins/{uuid}/driver-schema`` for server-side compilation.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Union

import yaml

CW_DRIVER_FILE_NAME = "cw-driver.yml"


_CW_DRIVER_ROOT_KEYS = (
    "registry_id",
    "registry_ids",
    "driver_family",
    "mqtt",
    "zenoh",
    "commands",
)


def extract_cw_driver_root(raw: dict[str, Any]) -> dict[str, Any]:
    """Return cw-driver catalog keys from a combined ROS+MQTT manifest or pure cw-driver."""
    if not isinstance(raw, dict):
        raise ValueError("driver manifest root must be a mapping")
    if "node_name" not in raw and "parameters" not in raw:
        return raw
    extracted = {key: raw[key] for key in _CW_DRIVER_ROOT_KEYS if key in raw}
    if "mqtt" not in extracted:
        raise ValueError("combined manifest must include top-level 'mqtt' mapping")
    return extracted


def load_cw_driver_yml(path: Path | str) -> dict[str, Any]:
    """Load a ``cw-driver.yml`` or combined ``manifest.yaml`` MQTT catalog."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"driver catalog file not found: {file_path}")
    with file_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"driver catalog root must be a mapping: {file_path}")
    return extract_cw_driver_root(raw)


def dump_cw_driver_yml(
    driver_config: dict[str, Any],
    path: Path | str,
    *,
    header_comment: str | None = None,
) -> Path:
    """Write a cw-driver.yml root dict to *path* (for review, seeding, or Docker COPY).

    Typical source: :meth:`BaseDriver.get_driver_manifest` with ``compiled=False`` or
    :attr:`BaseDriver.cw_driver` after ``define_interface`` is registered on the class.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(
        driver_config,
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


def _looks_like_compiled_bundle(value: dict[str, Any]) -> bool:
    """Detect compiled twin/asset metadata bundles, not cw-driver.yml root dicts.

    Uncompiled roots may include ``zenoh.channels`` (authoring shape). Only reject
    when the dict looks like ``metadata["mqtt"]`` or ``metadata["zenoh"]`` alone.
    """
    topics = value.get("topics")
    if isinstance(topics, dict) and topics:
        return True
    channels = value.get("channels")
    if isinstance(channels, dict) and channels and "mqtt" not in value:
        return True
    mqtt = value.get("mqtt")
    if isinstance(mqtt, dict) and isinstance(mqtt.get("topics"), dict):
        return True
    return False


def resolve_driver_config_dict(
    driver_config: Union[str, Path, dict[str, Any], type, Any],
) -> dict[str, Any]:
    """Normalize *driver_config* to an uncompiled cw-driver.yml root dict for the API."""
    if isinstance(driver_config, (str, Path)):
        return load_cw_driver_yml(driver_config)

    if isinstance(driver_config, dict):
        if _looks_like_compiled_bundle(driver_config):
            raise ValueError(
                "compiled driver catalogs are produced by the backend; "
                "pass a cw-driver.yml root dict or file path"
            )
        return copy.deepcopy(extract_cw_driver_root(driver_config))

    if isinstance(driver_config, type):
        from cyberwave.driver.base import BaseDriver

        if issubclass(driver_config, BaseDriver):
            return driver_config._manifest_probe().get_driver_manifest(compiled=False)

    if hasattr(driver_config, "get_driver_manifest"):
        return driver_config.get_driver_manifest(compiled=False)

    if hasattr(driver_config, "cw_driver"):
        candidate = driver_config.cw_driver
        if isinstance(candidate, dict):
            return copy.deepcopy(candidate)

    raise TypeError(
        "driver_config must be a cw-driver.yml path, root dict, or driver with cw_driver"
    )
