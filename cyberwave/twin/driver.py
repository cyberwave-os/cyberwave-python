"""Driver interface catalog handle for twins (read + ``set_schema``)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from ..driver.interface.cw_driver import resolve_driver_config_dict
from ..exceptions import CyberwaveError
from ..manifest.driver_config import (
    JOINT_UPDATE_TOPIC_SLUG,
    TWIN_COMMAND_TOPIC_SLUG,
    command_specs,
    extract_mqtt_bundle_from_metadata,
    extract_zenoh_bundle_from_metadata,
    mqtt_topic_slugs,
    supported_mqtt_commands,
    supported_transports_from_metadata,
    zenoh_channel_names,
)
from ._helpers import _get_twin_metadata
from .command_factory import rebind_catalog_commands

if TYPE_CHECKING:
    from .base import Twin


class TwinDriverHandle:
    """Driver interface catalogs stored on the twin (MQTT + optional Zenoh).

    Use this handle for **introspection** (getters) and **registering** a driver
    manifest via :meth:`set_schema`. Command invocation stays on
    :attr:`~cyberwave.twin.base.Twin.commands` (``twin.commands.<name>(...)``).
    """

    def __init__(self, twin: Twin) -> None:
        self._twin = twin

    def get_schemas(self, *, force_refresh: bool = False) -> dict[str, Any]:
        """Return compiled catalogs: ``mqtt`` (with ``joint_control`` when applicable) and ``zenoh``."""
        if not force_refresh and getattr(self._twin, "_driver_catalog_cache", None) is not None:
            return copy.deepcopy(self._twin._driver_catalog_cache)

        mqtt_bundle = self._load_mqtt_bundle()
        if mqtt_bundle is None:
            mqtt_view: dict[str, Any] = {"commands": {"supported": []}, "topics": {}}
        else:
            mqtt_view = copy.deepcopy(mqtt_bundle)
            self._merge_joint_control(mqtt_view)

        zenoh_bundle = self._load_zenoh_bundle()
        schemas = {
            "mqtt": mqtt_view,
            "zenoh": copy.deepcopy(zenoh_bundle) if zenoh_bundle is not None else {},
        }
        self._twin._driver_catalog_cache = schemas
        self._twin._mqtt_catalog_cache = copy.deepcopy(mqtt_view)
        return copy.deepcopy(schemas)

    def get_mqtt_schema(self, *, force_refresh: bool = False) -> dict[str, Any]:
        """MQTT catalog only (topics, commands, optional ``joint_control``)."""
        return copy.deepcopy(self.get_schemas(force_refresh=force_refresh)["mqtt"])

    def get_zenoh_schema(self, *, force_refresh: bool = False) -> dict[str, Any]:
        """Zenoh edge catalog only (``channels``, ``locality``, …). Empty when absent."""
        return copy.deepcopy(self.get_schemas(force_refresh=force_refresh)["zenoh"])

    def get_supported_commands(self) -> list[str]:
        """Command names from the MQTT catalog ``commands.supported`` list."""
        return supported_mqtt_commands(self._load_mqtt_bundle())

    def get_supported_topics(self) -> list[str]:
        """MQTT topic slug keys from the compiled catalog."""
        return mqtt_topic_slugs(self._load_mqtt_bundle())

    def get_supported_channels(self) -> list[str]:
        """Zenoh channel names from the compiled edge catalog."""
        return zenoh_channel_names(self._load_zenoh_bundle())

    def get_supported_transports(self) -> list[str]:
        """Transports with non-empty catalogs on this twin (``mqtt``, ``zenoh``)."""
        return supported_transports_from_metadata(_get_twin_metadata(self._twin._data))

    def get_command_specs(self) -> dict[str, dict[str, Any]]:
        """Per-command MQTT specs (``continuous``, ``rate_hz``, …)."""
        return command_specs(self._load_mqtt_bundle())

    def set_schema(
        self,
        driver_config: Union[str, Path, dict[str, Any], type, Any],
        *,
        merge: bool = True,
    ) -> dict[str, Any]:
        """Persist driver catalogs from ``cw-driver.yml`` root dict or a driver class.

        Posts an uncompiled cw-driver manifest to the platform API for compilation,
        writes ``metadata["mqtt"]`` / ``metadata["zenoh"]``, clears caches, and
        re-binds ``twin.commands.<name>`` methods.

        Args:
            driver_config: Path to ``cw-driver.yml``, cw-driver root dict, or a
                :class:`~cyberwave.driver.BaseDriver` subclass / instance.
            merge: Deep-merge into existing metadata blocks when ``True``; replace
                each present block when ``False``.

        Returns:
            :meth:`get_schemas` after persistence.

        Raises:
            CyberwaveError: REST client cannot update the twin.
            FileNotFoundError: *driver_config* path missing.
            ValueError: Invalid manifest.
        """
        driver_root = resolve_driver_config_dict(driver_config)

        twins_api = getattr(self._twin.client, "twins", None)
        if twins_api is None or not hasattr(twins_api, "set_driver_schema"):
            raise CyberwaveError(
                "Cannot persist driver catalog: Cyberwave client has no twins.set_driver_schema API"
            )

        try:
            updated = twins_api.set_driver_schema(
                self._twin.uuid,
                driver_config=driver_root,
                merge=merge,
            )
        except Exception as exc:
            raise CyberwaveError(
                f"Failed to update twin {self._twin.uuid!r} driver catalog: {exc}"
            ) from exc

        self._twin._data = updated
        self._clear_catalog_caches()
        rebind_catalog_commands(self._twin.commands)
        return self.get_schemas(force_refresh=True)

    def _clear_catalog_caches(self) -> None:
        self._twin._driver_catalog_cache = None
        self._twin._mqtt_catalog_cache = None

    def _load_mqtt_bundle(self) -> dict[str, Any] | None:
        return extract_mqtt_bundle_from_metadata(_get_twin_metadata(self._twin._data))

    def _load_zenoh_bundle(self) -> dict[str, Any] | None:
        return extract_zenoh_bundle_from_metadata(_get_twin_metadata(self._twin._data))

    def _merge_joint_control(self, schema: dict[str, Any]) -> None:
        from ..manifest.driver_config import has_joint_update_topic

        if not has_joint_update_topic(schema):
            return
        from .capabilities.joints import controllable_joint_names

        names = controllable_joint_names(self._twin)
        if not names:
            return
        schema["joint_control"] = {
            "controllable_joint_names": names,
            "joints": [{"name": name} for name in names],
        }

    def describe_section(self) -> dict[str, Any]:
        """Agent-facing summary of ``twin.driver`` (catalog introspection + ``set_schema``)."""
        schemas = self.get_schemas()
        mqtt = schemas.get("mqtt") if isinstance(schemas.get("mqtt"), dict) else {}
        zenoh = schemas.get("zenoh") if isinstance(schemas.get("zenoh"), dict) else {}
        topic_slugs = self.get_supported_topics()

        mqtt_section: dict[str, Any] = {
            "command_topic_slug": TWIN_COMMAND_TOPIC_SLUG,
            "has_joint_update_topic": JOINT_UPDATE_TOPIC_SLUG in set(topic_slugs),
            "supported_commands": self.get_supported_commands(),
            "topics": topic_slugs,
            "command_specs": self.get_command_specs(),
        }
        joint_control = mqtt.get("joint_control")
        if isinstance(joint_control, dict) and joint_control:
            mqtt_section["joint_control"] = joint_control

        zenoh_section: dict[str, Any] = {"channels": self.get_supported_channels()}
        if isinstance(zenoh.get("locality"), str) and zenoh["locality"]:
            zenoh_section["locality"] = zenoh["locality"]

        section: dict[str, Any] = {
            "access": "twin.driver",
            "role": "driver_catalog_introspection_and_registration",
            "methods": [
                "get_schemas",
                "get_mqtt_schema",
                "get_zenoh_schema",
                "get_supported_commands",
                "get_supported_topics",
                "get_supported_channels",
                "get_supported_transports",
                "get_command_specs",
                "set_schema",
            ],
            "transports": self.get_supported_transports(),
            "mqtt": mqtt_section,
            "zenoh": zenoh_section,
        }
        provenance = mqtt.get("provenance")
        if isinstance(provenance, dict) and provenance:
            section["mqtt"]["provenance"] = provenance
        if isinstance(zenoh.get("provenance"), dict) and zenoh.get("provenance"):
            section["zenoh"]["provenance"] = zenoh["provenance"]
        return section

    def __repr__(self) -> str:
        transports = ",".join(self.get_supported_transports()) or "none"
        return (
            f"TwinDriverHandle(twin={self._twin.uuid!r}, transports=[{transports}])"
        )
