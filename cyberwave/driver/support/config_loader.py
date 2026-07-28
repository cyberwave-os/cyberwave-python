"""DriverConfigLoader — typed config resolution with env > yaml > default.

Centralizes a per-field config resolution pattern every driver used to
hand-roll. Per-field precedence is **environment variable -> yaml file ->
built-in default**; env wins so the edge orchestrator can reconfigure a
driver container without editing files. Env var names default to
``{env_prefix}_{KEY}`` (upper-cased key), overridable per call.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_TRUTHY = ("1", "true", "yes", "on")


def _env(name: str) -> str | None:
    """Trimmed env value, or None when unset/blank."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


class ConfigSection:
    """Typed getters over one yaml section with env-first precedence."""

    def __init__(self, env_prefix: str, data: dict[str, Any]) -> None:
        self._prefix = env_prefix
        self._data = data

    def _env_name(self, key: str, env: str | None) -> str:
        return env or f"{self._prefix}_{key.upper()}"

    def get_str(self, key: str, default: str, *, env: str | None = None) -> str:
        value = self.get_optional_str(key, env=env)
        return value if value is not None else default

    def get_optional_str(self, key: str, *, env: str | None = None) -> str | None:
        e = _env(self._env_name(key, env))
        if e is not None:
            return e
        value = self._data.get(key)
        if value is not None:
            text = str(value).strip()
            return text or None
        return None

    def get_int(self, key: str, default: int, *, env: str | None = None) -> int:
        name = self._env_name(key, env)
        e = _env(name)
        if e is not None:
            try:
                return int(e)
            except ValueError:
                logger.warning("Invalid %s=%r; using config/default", name, e)
        value = self._data.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                logger.warning("Invalid config %s=%r; using %d", key, value, default)
        return default

    def get_bool(self, key: str, default: bool, *, env: str | None = None) -> bool:
        e = _env(self._env_name(key, env))
        if e is not None:
            return e.lower() in _TRUTHY
        if key in self._data and self._data[key] is not None:
            return bool(self._data[key])
        return default

    def get_optional_float(self, key: str, *, env: str | None = None) -> float | None:
        name = self._env_name(key, env)
        e = _env(name)
        if e is not None:
            try:
                return float(e)
            except ValueError:
                logger.warning("Invalid %s=%r; ignoring", name, e)
        value = self._data.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                logger.warning("Invalid config %s=%r; ignoring", key, value)
        return None


class DriverConfigLoader:
    """Locate + parse the driver yaml once; hand out typed sections."""

    def __init__(
        self,
        env_prefix: str,
        *,
        path_env: str | None = None,
        search_paths: tuple[Path, ...] = (),
    ) -> None:
        self._env_prefix = env_prefix
        self._path_env = path_env
        self._search_paths = search_paths
        self._data_cache: dict[str, Any] | None = None

    def _config_path(self) -> Path | None:
        if self._path_env:
            explicit = _env(self._path_env)
            if explicit:
                return Path(explicit)
        for candidate in self._search_paths:
            if candidate.is_file():
                return candidate
        return None

    def _data(self) -> dict[str, Any]:
        if self._data_cache is None:
            self._data_cache = self._load()
        return self._data_cache

    def _load(self) -> dict[str, Any]:
        path = self._config_path()
        if path is None or not path.is_file():
            return {}
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("Failed to read config %s: %s; using env/defaults", path, exc)
            return {}
        return data if isinstance(data, dict) else {}

    def section(self, name: str) -> ConfigSection:
        raw = self._data().get(name)
        return ConfigSection(
            self._env_prefix, raw if isinstance(raw, dict) else {}
        )
