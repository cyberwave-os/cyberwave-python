"""Shared test configuration.

When the auto-generated ``cyberwave.rest`` package is absent (it requires the
backend running + ``python-sdk-gen.sh``), we inject lightweight stubs into
``sys.modules`` so that importing ``cyberwave.data.*`` does not fail.

This file is loaded by pytest before any test module is collected, so the
stubs are in place before the import chain is triggered.
"""

import sys
import types
from typing import Any


def _rest_module_is_real() -> bool:
    """Return True if the auto-generated REST client is available.

    Scans ``sys.path`` directly for the ``cyberwave/rest/__init__.py`` file
    rather than importing the module, to avoid triggering the parent package's
    ``__init__.py`` (which would cause a circular import chain before the stub
    modules are pre-seeded).
    """
    from pathlib import Path

    for search_path in sys.path:
        candidate = Path(search_path) / "cyberwave" / "rest" / "__init__.py"
        try:
            if candidate.exists() and candidate.stat().st_size > 0:
                # Read only the first 4 KB — the import of DefaultApi always
                # appears near the top of the generated __init__.py.
                with candidate.open(encoding="utf-8", errors="ignore") as fh:
                    head = fh.read(4096)
                return "DefaultApi" in head
        except OSError:
            continue
    return False


def _to_plain_value(value: Any) -> Any:
    if isinstance(value, _SchemaStub):
        return value.to_dict()
    if isinstance(value, list):
        return [_to_plain_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain_value(item) for key, item in value.items()}
    return value


class _SchemaStub:
    """Small stand-in for generated OpenAPI schema models in unit tests."""

    def __init__(self, *a, **kw):
        if a and isinstance(a[0], dict):
            kw.update(a[0])
        for key, value in kw.items():
            setattr(self, key, value)

    def __getattr__(self, name):
        raise AttributeError(name)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def to_dict(self):
        return {
            key: _to_plain_value(value)
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }


class _ConfigurationStub:
    def __init__(self, *a, **kw):
        self.host = kw.get("host")
        self.api_key = {}
        self.api_key_prefix = {}
        self.verify_ssl = True


class _ResponseStub:
    data = None


class _ApiClientStub:
    def __init__(self, configuration=None, *a, **kw):
        self.configuration = configuration
        self.rest_client = types.SimpleNamespace(request=lambda *args, **kwargs: None)

    def response_deserialize(self, *a, **kw):
        return _ResponseStub()

    def call_api(
        self,
        method,
        url,
        header_params=None,
        body=None,
        post_params=None,
        _request_timeout=None,
        **kwargs,
    ):
        headers = dict(header_params or {})
        return self.rest_client.request(
            method=method,
            url=url,
            headers=headers,
            body=body,
            post_params=post_params,
            _request_timeout=_request_timeout,
            **kwargs,
        )

    def param_serialize(self, **kwargs):
        return (
            kwargs.get("method"),
            kwargs.get("resource_path"),
            kwargs.get("header_params") or {},
            kwargs.get("body"),
            kwargs.get("post_params") or [],
            kwargs,
        )


class _DefaultApiStub:
    def __init__(self, api_client=None, *a, **kw):
        self.api_client = api_client

    def _src_app_api_assets_list_assets_serialize(self, **kwargs):
        headers = kwargs.get("_headers") or {}
        return ("GET", "/api/v1/assets", headers, None, [], None)

    def __call__(self, *a, **kw):
        return _SchemaStub()


def _rest_attr(name: str):
    if name == "ApiClient":
        return _ApiClientStub
    if name == "Configuration":
        return _ConfigurationStub
    if name == "DefaultApi":
        return _DefaultApiStub
    return _SchemaStub


def _make_stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    setattr(mod, "__getattr__", lambda n: _rest_attr(n))
    setattr(mod, "__path__", [])
    return mod


def _inject_rest_stubs() -> None:
    """Pre-seed ``sys.modules`` with stubs for ``cyberwave.rest.*``."""
    if _rest_module_is_real():
        return

    _REST_MODULES = [
        "cyberwave.rest",
        "cyberwave.rest.models",
        "cyberwave.rest.models.twin_joint_calibration_schema",
        "cyberwave.rest.models.universal_schema_patch_schema",
        "cyberwave.rest.models.twin_universal_schema_patch_schema",
    ]

    for name in _REST_MODULES:
        sys.modules[name] = _make_stub_module(name)


_inject_rest_stubs()
