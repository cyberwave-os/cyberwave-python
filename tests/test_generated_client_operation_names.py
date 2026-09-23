"""Every operation ``resources.py`` calls must exist on the generated client.

``cyberwave/rest`` is generated from the backend's OpenAPI schema at release
time and is not committed, so a call site names an operation nothing in this
repository can check. The manager tests all stub the client with a bare
``MagicMock``, which answers to any spelling — so a name that drifts stays green
here and raises ``AttributeError`` in a user's process, where
``BaseResourceManager._handle_error`` flattens it into a generic
``CyberwaveAPIError`` that names the operation the user asked for rather than
the method that is missing.

django-ninja derives each operation id from the *module path* of the view, so
moving a view between API modules renames its operation without touching a line
of the view itself. That is how ``export_universal_schema_json`` and
``export_urdf_scene`` came to call ``src_app_api_environments_…`` after their
views moved into ``environments_exports.py``: both were dead in production for
months with a green suite.

One test over the whole file rather than a per-operation assertion in each test
module: the failure mode is a rename nobody thought to check, so the check has
to be the one nobody has to remember to add.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from _rest_probe import rest_module_is_real

pytestmark = pytest.mark.skipif(
    not rest_module_is_real(),
    reason="generated cyberwave.rest package not available",
)

_RESOURCES = Path(__file__).resolve().parents[1] / "cyberwave" / "resources.py"

# Matches ``self.api.<operation>``. Deliberately source-level rather than
# import-and-introspect: the call sites are inside method bodies, so nothing
# short of calling every manager method would reach them at runtime.
_CALL_SITE = re.compile(r"self\.api\.(src_app_api_[A-Za-z0-9_]*)")


def _referenced_operations() -> list[str]:
    return sorted(set(_CALL_SITE.findall(_RESOURCES.read_text(encoding="utf-8"))))


def test_resources_only_calls_operations_the_generated_client_exposes():
    from cyberwave.rest import DefaultApi

    operations = _referenced_operations()
    assert operations, "found no self.api.<operation> call sites — has the regex gone stale?"

    missing = [name for name in operations if not hasattr(DefaultApi, name)]
    assert not missing, (
        "resources.py calls operations the generated DefaultApi does not have: "
        + ", ".join(missing)
        + ". Either the backend view moved module (django-ninja derives the "
        "operation id from the view's module path, so the call site needs the "
        "new prefix) or cyberwave/rest is stale — regenerate it with "
        "cyberwave-sdks/python-sdk-gen.sh sdk."
    )
