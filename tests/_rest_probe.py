"""Single source of truth for "is the generated ``cyberwave.rest`` present?".

Both ``conftest`` (to decide whether to inject stubs) and the test modules that
skip when the real client is missing must agree on the answer. Deriving it twice
by different mechanisms is how the two drift apart, so it lives here.

Deliberately does **not** import the module. Importing would execute the parent
``cyberwave/__init__.py`` and pull in a circular import chain before the stubs
are pre-seeded, and — once ``conftest`` has stubbed ``cyberwave.rest`` into
``sys.modules`` — an import probe answers "yes" for the stub too, since the stub
module's ``__getattr__`` resolves every name. The filesystem gives the same
answer whenever it is asked.
"""

import sys
from pathlib import Path


def rest_module_is_real() -> bool:
    """Return True if the auto-generated REST client is available on ``sys.path``."""
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
