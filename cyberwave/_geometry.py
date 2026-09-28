"""Lazy access to the shared geometry core.

The SDK holds no quaternion arithmetic of its own -- it lives in
``cyberwave_geometry``, a ctypes binding over the C++ core in
``common/geometry``. But that package carries a native library, and importing
it at module scope makes it a hard requirement of ``import cyberwave`` itself:
``cyberwave/__init__`` -> ``compact`` -> ``client`` -> ``data`` -> ``fusion``
reaches it on every single import.

That is too much coupling for a leaf feature. It broke the camera-driver E2E,
which builds its own virtualenv, never interpolates a rotation, and has nothing
to do with geometry::

    File ".../cyberwave/data/fusion.py", line 31, in <module>
        from cyberwave_geometry import GeometryError as _CoreGeometryError
    ModuleNotFoundError: No module named 'cyberwave_geometry'

So the import happens on first *use* instead. This is **not** a fallback -- there
is no second implementation of the maths, deliberately, and there never will be
one. A caller that actually does rotation maths without the core installed gets
an ImportError saying what to install; a caller that never touches it is not
made to care.

Once ``cyberwave-geometry`` is published and can be declared as a real
dependency, this indirection is still worth keeping: it costs one cached
function call and it keeps a native library off the import path of every
consumer that does not need it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

__all__ = ["core"]

_CACHE: Any = None

_MISSING = (
    "cyberwave-geometry is not installed, so this SDK cannot do rotation "
    "maths.\n"
    "\n"
    "The Cyberwave SDK does not implement quaternion or transform maths; it "
    "calls the shared geometry core, which ships a native library.\n"
    "\n"
    "Install it with:\n"
    "    pip install cyberwave-geometry\n"
    "\n"
    "It is declared as a dependency of this SDK, so seeing this usually means "
    "the environment was assembled without it -- a partial install, or a "
    "platform with no wheel (wheels cover CPython 3.10-3.14 on Linux, macOS "
    "and Windows).\n"
    "\n"
    "Everything in the SDK that does not touch rotations works without it.\n"
)


def core() -> Any:
    """Return the shared geometry core, importing it on first call.

    The result is a namespace with ``quat`` (the ``quaternion`` module),
    ``transform``, ``compat``, and the ``Quaternion`` / ``Vector3`` /
    ``GeometryError`` types. Cached, so the cost after the first call is a
    global read.
    """
    global _CACHE
    if _CACHE is None:
        try:
            from cyberwave_geometry import GeometryError, Quaternion, Vector3, compat
            from cyberwave_geometry import quaternion as quaternion_module
            from cyberwave_geometry import transform as transform_module
        except ImportError as error:  # pragma: no cover - depends on the install
            raise ImportError(_MISSING) from error
        _CACHE = SimpleNamespace(
            quat=quaternion_module,
            transform=transform_module,
            compat=compat,
            Quaternion=Quaternion,
            Vector3=Vector3,
            GeometryError=GeometryError,
        )
    return _CACHE
