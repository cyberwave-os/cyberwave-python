"""cyberwave.yml project manifest: schema, loader, and validator.

(The ``cw-driver.yml`` driver-interface catalog I/O moved to
:mod:`cyberwave.driver.interface.cw_driver` — it is driver-export logic, not part
of the project manifest. Reading compiled catalogs from twin/asset metadata stays
in :mod:`cyberwave.manifest.driver_config`.)
"""

from .loader import from_dict, from_file
from .schema import (
    KNOWN_MANIFEST_FIELDS,
    MANIFEST_VERSION,
    ManifestSchema,
    ResourcesSchema,
    detect_dispatch_mode,
)
from .validator import ManifestFieldError, ManifestValidationResult, validate_manifest

__all__ = [
    "ManifestSchema",
    "ResourcesSchema",
    "detect_dispatch_mode",
    "MANIFEST_VERSION",
    "KNOWN_MANIFEST_FIELDS",
    "from_file",
    "from_dict",
    "validate_manifest",
    "ManifestValidationResult",
    "ManifestFieldError",
]
