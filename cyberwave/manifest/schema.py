"""Pydantic v2 manifest schema for cyberwave.yml."""

from __future__ import annotations

from pydantic import BaseModel, field_validator, model_validator

MANIFEST_VERSION = "1"

_SUPPORTED_VERSIONS = {"1"}


class ResourcesSchema(BaseModel):
    """Hardware resource constraints for a cloud node workload."""

    memory: str | None = None
    cpus: float | None = None


class ManifestSchema(BaseModel):
    """Validated schema for ``cyberwave.yml`` manifest files.

    All fields are optional with sensible defaults so that an empty
    ``cyberwave:`` block is a valid (albeit no-op) manifest.
    """

    # Schema housekeeping
    version: str = MANIFEST_VERSION

    # Identity / catalog
    name: str | None = None

    # Environment setup
    install: str | None = None
    install_script: str | None = None
    requirements: list[str] | None = None

    # Model pre-loading (parsed, execution deferred to CYB-1546)
    models: list[str] | None = None

    # On-demand dispatch paths
    inference: str | None = None
    training: str | None = None
    simulate: str | None = None

    # Continuous workers
    workers: list[str] | None = None

    # Input declaration
    input: list[str] | None = None

    # Hardware / profile routing
    gpu: bool = False
    runtime: str | None = None
    model: str | None = None
    profile_slug: str = "default"

    # Operational
    heartbeat_interval: int = 30
    upload_results: bool = True
    results_folder: str = "/results"
    resources: ResourcesSchema | None = None

    # MQTT overrides
    mqtt_host: str | None = None
    mqtt_port: int | None = None
    mqtt_use_tls: bool | None = None
    mqtt_tls_ca_certs: str | None = None
    mqtt_username: str | None = None
    mqtt_password: str | None = None  # Prefer env var substitution (e.g. ${MQTT_PASSWORD}) to avoid committing secrets to source control.

    model_config = {"extra": "forbid"}

    @field_validator("version")
    @classmethod
    def check_version(cls, v: str) -> str:
        if v not in _SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported manifest version '{v}'. "
                f"Supported: {sorted(_SUPPORTED_VERSIONS)}. "
                "Upgrade cyberwave to use a newer manifest version."
            )
        return v

    @field_validator("input", mode="before")
    @classmethod
    def normalise_input(cls, v: object) -> list[str] | None:
        if isinstance(v, str):
            return [v]
        return v  # type: ignore[return-value]

    @model_validator(mode="after")
    def check_runtime_requires_model_or_inference(self) -> ManifestSchema:
        if self.runtime and not self.model and not self.inference:
            raise ValueError(
                "'runtime' is set but neither 'model' nor 'inference' is provided. "
                "For zero-code mode, set 'model: path/to/model.pt'. "
                "For function mode, set 'inference: inference.py'."
            )
        return self

    @property
    def effective_install(self) -> str | None:
        """Return the install command, normalising ``install_script`` -> ``install``."""
        return self.install or self.install_script


# Exported constant so validator.py can reference it without reaching
# into loader internals.
KNOWN_MANIFEST_FIELDS = frozenset(ManifestSchema.model_fields.keys())


def detect_dispatch_mode(value: str) -> str:
    """Return ``'module'`` or ``'shell'`` for an inference/training field value.

    Module mode: value ends with ``.py`` and contains no spaces.
    Shell mode: everything else (contains spaces, ``{body}``, multi-word commands).
    """
    stripped = value.strip()
    if stripped.endswith(".py") and " " not in stripped:
        return "module"
    return "shell"
