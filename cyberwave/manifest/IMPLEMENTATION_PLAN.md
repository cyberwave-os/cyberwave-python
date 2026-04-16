# CYB-1550: cyberwave.yml Manifest Support — Implementation Plan

**Issue:** [CYB-1550](https://linear.app/cyberwave-spa/issue/CYB-1550/cyberwaveyml-manifest-support)
**Parent Epic:** [CYB-1498](https://linear.app/cyberwave-spa/issue/CYB-1498/epic-edge-ml-models) (Edge ML Models)
**Coordinates with:** CYB-1545 (SDK Worker API), CYB-1548 (Worker deployment + CLI), CYB-1549 (Cloud worker runtime)

---

## Overview

`cyberwave.yml` exists in two incompatible forms today:

1. **`cyberwave-cloud-node:` key** — the production path (plain dataclass, bash shell commands, no validation, unknown fields silently go to `extra`).
2. **`cyberwave:` key** — the root README spec (richer fields: `workers`, `requirements`, `models`, `input`, `gpu`, `resources`, etc.) that is currently aspirational and unimplemented.

This issue bridges that gap: a single Pydantic-validated schema, a normalising loader that accepts both key shapes for backward compat, a module dispatch path in `cloud_node.py` (so `inference: inference.py` calls `infer()` directly instead of shelling out), `workers:` list loading at startup, and a `cyberwave manifest validate` CLI command for local authoring feedback.

---

## Current State

### What exists

| Component | File | Status |
|-----------|------|--------|
| Cloud node config | `cyberwave-cloud-node/cyberwave_cloud_node/config.py` | `CloudNodeConfig` dataclass, `cyberwave-cloud-node:` key only, `extra: dict` catch-all, no validation |
| Cloud node dispatch | `cyberwave-cloud-node/cyberwave_cloud_node/cloud_node.py` | Bash-only: `{body}` params file, `subprocess.Popen` |
| SDK workers | `cyberwave-sdks/.../cyberwave/workers/` | `HookRegistry`, `HookContext`, `load_workers()` (CYB-1545) |
| Manifest schema | — | **Does not exist** |
| Manifest tests | — | **Does not exist** |

### What does NOT exist

- `cyberwave:` key is documented in README but not parsed anywhere
- No field-level schema validation — unknown keys land in `extra` silently
- No module dispatch — `inference`/`training` values are always treated as bash
- No `workers:` list loading at node startup
- No `cyberwave manifest validate` command
- No schema versioning

---

## Architecture Decisions

### AD-1: Pydantic v2 schema with `extra = "forbid"`

All manifest fields are declared as optional Pydantic fields with defaults. Unknown fields raise `ValidationError` with the field name — satisfying "fail fast with precise field-level diagnostics." This is the key behavioural difference from the current `extra: dict` catch-all.

A controlled `extra = "ignore"` escape hatch is available via an explicit `--lenient` flag in the CLI validator, but the default and the runtime path both reject unknown fields.

### AD-2: Key normalisation — both `cyberwave:` and `cyberwave-cloud-node:` work indefinitely

The loader tries `cyberwave:` first, then `cyberwave-cloud-node:`, then interprets a key-less flat dict as manifest fields. Production profiles with `cyberwave-cloud-node:` continue to work without changes. Migration is opt-in and documented.

### AD-3: Dispatch mode is auto-detected from the field value, not a separate flag

| Value shape | Detected as | Behaviour |
|---|---|---|
| `inference.py` (`.py` suffix, no spaces) | **module** | Import once, call `infer(input, **params)` |
| `python server.py --params {body}` (spaces or `{body}`) | **shell** | Spawn subprocess — existing path, unchanged |

No new YAML key needed. Auto-detection is deterministic and reversible.

### AD-4: Module dispatch runs in a thread pool executor

`infer()` / `train()` are synchronous functions. To avoid blocking the asyncio event loop in `cloud_node.py`, they run via `loop.run_in_executor(None, ...)`. The module is imported once and cached in `sys.modules` — model warm-up at import time is the intended design.

### AD-5: `workers:` loading uses the existing SDK `load_workers()` from CYB-1545

The manifest's `workers:` list resolves paths relative to the `working_dir` of the `CloudNode`. Each path is passed to `load_workers()` from `cyberwave.workers.loader`. If the SDK is not installed or CYB-1545 has not yet merged, the workers block degrades gracefully with a logged warning.

### AD-6: Schema versioning is opt-in via `version:` field

`version` defaults to `"1"` — existing manifests without a version key parse as version 1. Unsupported versions fail immediately with a message listing supported versions. Unknown fields in a supported version fail via `extra = "forbid"`.

---

## File Layout

### New files

```
cyberwave-sdks/cyberwave-python/cyberwave/manifest/
├── IMPLEMENTATION_PLAN.md      ← this file
├── __init__.py                 # Re-exports: ManifestSchema, from_file, from_dict, validate_manifest
├── schema.py                   # Pydantic ManifestSchema + sub-models + detect_dispatch_mode()
├── loader.py                   # YAML loading, key normalisation, from_file(), from_dict()
└── validator.py                # validate_manifest() → ManifestValidationResult, pretty printer

cyberwave-clis/cyberwave-python-cli/cyberwave_cli/commands/
└── manifest.py                 # NEW — `cyberwave manifest validate [path]` command

docs-mintlify/use-cyberwave/
└── manifest.mdx                # NEW (stub) — cyberwave.yml reference page
```

### Modified files

```
cyberwave-cloud-nodes/cyberwave-cloud-node/cyberwave_cloud_node/
├── config.py                   # MINOR — unchanged API; ManifestSchema loaded alongside CloudNodeConfig
└── cloud_node.py               # ADD module dispatch path + workers: startup loading

cyberwave-clis/cyberwave-python-cli/cyberwave_cli/main.py
└──                             # Register `manifest` command group

cyberwave-sdks/cyberwave-python/cyberwave/__init__.py
└──                             # Re-export ManifestSchema

cyberwave-sdks/cyberwave-python/tests/
├── test_manifest_schema.py     # Positive / negative validation matrix
├── test_manifest_loader.py     # Key normalisation, from_file(), flat-format guard, compat
├── test_manifest_validator.py  # Warnings, lenient mode, legacy key detection on file path
└── test_manifest_dispatch.py   # dispatch mode detection + end-to-end module call + install bridging
```

---

## Step-by-step Plan

### Step 1 — `schema.py`: Pydantic manifest model

```python
# cyberwave/manifest/schema.py
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Union

MANIFEST_VERSION = "1"


class ResourcesSchema(BaseModel):
    memory: str | None = None   # e.g. "4g", "512m"
    cpus: float | None = None   # e.g. 2.0


class ManifestSchema(BaseModel):
    # Schema housekeeping
    version: str = MANIFEST_VERSION

    # Identity / catalog
    name: str | None = None

    # Environment setup
    install: str | None = None             # shell command, runs once at startup (README spec name)
    install_script: str | None = None      # legacy alias used by cyberwave-cloud-node: key
    requirements: list[str] | None = None  # pip package specs (alternative to install)

    # Model pre-loading
    models: list[str] | None = None        # catalog model IDs to pre-download

    # On-demand dispatch paths
    inference: str | None = None           # module path (.py) or shell command
    training: str | None = None            # module path (.py) or shell command
    simulate: str | None = None            # shell command (legacy compat — not in README spec)

    # Continuous workers
    workers: list[str] | None = None       # .py files using @cw.on_frame / @cw.on_data hooks

    # Input declaration (for auto-generated worker and docs)
    input: list[str] | None = None         # normalised from str or list

    # Hardware / profile routing
    gpu: bool = False
    runtime: str | None = None             # known runtime ID for zero-code level 0
    model: str | None = None               # model file path (used with `runtime`)
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
    mqtt_password: str | None = None

    model_config = {"extra": "forbid"}

    @field_validator("version")
    @classmethod
    def check_version(cls, v: str) -> str:
        supported = {"1"}
        if v not in supported:
            raise ValueError(
                f"Unsupported manifest version '{v}'. "
                f"Supported: {sorted(supported)}. "
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
    def check_runtime_requires_model_or_inference(self) -> "ManifestSchema":
        if self.runtime and not self.model and not self.inference:
            raise ValueError(
                "'runtime' is set but neither 'model' nor 'inference' is provided. "
                "For zero-code mode, set 'model: path/to/model.pt'. "
                "For function mode, set 'inference: inference.py'."
            )
        return self

    @property
    def effective_install(self) -> str | None:
        """Return the install command, normalising install_script → install."""
        return self.install or self.install_script


def detect_dispatch_mode(value: str) -> str:
    """Return 'module' or 'shell' for an inference/training field value.

    Module mode: value ends with '.py' and contains no spaces.
    Shell mode: everything else (contains spaces, '{body}', multi-word commands).
    """
    stripped = value.strip()
    if stripped.endswith(".py") and " " not in stripped:
        return "module"
    return "shell"
```

**Design notes:**

- All fields optional with defaults — an empty `cyberwave:` block is valid.
- `simulate` included for backward compat with existing cloud-node profiles (not in README spec).
- `input` is normalised from `str` to `list[str]` so consumers always see a list.
- `extra = "forbid"`: unknown field `foo: bar` → `ValidationError` naming `foo` with message `"Extra inputs are not permitted"`.
- Cross-field validator: `runtime:` without `model:` or `inference:` → clear authoring error immediately.
- `detect_dispatch_mode` is a pure function; testable in isolation and used in both `cloud_node.py` and the CLI.
- **`install` vs `install_script` field name**: the README spec uses `install:`, but all production `cyberwave.yml` files use `install_script:` (the `cyberwave-cloud-node:` convention). Both fields are accepted in the schema; `effective_install` is the canonical accessor used in `cloud_node.py` so neither form is silently dropped.
- **`requirements:` and `models:` — parsed but not executed by this issue**: the schema stores them, but pip installation of `requirements:` and pre-download of `models:` are CYB-1546 (Edge Core model manager) territory. `cloud_node.py` explicitly passes `manifest.requirements` and `manifest.models` to the model manager hook defined in CYB-1546 — this coordination is described in Step 5.
- **`install_script` field stays in schema for backward compat**: existing yml files with `cyberwave-cloud-node: {install_script: ./install.sh}` load correctly without any migration.

---

### Step 2 — `loader.py`: YAML loading and key normalisation

```python
# cyberwave/manifest/loader.py
from pathlib import Path
from typing import Optional
import yaml
from cyberwave.manifest.schema import ManifestSchema

CONFIG_FILE_NAME = "cyberwave.yml"
_MANIFEST_KEYS = ("cyberwave", "cyberwave-cloud-node")


_KNOWN_MANIFEST_FIELDS = {
    "version", "name", "install", "install_script", "requirements", "models",
    "inference", "training", "simulate", "workers", "input", "gpu", "runtime",
    "model", "profile_slug", "heartbeat_interval", "upload_results",
    "results_folder", "resources", "mqtt_host", "mqtt_port", "mqtt_use_tls",
    "mqtt_tls_ca_certs", "mqtt_username", "mqtt_password",
}


def _extract_manifest_data(raw: dict) -> dict:
    """Return the inner manifest dict regardless of which top-level key was used.

    Tries known wrapper keys first. Falls back to flat format only when every
    top-level key looks like a manifest field — prevents accidentally loading
    an unrelated YAML file (e.g. docker-compose.yml) as a manifest.
    """
    for key in _MANIFEST_KEYS:
        if key in raw:
            data = raw[key]
            return data if isinstance(data, dict) else {}
    # Flat format: only use if all keys are known manifest fields
    if raw.keys() <= _KNOWN_MANIFEST_FIELDS:
        return raw
    raise ValueError(
        "No 'cyberwave:' or 'cyberwave-cloud-node:' key found in the file, "
        "and the top-level keys do not match the manifest schema. "
        "Wrap your manifest under a 'cyberwave:' key."
    )


def from_dict(data: dict) -> ManifestSchema:
    """Parse and validate a manifest from a raw dictionary.

    Raises:
        pydantic.ValidationError: if the manifest fails schema validation.
    """
    manifest_data = _extract_manifest_data(data)
    return ManifestSchema.model_validate(manifest_data)


def from_file(path: Optional[Path] = None) -> ManifestSchema:
    """Load, parse, and validate a manifest from a YAML file.

    Raises:
        FileNotFoundError: if the file does not exist.
        yaml.YAMLError: if the YAML is malformed.
        pydantic.ValidationError: if the manifest fails schema validation.
    """
    if path is None:
        path = Path.cwd() / CONFIG_FILE_NAME

    if not path.exists():
        # Backward compat: try README.md YAML front matter
        readme_path = path.parent / "README.md"
        if readme_path.exists():
            text = readme_path.read_text()
            parts = text.split("---", 2)
            if len(parts) >= 3:
                raw = yaml.safe_load(parts[1]) or {}
                return from_dict(raw)
        raise FileNotFoundError(f"Manifest file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    return from_dict(raw)
```

**Key normalisation design:**

- `cyberwave:` is tried before `cyberwave-cloud-node:` — production profiles with the old key keep working.
- Flat format (no wrapper key, e.g. `inference: python server.py`) is supported for one-liner configs.
- README front-matter fallback preserved from `CloudNodeConfig.from_file()` for full backward compat.

---

### Step 3 — `validator.py`: structured validation result + pretty printer

```python
# cyberwave/manifest/validator.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pydantic import ValidationError as PydanticValidationError

from cyberwave.manifest.loader import from_dict, from_file
from cyberwave.manifest.schema import ManifestSchema


@dataclass
class ManifestFieldError:
    field_path: str      # e.g. "version", "resources.memory"
    message: str         # human-readable description
    value: object = None # the offending value, if available


@dataclass
class ManifestValidationResult:
    valid: bool
    manifest: Optional[ManifestSchema]
    errors: list[ManifestFieldError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def format_errors(self) -> str:
        if not self.errors:
            return ""
        lines = ["Manifest validation failed:"]
        for e in self.errors:
            line = f"  • {e.field_path}: {e.message}"
            if e.value is not None:
                line += f"  (got: {e.value!r})"
            lines.append(line)
        if self.warnings:
            lines.append("")
            for w in self.warnings:
                lines.append(f"  ⚠  {w}")
        return "\n".join(lines)


def _check_legacy_key(raw: dict) -> list[str]:
    """Return a warning list if the old cyberwave-cloud-node: key is detected."""
    if "cyberwave-cloud-node" in raw and "cyberwave" not in raw:
        return [
            "Key 'cyberwave-cloud-node:' detected. "
            "Consider migrating to 'cyberwave:' for full feature support."
        ]
    return []


def validate_manifest(
    path: Optional[Path] = None,
    data: Optional[dict] = None,
    *,
    lenient: bool = False,
) -> ManifestValidationResult:
    """Validate a manifest from a file path or raw dict.

    Args:
        path: Path to cyberwave.yml. Defaults to ./cyberwave.yml.
        data: Raw dict (overrides path).
        lenient: If True, unknown fields are demoted to warnings instead of errors.
                 Useful during migration from undocumented custom fields.

    Returns a ManifestValidationResult. On success, .manifest is set.
    On failure, .errors contains one entry per failing field.
    """
    import yaml as _yaml

    try:
        if data is not None:
            warnings = _check_legacy_key(data)
            raw = data
        else:
            # Load raw YAML first so we can check for the legacy key before parsing
            if path is None:
                path = Path.cwd() / "cyberwave.yml"
            with open(path, "r") as f:
                raw = _yaml.safe_load(f) or {}
            warnings = _check_legacy_key(raw)

        if lenient:
            # Validate with extra="ignore" to collect warnings instead of errors
            from cyberwave.manifest.loader import _extract_manifest_data
            from cyberwave.manifest.schema import ManifestSchema as _MS
            inner = _extract_manifest_data(raw)
            # Find unknown fields before ignoring them
            from cyberwave.manifest.schema import _KNOWN_MANIFEST_FIELDS  # exported constant
            unknown = set(inner.keys()) - _KNOWN_MANIFEST_FIELDS
            for u in sorted(unknown):
                warnings.append(f"Unknown field ignored (--lenient mode): '{u}'")
            # Re-parse without the unknown fields
            known_only = {k: v for k, v in inner.items() if k in _KNOWN_MANIFEST_FIELDS}
            manifest = ManifestSchema.model_validate(known_only)
        else:
            manifest = from_dict(raw)

        return ManifestValidationResult(valid=True, manifest=manifest, warnings=warnings)

    except PydanticValidationError as exc:
        errors = [
            ManifestFieldError(
                field_path=".".join(str(loc) for loc in err["loc"]),
                message=err["msg"],
                value=err.get("input"),
            )
            for err in exc.errors()
        ]
        return ManifestValidationResult(valid=False, manifest=None, errors=errors, warnings=warnings)

    except FileNotFoundError as exc:
        return ManifestValidationResult(
            valid=False,
            manifest=None,
            errors=[ManifestFieldError(field_path="path", message=str(exc))],
            warnings=[],
        )
    except _yaml.YAMLError as exc:
        return ManifestValidationResult(
            valid=False,
            manifest=None,
            errors=[ManifestFieldError(field_path="yaml", message=f"Invalid YAML: {exc}")],
            warnings=[],
        )
    except ValueError as exc:
        # Raised by _extract_manifest_data for missing wrapper key
        return ManifestValidationResult(
            valid=False,
            manifest=None,
            errors=[ManifestFieldError(field_path="structure", message=str(exc))],
            warnings=[],
        )
```

---

### Step 4 — `__init__.py`: re-exports

```python
# cyberwave/manifest/__init__.py
from .schema import ManifestSchema, ResourcesSchema, detect_dispatch_mode, MANIFEST_VERSION
from .loader import from_file, from_dict
from .validator import validate_manifest, ManifestValidationResult, ManifestFieldError

__all__ = [
    "ManifestSchema",
    "ResourcesSchema",
    "detect_dispatch_mode",
    "MANIFEST_VERSION",
    "from_file",
    "from_dict",
    "validate_manifest",
    "ManifestValidationResult",
    "ManifestFieldError",
]
```

---

### Step 5 — `cloud_node.py`: module dispatch path + `workers:` startup loading

This is the largest runtime change. The existing bash dispatch is preserved unchanged. A parallel module dispatch path is added, gated on `detect_dispatch_mode()`.

**Loading the manifest alongside `CloudNodeConfig`:**

```python
# In CloudNode.from_config_file() and __init__:
try:
    from cyberwave.manifest.loader import from_file as _load_manifest
    _manifest_path = config_path or Path.cwd() / "cyberwave.yml"
    self._manifest = _load_manifest(_manifest_path)
except Exception:
    self._manifest = None  # graceful degradation — bash path still works
```

`CloudNodeConfig` API is left unchanged (it is a published package with its own version). The manifest is an additive attribute on `CloudNode`.

**Bridging `install` → `install_script` (run_async):**

`cloud_node.py` currently reads `self.config.install_script`. When a `cyberwave:` key is used, `install_script` in `CloudNodeConfig` will be `None` (because `from_dict` inside `CloudNodeConfig` only reads `cyberwave-cloud-node:` sub-fields). The bridge:

```python
async def run_async(self) -> None:
    # ... existing code ...
    # Step 1: Run install script — prefer manifest.effective_install, fall back to config
    install_cmd = (
        self._manifest.effective_install if self._manifest else None
    ) or self.config.install_script
    if install_cmd:
        await self._run_install_script(install_cmd)
```

Update `_run_install_script` to accept an explicit command:

```python
async def _run_install_script(self, cmd: str | None = None) -> None:
    command = cmd or self.config.install_script
    if not command:
        return
    logger.info(f"Running install: {command}")
    result = await self._run_command(command, workload_type="install")
    if not result.success:
        raise CloudNodeError(f"Install failed (code {result.return_code}): {result.error}")
```

**`requirements:` and `models:` — explicit coordination note:**

`manifest.requirements` (list of pip specs) and `manifest.models` (list of model IDs to pre-download) are parsed and stored in `ManifestSchema` but **not executed by CYB-1550**. Execution is CYB-1546 (Edge Core model manager) territory. The cloud node passes them through via log messages for now:

```python
if self._manifest and self._manifest.requirements:
    logger.info("manifest.requirements (install via 'install:' or manually): %s", self._manifest.requirements)
if self._manifest and self._manifest.models:
    logger.info("manifest.models (pre-download handled by CYB-1546 model manager): %s", self._manifest.models)
```

This is documented as an explicit TODO in the implementation so it is not silently forgotten.

**Dispatch routing in `_spawn_workload_process()`:**

```python
async def _spawn_workload_process(self, workload_type, params, request_id):
    if self._manifest is not None:
        value = getattr(self._manifest, workload_type, None)
        if value and detect_dispatch_mode(value) == "module":
            await self._dispatch_module_workload(workload_type, params, request_id)
            return
    # Fall through to existing bash path (unchanged)
    # ... existing _spawn_workload_process body ...
```

**`_dispatch_module_workload()` — new method:**

```python
# Platform-internal keys that must not be passed to user functions
_PLATFORM_PARAMS = frozenset({"workload_uuid", "command_type", "status"})


async def _dispatch_module_workload(self, workload_type, params, request_id):
    import importlib.util, sys

    module_path_str = getattr(self._manifest, workload_type)
    abs_path = self.working_dir / module_path_str

    if not abs_path.exists():
        self._publish_response(
            request_id, success=False,
            error=f"Module not found: {abs_path}. "
                  f"Check '{workload_type}: {module_path_str}' in cyberwave.yml."
        )
        return

    module_key = f"_cyberwave_{workload_type}_module"
    if module_key not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_key, abs_path)
        if spec is None or spec.loader is None:
            self._publish_response(
                request_id, success=False,
                error=f"Cannot load module: {abs_path}"
            )
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_key] = module
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            del sys.modules[module_key]
            self._publish_response(
                request_id, success=False,
                error=f"Module import failed: {e}"
            )
            return

    fn_name = "infer" if workload_type == "inference" else "train"
    fn = getattr(sys.modules[module_key], fn_name, None)
    if fn is None:
        self._publish_response(
            request_id, success=False,
            error=f"Module {module_path_str} does not export {fn_name}(). "
                  "Add 'def infer(input, **params): ...' to the module."
        )
        return

    workload_uuid = params.get("workload_uuid")

    # Signal workload started (mirrors the bash path's update_workload_status("running"))
    if workload_uuid and self._mqtt_client:
        try:
            await self._mqtt_client.update_workload_status(
                workload_uuid=workload_uuid,
                status="running",
                additional_data={"message": f"{workload_type} module executing"},
            )
        except Exception as e:
            logger.warning("Failed to send running status for module workload: %s", e)

    # Strip platform-internal keys before calling user function
    user_params = {k: v for k, v in params.items() if k not in _PLATFORM_PARAMS}

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, lambda: fn(**user_params))
        output = json.dumps(result) if not isinstance(result, str) else result
        self._publish_response(request_id, success=True, output=output)
        if workload_uuid and self._mqtt_client:
            await self._mqtt_client.complete_workload(
                workload_uuid=workload_uuid, success=True, exit_code=0, timeout=30.0
            )
    except Exception as e:
        logger.exception("Module workload %s failed", workload_type)
        self._publish_response(request_id, success=False, error=str(e))
        if workload_uuid and self._mqtt_client:
            await self._mqtt_client.complete_workload(
                workload_uuid=workload_uuid, success=False, exit_code=1, timeout=30.0
            )
```

**`workers:` startup loading — new method `_load_manifest_workers()`:**

The plan previously referenced `self._build_cw_proxy()` which does not exist in `cloud_node.py`. The correct approach: use the full `Cyberwave` SDK client if available (it is — `cyberwave>=0.3.31` is already a dependency); otherwise construct a minimal shim from the existing MQTT client.

```python
async def _load_manifest_workers(self) -> None:
    if not self._manifest or not self._manifest.workers:
        return
    try:
        from cyberwave.workers.loader import load_workers
        from cyberwave import Cyberwave
    except ImportError:
        logger.warning(
            "cyberwave SDK worker loader not available. "
            "Ensure cyberwave>=0.3.46 to enable 'workers:' in cyberwave.yml."
        )
        return

    # Construct a Cyberwave client instance wired to this node's MQTT connection.
    # Workers use cw.publish_event(), cw.models.load(), @cw.on_frame(), etc.
    # The Cyberwave client is initialized from env vars (CYBERWAVE_API_KEY, CYBERWAVE_MQTT_*)
    # which are already set when the cloud node starts.
    try:
        cw = Cyberwave()
    except Exception as e:
        logger.warning("Cannot initialise Cyberwave client for workers: %s. Skipping.", e)
        return

    loaded = 0
    for worker_rel_path in self._manifest.workers:
        abs_path = (self.working_dir / worker_rel_path).resolve()
        if not abs_path.exists():
            logger.warning("Worker file not found (skipping): %s", abs_path)
            continue
        # load_workers loads all .py files from a directory;
        # pass the file's parent dir and filter by exact filename via glob
        count = load_workers(abs_path.parent, cw_module=cw)
        loaded += count
        logger.info("Loaded %d hook(s) from worker: %s", count, abs_path.name)

    logger.info("Manifest workers: %d module(s) loaded total", loaded)
```

Call site in `run_async()`:

```python
# After install, before MQTT connect:
if self._manifest and self._manifest.workers:
    await self._load_manifest_workers()
```

---

### Step 6 — `cyberwave manifest validate` CLI command

```python
# cyberwave_cli/commands/manifest.py
import click
from pathlib import Path


@click.group()
def manifest():
    """Manage and validate cyberwave.yml manifests."""


@manifest.command("validate")
@click.argument("path", default="cyberwave.yml", type=click.Path())
@click.option(
    "--lenient",
    is_flag=True,
    help="Treat unknown fields as warnings instead of errors (useful during migration).",
)
def validate(path: str, lenient: bool) -> None:
    """Validate a cyberwave.yml manifest.

    PATH defaults to cyberwave.yml in the current directory.

    Exit code 0 if valid, 1 if invalid.
    """
    from cyberwave.manifest.validator import validate_manifest
    from cyberwave.manifest.schema import detect_dispatch_mode

    result = validate_manifest(Path(path), lenient=lenient)

    if result.warnings:
        for w in result.warnings:
            click.secho(f"⚠  {w}", fg="yellow")

    if result.valid:
        click.secho(f"✓  {path} is valid", fg="green")
        m = result.manifest
        if m.inference:
            mode = detect_dispatch_mode(m.inference)
            click.echo(f"   inference: {m.inference!r}  [{mode} mode]")
        if m.training:
            mode = detect_dispatch_mode(m.training)
            click.echo(f"   training:  {m.training!r}  [{mode} mode]")
        if m.workers:
            click.echo(f"   workers:   {m.workers}")
        if m.models:
            click.echo(f"   models:    {m.models}")
    else:
        click.secho(f"✗  {path} failed validation", fg="red", err=True)
        click.echo(result.format_errors(), err=True)
        raise SystemExit(1)
```

Registration in `main.py`:

```python
from cyberwave_cli.commands.manifest import manifest as manifest_cmd
cli.add_command(manifest_cmd, "manifest")
```

---

### Step 7 — Tests

#### `test_manifest_schema.py` — positive / negative schema matrix

**Positive cases:**

| Test | Input | Expected |
|------|-------|----------|
| `test_empty_block_is_valid` | `{}` | Valid, all defaults |
| `test_full_readme_example` | Full Level 2 YAML from README | Valid |
| `test_cloud_node_key_compat` | `cyberwave-cloud-node: {inference: ...}` | Valid |
| `test_install_field_accepted` | `cyberwave: {install: pip install X}` | `manifest.effective_install == "pip install X"` |
| `test_install_script_field_accepted` | `cyberwave-cloud-node: {install_script: ./install.sh}` | `manifest.effective_install == "./install.sh"` |
| `test_install_takes_priority_over_install_script` | both present | `manifest.effective_install == install value` |
| `test_input_string_normalised` | `input: image` | `input == ["image"]` |
| `test_input_list_preserved` | `input: [image, depth]` | `input == ["image", "depth"]` |
| `test_version_1` | `version: "1"` | Valid |
| `test_bash_inference_valid` | `inference: "python server.py {body}"` | Valid, `detect_dispatch_mode → "shell"` |
| `test_module_inference_valid` | `inference: inference.py` | Valid, `detect_dispatch_mode → "module"` |
| `test_runtime_with_model_valid` | `runtime: ultralytics, model: yolov8n.pt` | Valid |
| `test_simulate_field_valid` | `simulate: "./sim.sh"` | Valid (compat field) |
| `test_requirements_stored` | `requirements: [ultralytics>=8.0]` | `manifest.requirements == ["ultralytics>=8.0"]` |
| `test_models_stored` | `models: [yolov8n]` | `manifest.models == ["yolov8n"]` |

**Negative cases:**

| Test | Input | Expected error |
|------|-------|----------------|
| `test_unknown_field_rejected` | `inference_timeout: 30` | `ValidationError` mentioning `inference_timeout` |
| `test_unsupported_version` | `version: "99"` | `ValidationError` with supported versions listed |
| `test_runtime_without_model_or_inference` | `runtime: ultralytics` | `ValidationError` with actionable message |
| `test_heartbeat_interval_not_string` | `heartbeat_interval: "fast"` | `ValidationError` mentioning `heartbeat_interval` |
| `test_workers_must_be_list` | `workers: detector.py` | `ValidationError` mentioning `workers` |
| `test_requirements_must_be_list` | `requirements: "ultralytics"` | `ValidationError` mentioning `requirements` |

#### `test_manifest_loader.py` — loading and key normalisation

- `test_cyberwave_key_loaded` — `cyberwave:` parsed correctly
- `test_cloud_node_key_loaded` — `cyberwave-cloud-node:` parsed correctly
- `test_cyberwave_key_takes_priority` — both keys present → `cyberwave:` used
- `test_flat_format_all_known_fields_loaded` — dict with only known field names → parsed as manifest
- `test_flat_format_with_unknown_fields_raises` — dict with `services:` (docker-compose) → `ValueError`, not silent pass
- `test_missing_file_raises` — `FileNotFoundError` propagated
- `test_readme_front_matter_fallback` — front-matter YAML parsed correctly
- `test_malformed_yaml_raises` — `yaml.YAMLError` propagated
- `test_null_block_returns_defaults` — `cyberwave: ~` → all defaults

#### `test_manifest_validator.py` — validator, warnings, and lenient mode

- `test_legacy_key_warning_from_dict` — `data={"cyberwave-cloud-node": {...}}` → `warnings` non-empty
- `test_legacy_key_warning_from_file` — file with `cyberwave-cloud-node:` key → `warnings` non-empty (ensures warning is also emitted on the file path, not just the dict path)
- `test_no_warning_for_cyberwave_key` — `data={"cyberwave": {...}}` → `warnings` empty
- `test_lenient_mode_unknown_field` — `validate_manifest(data={"cyberwave": {"foo": 1}}, lenient=True)` → `valid=True`, warning about `foo`
- `test_strict_mode_unknown_field` — same input without `lenient` → `valid=False`, error mentioning `foo`
- `test_yaml_error_captured` — write a file with invalid YAML → result has `valid=False`, error mentions YAML

#### `test_manifest_dispatch.py` — dispatch mode detection + integration

**Dispatch mode unit tests:**

- `"inference.py"` → `"module"`
- `"./models/inference.py"` → `"module"`
- `"python server.py --params {body}"` → `"shell"`
- `"source activate && python run.py {body}"` → `"shell"`
- `"inference.py --extra arg"` (has space) → `"shell"`
- `"train.py"` → `"module"`

**Integration test — manifest → `infer()` function call:**

1. Write a temporary `cyberwave.yml` with `cyberwave: {inference: inference.py}` to a temp dir.
2. Write a temporary `inference.py` that exports `infer(**params)` returning `{"result": "ok"}`.
3. Construct a `CloudNode` pointing at the temp dir (with mocked MQTT + HTTP clients).
4. Call `_dispatch_module_workload("inference", {"input": "test"}, "req-1")`.
5. Assert `_publish_response` was called with `success=True` and `output` contains `"ok"`.
6. Assert `update_workload_status(status="running")` was called before `infer()` (mirrors bash path).
7. Assert `workload_uuid` is NOT passed as a kwarg to `infer()` (platform keys stripped).

**Test: install field bridging:**

1. Write `cyberwave.yml` with `cyberwave: {install: echo "hello"}`.
2. Assert `CloudNode._manifest.effective_install == "echo \"hello\""`.
3. Assert `run_async()` calls `_run_install_script("echo \"hello\"")` rather than reading `config.install_script`.

This validates the full manifest → parser → dispatch → module call chain.

---

## Scope Boundaries

| In scope (this issue) | Out of scope |
|---|---|
| `ManifestSchema` Pydantic model covering all README fields | `runtime:` zero-code built-in wrappers beyond Ultralytics (CYB-1551) |
| Key normalisation: both `cyberwave:` and `cyberwave-cloud-node:` | Auto-generation of continuous worker from `infer()` + `input:` |
| Module dispatch path: `infer()` / `train()` function calls in thread executor | Container orchestration / model weight download (CYB-1546) |
| `workers:` list loaded at `CloudNode` startup | Workflow-generated worker codegen (CYB-1548) |
| `cyberwave manifest validate` CLI command | GPU scheduling / routing (CYB-1549) |
| Schema `version:` field + forward-compat behaviour | `cyberwave manifest init` scaffold command |
| Positive/negative schema test matrix + integration test | Federated training orchestration |
| Backward compat: `CloudNodeConfig` dataclass API unchanged | |
| Internal docs: migration notes in `cyberwave-cloud-nodes/README.md` | |
| External docs: `docs-mintlify/use-cyberwave/manifest.mdx` stub | |

---

## Dependency Graph

```
CYB-1545: SDK Worker API
  load_workers() + HookRegistry
    │
    ▼
CYB-1550 (this issue): Manifest Support
  ManifestSchema (Pydantic) + loader + validator
  Module dispatch path in cloud_node.py
  Workers list loading at startup
  CLI validate command
    │
    ├──▶ CYB-1548: Worker deployment paths
    │     `cyberwave worker add/list` + file watcher
    │     Uses ManifestSchema to locate worker files on edge sync
    │
    └──▶ CYB-1549: Cloud worker runtime
          Cloud dispatch reads ManifestSchema for module vs shell routing
          @cw.on_inference_request hooks are the next abstraction layer
```

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `extra = "forbid"` breaks existing manifests with undocumented fields | High — deployed profiles may use fields not yet in schema | All known `cyberwave-cloud-node` legacy fields added to schema (`simulate`, `install_script`); `--lenient` CLI flag demotes unknown-field errors to warnings |
| `install:` (README name) vs `install_script:` (yml files) mismatch | Medium — `install:` in a `cyberwave:` block would be silently ignored without bridging | `effective_install` property normalises both; `run_async` reads `manifest.effective_install or config.install_script` |
| `requirements:` and `models:` parsed but not executed | Medium — author sets them, nothing happens; silent | Explicit log messages at startup; plan documents these as deferred to CYB-1546 |
| `_build_cw_proxy()` doesn't exist in `cloud_node.py` | Build failure if left as-is | Replaced with explicit `Cyberwave()` construction using env vars already set when node starts |
| Platform keys (`workload_uuid`) leaking into `infer()` | `TypeError` in user function if signature uses `**params` strictly | `_PLATFORM_PARAMS` set stripped before calling user function |
| Missing `running` status notification in module path | Backend workload stays in pending state forever | `update_workload_status("running")` sent before `run_in_executor` call, mirroring bash path |
| Legacy key warning only emitted for `data=` path | Silent for file-based calls | `validate_manifest(path=...)` now loads raw YAML first and checks keys before parsing |
| Flat-format loading of arbitrary YAML files | `docker-compose.yml` in same dir parsed as manifest, confusing errors | `_extract_manifest_data` only falls back to flat format if all keys are in `_KNOWN_MANIFEST_FIELDS`; otherwise raises `ValueError` |
| `--lenient` flag accepted but not wired | Flag silently has no effect | `validate_manifest(lenient=bool)` added; CLI passes it through |
| Module-level `infer()` cached in `sys.modules` across workloads | Stale model if weights change on disk | Document that module reload requires node restart; expose `cyberwave manifest reload` later |
| Pydantic version mismatch | `cyberwave` SDK already pins `pydantic = "^2"` | No compat shim needed; verified by checking SDK pyproject.toml |

---

## Implementation Order

1. `cyberwave/manifest/schema.py` — Pydantic model, no deps, everything else builds on it; includes `_KNOWN_MANIFEST_FIELDS` constant
2. `cyberwave/manifest/loader.py` — YAML + key normalisation with flat-format guard
3. `cyberwave/manifest/validator.py` — structured result + pretty printer; `lenient=` param; raw YAML pre-load for legacy key detection
4. `cyberwave/manifest/__init__.py` — re-exports
5. Tests: `test_manifest_schema.py`, `test_manifest_loader.py`, `test_manifest_validator.py`
6. `cloud_node.py` — manifest load + `effective_install` bridging + `_dispatch_module_workload()` (with `running` status + platform-key stripping) + `_load_manifest_workers()` (with real `Cyberwave()` client)
7. Tests: `test_manifest_dispatch.py` including end-to-end integration test and install bridging test
8. `cyberwave_cli/commands/manifest.py` — `validate` command + registration in `main.py`
9. Internal docs: `cyberwave-cloud-nodes/README.md` migration notes
10. External docs: `docs-mintlify/use-cyberwave/manifest.mdx` (stub)

---

## Acceptance Criteria Cross-Reference

| Acceptance criterion | How satisfied |
|---|---|
| Valid manifests register/deploy successfully | Step 6 module dispatch + integration test; `effective_install` bridging ensures `install:` executes |
| Invalid manifests fail fast with field-level diagnostics | `extra = "forbid"` + `ManifestFieldError.field_path` + CLI output; `--lenient` for migration |
| README/spec examples executable or explicitly marked target-state | All Level 0–3 examples validate against the schema; auto-generated worker and `models:` pre-download annotated as future-state (CYB-1546) |
| Integration with workflow/codegen path documented | Scope boundary table; explicit note that `wf_*.py` bypasses `cyberwave.yml` |
| Positive/negative schema test matrix | `test_manifest_schema.py`, `test_manifest_loader.py`, `test_manifest_validator.py` test cases |
| Integration test: manifest → runnable binding | `test_manifest_dispatch.py` end-to-end test |
| Compat tests across schema versions | `version: "99"` negative test + `version: "1"` positive test |
| `running` status lifecycle preserved in module path | `update_workload_status("running")` in `_dispatch_module_workload`; asserted in integration test |
| Platform keys not leaked to user functions | `_PLATFORM_PARAMS` stripping; asserted in integration test |
| `install:` and `install_script:` both work | `effective_install` property; `test_install_field_accepted` + `test_install_script_field_accepted` |
