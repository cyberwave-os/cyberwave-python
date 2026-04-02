# CYB-1545: SDK Worker API (Hooks, Models, Runtime) — Implementation Plan

## Context

This issue is the second layer of the [CYB-1498 — EPIC: Edge ML models](https://linear.app/cyberwave-spa/issue/CYB-1498). It builds the worker-facing SDK contract on top of the data layer created by CYB-1544.

| Issue | Title | Relationship |
|-------|-------|-------------|
| CYB-1544 | SDK Data Layer (Zenoh Transport Abstraction) | **Blocked by.** Provides `DataBackend`, `Sample`, `cw.data.publish/subscribe/latest` |
| **CYB-1545** | **SDK Worker API (Hooks, Models, Runtime)** | **This issue.** Hook decorators, model loading API, worker runtime entrypoint |
| CYB-1546 | Edge Core worker container + model manager + Zenoh infra | **Blocks.** Consumes the worker runtime entrypoint and model loader |
| CYB-1548 | Worker deployment paths (CLI + workflow codegen + sync) | **Blocks.** Generated workers use the hook decorators and model API from this issue |

**Current state:** No hook, model-loading, or worker-runtime module exists in the SDK. The `Cyberwave` client class (`client.py`) manages REST + MQTT. The README.md contains the full target spec: hook decorator list, `(sample, ctx)` callback signature, `cw.models.load()` API, and the runtime boundary where worker `.py` files never call `cw.run()`.

**Important dependency note:** This issue does **not** add `cw.data` to the `Cyberwave` client — that comes from CYB-1544 (specifically CYB-1554, the public API facade). Until CYB-1544 lands, the worker runtime's `_subscribe_hook` will log a warning instead of creating subscriptions. Hook registration, model loading, `publish_event`, and the runtime entrypoint all work standalone without the data layer. Similarly, hook callbacks receive **raw `bytes`** (the `Sample.payload` from the data layer) in Phase 1. Deserialization to numpy arrays / JSON is the responsibility of the wire format layer (CYB-1553). The plan notes where this boundary applies.

**`cw.data.latest()` in worker callbacks:** README examples show workers calling `cw.data.latest("depth")` inside hook callbacks for multi-sensor fusion. This is a pass-through to the data layer's `latest()` method — it requires `cw.data` to exist. Until CYB-1544 lands, this call is not available. This issue does **not** implement or mock `cw.data.latest()`. Workers that need multi-sensor fusion will work once CYB-1544 is merged.

---

## Terminology

| Term | Meaning | Cardinality |
|------|---------|-------------|
| **Worker module** | A single `.py` file containing hook registrations and business logic (e.g. `detect_people.py`, `wf_abc123_person_alert.py`). | Many per device |
| **Worker runtime** | The SDK-provided process entrypoint (`WorkerRuntime`) that loads modules, wires hooks to the data layer, and runs the dispatch loop via `cw.run()`. | One per container |
| **Worker container** | The Docker container started by Edge Core. Runs exactly one `WorkerRuntime` instance, which loads all worker modules from `/app/workers/`. | One per edge device |

The relationship is: **one edge device → one worker container → one runtime → many modules**. All modules share a single process, a single Zenoh session, a single MQTT connection, and a single model cache. Hook callbacks from different modules are isolated at the dispatch level (each wrapped in `try/except`) but share the same thread pool and resources.

---

## Deliverables

1. Hook decorator registry and dispatch pipeline
2. Callback context type (`HookContext`)
3. Model loading API (`cw.models.load()`, `LoadedModel`, runtime backends)
4. Model prediction output types (`Detection`, `PredictionResult`)
5. Worker runtime entrypoint (`cw.run()`)
6. Worker module loader (imports `.py` files, collects registered hooks)
7. Integration with `Cyberwave` client (`cw.on_frame`, `cw.models`, `cw.run`, `cw.publish_event`)
8. Unit and integration tests
9. Example hand-written worker (`examples/edge_worker_detect_people.py`)

---

## File Layout

New files under `cyberwave-sdks/cyberwave-python/cyberwave/`:

```
cyberwave/
├── workers/
│   ├── __init__.py              # Re-exports: HookRegistry, HookContext, run, load_workers
│   ├── hooks.py                 # Hook registry, decorator factory, dispatch pipeline
│   ├── context.py               # HookContext dataclass
│   ├── runtime.py               # Worker runtime entrypoint: run(), load_workers()
│   └── loader.py                # Module loader: import .py files, collect hooks
├── models/
│   ├── __init__.py              # Re-exports: ModelManager, LoadedModel, Detection
│   ├── manager.py               # ModelManager: load(), load_from_file()
│   ├── loaded_model.py          # LoadedModel base + .predict() contract
│   ├── types.py                 # Detection, BoundingBox, PredictionResult dataclasses
│   └── runtimes/
│       ├── __init__.py          # Runtime registry + get_runtime()
│       ├── base.py              # ModelRuntime ABC
│       └── ultralytics_rt.py    # UltralyticsRuntime (YOLOv8/v11)
```

Tests:

```
tests/
├── test_hook_registration.py       # Decorator registers hooks correctly
├── test_hook_dispatch.py           # Dispatch pipeline invokes callbacks
├── test_hook_context.py            # HookContext fields are populated
├── test_model_manager.py           # ModelManager.load() and load_from_file()
├── test_loaded_model.py            # LoadedModel.predict() contract
├── test_model_types.py             # Detection / BoundingBox dataclass behavior
├── test_runtime_boundary.py        # Importing a worker module does NOT start the loop
├── test_worker_loader.py           # load_workers() finds and imports .py files
├── test_worker_runtime.py          # run() starts dispatch loop, stop() exits cleanly
└── test_worker_integration.py      # End-to-end: publish sample → hook fires → event emitted
```

---

## Step-by-step Plan

### Step 1: Define hook context type (`context.py`)

The `HookContext` carries per-sample metadata from the runtime into user callbacks. Every hook callback receives `(sample, ctx)` — the sample is the deserialized payload; `ctx` is a `HookContext`.

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HookContext:
    """Per-sample context passed as the second argument to every hook callback."""
    timestamp: float
    channel: str
    sensor_name: str = "default"
    twin_uuid: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
```

**Design notes:**

- `timestamp` is the sample time from the data layer (`Sample.timestamp`), not wall-clock at dispatch.
- `channel` is the resolved data channel (e.g. `"frames/default"`).
- `sensor_name` is extracted from the channel suffix (e.g. `"front"` from `"frames/front"`).
- `twin_uuid` is set by the runtime from `cw.config.twin_uuid`.
- `metadata` carries any extra fields the data layer attached (e.g. frame dimensions, encoding).

---

### Step 2: Hook registry and decorator factory (`hooks.py`)

The hook registry is the core of the worker API. Decorators like `@cw.on_frame(twin_uuid)` register a callback in a global registry. The runtime later reads the registry and creates data-layer subscriptions for each registered hook.

```python
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class HookRegistration:
    """One registered hook."""
    channel: str              # e.g. "frames/default", "depth/default"
    twin_uuid: str            # which twin this hook subscribes to
    callback: Callable        # user function: (sample, ctx) -> None
    hook_type: str            # e.g. "frame", "depth", "audio", "data"
    sensor_name: str = "default"
    options: dict[str, Any] = field(default_factory=dict)  # e.g. {"fps": 15}


class HookRegistry:
    """Collects hook registrations. One per Cyberwave instance."""

    def __init__(self) -> None:
        self._hooks: list[HookRegistration] = []

    @property
    def hooks(self) -> list[HookRegistration]:
        return list(self._hooks)

    def register(self, registration: HookRegistration) -> None:
        self._hooks.append(registration)

    def clear(self) -> None:
        self._hooks.clear()

    # ── Decorator factories ──────────────────────────────────────

    def on_frame(
        self,
        twin_uuid: str,
        *,
        camera: str = "default",
        fps: int | None = None,
    ) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self.register(HookRegistration(
                channel=f"frames/{camera}",
                twin_uuid=twin_uuid,
                callback=fn,
                hook_type="frame",
                sensor_name=camera,
                options={"fps": fps} if fps else {},
            ))
            return fn
        return decorator

    def on_depth(self, twin_uuid: str, *, sensor: str = "default") -> Callable:
        def decorator(fn: Callable) -> Callable:
            self.register(HookRegistration(
                channel=f"depth/{sensor}",
                twin_uuid=twin_uuid,
                callback=fn,
                hook_type="depth",
                sensor_name=sensor,
            ))
            return fn
        return decorator

    def on_audio(self, twin_uuid: str, *, sensor: str = "default") -> Callable:
        def decorator(fn: Callable) -> Callable:
            self.register(HookRegistration(
                channel=f"audio/{sensor}",
                twin_uuid=twin_uuid,
                callback=fn,
                hook_type="audio",
                sensor_name=sensor,
            ))
            return fn
        return decorator

    def on_pointcloud(self, twin_uuid: str, *, sensor: str = "default") -> Callable:
        def decorator(fn: Callable) -> Callable:
            self.register(HookRegistration(
                channel=f"pointcloud/{sensor}",
                twin_uuid=twin_uuid,
                callback=fn,
                hook_type="pointcloud",
                sensor_name=sensor,
            ))
            return fn
        return decorator

    def on_imu(self, twin_uuid: str) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self.register(HookRegistration(
                channel="imu",
                twin_uuid=twin_uuid,
                callback=fn,
                hook_type="imu",
            ))
            return fn
        return decorator

    def on_force_torque(self, twin_uuid: str) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self.register(HookRegistration(
                channel="force_torque",
                twin_uuid=twin_uuid,
                callback=fn,
                hook_type="force_torque",
            ))
            return fn
        return decorator

    def on_joint_states(self, twin_uuid: str) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self.register(HookRegistration(
                channel="joint_states",
                twin_uuid=twin_uuid,
                callback=fn,
                hook_type="joint_states",
            ))
            return fn
        return decorator

    def on_attitude(self, twin_uuid: str) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self.register(HookRegistration(
                channel="attitude",
                twin_uuid=twin_uuid,
                callback=fn,
                hook_type="attitude",
            ))
            return fn
        return decorator

    def on_gps(self, twin_uuid: str) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self.register(HookRegistration(
                channel="gps",
                twin_uuid=twin_uuid,
                callback=fn,
                hook_type="gps",
            ))
            return fn
        return decorator

    def on_end_effector_pose(self, twin_uuid: str) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self.register(HookRegistration(
                channel="end_effector_pose",
                twin_uuid=twin_uuid,
                callback=fn,
                hook_type="end_effector_pose",
            ))
            return fn
        return decorator

    def on_gripper_state(self, twin_uuid: str) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self.register(HookRegistration(
                channel="gripper_state",
                twin_uuid=twin_uuid,
                callback=fn,
                hook_type="gripper_state",
            ))
            return fn
        return decorator

    def on_map(self, twin_uuid: str) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self.register(HookRegistration(
                channel="map",
                twin_uuid=twin_uuid,
                callback=fn,
                hook_type="map",
            ))
            return fn
        return decorator

    def on_battery(self, twin_uuid: str) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self.register(HookRegistration(
                channel="battery",
                twin_uuid=twin_uuid,
                callback=fn,
                hook_type="battery",
            ))
            return fn
        return decorator

    def on_temperature(self, twin_uuid: str) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self.register(HookRegistration(
                channel="temperature",
                twin_uuid=twin_uuid,
                callback=fn,
                hook_type="temperature",
            ))
            return fn
        return decorator

    def on_data(self, twin_uuid: str, channel: str) -> Callable:
        """Generic hook for any custom data channel."""
        def decorator(fn: Callable) -> Callable:
            self.register(HookRegistration(
                channel=channel,
                twin_uuid=twin_uuid,
                callback=fn,
                hook_type="data",
            ))
            return fn
        return decorator
```

**Design notes:**

- Each decorator is a factory: `@cw.on_frame(twin_uuid)` → returns a decorator → returns the original function unchanged. The side-effect is a registration in `_hooks`.
- The registry is stateful but passive — it collects declarations. The runtime (Step 5) reads and activates them.
- `on_frame` supports an extra `camera` keyword (default `"default"`) and optional `fps` for rate limiting.
- `on_depth`, `on_audio`, `on_pointcloud` accept a `sensor` keyword for multi-sensor twins.
- `on_data` is the generic catch-all for custom channels (README: `@cw.on_data(twin_uuid, "custom")`).
- Stateless channels (`imu`, `force_torque`, `joint_states`, `attitude`, `gps`, `end_effector_pose`, `gripper_state`, `map`, `battery`, `temperature`) have no sensor qualifier — they map 1:1 to fixed Zenoh key suffixes.
- The decorated function is returned unmodified (no wrapping) — the hook is just a side-effect registration.

---

### Step 3: Define model output types (`models/types.py`)

Stable output types that downstream logic and generated workers can depend on.

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class Detection:
    """A single object detection from a model prediction."""
    label: str
    confidence: float
    bbox: BoundingBox
    area_ratio: float = 0.0
    mask: Any | None = None      # optional segmentation mask (numpy array)
    keypoints: Any | None = None # optional pose keypoints
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Container for model prediction output."""
    detections: list[Detection] = field(default_factory=list)
    raw: Any | None = None       # backend-specific raw result (e.g. ultralytics Results)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.detections)

    def __len__(self) -> int:
        return len(self.detections)

    def __bool__(self) -> bool:
        return len(self.detections) > 0
```

**Design notes:**

- `Detection` matches the README examples: `det.label`, `det.confidence`, `det.bbox`, `det.area_ratio`, `det.mask`.
- `PredictionResult` is iterable (yields `Detection`s) and truthy when non-empty, matching README usage: `if detections:`, `for det in detections:`.
- `raw` preserves the backend-specific result object for advanced users who need fields the SDK doesn't normalize.
- `area_ratio` is the detection's bounding box area divided by the frame area. Computed by the runtime adapters, default `0.0` if frame dimensions are unknown.

---

### Step 4: Define model runtime abstraction (`models/runtimes/base.py`)

Each ML runtime (Ultralytics, ONNX Runtime, OpenCV, etc.) is wrapped in a common interface. Phase 1 ships `ultralytics` only; others are stubbed.

```python
from abc import ABC, abstractmethod
from typing import Any

from cyberwave.models.types import PredictionResult


class ModelRuntime(ABC):
    """Abstract interface for an ML runtime backend."""

    name: str  # e.g. "ultralytics", "onnxruntime"

    @abstractmethod
    def load(self, model_path: str, *, device: str | None = None, **kwargs: Any) -> Any:
        """Load model weights from a file path. Returns a runtime-specific model handle."""
        ...

    @abstractmethod
    def predict(
        self,
        model_handle: Any,
        input_data: Any,
        *,
        confidence: float = 0.5,
        classes: list[str] | None = None,
        **kwargs: Any,
    ) -> PredictionResult:
        """Run inference. Returns a normalized PredictionResult."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the runtime's dependencies are importable."""
        ...
```

**Design notes:**

- `load()` returns an opaque handle. The `LoadedModel` wrapper (Step 5) holds onto it.
- `predict()` always returns a `PredictionResult` with normalized `Detection` objects — runtime-specific result shapes are adapted inside each runtime implementation.
- `is_available()` checks `import ultralytics` (or equivalent) and returns `False` if the package is missing. This allows the model manager to give a clear error message.

---

### Step 5: Ultralytics runtime implementation (`models/runtimes/ultralytics_rt.py`)

Phase 1 ships with Ultralytics support (YOLOv8, YOLOv11).

```python
from cyberwave.models.runtimes.base import ModelRuntime
from cyberwave.models.types import PredictionResult, Detection, BoundingBox


class UltralyticsRuntime(ModelRuntime):
    name = "ultralytics"

    def is_available(self) -> bool:
        try:
            import ultralytics  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self, model_path, *, device=None, **kwargs):
        from ultralytics import YOLO
        model = YOLO(model_path)
        if device:
            model.to(device)
        return model

    def predict(self, model_handle, input_data, *, confidence=0.5, classes=None, **kwargs):
        results = model_handle(input_data, conf=confidence, verbose=False)
        detections = []
        for result in results:
            frame_area = result.orig_shape[0] * result.orig_shape[1] if result.orig_shape else 1
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    label = result.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    if classes and label not in classes:
                        continue
                    bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                    detections.append(Detection(
                        label=label,
                        confidence=conf,
                        bbox=bbox,
                        area_ratio=bbox.area / frame_area if frame_area else 0.0,
                    ))
        return PredictionResult(detections=detections, raw=results)
```

**Design notes:**

- `classes` filtering is applied post-prediction. Ultralytics supports a `classes` arg too (as integer indices), but filtering by label name here is more user-friendly and consistent across runtimes.
- `area_ratio` is computed using `result.orig_shape` (H, W). This matches README usage: `if det.area_ratio > 0.3`.
- The raw `ultralytics.Results` list is preserved in `PredictionResult.raw` for power users.

---

### Step 6: Runtime registry (`models/runtimes/__init__.py`)

Maps runtime names to implementations. Extensible for future runtimes (ONNX, OpenCV, TFLite, etc.).

```python
_RUNTIME_REGISTRY: dict[str, type[ModelRuntime]] = {}

def register_runtime(runtime_class: type[ModelRuntime]) -> None:
    _RUNTIME_REGISTRY[runtime_class.name] = runtime_class

def get_runtime(name: str) -> ModelRuntime:
    if name not in _RUNTIME_REGISTRY:
        available = ", ".join(sorted(_RUNTIME_REGISTRY.keys())) or "(none)"
        raise ValueError(
            f"Unknown model runtime '{name}'. Available: {available}. "
            f"Install the required package and ensure it is registered."
        )
    cls = _RUNTIME_REGISTRY[name]
    instance = cls()
    if not instance.is_available():
        raise ImportError(
            f"Runtime '{name}' is registered but its dependencies are not installed. "
            f"Install with: pip install {name}"
        )
    return instance

def available_runtimes() -> list[str]:
    return [name for name, cls in _RUNTIME_REGISTRY.items() if cls().is_available()]

# Auto-register built-in runtimes
from cyberwave.models.runtimes.ultralytics_rt import UltralyticsRuntime
register_runtime(UltralyticsRuntime)
```

---

### Step 7: `LoadedModel` wrapper (`models/loaded_model.py`)

The user-facing object returned by `cw.models.load()`. Wraps a runtime handle with a clean `.predict()` API.

```python
from typing import Any
from cyberwave.models.runtimes.base import ModelRuntime
from cyberwave.models.types import PredictionResult


class LoadedModel:
    """A loaded ML model ready for inference."""

    def __init__(
        self,
        *,
        name: str,
        runtime: ModelRuntime,
        model_handle: Any,
        device: str = "cpu",
        model_path: str = "",
    ) -> None:
        self._name = name
        self._runtime = runtime
        self._model_handle = model_handle
        self._device = device
        self._model_path = model_path

    @property
    def name(self) -> str:
        return self._name

    @property
    def runtime(self) -> str:
        return self._runtime.name

    @property
    def device(self) -> str:
        return self._device

    def predict(
        self,
        input_data: Any,
        *,
        confidence: float = 0.5,
        classes: list[str] | None = None,
        **kwargs: Any,
    ) -> PredictionResult:
        """Run inference on the input data."""
        return self._runtime.predict(
            self._model_handle,
            input_data,
            confidence=confidence,
            classes=classes,
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"LoadedModel(name={self._name!r}, runtime={self.runtime!r}, device={self._device!r})"
```

**Design notes:**

- `model.name`, `model.runtime`, `model.device` match README: `print(model.name, model.runtime, model.device)`.
- `predict()` delegates to the runtime, passing through `confidence` and `classes`.
- The `model_handle` is opaque to the user. Power users can access it via `model._model_handle` (conventional underscore-private).

---

### Step 8: Model manager (`models/manager.py`)

Owns model resolution, caching, and loading. Exposed as `cw.models` on the `Cyberwave` client.

```python
import os
import logging
from pathlib import Path
from typing import Any

from cyberwave.models.loaded_model import LoadedModel
from cyberwave.models.runtimes import get_runtime, available_runtimes

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = "/app/models"
FALLBACK_MODEL_DIR = os.path.expanduser("~/.cyberwave/models")


class ModelManager:
    """Manages model loading, caching, and runtime selection."""

    def __init__(
        self,
        *,
        model_dir: str | None = None,
        default_device: str | None = None,
    ) -> None:
        dir_from_env = os.environ.get("CYBERWAVE_MODEL_DIR")
        if model_dir:
            self._model_dir = Path(model_dir)
        elif dir_from_env:
            self._model_dir = Path(dir_from_env)
        elif Path(DEFAULT_MODEL_DIR).is_dir():
            self._model_dir = Path(DEFAULT_MODEL_DIR)
        else:
            self._model_dir = Path(FALLBACK_MODEL_DIR)

        self._default_device = default_device or os.environ.get("CYBERWAVE_MODEL_DEVICE")
        self._loaded: dict[str, LoadedModel] = {}

    def load(
        self,
        model_id: str,
        *,
        runtime: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> LoadedModel:
        """Load a model by catalog ID.

        If the model is already loaded, returns the cached instance.
        The runtime is auto-detected from the model file extension if not specified.
        """
        cache_key = f"{model_id}:{runtime or 'auto'}:{device or 'auto'}"
        if cache_key in self._loaded:
            return self._loaded[cache_key]

        resolved_runtime = runtime or self._detect_runtime(model_id)
        rt = get_runtime(resolved_runtime)
        model_path = self._resolve_model_path(model_id, resolved_runtime)
        device = device or self._default_device or self._detect_device()

        logger.info("Loading model '%s' with runtime '%s' on device '%s'", model_id, resolved_runtime, device)
        handle = rt.load(str(model_path), device=device, **kwargs)

        loaded = LoadedModel(
            name=model_id,
            runtime=rt,
            model_handle=handle,
            device=device,
            model_path=str(model_path),
        )
        self._loaded[cache_key] = loaded
        return loaded

    def load_from_file(
        self,
        path: str,
        *,
        runtime: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> LoadedModel:
        """Load a model directly from a file path."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        resolved_runtime = runtime or self._detect_runtime_from_extension(p.suffix)
        rt = get_runtime(resolved_runtime)
        device = device or self._default_device or self._detect_device()

        handle = rt.load(str(p), device=device, **kwargs)
        return LoadedModel(
            name=p.stem,
            runtime=rt,
            model_handle=handle,
            device=device,
            model_path=str(p),
        )

    def _resolve_model_path(self, model_id: str, runtime: str) -> Path:
        """Resolve a catalog model ID to a local file path."""
        # Look for exact file match first
        for ext in self._runtime_extensions(runtime):
            candidate = self._model_dir / f"{model_id}{ext}"
            if candidate.exists():
                return candidate

        # Look inside a model subdirectory
        model_subdir = self._model_dir / model_id
        if model_subdir.is_dir():
            for ext in self._runtime_extensions(runtime):
                for f in model_subdir.iterdir():
                    if f.suffix == ext:
                        return f

        # For Ultralytics, the model_id itself is a valid model name (downloads from hub)
        if runtime == "ultralytics":
            return Path(model_id)

        raise FileNotFoundError(
            f"Model '{model_id}' not found in {self._model_dir}. "
            f"Ensure edge core has downloaded the model weights, or use load_from_file()."
        )

    @staticmethod
    def _detect_runtime(model_id: str) -> str:
        """Heuristic: detect runtime from model ID. Raises if unknown."""
        model_id_lower = model_id.lower()
        if any(k in model_id_lower for k in ("yolo", "yolov8", "yolov11")):
            return "ultralytics"
        if any(k in model_id_lower for k in ("background-subtraction", "haar", "cascade")):
            return "opencv"
        raise ValueError(
            f"Cannot auto-detect runtime for model '{model_id}'. "
            f"Pass runtime= explicitly, e.g.: cw.models.load('{model_id}', runtime='ultralytics')"
        )

    @staticmethod
    def _detect_runtime_from_extension(ext: str) -> str:
        """Map file extension to runtime."""
        mapping = {
            ".pt": "ultralytics",
            ".onnx": "onnxruntime",
            ".tflite": "tflite",
            ".xml": "opencv",
            ".engine": "tensorrt",
        }
        return mapping.get(ext.lower(), "ultralytics")

    @staticmethod
    def _runtime_extensions(runtime: str) -> list[str]:
        mapping = {
            "ultralytics": [".pt", ".onnx", ".engine"],
            "onnxruntime": [".onnx"],
            "opencv": [".xml", ".caffemodel"],
            "tflite": [".tflite"],
            "torch": [".pt", ".pth"],
            "tensorrt": [".engine", ".trt"],
        }
        return mapping.get(runtime, [".pt"])

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
        except ImportError:
            pass
        return "cpu"
```

**Design notes:**

- `load()` caches by `(model_id, runtime, device)` to prevent duplicate loads. README example loads at module level: `model = cw.models.load("yolov8n")` — called once per worker import, cached forever.
- `_resolve_model_path()` searches `CYBERWAVE_MODEL_DIR` (container: `/app/models/`, host: `~/.cyberwave/models/`). For Ultralytics, passes the model ID directly (Ultralytics auto-downloads from hub).
- `_detect_runtime()` raises `ValueError` for unknown model IDs instead of silently defaulting to a runtime. Users must pass `runtime=` explicitly for non-standard models. Future phases will query the backend catalog for the model's `edge_runtime` metadata field.
- `_detect_device()` uses `torch.cuda.is_available()` if torch is installed, else defaults to `"cpu"`.

---

### Step 9: Worker module loader (`loader.py`)

Imports worker `.py` files from a directory and returns the hooks they registered.

```python
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def load_workers(
    workers_dir: str | Path,
    *,
    cw_module: object,
) -> int:
    """Import all .py files from workers_dir.

    Worker modules use the `cw` variable (the Cyberwave client instance
    injected into their module namespace). This function:
      1. Injects `cw_module` as `builtins.cw` so worker code can use
         bare `cw.on_frame(...)` without an import.
      2. Imports each .py file as a standalone module.
      3. Returns the number of successfully loaded modules.

    Args:
        workers_dir: Path to the directory containing worker .py files.
        cw_module: The Cyberwave client instance to inject as `cw`.

    Returns:
        Number of worker modules successfully loaded.
    """
    import builtins
    workers_dir = Path(workers_dir)

    if not workers_dir.is_dir():
        logger.warning("Workers directory does not exist: %s", workers_dir)
        return 0

    builtins.cw = cw_module  # type: ignore[attr-defined]

    loaded = 0
    for py_file in sorted(workers_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        module_name = f"cyberwave_worker_{py_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                logger.warning("Cannot load worker module: %s", py_file)
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            loaded += 1
            logger.info("Loaded worker: %s", py_file.name)
        except Exception:
            logger.exception("Failed to load worker: %s", py_file.name)

    return loaded
```

**Design notes:**

- Workers use bare `cw` (no import). The loader injects the client as `builtins.cw`. This matches the README pattern where workers write `model = cw.models.load("yolov8n")` at module level.
- Files starting with `_` are skipped (conventional Python private modules, `__init__.py`, etc.).
- Each module is imported into `sys.modules` with a `cyberwave_worker_` prefix to avoid name collisions.
- Import failures are logged but don't crash the runtime — one broken worker shouldn't take down the whole container.
- Worker modules are sorted alphabetically for deterministic load order.

---

### Step 10: Worker runtime entrypoint (`runtime.py`)

The runtime is the process-level entry point. It creates the `Cyberwave` client, loads workers, wires hooks to the data layer, and runs the dispatch loop. Worker `.py` files never call this — only the container entrypoint does.

```python
import logging
import os
import signal
import threading
from pathlib import Path

from cyberwave.workers.hooks import HookRegistry
from cyberwave.workers.context import HookContext
from cyberwave.workers.loader import load_workers

logger = logging.getLogger(__name__)

DEFAULT_WORKERS_DIR = "/app/workers"
FALLBACK_WORKERS_DIR = os.path.expanduser("~/.cyberwave/workers")


class WorkerRuntime:
    """Manages the lifecycle of a worker process."""

    def __init__(self, cw_client: "Cyberwave") -> None:
        self._cw = cw_client
        self._registry: HookRegistry = cw_client._hook_registry
        self._subscriptions: list = []
        self._stop_event = threading.Event()

    def load(self, workers_dir: str | Path | None = None) -> int:
        """Load worker modules from disk."""
        if workers_dir is None:
            env_dir = os.environ.get("CYBERWAVE_WORKERS_DIR")
            if env_dir:
                workers_dir = env_dir
            elif Path(DEFAULT_WORKERS_DIR).is_dir():
                workers_dir = DEFAULT_WORKERS_DIR
            else:
                workers_dir = FALLBACK_WORKERS_DIR

        return load_workers(workers_dir, cw_module=self._cw)

    def start(self) -> None:
        """Wire registered hooks to data-layer subscriptions and start dispatch."""
        for hook in self._registry.hooks:
            full_channel = f"{hook.twin_uuid}/data/{hook.channel}"
            self._subscribe_hook(full_channel, hook)
            logger.info(
                "Activated hook: @cw.on_%s(%s) → %s",
                hook.hook_type,
                hook.twin_uuid[:8] + "...",
                hook.callback.__name__,
            )
        logger.info("Worker runtime started with %d hooks", len(self._registry.hooks))

    def run(self) -> None:
        """Block until stop() is called or a signal is received."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logger.info("Worker runtime running. Press Ctrl+C to stop.")
        self._stop_event.wait()

    def stop(self) -> None:
        """Stop the runtime: unsubscribe all hooks and release resources."""
        logger.info("Stopping worker runtime...")
        for sub in self._subscriptions:
            try:
                sub.close()
            except Exception:
                logger.exception("Error closing subscription")
        self._subscriptions.clear()
        self._stop_event.set()

    def _subscribe_hook(self, channel: str, hook) -> None:
        """Create a data-layer subscription that dispatches to the hook callback."""
        def on_sample(sample):
            parts = hook.channel.rsplit("/", 1)
            sensor_name = parts[1] if len(parts) > 1 else "default"
            ctx = HookContext(
                timestamp=sample.timestamp,
                channel=hook.channel,
                sensor_name=sensor_name,
                twin_uuid=hook.twin_uuid,
                metadata=sample.metadata or {},
            )
            try:
                hook.callback(sample.payload, ctx)
            except Exception:
                logger.exception(
                    "Error in hook %s for channel %s",
                    hook.callback.__name__,
                    hook.channel,
                )

        if hasattr(self._cw, 'data') and self._cw.data is not None:
            sub = self._cw.data.subscribe(channel, on_sample)
            self._subscriptions.append(sub)
        else:
            logger.warning(
                "Data backend not available. Hook '%s' on channel '%s' will not receive samples. "
                "Set CYBERWAVE_DATA_BACKEND to enable the data layer.",
                hook.callback.__name__,
                channel,
            )

    def _signal_handler(self, signum, frame) -> None:
        logger.info("Received signal %s, stopping...", signum)
        self.stop()
```

**Design notes:**

- `run()` blocks the main thread using `threading.Event.wait()`. The data layer's subscriptions fire callbacks on their own threads (same model as paho-mqtt). Worker callbacks execute on data-layer threads — no extra thread pool.
- `stop()` closes all subscriptions and sets the stop event. Triggered by SIGINT/SIGTERM for clean container shutdown.
- The hook dispatch wraps each callback in a `try/except` so one failing hook doesn't crash the runtime.
- `_subscribe_hook` constructs the full Zenoh key expression (`{twin_uuid}/data/{channel}`) and delegates to `cw.data.subscribe()`. If the data backend is not configured (i.e. CYB-1544 not merged yet), it logs a warning instead of crashing — useful for testing workers without a live data bus.
- **Raw bytes boundary:** the callback receives `sample.payload` as raw `bytes`. Deserialization to numpy arrays, JSON, or typed structs is the wire format layer's job (CYB-1553). In Phase 1, workers that need decoded frames must handle deserialization themselves. Once CYB-1553 lands, the dispatch pipeline will decode before calling the user callback.
- The runtime deliberately does **not** own the `Cyberwave()` construction — it receives an already-configured client. This supports both the container entrypoint (which builds the client from env vars) and test harnesses (which inject a mock).

---

### Step 11: Integrate with `Cyberwave` client

Wire the hook registry, model manager, and runtime into the existing `Cyberwave` class.

**Changes to `client.py`:**

```python
class Cyberwave:
    def __init__(self, ...):
        # ... existing init ...
        self._hook_registry = HookRegistry()
        self.models = ModelManager()

    # Delegate hook decorators to registry
    @property
    def on_frame(self):
        return self._hook_registry.on_frame
    @property
    def on_depth(self):
        return self._hook_registry.on_depth
    @property
    def on_audio(self):
        return self._hook_registry.on_audio
    @property
    def on_pointcloud(self):
        return self._hook_registry.on_pointcloud
    @property
    def on_imu(self):
        return self._hook_registry.on_imu
    @property
    def on_force_torque(self):
        return self._hook_registry.on_force_torque
    @property
    def on_joint_states(self):
        return self._hook_registry.on_joint_states
    @property
    def on_attitude(self):
        return self._hook_registry.on_attitude
    @property
    def on_gps(self):
        return self._hook_registry.on_gps
    @property
    def on_end_effector_pose(self):
        return self._hook_registry.on_end_effector_pose
    @property
    def on_gripper_state(self):
        return self._hook_registry.on_gripper_state
    @property
    def on_map(self):
        return self._hook_registry.on_map
    @property
    def on_battery(self):
        return self._hook_registry.on_battery
    @property
    def on_temperature(self):
        return self._hook_registry.on_temperature
    @property
    def on_data(self):
        return self._hook_registry.on_data

    def publish_event(
        self,
        twin_uuid: str,
        event_type: str,
        data: dict,
        *,
        source: str = "edge_node",
    ) -> None:
        """Publish a business event via MQTT.

        Payload shape must match what the backend's mqtt_consumer.handle_business_event
        expects: {"event_type": ..., "source": ..., "data": ..., "timestamp": ...}.
        This mirrors the existing BaseEdgeNode.publish_event() implementation.
        """
        import time
        self.mqtt.publish(
            f"{self.mqtt.topic_prefix}cyberwave/twin/{twin_uuid}/event",
            {
                "event_type": event_type,
                "source": source,
                "data": data,
                "timestamp": time.time(),
            },
        )

    def run(self, workers_dir: str | None = None) -> None:
        """Start the worker runtime: load workers, activate hooks, block until stopped."""
        runtime = WorkerRuntime(self)
        runtime.load(workers_dir)
        runtime.start()
        runtime.run()
```

**Changes to `config.py`:**

```python
@dataclass
class CyberwaveConfig:
    # ... existing fields ...
    twin_uuid: str | None = None  # NEW — set by the worker container via CYBERWAVE_TWIN_UUID

    def __post_init__(self):
        # ... existing env var loading ...
        if not self.twin_uuid:
            self.twin_uuid = os.getenv("CYBERWAVE_TWIN_UUID")
```

This adds `cw.config.twin_uuid`, matching README usage: `twin_uuid = cw.config.twin_uuid`.

**Note on `on_*` property boilerplate:** 15 explicit `@property` delegations is verbose. An alternative is `__getattr__` forwarding `on_*` to the registry — less code but worse IDE autocomplete and type checking. The explicit approach is chosen for discoverability: users see all hook decorators in IDE autocompletion on `cw.`, and type checkers can validate usage. If the list grows unwieldy, a `__getattr__` fallback can be added later without breaking the API.

---

### Step 12: Package `__init__.py` re-exports

**`cyberwave/workers/__init__.py`:**

```python
from .hooks import HookRegistry, HookRegistration
from .context import HookContext
from .runtime import WorkerRuntime
from .loader import load_workers

__all__ = [
    "HookRegistry",
    "HookRegistration",
    "HookContext",
    "WorkerRuntime",
    "load_workers",
]
```

**`cyberwave/models/__init__.py`:**

```python
from .manager import ModelManager
from .loaded_model import LoadedModel
from .types import Detection, BoundingBox, PredictionResult

__all__ = [
    "ModelManager",
    "LoadedModel",
    "Detection",
    "BoundingBox",
    "PredictionResult",
]
```

**Update `cyberwave/__init__.py`:**

Add re-exports so users can do `from cyberwave import Detection, LoadedModel` etc.

```python
# Worker API
from .workers import HookContext

# Model API
from .models import ModelManager, LoadedModel, Detection, BoundingBox, PredictionResult
```

---

### Step 13: Example worker (`examples/edge_worker_detect_people.py`)

A concrete example demonstrating the full worker pattern. This file is **not** loaded by the runtime (it's in `examples/`, not `/app/workers/`) — it's documentation-as-code.

```python
"""
Example edge worker: detect people near the robot and publish events.

This file demonstrates the worker module pattern. In production, it would
live at /app/workers/detect_people.py inside the worker container.

Prerequisites:
  - Worker runtime injects `cw` as a builtin (no import needed)
  - Ultralytics installed: pip install cyberwave[ml]
  - Model weights available in the model cache

Usage:
  This file is loaded by the worker runtime, not run directly.
  The runtime calls cw.run() after importing all worker modules.
"""

# cw is injected by the worker runtime — no import needed.
# For IDE support, uncomment the following:
# from cyberwave import Cyberwave; cw: Cyberwave

model = cw.models.load("yolov8n")
twin_uuid = cw.config.twin_uuid


@cw.on_frame(twin_uuid, camera="front")
def detect_people(frame, ctx):
    """Called for every new frame from the front camera."""
    results = model.predict(frame, classes=["person"], confidence=0.5)

    for det in results:
        if det.area_ratio > 0.3:
            cw.publish_event(twin_uuid, "person_too_close", {
                "distance_estimate": "near",
                "detections": len(results),
                "model": "yolov8n",
                "frame_ts": ctx.timestamp,
            })
```

---

### Step 14: Update `pyproject.toml`

Add optional ML runtime extras:

```toml
[tool.poetry.dependencies]
# ... existing deps ...
ultralytics = { version = ">=8.0.0", optional = true }

[tool.poetry.extras]
camera = ["aiortc", "av", "opencv-python"]
realsense = ["aiortc", "av", "opencv-python", "pyrealsense2"]
zenoh = ["eclipse-zenoh"]
ml = ["ultralytics"]          # NEW — ML runtime dependencies for edge workers
```

The `ModelManager` and hook system work without `ultralytics` installed — they only fail at `load()` time if the requested runtime is missing, with a clear error message.

---

### Step 14: Tests

#### 14a. Hook registration tests (`test_hook_registration.py`)

- `test_on_frame_registers_hook` — `@cw.on_frame(uuid)` adds a `HookRegistration` with `channel="frames/default"`
- `test_on_frame_custom_camera` — `@cw.on_frame(uuid, camera="front")` → `channel="frames/front"`
- `test_on_depth_registers_hook` — `@cw.on_depth(uuid)` → `channel="depth/default"`
- `test_on_data_custom_channel` — `@cw.on_data(uuid, "custom")` → `channel="custom"`
- `test_all_decorators_register` — each of the 15 decorators creates a valid registration
- `test_decorator_returns_original_function` — decorated function is unchanged
- `test_multiple_hooks_same_channel` — two callbacks on the same channel both register
- `test_registry_clear` — `clear()` removes all registrations

#### 14b. Hook dispatch tests (`test_hook_dispatch.py`)

- `test_dispatch_invokes_callback` — publish a sample → registered callback fires with correct args
- `test_dispatch_provides_context` — `ctx.timestamp`, `ctx.channel`, `ctx.twin_uuid` populated
- `test_dispatch_error_does_not_crash` — callback raises → runtime logs error, continues
- `test_multiple_hooks_all_fire` — two hooks on different channels both receive their samples
- `test_hook_ordering_deterministic` — hooks fire in registration order

#### 14c. Model types tests (`test_model_types.py`)

- `test_bounding_box_properties` — `width`, `height`, `area` computed correctly
- `test_detection_defaults` — `area_ratio=0.0`, `mask=None`
- `test_prediction_result_iterable` — `for det in result:` works
- `test_prediction_result_bool` — empty → `False`, non-empty → `True`
- `test_prediction_result_len` — `len(result)` matches detection count

#### 14d. Model manager tests (`test_model_manager.py`)

- `test_load_caches_model` — second `load()` returns same instance
- `test_load_from_file_not_found` — `FileNotFoundError` with clear message
- `test_detect_runtime_yolo` — `"yolov8n"` → `"ultralytics"`
- `test_detect_runtime_from_extension` — `.onnx` → `"onnxruntime"`, `.pt` → `"ultralytics"`
- `test_model_dir_from_env` — `CYBERWAVE_MODEL_DIR` respected
- `test_device_detection` — returns `"cpu"` when torch unavailable

#### 14e. Runtime boundary test (`test_runtime_boundary.py`)

**Critical acceptance criterion:** importing a worker module must NOT start the event loop.

- `test_import_worker_does_not_start_loop` — write a temp `.py` worker file, import it, verify no subscriptions created and no threads spawned
- `test_cw_run_not_in_worker` — worker module that calls `cw.run()` → runtime logs a warning or raises

#### 14f. Worker loader tests (`test_worker_loader.py`)

- `test_load_workers_finds_py_files` — creates temp dir with `.py` files, `load_workers()` imports them
- `test_load_workers_skips_underscore` — `_helper.py` is not loaded
- `test_load_workers_bad_file_continues` — one broken `.py` → logged, others still load
- `test_load_workers_missing_dir` — missing directory → returns 0, logs warning
- `test_cw_injected_as_builtin` — worker code can use bare `cw` variable

#### 14g. `publish_event` tests (`test_publish_event.py`)

- `test_publish_event_correct_topic` — publishes to `{prefix}cyberwave/twin/{uuid}/event`
- `test_publish_event_payload_shape` — payload contains `event_type`, `source`, `data`, `timestamp` as separate keys (not spread)
- `test_publish_event_default_source` — default `source` is `"edge_node"`
- `test_publish_event_custom_source` — `source="sensor"` overrides the default

#### 14h. Integration test (`test_worker_integration.py`)

End-to-end with a mock data backend:

1. Create a `Cyberwave` instance with a mock data backend
2. Load a worker module that registers `@cw.on_frame` and records calls
3. Start the runtime
4. Publish a sample to the mock backend
5. Assert the hook callback was invoked with the correct payload and context
6. Stop the runtime

This validates the full pipeline: registration → subscription → dispatch → callback.

---

## Dependency Graph (within CYB-1498 epic)

```
CYB-1544: Data Layer
  DataBackend ABC + ZenohBackend + FilesystemBackend
    │
    ▼
CYB-1545 (this issue): Worker API
  Hook decorators + Model loader + Worker runtime
    │
    ├──▶ CYB-1546: Edge Core worker container + model manager
    │     Uses WorkerRuntime as container entrypoint
    │     Uses ModelManager to pre-download weights
    │
    ├──▶ CYB-1548: Worker deployment (CLI + workflow codegen)
    │     Generated workers use @cw.on_frame, cw.models.load(), cw.publish_event()
    │     CLI uses load_workers() to validate worker files
    │
    └──▶ CYB-1549: Cloud worker runtime evolution
          Reuses hook decorators (@cw.on_inference_request)
          Reuses ModelManager (cw.models.load())
          Adds cloud-specific hooks and transport
```

---

## Boundaries and Out-of-Scope

| In scope (this issue) | Out of scope (later issues) |
|---|---|
| Hook decorator registry (`@cw.on_frame`, etc.) | Edge Core container orchestration (CYB-1546) |
| `(sample, ctx)` callback signature and `HookContext` | Workflow code generation (CYB-1548) |
| `cw.models.load(model_id, runtime=None)` → `LoadedModel` | Driver Zenoh publishing migration (CYB-1547) |
| `LoadedModel.predict()` → `PredictionResult`/`Detection` | Advanced runtimes beyond Ultralytics (CYB-1551 Phase 3) |
| Ultralytics runtime backend (Phase 1 only) | `cw.models.finetune()` / training API (CYB-1549) |
| Worker runtime entrypoint (`cw.run()`) | Cloud-specific hooks (`@cw.on_inference_request`) (CYB-1549) |
| Worker module loader (`load_workers()`) | Hot-reload / file watching (CYB-1546) |
| `cw.config.twin_uuid` from env | `cw.data.record()` / `cw.data.replay()` (CYB-1555) |
| `cw.publish_event()` surfaced on client | Wire format / SDK header (CYB-1553) |

---

## Decision Record: Hook dispatch on data-layer threads (no dedicated thread pool)

### Decision

Hook callbacks execute directly on the data backend's subscriber threads. Do **not** introduce a separate dispatch thread pool in this issue.

### Context

When the data backend (Zenoh or filesystem) calls the subscriber callback, the SDK can either:

1. **Invoke the user callback directly** on the subscriber thread (Zenoh internal thread or filesystem polling thread).
2. **Queue the callback to a dedicated worker thread pool** for isolation.

### Arguments considered

**For a dedicated thread pool:**

- Isolates user callback execution from the transport layer — a slow callback doesn't block Zenoh's internal thread.
- More predictable concurrency model for users (fixed pool size, backpressure).
- Aligns with paho-mqtt's `loop_start()` model where callbacks run on a single thread.

**Against (decisive):**

1. **Zenoh already handles this.** Zenoh's subscriber model delivers samples on its own managed threads. The subscriber thread is not shared with the session's I/O — blocking it doesn't block other subscriptions.
2. **Thread pool adds latency.** An extra queue hop adds measurable overhead for high-frequency streams (1kHz+ IMU data). The zero-copy shared-memory path loses its latency advantage if we bounce through a pool.
3. **Complexity budget.** A thread pool needs sizing, backpressure policy (`"latest"` vs `"fifo"` already handled at the data layer), and lifecycle management. This is premature — users can wrap their own callbacks in a pool if needed.
4. **Filesystem backend already has a polling thread.** Adding another pool on top doubles threading complexity for the fallback path.
5. **Consistent with paho-mqtt pattern.** The existing SDK's MQTT callbacks (`on_message`) run on paho's network thread. Users are familiar with this model.

### Revisit condition

If profiling shows that user callbacks block Zenoh's internal threads and cause sample drops, introduce a single-threaded dispatch queue (not a pool) as an opt-in at the `WorkerRuntime` level. This can be added without changing the hook decorator API.

### Status

**Accepted.** Callbacks run on data-layer threads. Revisit if Zenoh subscriber thread contention is observed.

---

## Decision Record: `builtins.cw` injection for worker modules

### Decision

The worker module loader injects the `Cyberwave` client instance as `builtins.cw` so that worker `.py` files can use a bare `cw` variable without imports.

### Context

README examples show workers using `cw.models.load(...)`, `@cw.on_frame(...)`, `cw.publish_event(...)`, and `cw.config.twin_uuid` without any `from cyberwave import ...` statement. This is a deliberate design choice for developer ergonomics.

### Alternatives considered

| Approach | Pros | Cons |
|----------|------|------|
| `builtins.cw` injection (chosen) | Matches README; zero boilerplate; works for module-level code like `model = cw.models.load(...)` | Implicit global; type checkers can't see it; IDE autocomplete needs a stub |
| `from cyberwave import cw` (module-level singleton) | Explicit import; type-checkable | Module-level `cw.models.load()` runs at import time before the client is configured |
| Pass `cw` as argument to a `setup(cw)` function | Clean dependency injection | Breaks the "hooks-only" pattern; requires wrapping all worker code in a function |

### Mitigation for IDE support

Ship a `cw.pyi` type stub that declares the `cw` variable with type `Cyberwave`. Workers can optionally add `from cyberwave import Cyberwave; cw: Cyberwave` as a type annotation (ignored at runtime, helps IDE).

### Status

**Accepted.** Matches the README contract exactly. Type stub planned as a follow-up.

---

## Decision Record: Actuator commands stay on MQTT (no Zenoh command path for now)

### Decision

Worker-to-driver actuator commands use `cw.mqtt.publish()` over the existing MQTT command topics. Do **not** introduce Zenoh-based command channels in this issue.

### Context

The Zenoh data bus is asymmetric by design: **sensor data flows from drivers to workers** (high-frequency, local-only, zero-copy), while **actuator commands flow from workers (or cloud) to drivers** via MQTT. Workers have two output paths:

| Output | Transport | Destination |
|--------|-----------|-------------|
| `cw.publish_event(twin_uuid, event_type, data)` | MQTT | Cloud backend → workflows, alerts |
| `cw.mqtt.publish(topic, payload)` | MQTT | Driver containers → robot hardware |

README examples confirm this pattern — e.g. a collision-detection worker publishes both an event and a motion stop command over MQTT.

### Arguments for keeping commands on MQTT

1. **Commands need cloud visibility.** The backend logs commands, workflows depend on command state, the frontend shows command history. MQTT already delivers to both local drivers and the cloud broker simultaneously.
2. **Commands are low-frequency.** A "stop" command fires once. A motion trajectory is maybe 10–100 Hz. MQTT's ~1ms local latency is fine for these rates.
3. **Existing driver infrastructure.** Every driver (ROS bridge `command_handler.py`, UGV plugin, etc.) already subscribes to MQTT command topics and has handler registries. Switching to Zenoh would require rewriting all drivers for no user-visible benefit.
4. **Reliability semantics.** MQTT QoS 1/2 provides delivery acknowledgment. For safety-critical commands (emergency stop), you want confirmation — MQTT's request/response pattern handles this. Zenoh `put()` is fire-and-forget by default.

### Future consideration: Zenoh for tight local control loops

There is one scenario where MQTT command latency could matter — a **tight local control loop** where a worker detects a collision risk and needs to stop the robot within single-digit milliseconds. MQTT through a local Mosquitto broker adds ~1–2ms; Zenoh shared memory would be ~10 microseconds.

The README does not spec Zenoh command channels today. If this becomes a requirement (e.g. real-time visual servoing where a worker adjusts joint velocities at 1kHz), the architecture supports adding it: drivers could subscribe to Zenoh command keys alongside their MQTT subscriptions, and the SDK could expose a `cw.data.publish("commands/joint_velocities", payload)` path. This would be an additive change — the MQTT command path remains for cloud-visible commands, Zenoh adds a local fast path for latency-critical actuation.

### Status

**Accepted.** Commands stay on MQTT. Revisit if a concrete use case requires sub-millisecond local command latency.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| CYB-1544 data layer not merged yet | Hook dispatch has nothing to subscribe to | Hooks register without error; `_subscribe_hook` logs a warning if `cw.data` is unavailable. Model loading and hook registration work standalone. |
| Ultralytics API changes across versions | `predict()` adapter breaks | Pin `>=8.0.0`; wrap result parsing in `try/except`; preserve `raw` result for fallback |
| `builtins.cw` injection conflicts with user code | Name collision if user has a `cw` variable | Extremely unlikely given the context. Document the convention. The loader could also use a more specific name (e.g. `builtins._cyberwave_cw`) but this would break the README contract. |
| Worker module import side-effects | Slow module-level `cw.models.load()` blocks the loader | Expected and acceptable — model loading is deliberately at import time (pre-warm). Timeout protection can be added later. |
| Thread safety of `HookRegistry` | Concurrent decorator calls from multi-threaded module loading | Module loading is sequential (single `load_workers()` call). Registry is append-only during load phase. No lock needed. |
| `predict()` output schema stability | Downstream codegen depends on `Detection` fields | `Detection` and `PredictionResult` are dataclasses with explicit fields — stable contract. `raw` field handles runtime-specific extensions. |

---

## Implementation Order

1. `models/types.py` — no dependencies, used by everything else
2. `workers/context.py` — no dependencies, used by hooks and runtime
3. `workers/hooks.py` — `HookRegistry`, `HookRegistration`, decorator factories
4. `models/runtimes/base.py` — `ModelRuntime` ABC
5. `models/runtimes/ultralytics_rt.py` — Ultralytics implementation
6. `models/runtimes/__init__.py` — runtime registry
7. `models/loaded_model.py` — `LoadedModel` wrapper
8. `models/manager.py` — `ModelManager`
9. `models/__init__.py` — re-exports
10. `workers/loader.py` — module loader
11. `workers/runtime.py` — `WorkerRuntime`
12. `workers/__init__.py` — re-exports
13. `client.py` changes — integrate hooks, models, `publish_event()`, `run()`
14. `config.py` changes — add `twin_uuid`
15. `cyberwave/__init__.py` changes — re-exports
16. `pyproject.toml` update — add `ml` extra
17. `examples/edge_worker_detect_people.py` — example worker
18. Tests — in order: types → hooks → model manager → publish_event → runtime boundary → loader → integration
