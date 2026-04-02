<p align="center">
  <a href="https://cyberwave.com">
    <img src="https://cyberwave.com/cyberwave-logo-black.svg" alt="Cyberwave logo" width="240" />
  </a>
</p>

# Cyberwave Python SDK

This module is part of **Cyberwave: Making the physical world programmable**.

The official Python SDK for Cyberwave. Create, control, and simulate robotics with ease.

[![License](https://img.shields.io/badge/License-MIT-orange.svg)](https://github.com/cyberwave-os/cyberwave-python/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-docs.cyberwave.com-orange)](https://docs.cyberwave.com)
[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label&color=orange)](https://discord.gg/dfGhNrawyF)
[![PyPI version](https://img.shields.io/pypi/v/cyberwave.svg)](https://pypi.org/project/cyberwave/)
[![PyPI Python versions](https://img.shields.io/pypi/pyversions/cyberwave.svg)](https://pypi.org/project/cyberwave/)
[![Build](https://github.com/cyberwave-os/cyberwave-python/actions/workflows/test.yml/badge.svg)](https://github.com/cyberwave-os/cyberwave-python/actions/workflows/test.yml)
[![Python 3.10](https://img.shields.io/github/actions/workflow/status/cyberwave-os/cyberwave-python/test.yml?label=Python%203.10&logo=python&branch=main)](https://github.com/cyberwave-os/cyberwave-python/actions/workflows/test.yml)
[![Python 3.11](https://img.shields.io/github/actions/workflow/status/cyberwave-os/cyberwave-python/test.yml?label=Python%203.11&logo=python&branch=main)](https://github.com/cyberwave-os/cyberwave-python/actions/workflows/test.yml)
[![Python 3.12](https://img.shields.io/github/actions/workflow/status/cyberwave-os/cyberwave-python/test.yml?label=Python%203.12&logo=python&branch=main)](https://github.com/cyberwave-os/cyberwave-python/actions/workflows/test.yml)
[![Python 3.13](https://img.shields.io/github/actions/workflow/status/cyberwave-os/cyberwave-python/test.yml?label=Python%203.13&logo=python&branch=main)](https://github.com/cyberwave-os/cyberwave-python/actions/workflows/test.yml)
[![Python 3.14](https://img.shields.io/github/actions/workflow/status/cyberwave-os/cyberwave-python/test.yml?label=Python%203.14&logo=python&branch=main)](https://github.com/cyberwave-os/cyberwave-python/actions/workflows/test.yml)

## Installation

```bash
pip install cyberwave
```

## Quick Start

### 1. Get Your API Key

Get your API key from the Cyberwave platform:

- Log in to your Cyberwave instance
- Navigate to [Profile](https://cyberwave.com/profile) → API Tokens
- Create an API key and copy it

### 2. Create Your First Digital Twin

```python
from cyberwave import Cyberwave

# Configure with your API key
cw = Cyberwave(
    api_key="your_api_key_here",
)

# For simulator-first scripts, initialize with simulation defaults
sim_cw = Cyberwave(
    api_key="your_api_key_here",
    mode="simulation",
)

# Create a digital twin from an asset
robot = cw.twin("the-robot-studio/so101")

# If no default environment is configured, this creates a "Quickstart Environment"
# and places the twin there automatically.

# Change position and rotation in the environment editor
robot.edit_position(x=1.0, y=0.0, z=0.5)
robot.edit_rotation(yaw=90)  # degrees

# For locomotion twins, declare whether you want to affect the simulation
# or the live robot, then call movement methods without source_type arguments
rover = cw.twin("unitree/go2")
cw.affect("simulation")   # or cw.affect("live") for the physical robot
rover.move_forward(distance=1.0)
rover.turn_left(angle=1.57)

# Move the robot arm to 30 degrees
robot.joints.set("1", 30)

# Get current joint positions
print(robot.joints.get_all())
```

## Core Features

### Working with Workspaces and Projects

```python
from cyberwave import Cyberwave

cw = Cyberwave(
    api_key="your_api_key_here"
)

# You can also set your API key as an environment variable: export CYBERWAVE_API_KEY=your_api_key_here
# in that case, you can simply do:
cw = Cyberwave()

# List workspaces
workspaces = cw.workspaces.list()
print(f"Found {len(workspaces)} workspaces")

# Create a project
project = cw.projects.create(
    name="My Robotics Project",
    workspace_id=workspaces[0].uuid
)

# Create an environment
environment = cw.environments.create(
    name="Development",
    project_id=project.uuid
)
```

### Managing Assets and Twins

```python
# To instantiate a twin, you can query the available assets from the catalog.
# This query will return both the public assets availaable at cyberwave.com/catalog and the private assets available to your organization.
assets = cw.assets.search("so101")
robot = cw.twin(assets[0].registry_id) # the registry_id is the canonical asset identifier, e.g. the-robot-studio/so101

# Standard assets can also expose a shorter registry ID alias for SDK ergonomics.
# For example, a catalog camera asset may be callable with:
camera = cw.twin("camera")

# Edit the twin to a specific position
robot.edit_position([1.0, 0.5, 0.0])

# Navigate locomotion twins
# Use affect() to set whether commands go to the simulation or the live robot
cw.affect("simulation")   # or cw.affect("live")
rover = cw.twin("unitree/go2")
rover.move_forward(distance=1.0)
rover.move_backward(distance=0.5)
rover.turn_left(angle=1.57)
rover.turn_right(angle=1.57)

# Update scale
robot.edit_scale(x=1.5, y=1.5, z=1.5)

# Move a joint to a specific position using radians
robot.joints.set("shoulder_joint", math.pi/4)

# You can also use degrees:
robot.joints.set("shoulder_joint", 45, degrees=True)

# You can also go a get_or_create for a specific twin an environment you created:
 robot = cw.twin("the-robot-studio/so101", environment_id="YOUR_ENVIRONMENT_ID")
```

Move an existing twin to another environment:

```python
robot = cw.twin("the-robot-studio/so101")
robot.add_to_environment("TARGET_ENVIRONMENT_UUID")
```

`add_to_environment()` creates a deep copy of the twin in the target environment, marks the original twin as deleted, and also deletes the source environment if it has no twins left.

`move()` and `rotate()` are deprecated. Use `move_forward()`, `move_backward()`, `turn_left()`, and `turn_right()` for locomotion commands, or `edit_rotation()` to directly set orientation.

### Uploading Large GLB Assets

The SDK supports large GLB uploads by automatically switching to an attachment + signed URL flow when files exceed the standard upload limit.

```python
from cyberwave import Cyberwave

cw = Cyberwave()

asset = cw.assets.create(
    name="Warehouse Shelf",
    description="Large GLB upload example",
)

# Automatically chooses direct upload (small files) or signed URL flow (large files)
updated_asset = cw.assets.upload_glb(asset.uuid, "/path/to/warehouse_shelf.glb")
print(updated_asset.glb_file)
```

### Grab a Frame

Capture the latest camera frame in 3 lines — no streaming setup required:

```python
from cyberwave import Cyberwave

cw = Cyberwave()
robot = cw.twin("the-robot-studio/so101")

# Grab the latest frame (default: saved to a temp JPEG file)
path = robot.capture_frame()                   # returns temp file path
frame = robot.capture_frame("numpy")           # numpy BGR array (requires numpy + opencv-python)
image = robot.capture_frame("pil")             # PIL.Image (requires Pillow)
raw = robot.capture_frame("bytes")             # raw JPEG bytes

# Batch capture: grab 5 frames 200ms apart
folder = robot.capture_frames(5, interval_ms=200)           # folder of JPEGs
frames = robot.capture_frames(5, format="numpy")             # list of arrays

# For multi-camera twins, specify a sensor
wrist = robot.capture_frame("numpy", sensor_id="wrist_cam")
```

`capture_frame()` and `get_latest_frame()` follow your active `cw.affect(...)` mode by default:

- `cw.affect("real-world")` → returns the latest real sensor frame
- `cw.affect("simulation")` → returns the rendered virtual camera frame

You can still override per call with `source_type`:

```python
cw.affect("simulation")
sim_raw = robot.get_latest_frame()                    # virtual camera frame
real_raw = robot.get_latest_frame(source_type="tele")  # force real-world frame
```

There's also a `twin.camera` namespace with convenience methods:

```python
frame = robot.camera.read()              # numpy array (default)
path  = robot.camera.snapshot()           # save JPEG to temp file
path  = robot.camera.snapshot("out.jpg")  # save to a specific path
```

### Environment Variables

If you are always using the same environment, you can set it as a default with the CYBERWAVE_ENVIRONMENT_ID environment variable:

```bash
export CYBERWAVE_ENVIRONMENT_ID="YOUR_ENVIRONMENT_ID"
export CYBERWAVE_API_KEY="YOUR_TOKEN"
python your_script.py
```

And then you can simply do:

```python
from cyberwave import Cyberwave

cw = Cyberwave()
robot = cw.twin("the-robot-studio/so101")
```

This code will return you the first SO101 twin in your environment, or create it if it doesn't exist.

### Video Streaming (WebRTC)

Stream camera feeds to your digital twins using WebRTC. The SDK supports both standard USB/webcam cameras (via OpenCV) and Intel RealSense cameras with RGB and depth streaming.

#### Prerequisites

Install FFMPEG if you don't have it:

```bash
# Mac
brew install ffmpeg pkg-config

# Ubuntu
sudo apt-get install ffmpeg
```

Install camera dependencies:

```bash
# Standard cameras (OpenCV)
pip install cyberwave[camera]

# Intel RealSense cameras
pip install cyberwave[realsense]
```

> **Note for ARM64/Raspberry Pi**: The `pip install cyberwave[realsense]` command installs the Python wrapper, but you'll still need the librealsense SDK installed on your system. On x86_64 systems, you can install it via `sudo apt install librealsense2` or use pre-built wheels. **On Raspberry Pi OS (ARM64), you must build librealsense from source** - see our [Raspberry Pi Installation Guide](install_realsense_raspian_os.md).

#### Quick Start

```python
import asyncio
import os
from cyberwave import Cyberwave
cw = Cyberwave()
camera = cw.twin("cyberwave/standard-cam")

try:
    print(f"Streaming to twin {camera.uuid}... (Ctrl+C to stop)")
    await camera.stream_video_background()

    while True:
        await asyncio.sleep(1)
except (KeyboardInterrupt, asyncio.CancelledError):
    print("\nStopping...")
finally:
    await camera.stop_streaming()
    cw.disconnect()
```

If you have a depth camera - that streams also a point cloud - it's the same thing! You just change the twin name and Cyberwave takes care of the rest:

```python
import asyncio
import os
from cyberwave import Cyberwave
cw = Cyberwave()
camera = cw.twin("intel/realsensed455")

try:
    print(f"Streaming to twin {camera.uuid}... (Ctrl+C to stop)")
    await camera.stream_video_background()

    while True:
        await asyncio.sleep(1)
except (KeyboardInterrupt, asyncio.CancelledError):
    print("\nStopping...")
finally:
    await camera.stop_streaming()
    cw.disconnect()
```

### Audio streaming (microphone)

Stream microphone audio to a digital twin over WebRTC (Opus, 48 kHz mono). Use a `get_audio()` callback that returns 20 ms of s16 mono audio (1920 bytes) or `None` for silence.

```python
import asyncio
from cyberwave import Cyberwave
from cyberwave.sensor.microphone import MicrophoneAudioStreamer

def get_audio() -> bytes | None:
    # Return 20 ms s16 mono 48 kHz (1920 bytes), or None for silence
    # e.g. read from PyAudio, sounddevice, or robot WebRTC (Go2)
    return bytes(1920)  # or your capture

cw = Cyberwave()
streamer = MicrophoneAudioStreamer(
    cw.mqtt,
    get_audio,
    twin_uuid="your-twin-uuid",
    sensor_name="mic",
)
await streamer.start()
# ... run until done ...
await streamer.stop()
cw.disconnect()
```

For Unitree Go2 robot microphone streaming, see [examples/audio_stream.py](examples/audio_stream.py) (requires `unitree_webrtc_connect`). For combined video + audio, see [examples/multimedia_stream.py](examples/multimedia_stream.py).

## Examples

Check the [examples](examples) directory for complete examples:

- Basic twin control
- Multi-robot coordination
- Real-time synchronization
- Joint manipulation for robot arms
- Audio streaming (e.g. Go2 microphone) — [audio_stream.py](examples/audio_stream.py), [multimedia_stream.py](examples/multimedia_stream.py)

## Advanced Usage

### Joint Control

You can change a specific joint actuation. You can use degrees or radiants:

```python
robot = cw.twin("the-robot-studio/so101")

# Set individual joints (degrees by default)
robot.joints.set("shoulder_joint", 45, degrees=True)

# Or use radians
import math
robot.joints.set("elbow_joint", math.pi/4, degrees=False)

# Get current joint position
angle = robot.joints.get("shoulder_joint")

# List all joints
joint_names = robot.joints.list()

# Get all joint states at once
all_joints = robot.joints.get_all()
```

To check out the available endpoints and their parameters, you can refer to the full API reference [here](https://docs.cyberwave.com/api-reference/overview).

### Simulation vs. Live

Use `mode="simulation"` when you want a simulator-first client with simulation defaults for state publishing, or use `cw.affect()` to switch locomotion/control commands between the simulation and the live robot on an existing client.

```python
from cyberwave import Cyberwave

cw = Cyberwave(mode="simulation")

# In simulation mode, state/telemetry publishers default to source_type="sim"
# and locomotion/control helpers target the simulator.
rover = cw.twin("unitree/go2")
rover.move_forward(1.0)   # moves the digital twin in Cyberwave

# Switch command helpers to the live robot
cw.affect("live")
rover.move_forward(1.0)   # moves the physical robot

# You can still override per-call when needed
rover.move_forward(1.0, source_type="tele")
```

`affect()` is chainable: `cw.affect("simulation").twin("unitree/go2")`.

`"real-world"` is also accepted as an alias for `"live"`.

Runtime mode defaults:

- Live mode publishes state updates as `edge`.
- Simulation mode publishes state updates as `sim`.
- Locomotion and other control helpers target `tele` in live mode and `sim_tele` in simulation mode.

For lower-level integrations you can still pass `source_type` explicitly or use the `CYBERWAVE_SOURCE_TYPE` environment variable. Accepted raw values are `"sim"`, `"sim_tele"`, `"tele"`, `"edge"`, and `"edit"`. Legacy locomotion calls that pass `"sim"` are still accepted and normalized to `"sim_tele"` for simulator control commands.

This same mode selection is also used by camera frame grabs (`get_latest_frame()` and `capture_frame()`), so frame retrieval stays consistent with the mode your script is affecting.

### Camera & Sensor discovery

You can leverage the SDK to discover the CV2 (standard webcameras) attached to your device:

```python
from cyberwave.sensor import CV2VideoTrack, CV2CameraStreamer, CameraConfig, Resolution

# Check supported resolutions for a camera
supported = CV2VideoTrack.get_supported_resolutions(camera_id=0)
print(f"Supported: {[str(r) for r in supported]}")

# Get camera info
info = CV2VideoTrack.get_camera_info(camera_id=0)
print(f"Camera: {info}")

# Using CameraConfig
config = CameraConfig(resolution=Resolution.HD, fps=30, camera_id=0)
streamer = CV2CameraStreamer.from_config(cw.mqtt, config, twin_uuid="...")
```

### RealSense Camera (RGB + Depth)

You can also discover and set up RGD+D (Depth) cameras.

> **Raspberry Pi / ARM64 users**: If you're running on Raspberry Pi OS or other ARM64 systems, you'll need to manually build librealsense from source, as pre-built packages aren't available. See our [Raspberry Pi Installation Guide](install_realsense_raspian_os.md) for detailed instructions.

The SDK supports dynamic discovery of RealSense device capabilities:

```python
from cyberwave.sensor import (
    RealSenseDiscovery,
    RealSenseConfig,
    RealSenseStreamer,
    Resolution
)

# Check if RealSense SDK is available
if RealSenseDiscovery.is_available():
    # List connected devices
    devices = RealSenseDiscovery.list_devices()
    for dev in devices:
        print(f"{dev.name} (SN: {dev.serial_number})")

    # Get detailed device info with all supported profiles
    info = RealSenseDiscovery.get_device_info()
    print(f"Color resolutions: {info.get_color_resolutions()}")
    print(f"Depth resolutions: {info.get_depth_resolutions()}")
    print(f"Sensor options: {info.sensor_options}")

# Auto-detect and create streamer from device capabilities
streamer = RealSenseStreamer.from_device(
    cw.mqtt,
    prefer_resolution=Resolution.HD,
    prefer_fps=30,
    enable_depth=True,
    twin_uuid="your_twin_uuid"
)

# Or use manual configuration with validation
config = RealSenseConfig(
    color_resolution=Resolution.HD,
    depth_resolution=Resolution.VGA,
    color_fps=30,
    depth_fps=15,
    enable_depth=True
)

# Validate against device
is_valid, errors = config.validate()
if not is_valid:
    print(f"Config errors: {errors}")

streamer = RealSenseStreamer.from_config(cw.mqtt, config, twin_uuid="...")
```

#### RealSense Device Discovery

Query detailed device capabilities:

```python
info = RealSenseDiscovery.get_device_info()

# Check if a specific profile is supported
if info.supports_color_profile(1280, 720, 30, "BGR8"):
    print("HD @ 30fps with BGR8 is supported")

# Get available FPS for a resolution
fps_options = info.get_color_fps_options(1280, 720)
print(f"Available FPS for HD: {fps_options}")

# Get sensor options (exposure, gain, laser power, etc.)
for sensor_name, options in info.sensor_options.items():
    print(f"\n{sensor_name}:")
    for opt in options:
        print(f"  {opt.name}: {opt.value} (range: {opt.min_value}-{opt.max_value})")
```

### Edge Management

Edges are physical devices (e.g. Raspberry Pi, Jetson) that run the Cyberwave Edge Core. You can manage them programmatically via `cw.edges`.

```python
from cyberwave import Cyberwave

cw = Cyberwave()

# List all edges registered to your account
edges = cw.edges.list()
for edge in edges:
    print(edge.uuid, edge.name, edge.fingerprint)

# Get a specific edge
edge = cw.edges.get("your-edge-uuid")

# Register a new edge with a hardware fingerprint
edge = cw.edges.create(
    fingerprint="linux-a1b2c3d4e5f60000",   # stable hardware identifier
    name="lab-rpi-001",                       # optional human-readable name
    workspace_id="your-workspace-uuid",       # optional, scopes the edge to a workspace
    metadata={"location": "lab-shelf-2"},     # optional arbitrary metadata
)

# Update edge name or metadata
edge = cw.edges.update(edge.uuid, {"name": "lab-rpi-001-renamed"})

# Delete an edge
cw.edges.delete(edge.uuid)
```

The fingerprint is a stable identifier derived from the host hardware (hostname, OS, architecture, and MAC address). The Edge Core generates and persists it automatically on first boot at `/etc/cyberwave/fingerprint.json`. When a twin has `metadata.edge_fingerprint` set to the same value, the Edge Core will automatically pull and start its driver container on boot.

### Workflows

List, trigger, and monitor workflows programmatically. Useful for building custom automations on top of Cyberwave's visual workflow engine.

```python
cw = Cyberwave()

# List available workflows
for wf in cw.workflows.list():
    print(f"{wf.name} ({wf.uuid}) — {wf.status}")

# Trigger a workflow
run = cw.workflows.trigger(
    "workflow-uuid",
    inputs={"target_position": [1.0, 2.0, 0.0], "speed": 0.5},
)

# Poll until done (blocks up to 60 s)
run.wait(timeout=60)
print(run.status, run.result)

# Or check manually
run.refresh()
if run.error:
    print(f"Failed: {run.error}")
```

You can also start from a `Workflow` object:

```python
wf = cw.workflows.get("workflow-uuid")
run = wf.trigger(inputs={"speed": 1.0})
run.wait()
```

List and filter past runs:

```python
runs = cw.workflow_runs.list(workflow_id="workflow-uuid", status="error")
for r in runs:
    print(r.uuid, r.status, r.error)
```

### Alerts

Create, list, and manage alerts directly from a twin. Alerts notify operators that action is needed (e.g. a robot needs calibration or a sensor reading is out of range).

```python
twin = cw.twin(twin_id="your_twin_uuid")

# Create an alert
alert = twin.alerts.create(
    name="Calibration needed",
    description="Joint 3 is drifting beyond tolerance",
    severity="warning",          # info | warning | error | critical
    alert_type="calibration_needed",
    source_type="edge",          # edge | cloud | workflow
)

# If you need to bypass backend deduplication and always create a new row:
forced_alert = twin.alerts.create(
    name="Calibration needed",
    description="Joint 3 is drifting beyond tolerance",
    alert_type="calibration_needed",
    force=True,
)

# List active alerts for this twin
for a in twin.alerts.list(status="active"):
    print(a.name, a.severity, a.status)

# Lifecycle actions
alert.acknowledge()   # operator has seen it
alert.resolve()       # root cause addressed

# Other operations
alert.silence()       # suppress without resolving
alert.update(severity="critical")
alert.delete()
```

## Data Layer (`cw.data`) — *stub*

Transport-agnostic pub/sub for edge sensor data. Supports Zenoh (primary) and filesystem (fallback) backends.

Requires `CYBERWAVE_TWIN_UUID` to be set when accessing `cw.data`.

```python
cw = Cyberwave(api_key="...")

# Publish a numpy frame
import numpy as np
frame = np.zeros((480, 640, 3), dtype=np.uint8)
cw.data.publish("frames", frame)

# Get the latest value (with staleness check)
depth = cw.data.latest("depth", max_age_ms=50)
if depth is None:
    print("Depth sample too stale or not yet available")

# Subscribe to decoded data
def on_joints(data: dict):
    print("Joints:", data)

sub = cw.data.subscribe("joint_states", on_joints)
# ... later
sub.close()
```

Backend selection via `CYBERWAVE_DATA_BACKEND` env var (`zenoh` or `filesystem`).

For Zenoh: `pip install cyberwave[zenoh]` or `pip install eclipse-zenoh`.

### Time-aware fusion

Interpolated point reads and time-window queries for multi-sensor workers.

```python
from cyberwave.data import Quaternion

joints = cw.data.at("joint_states", t=ctx.timestamp, interpolation="linear")
pose = cw.data.at("orientation", t=ctx.timestamp, interpolation="slerp")  # requires Quaternion values
imu_samples = cw.data.window("imu", from_t=prev_frame_ts, to_t=ctx.timestamp)
recent_ft = cw.data.window("force_torque", duration_ms=100)
```

| Strategy | Value types | Behavior |
| --- | --- | --- |
| `"linear"` | scalar, vector, dict, numpy, `Quaternion` | Element-wise lerp (NLERP for quaternions) |
| `"slerp"` | `Quaternion` instances | Spherical linear interpolation |
| `"nearest"` | any | Returns closest sample by time |
| `"none"` | any | Exact timestamp match or `None` |

Wrap orientation data in `Quaternion(x, y, z, w)` for type-safe SLERP. Convention: Hamilton `(x, y, z, w)`, same as ROS/MuJoCo.

See the [data fusion docs](https://docs.cyberwave.com/sdks/data-fusion) for details.

### Synchronized multi-channel hooks

`@cw.on_synchronized` fires only when samples from all listed channels arrive within a configurable time tolerance — an approximate time synchronizer for multi-sensor fusion.

```python
@cw.on_synchronized(twin_uuid, ["frames/front", "depth/default", "joint_states"], tolerance_ms=50)
def detect_collision(samples, ctx):
    frame = samples["frames/front"].payload
    depth = samples["depth/default"].payload
    joints = samples["joint_states"].payload
    # All three are within 50ms of each other
```

See the [synchronized hooks docs](https://docs.cyberwave.com/sdks/data-synchronized-hooks) for details.

## Testing

### Unit Tests

Run basic import tests:

```bash
poetry install
poetry run python tests/test_imports.py
```

## Contributing

Contributions are welcome. If you have an idea, bug report, or improvement request, please open an issue or submit a pull request.

## Support

- **Documentation**: [docs.cyberwave.com](https://docs.cyberwave.com)
- **Issues**: [GitHub Issues](https://github.com/cyberwave-os/cyberwave-python/issues)
- **Community**: [Discord](https://discord.gg/dfGhNrawyF)
