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

### Managing Environment Waypoints

```python
waypoints = cw.environments.create_waypoint(
    environment.uuid,
    waypoint_id="dock-a",
    name="Dock A",
    position={"x": 1.0, "y": 2.0, "z": 0.0},
    rotation={"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    metadata={"priority": "high"},
)

waypoints = cw.environments.get_waypoints(environment.uuid)
waypoints = cw.environments.delete_waypoint(environment.uuid, "dock-a")
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

### Using Unified Slugs

Every major entity (asset, twin, environment, workflow) has a **unified slug** — a human-readable identifier in the format `{workspace-slug}/{type-prefix}/{entity-slug}`. Slugs can be used interchangeably with UUIDs across the SDK.

```python
from cyberwave import Cyberwave

cw = Cyberwave()

# Fetch a twin by its slug instead of UUID
robot = cw.twin(twin_id="acme/twins/my-arm")

# Create a twin using an asset slug
robot = cw.twin("acme/catalog/so101")

# Pass an environment slug
robot = cw.twin("acme/catalog/so101", environment_id="acme/envs/production")

# Fetch an environment by slug
env = cw.environments.get_by_slug("acme/envs/production")

# Fetch a workflow by slug and trigger it
wf = cw.workflows.get_by_slug("acme/workflows/pick-and-place")
run = cw.workflows.trigger("acme/workflows/pick-and-place", inputs={"speed": 1.0})

# Check slug availability
result = cw.assets.check_slug("acme/catalog/my-new-robot")
if result["available"]:
    print("Slug is available!")

# Access an entity's slug
print(robot.slug)  # e.g. "acme/twins/my-arm"
```

| Entity       | Type Prefix   | Slug Example                          |
|------------- |-------------- |---------------------------------------|
| Asset        | `catalog`     | `acme/catalog/my-robot-arm`           |
| Twin         | `twins`       | `acme/twins/arm-station-1`            |
| Environment  | `envs`        | `acme/envs/production-floor`          |
| Workflow     | `workflows`   | `acme/workflows/pick-and-place`       |
| ML Model     | `models`      | `acme/models/yolov8-custom`           |
| Controller   | `controllers` | `acme/controllers/keyboard-teleop`    |

### Listing Primitive Assets (catalog assets with shortcuts)

Primitive assets are curated public catalog entries with a short **alias** (e.g. `camera`, `lidar`). They can be instantiated directly by alias without knowing the full `vendor/slug` registry ID — making it easy to quickly populate an environment.

```python
from cyberwave import Cyberwave

cw = Cyberwave()

# List all catalog primitives (public assets with a registry_id_alias)
primitives = cw.assets.list_primitives()
for asset in primitives:
    print(f"{asset.registry_id_alias!r:20} -> {asset.registry_id}")

# Instantiate a primitive directly by its alias
camera = cw.twin("camera")
lidar  = cw.twin("lidar")
```

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

### Drones and Flying Twins

Twins whose asset has `can_fly: true` are returned as `FlyingTwin` instances.
They expose `takeoff()`, `land()`, and `hover()` MQTT commands, plus helpers to
read and persist the hovering state in the twin's metadata.

```python
from cyberwave import Cyberwave
from cyberwave.twin import FlyingTwin

cw = Cyberwave()
drone: FlyingTwin = cw.twin("cyberwave/px4vision")  # type: ignore[assignment]

# Send the takeoff command
ALTITUDE = 2.0  # metres
drone.takeoff(altitude=ALTITUDE)

# Read the hovering status back
if drone.is_hovering():
    status = drone.get_hovering_status()
    print(f"Hovering at {status['controller_requested_hovering_altitude']} m")

# Land and clear the hovering state
drone.land()
drone.set_hovering_status(hovering=False)
```

The hovering state is stored under `twin.metadata.status`:

```json
{
  "status": {
    "controller_requested_hovering": true,
    "controller_requested_hovering_altitude": 2.0
  }
}
```

The `controller_requested_` prefix makes it clear these are controller
intentions, not ground-truth sensor readings from the drone.

In the Cyberwave **playground simulate** mode, setting `controller_requested_hovering=True`
disables gravity for that twin so it stays at its current altitude visually.

See the full example in [examples/drone_hovering.py](examples/drone_hovering.py).

## Examples

Check the [examples](examples) directory for complete examples:

- Basic twin control
- Multi-robot coordination
- Real-time synchronization
- Joint manipulation for robot arms
- Audio streaming (e.g. Go2 microphone) — [audio_stream.py](examples/audio_stream.py), [multimedia_stream.py](examples/multimedia_stream.py)
- Drone takeoff, hovering status, and landing — [drone_hovering.py](examples/drone_hovering.py)

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

# Get all joint states at once as a dict {name: radians}
all_joints = robot.joints.get_all()

# Print all joint states in a human-readable table (radians + degrees)
robot.joints.print_joint_states()
```

`print_joint_states()` fetches the latest states from the server and prints a formatted table:

```
Joint states for twin <twin-uuid>:
------------------------------------------------------
Joint                       Radians       Degrees
------------------------------------------------------
elbow_joint               0.0000 rad       0.00 °
shoulder_joint            0.7854 rad      45.00 °
------------------------------------------------------
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

Generated `run_on_edge` workers with schedule triggers use
`@cw.on_schedule(...)`. The edge worker runtime evaluates the cron locally and
calls the generated `run(...)` entrypoint when the schedule is due.
Install `cyberwave[schedule]` when running scheduled workers outside the
standard Edge worker image.

Custom edge workers can use the same scheduler directly:

```python
@cw.on_schedule("*/5 * * * *", timezone="UTC")
def every_five_minutes(ctx):
    print("schedule fired", ctx.timestamp)
```

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

### Datasets

Import and manage robotics datasets from HuggingFace or local files. Supports LeRobot and RLDS formats.

```python
# Import a dataset from HuggingFace
ds = cw.datasets.add("lerobot/pusht", name="pusht")
print(f"Dataset {ds.uuid} is {ds.processing_status}")

# Import with specific revision/subset
ds = cw.datasets.add(
    "lerobot/aloha_sim_insertion_human",
    name="aloha-insertion",
    hf_revision="main",
    hf_subset="default",
)

# Upload a local dataset (directory or zip)
ds = cw.datasets.add("./my_lerobot_dataset", name="my-dataset")
ds = cw.datasets.add("./recordings.zip", name="recordings")

# List datasets with pagination and filters
datasets = cw.datasets.list(limit=20, offset=0)
datasets = cw.datasets.list(processing_status="completed")
datasets = cw.datasets.list(environment="env-uuid")

# Get and delete
ds = cw.datasets.get("dataset-uuid")
cw.datasets.delete("dataset-uuid")

# Open in browser for visualization
cw.datasets.visualize(ds)
# Prints: View dataset at: https://cyberwave.com/acme/datasets/pusht

# Poll for completion (HuggingFace imports are async)
import time
while ds.processing_status == "pending":
    time.sleep(5)
    ds = cw.datasets.get(ds.uuid)
print(f"Dataset ready: {ds.processing_status}")
```

**Supported formats:**
- **LeRobot v2.1** — HuggingFace datasets with `lerobot` metadata
- **LeRobot v3** — Latest LeRobot format with parquet episodes  
- **RLDS** — TensorFlow Datasets format (zip upload only)

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

**Cross-twin mode** — synchronize channels from different twins (e.g. stereo cameras):

```python
@cw.on_synchronized(
    twin_channels={
        "left": (CAMERA_LEFT, "frames/default"),
        "right": (CAMERA_RIGHT, "frames/default"),
    },
    tolerance_ms=50.0,
)
def on_stereo_pair(samples, ctx):
    left = samples["left"]   # Sample from CAMERA_LEFT
    right = samples["right"] # Sample from CAMERA_RIGHT
```

See the [synchronized hooks docs](https://docs.cyberwave.com/sdks/data-synchronized-hooks) for details.

## ML Models (`cw.models`) — *stub*

Load and run ML models on edge devices. Models are loaded via `cw.models.load()` and expose a `predict()` method.

```python
cw = Cyberwave()
model = cw.models.load("yolov8n")
result = model.predict(frame)
for det in result.detections:
    print(f"{det.label}: {det.confidence:.2f}")
```

YOLO ONNX postprocessing applies per-class non-max suppression with the same default IoU threshold (`0.7`) as Ultralytics' `YOLO.predict`, so swapping `yolov8s.pt` for `yolov8s.onnx` produces the same number of boxes per object instead of a cluster of overlapping anchors. Tune via `model.predict(frame, iou=0.5)` (stricter) or pass `iou=1.0` to disable NMS entirely (raw output for custom trackers).

### Model warm-up — *stub*

Eliminate cold-start latency by calling `warm_up()` after loading:

```python
model = cw.models.load("yolov8n")
cold_ms, warm_ms = model.warm_up()  # two dummy inferences
# cold_ms ≈ 150 ms (JIT/allocation), warm_ms ≈ 8 ms (steady-state)
```

The worker runtime calls `warm_up()` automatically on startup for all loaded models.

### Frame resolution scaling — *stub*

Set `CYBERWAVE_WORKER_INPUT_RESOLUTION` to downscale frames before inference without changing the camera driver's publish resolution:

```bash
export CYBERWAVE_WORKER_INPUT_RESOLUTION=640x480  # 4K camera → 640x480 for YOLO nano
```

### Automatic detection publishing

Every call to `model.predict()` automatically publishes its result to a `detections/<runtime>` Zenoh channel (e.g. `detections/ultralytics`, `detections/onnxruntime`) as structured JSON. Empty results are published as `{"detections": []}` heartbeats at the worker's inference cadence so overlay consumers (e.g. the OBSBOT camera driver) see a steady signal and don't fall into their staleness cutoff when the scene transiently has nothing to detect. Drivers that subscribe to `detections/*` can draw bounding box overlays directly on the video stream — no extra worker code needed.

**Multi-camera routing:** Pass `twin_uuid=ctx.twin_uuid` to `model.predict()` to route detections to the correct twin when handling multiple cameras:

```python
@cw.on_frame(CAMERA_A)
def on_cam_a(frame, ctx):
    model.predict(frame, confidence=0.5, twin_uuid=ctx.twin_uuid)
```

Omitting `sensor=` subscribes to `frames/**` on the twin, so the hook picks
up whatever sensor name the driver publishes (e.g. `color_camera`,
`depth_camera`). `ctx.sensor_name` is populated from the observed key, so
a single handler can disambiguate multi-sensor twins. Pass
`sensor="<name>"` explicitly when you need to target one specific sensor.

This requires a Zenoh data bus (`pip install cyberwave[zenoh]` and `CYBERWAVE_TWIN_UUID` set). If unavailable, auto-publish is silently skipped.

### Privacy-preserving workers — *stub*

Use `cyberwave.vision.anonymize_frame()` together with a pose model (e.g. `yolov8n-pose-onnx`) to **obscure every person** in the stream and overlay a colour-coded pose skeleton **before** the frame leaves the edge:

```python
from cyberwave.vision import anonymize_frame

model = cw.models.load("yolov8n-pose-onnx")

from cyberwave.data import FILTERED_FRAME_CHANNEL

@cw.on_frame(cw.config.twin_uuid, sensor="default")
def anonymise(frame, ctx):
    result = model.predict(frame, classes=["person"], confidence=0.4)
    # Defaults: pixelate mosaic + per-bodypart skeleton palette.
    # mode={"pixelate"|"redact"|"blur"|"bbox"}, pixel_size=int|None,
    # blur_kernel=int, color=BGR (used by bbox + redact, and as the
    # small-ROI fallback fill for any mode).
    out = anonymize_frame(frame, result.detections)
    cw.data.publish(FILTERED_FRAME_CHANNEL, out, twin_uuid=ctx.twin_uuid)
```

For a runnable end-to-end demo against your local webcam — useful for tuning the mode / threshold knobs interactively — see [`examples/webcam_pose_anonymize.py`](./examples/webcam_pose_anonymize.py).

Pair the worker with a generic-camera driver configured to consult `frames/processed` (`CYBERWAVE_DRIVER_FRAME_FILTER=frames/processed`); the driver substitutes the obscured frame into the WebRTC stream before encoding. Raw `frames/*` channels stay local — they are not forwarded over MQTT by default.

> **Privacy note:** the default `pixelate` mode is intended for casual visual obscuring, not as a cryptographic anonymisation primitive — modern depixelation networks can recover recognisable faces from low-density mosaics. For stronger guarantees use `mode="blur"` (heavier irreversible degradation), `mode="redact"` (grid of solid `color` blocks with visible separators — the "censored document" look), or `mode="bbox"` (single solid fill, destroys the underlying pixels entirely). See the `cyberwave.vision.anonymize` module docstring for the full caveat.

> **Note:** the driver-side `frames/processed` consumer and the
> `CYBERWAVE_DRIVER_FRAME_FILTER` knob ship in the follow-up PRs (driver
> wiring + end-to-end example). This snippet shows the eventual shape of
> the API; today, only the SDK building blocks (`anonymize_frame`,
> `draw_skeleton`, the return-aware `frame_callback`) are wired up.

A complete two-camera example lives at [`examples/security_pipeline/`](./examples/security_pipeline). See the [Security Pipeline](https://docs.cyberwave.com/edge/drivers/security-pipeline) and [Frame Filters](https://docs.cyberwave.com/edge/drivers/frame-filters) docs for the full picture.

## Zenoh-MQTT Bridge (`cyberwave.zenoh_mqtt`) — *stub*

Bidirectional forwarder between the local Zenoh data bus and the cloud MQTT broker. Runs on edge devices alongside Edge Core.

**Outbound** (edge to cloud): subscribes to Zenoh channels and publishes to MQTT.
**Inbound** (cloud to edge): subscribes to MQTT command topics and republishes into the local Zenoh session.

When the MQTT connection drops, outbound messages are buffered to a persistent file-backed queue and drained in FIFO order on reconnect.

```python
from cyberwave.zenoh_mqtt import ZenohMqttBridge, BridgeConfig

bridge = ZenohMqttBridge(
    config=BridgeConfig(
        twin_uuids=["<twin_uuid>"],
        outbound_channels=["model_output", "event", "model_health"],
    ),
    mqtt_host="<mqtt_host>",
    mqtt_port=8883,
    mqtt_password="<api_key>",
)
bridge.start()
# bridge runs until stopped
bridge.stop()
```

### Default topic mapping

| Zenoh key | MQTT topic | Direction |
|---|---|---|
| `cw/{twin}/data/model_output` | `cyberwave/twin/{twin}/model_output` | Edge to Cloud |
| `cw/{twin}/data/event` | `cyberwave/twin/{twin}/event` | Edge to Cloud |
| `cw/{twin}/data/model_health` | `cyberwave/twin/{twin}/model_health` | Edge to Cloud |
| MQTT `cyberwave/twin/{twin}/commands/sync_workflows` | `cw/{twin}/data/commands_sync_workflows` | Cloud to Edge |

### Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `CYBERWAVE_BRIDGE_ENABLED` | `false` | Master switch |
| `CYBERWAVE_BRIDGE_TWIN_UUIDS` | — | Comma-separated twin UUIDs to bridge |
| `CYBERWAVE_BRIDGE_OUTBOUND_CHANNELS` | `model_output,event,model_health` | Zenoh channels forwarded to MQTT |
| `CYBERWAVE_BRIDGE_INBOUND_TOPICS` | `commands/sync_workflows` | MQTT suffixes forwarded to Zenoh |
| `CYBERWAVE_BRIDGE_QUEUE_DIR` | `/tmp/cyberwave_bridge_queue` | Persistent queue directory |
| `CYBERWAVE_BRIDGE_QUEUE_MAX_BYTES` | `52428800` (50 MiB) | Max offline queue size |

Requires `pip install cyberwave[zenoh]` (Zenoh) and `paho-mqtt` (already a core dependency).

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
