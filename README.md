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

## Contents

- [Installation](#installation)
- [Try on Google Colab](#try-on-google-colab)
- [Quick Start](#quick-start)
- [Core Features](#core-features)
- [Twin control guide](#twin-control-guide)
- [Examples](#examples)
- [Advanced Usage](#advanced-usage)
- [Data Layer (`cw.data`)](#data-layer-cwdata--stub)
- [ML Models (`cw.models`)](#ml-models-cwmodels)
- [Zenoh-MQTT Bridge](#zenoh-mqtt-bridge-cyberwavezenoh_mqtt--stub)
- [Testing](#testing)
- [Version 0.5.0 twin API changelog](#version-050-twin-api-changelog-in-progress)
- [Contributing](#contributing)

## Try on Google Colab

These notebooks use **`pip install "cyberwave[ml]"`** and walk through **`cw.models` / `cyberwave.models`**: local YOLO weights, **`PredictionResult`**, overlays, and (in the minimal notebook) a **live twin** JPEG → predict flow.

| Notebook | Open in Colab | In this repo |
| --- | --- | --- |
| **YOLO26 tasks** — detection, segmentation, pose, classification, cascade demos | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cyberwave-os/cyberwave-python/blob/main/examples/ai/yolo.ipynb) | [`examples/ai/yolo.ipynb`](examples/ai/yolo.ipynb) |
| **Minimal twin** — `Twin.get_latest_frame()` → predict → table + overlay | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cyberwave-os/cyberwave-python/blob/main/examples/ai/yolo_minimal_on_twin.ipynb) | [`examples/ai/yolo_minimal_on_twin.ipynb`](examples/ai/yolo_minimal_on_twin.ipynb) |

On Colab, run the install cell first, then set **`CYBERWAVE_API_KEY`** (or use **`cyberwave login --token`** as in the twin notebook). The twin notebook needs a camera twin with Edge Core and a publishing driver; the YOLO task notebook is self-contained offline once weights are downloaded.

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

# Move a joint by name (from joints.list())
joint_names = robot.joints.list()
if joint_names:
    robot.set_joints({joint_names[0]: -0.2})
    print(joint_names[0], ":", robot.joints[joint_names[0]])

# Or use the joints handle directly
print(robot.get_joints())
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

``twin.get_frame()`` is the single entry point for still images. It has two independent parameters:

- ``source=`` — **where** the pixels are produced (default ``"cloud"``).
- ``format=`` — **how** you want them back: ``"bytes"``, ``"numpy"``, ``"pil"``, or ``"path"`` (write a JPEG and return the file path; replaces the old ``snapshot`` helpers). Default: ``"bytes"``.

#### 1. Runtime mode — ``cw.affect()`` (cloud only)

Sim vs live is **not** a ``source=`` option. Set it once on the client with ``cw.affect()`` and it will affect twin.get_frame() as well as other twin related methods for getters and setters. ``cw.affect("sim")`` will make your code work on simulation, while ``cw.affect("live")`` will make the same code work on real hardware. Same code, just one setting to update. By default, the runtime mode is ``live``.

| ``cw.affect(...)`` | What ``robot.get_frame()`` sees (``source="cloud"``) |
| --- | --- |
| ``"simulation"`` | Latest frame from the **simulator** bucket (virtual camera in the scene). |
| ``"live"`` | Latest frame from the **live / teleoperation** bucket (edge or active stream mirrored to the platform). |

```python
cw.affect("simulation")
sim_jpeg = robot.get_frame()          # sim latest-frame — no edge hardware required

cw.affect("live")
live_jpeg = robot.get_frame()         # live latest-frame — what the UI / teleop path sees
```

When the client is in **simulation runtime mode**, only ``source="cloud"`` is allowed. ``local``, ``zenoh``, and ``remote_edge`` raise — those transports are for on-machine / edge workflows, not the sim-first client default.

#### 2. Frame sources — ``source=``

Pick the transport that matches **where your script runs** relative to the camera.

##### ``"cloud"`` (default) — works from anywhere

``robot.get_frame()`` with no ``source`` calls the platform ``/latest-frame`` API. Use this in notebooks, Colab, CI, or any machine with API access. No local driver, Zenoh, or stream required.

Fail-soft: returns ``None`` if the endpoint has no frame yet. Which sim vs live bucket is used is determined by ``cw.affect()`` as above.

```python
jpeg = robot.get_frame()                    # bytes from latest-frame
arr = robot.get_frame("numpy")              # decoded array
wrist = robot.get_frame("numpy", sensor_id="wrist_cam")  # multi-camera twins
```

##### ``"remote_edge"`` — still from hardware over MQTT

Sends the twin’s MQTT ``take_photo`` command and waits for a JPEG on the driver photo topic. Useful when there is **no active video stream** to the platform but the edge driver can capture a single image (e.g. snapshot on demand).

Raises on timeout or driver error (unlike cloud, which is fail-soft).

```python
edge_jpeg = robot.get_frame("bytes", source="remote_edge")
```

##### ``"local"`` — same process as your stream (**advanced**, driver authoring)

For scripts that **start the camera stream inside the same Python process** — typical when writing or testing drivers. After you call ``twin.start_streaming()`` (or ``twin.stream()``, which delegates to it), ``source="local"`` reads the current frame from the in-process ``CameraStreamer`` (``_camera_streamer``) **without** a round trip to Cyberwave servers.

```python
robot.start_streaming()   # publishes via MQTT; streamer lives on this process
frame = robot.get_frame("numpy", source="local")   # from CameraStreamer, not REST
```

**Scope:** only meaningful in the process that owns the stream. It will not fetch another machine’s stream.

If no streamer is running, ``local`` falls back to a **single** OpenCV grab from device ``idx`` (default ``0``) — handy for a one-off USB frame, not a substitute for cloud or edge streaming.

##### ``"zenoh"`` — same machine as the driver (**advanced**)

Reads the latest sample on the twin’s ``cw.data`` ``frames`` channel. Use when your script runs on the **same device (or Zenoh-reachable LAN)** as the edge camera driver that publishes frames over Zenoh — e.g. your driver process and your test script on the robot’s onboard computer.

**Scope:** does not work from arbitrary remote hosts (Colab, your laptop) unless Zenoh is explicitly reachable there. Treat as an on-robot / co-located integration path.

```python
frame = robot.get_frame("numpy", source="zenoh")
```

#### Quick reference

| ``source`` | Runs from | Needs |
| --- | --- | --- |
| ``cloud`` | Anywhere with API key | Twin has a latest-frame on the platform; ``cw.affect()`` for sim vs live |
| ``remote_edge`` | Anywhere with MQTT to twin | Driver supports ``take_photo`` |
| ``local`` | Same process as ``start_streaming()`` | Active ``CameraStreamer``, or local OpenCV device |
| ``zenoh`` | Same host / Zenoh LAN as driver | Driver publishing ``cw.data`` frames |

```python
from cyberwave import Cyberwave

cw = Cyberwave()
robot = cw.twin("the-robot-studio/so101")

# General-purpose (default)
cloud = robot.get_frame()

# On-robot / driver development (advanced)
robot.start_streaming()
in_process = robot.get_frame("numpy", source="local")
on_device = robot.get_frame("numpy", source="zenoh")

# Edge still without live video
edge = robot.get_frame("bytes", source="remote_edge")
```

#### Output format and bursts

```python
path = robot.get_frame("path")                              # temp JPEG on disk
path = robot.get_frame("path", path="/tmp/frame.jpg", source="local")

folder = robot.get_frames(5, interval_ms=200)               # numbered JPEGs in a temp dir
frames = robot.get_frames(5, format="bytes")                # list of raw JPEG bytes
```

``twin.get_latest_frame()`` remains for backward compatibility (raises on error). Deprecated aliases on ``twin.camera`` (``latest_frame``, ``capture``, ``read``, ``snapshot``, ``edge_photo``, …) delegate to ``get_frame`` with a ``DeprecationWarning``.

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

Install host microphone dependencies:

```bash
pip install cyberwave[microphone]
```

```python
import asyncio
from cyberwave import Cyberwave
from cyberwave.sensor.microphone import HostMicrophoneCapture, MicrophoneAudioStreamer

capture = HostMicrophoneCapture()
capture.start()

cw = Cyberwave()
streamer = MicrophoneAudioStreamer(
    cw.mqtt,
    capture.get_audio,
    twin_uuid="your-twin-uuid",
    mic_name="mic",
)
await streamer.start()
# ... run until done ...
await streamer.stop()
capture.stop()
cw.disconnect()
```

For custom or robot-provided sources, pass your own `get_audio()` callback. For Unitree Go2 robot microphone streaming, see [examples/audio_stream.py](examples/audio_stream.py) (requires `unitree_webrtc_connect`). For combined video + audio, see [examples/multimedia_stream.py](examples/multimedia_stream.py).

### Audio playback (speaker)

The speaker counterpart consumes a WebRTC downstream leg from the media-service (or any user-supplied source) and plays it through a host `sounddevice.OutputStream`. The SDK ships the same lifecycle abstractions as the microphone — `SpeakerAudioStreamer` subclasses the shared `BaseAudioStreamer`, and `HostSpeakerCapture` wraps the device with idempotent `start()` / `stop()`. See the [native speaker driver](https://docs.cyberwave.com/feature-reference/edge/drivers/native-speaker-driver) page for the containerised edge build.

Install host speaker dependencies:

```bash
pip install cyberwave[speaker]
```

Play a local file through the default output device:

```python
from cyberwave.sensor.speaker import play_file

play_file("hello.mp3")  # supports mp3, wav, flac, ogg, ...
```

Subscribe a speaker twin to a peer microphone twin (Zenoh or WebRTC, your choice):

```python
import asyncio
from cyberwave import Cyberwave
from cyberwave.sensor.speaker import associate_speaker_to_microphone

cw = Cyberwave()
session = await associate_speaker_to_microphone(
    cw,
    speaker_twin_uuid="speaker-twin-uuid",
    microphone_twin_uuid="mic-twin-uuid",
    transport="webrtc",  # or "zenoh"
)
# ... run until done ...
await session.stop()
```

The lower-level building blocks (`SpeakerAudioStreamer`, `SpeakerAudioTrack`, `HostSpeakerCapture`, `list_host_sound_devices`, plus volume/gain/routing helpers) are available directly from `cyberwave.sensor.speaker` for custom integrations.

### Drones and Flying Twins

Twins whose asset has `can_fly: true` are returned as `FlyingTwin` instances.
`FlyingTwin` inherits from `LocomoteTwin`, so flying twins also expose the
locomotion verbs (`move_forward`, `move_backward`, `turn_left`, `turn_right`)
on top of their aerial-specific surface:

| Surface | Methods |
| --- | --- |
| Flight phase | `takeoff()`, `land()`, `hover()`, `cancel_takeoff()`, `cancel_landing()` |
| Return to home | `return_to_home()`, `cancel_return_to_home()`, `set_home_here()` |
| Service / safety | `start_compass_calibration()`, `stop_compass_calibration()`, `reboot()`, `emergency_stop()` |
| Gimbal | `gimbal_rotate(pitch=..., roll=..., yaw=..., mode=..., duration=...)`, `gimbal_recenter()`, `gimbal_rotate_speed(pitch=..., roll=..., yaw=...)` |
| Locomotion (inherited) | `move_forward()`, `move_backward()`, `turn_left()`, `turn_right()` |

All commands publish on the canonical `{topic_prefix}cyberwave/twin/{uuid}/command`
topic with the standard `{source_type, command, data, timestamp}` envelope —
the same contract every Cyberwave drone driver listens on.

```python
from cyberwave import Cyberwave
from cyberwave.twin import FlyingTwin

cw = Cyberwave()
cw.affect("real-world")  # or "simulation" for a dry run

drone: FlyingTwin = cw.twin("SZ-DJI-Technology/DJI-Mini-4-Pro")  # type: ignore[assignment]

drone.takeoff(altitude=2.0)
drone.move_forward(1.5)                          # locomotion (sim + future off-RC teleop)
drone.gimbal_rotate(pitch=-45.0, duration=1.5)   # tilt camera 45° down
drone.gimbal_rotate_speed(pitch=50.0)            # cinematic pan @ 5°/s
drone.gimbal_recenter()                          # back to 0° / absolute
drone.return_to_home()                           # KeyStartGoHome (with confirm flow)
drone.land()                                     # auto-arms landing-confirmation flow
```

Hovering helpers persist the controller's *intent* in `twin.metadata.status`
(useful in the Cyberwave playground simulator, where setting
`controller_requested_hovering=True` disables gravity for that twin so it stays
at its current altitude visually):

```python
if drone.is_hovering():
    status = drone.get_hovering_status()
    print(f"Hovering at {status['controller_requested_hovering_altitude']} m")
```

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

See the full DJI Mini 4 Pro walkthrough in
[examples/drone_dji_mini.py](examples/drone_dji_mini.py), and the simpler
hovering-only flow in [examples/drone_hovering.py](examples/drone_hovering.py).

## Twin control guide

How to command twins in SDK **0.5.x**: grouped capability handles, asset-driven MQTT catalogs, and separate **simulation** vs **live** state buckets. Runnable scripts for each topic live under [examples/](examples) ([examples/README.md](examples/README.md)).

### Prerequisites

```bash
export CYBERWAVE_API_KEY=your_key   # or: cyberwave login --token …
pip install cyberwave
```

Use **asset slugs** (e.g. `the-robot-studio/so101`, `unitree/go2`) with `cw.twin(slug)`. Call `cw.affect("simulation")` or `cw.affect("live")` before locomotion and joint MQTT so commands and cached state target the right runtime.

### Scene layout

Editor placement does **not** go over MQTT. Use `edit_position` / `edit_rotation` to move the twin in the environment scene.

```python
arm = cw.twin("the-robot-studio/so101")
arm.edit_position(x=1, y=0, z=0.5)
arm.edit_rotation(yaw=90)
```

→ [examples/quickstart.py](examples/quickstart.py)

### Joints

Joint names come from `twin.joints.list()`. Read/write by **name** (not numeric index). Positions are radians unless you pass `degrees=True` on `joints.set`.

```python
joint_names = arm.joints.list()
arm.set_joints({joint_names[0]: -0.2})
print(arm.get_joints())
# arm.get_joints(what_data=("position", "velocity", "effort"))
```

→ [examples/joints.py](examples/joints.py), [examples/compact.py](examples/compact.py). Deeper API: [Joint Control](#joint-control) below.

### Locomotion

Locomotion twins (Go2, UGV, …) accept velocity-style commands. The SDK publishes a **burst** of MQTT messages plus an explicit `stop` so edge drivers with velocity watchdogs stay alive.

```python
cw.affect("simulation")  # or "live"
robot = cw.twin("unitree/go2")
robot.locomotion.move_forward(0.3, duration=0.5, rate_hz=10)
robot.turn_left(0.5, duration=0.3)
# Top-level shortcuts: robot.move_forward(0.3, duration=0.5)
```

Speed is **m/s** (forward/back) or **rad/s** (turns), not travel distance. Do not use deprecated `move()` / `rotate()` for locomotion — use `edit_rotation()` for editor layout only.

→ [examples/locomotion.py](examples/locomotion.py)

### Command catalog

Each asset can declare MQTT commands in `cw-driver.yml` (`metadata["mqtt"]["commands"]`). The SDK binds `twin.commands.<name>(**kwargs)` at runtime. Locomotion names delegate to `twin.locomotion` (burst); other names publish once.

```python
dog = cw.twin("unitree/go2")
print(dog.commands.get_schema()["commands"]["supported"])
dog.commands.move_forward(linear_x=0.3, duration=0.5, rate_hz=10)
# dog.commands.sit_down()  # catalog-only, single publish
```

→ [examples/commands_catalog.py](examples/commands_catalog.py). Introspection: `twin.describe()`, `twin.commands.get_schema()`.

### Inbound MQTT (`listen`)

Subscribe to catalog **inbound** topics with `twin.listen(filters=[…])`. Returns a session; call `session.stop()` when done. Cached state is read with `get_joints()`, `pose.get()`, etc.

```python
session = arm.listen(filters=["joints", "pose"])
time.sleep(2)
print(arm.get_joints())
session.stop()
```

Filter slugs match the asset driver manifest (e.g. `joints`, `pose`, `power`). Replaces deprecated `subscribe_position()` / `subscribe_rotation()`.

→ [examples/listen_mqtt.py](examples/listen_mqtt.py)

### Runtime mode (`affect`)

`cw.affect("simulation")` and `cw.affect("live")` set `config.runtime_mode` and the default control `source_type`. Inbound MQTT updates land in separate **buckets**; `get_joints()` reads the bucket for the active mode.

```python
cw.affect("simulation")
arm.set_joints({joint_names[0]: -0.2})
cw.affect("live")
arm.set_joints({joint_names[-1]: 0.2})
```

→ [examples/runtime_mode.py](examples/runtime_mode.py). Also: [Simulation vs. Live](#simulation-vs-live).

### Teleop policy

In **live** mode, attach a workspace controller policy before joint/locomotion MQTT when the platform expects teleop routing.

```python
cw.affect("live")
arm.policy.ensure_attached()
# arm.policy.assign(arm.policy.list()[0])
arm.set_joints({joint_names[-1]: 0.2})
```

→ [examples/policy_assign.py](examples/policy_assign.py)

### Pose reads

| Twin kind | `get_pose()` meaning | Preferred handle |
| --- | --- | --- |
| Manipulator (arm) | Joint-space (alias for `get_joints()`) | `twin.get_joints()` / `twin.joints.get()` |
| Locomote (Go2, …) | World pose from MQTT | `twin.pose.get()` |

```python
print(arm.get_pose())           # joint-space on SO101
print(dog.pose.get())           # Cartesian on Go2
```

→ [examples/get_pose.py](examples/get_pose.py)

### Outbound MQTT rate limits

The MQTT client throttles **outbound** telemetry (not `listen` inbound):

| Channel | Limit |
| --- | --- |
| Position, rotation, scale, joints | ~40 Hz per twin/channel (`time.monotonic()` window) |
| GPS (`update_twin_gps`) | 2 Hz per twin; `fix_type="none"` is dropped without consuming the slot |

Duplicate position/rotation payloads are deduplicated before publish.

### Discovery

```python
print(twin.describe())  # handles, catalog commands, routing (via, continuous)
```

---

## Examples

Runnable scripts in [examples/](examples). Full index: [examples/README.md](examples/README.md).

### Running examples

```bash
cd cyberwave-sdks/cyberwave-python
export CYBERWAVE_API_KEY=your_key
poetry install
poetry run python examples/quickstart.py
```

Adjust twin slugs (`the-robot-studio/so101`, `unitree/go2`, …) to match your workspace catalog.

### Twin API scripts

| Script | Section |
| --- | --- |
| [quickstart.py](examples/quickstart.py) | [Scene layout](#scene-layout-rest), [Joints](#joints), [Locomotion](#locomotion) |
| [joints.py](examples/joints.py) | [Joints](#joints) |
| [compact.py](examples/compact.py) | [Joints](#joints) (one-liner) |
| [locomotion.py](examples/locomotion.py) | [Locomotion](#locomotion) |
| [commands_catalog.py](examples/commands_catalog.py) | [Command catalog](#command-catalog) |
| [listen_mqtt.py](examples/listen_mqtt.py) | [Inbound MQTT](#inbound-mqtt-listen) |
| [runtime_mode.py](examples/runtime_mode.py) | [Runtime mode](#runtime-mode-affect) |
| [policy_assign.py](examples/policy_assign.py) | [Teleop policy](#teleop-policy) |
| [get_pose.py](examples/get_pose.py) | [Pose reads](#pose-reads) |

`go2_locomotion.py` was removed — use [locomotion.py](examples/locomotion.py) or [commands_catalog.py](examples/commands_catalog.py).

### Streaming and cameras

- [camera_stream.py](examples/camera_stream.py) — WebRTC / twin frames
- [capture_frame.py](examples/capture_frame.py) — single frame capture
- [audio_stream.py](examples/audio_stream.py), [multimedia_stream.py](examples/multimedia_stream.py) — Go2 mic + A/V
- [realsense_stream.py](examples/realsense_stream.py) — RGB + depth
- [webcam_pose_anonymize.py](examples/webcam_pose_anonymize.py) — local webcam + frame filters

### Drones and flight

- [drone_hovering.py](examples/drone_hovering.py) — takeoff, hover, land ([Drones](#drones-and-flying-twins))
- [drone_dji_mini.py](examples/drone_dji_mini.py) — DJI Mini 4 Pro flight + gimbal
- [flying.py](examples/flying.py) — general flight helpers

### Edge, Zenoh, and workers

- [edge_worker_detect_people.py](examples/edge_worker_detect_people.py), [edge_worker_hailo_detect.py](examples/edge_worker_hailo_detect.py) — edge ML workers
- [zenoh_triad.py](examples/zenoh_triad.py), [zenoh_fanout.py](examples/zenoh_fanout.py), [zenoh_bench.py](examples/zenoh_bench.py) — Zenoh pub/sub
- [command_receiver_simple.py](examples/command_receiver_simple.py) — inbound MQTT on edge

### Workflows, missions, and data

- [workflows.py](examples/workflows.py) — list, trigger, poll runs
- [missions.py](examples/missions.py), [datasets.py](examples/datasets.py)
- [alerts_example.py](examples/alerts_example.py)

### Machine learning (Colab)

- [examples/ai/yolo.ipynb](examples/ai/yolo.ipynb) — YOLO26 tasks ([Try on Google Colab](#try-on-google-colab))
- [examples/ai/yolo_minimal_on_twin.ipynb](examples/ai/yolo_minimal_on_twin.ipynb) — live twin frame → predict

## Advanced Usage

### Joint Control

> **Getting started:** see [Joints](#joints) in the Twin control guide and [examples/joints.py](examples/joints.py).

You can change a specific joint actuation. Positions are **radians by default**; pass ``degrees=True`` for degrees:

```python
import math

robot = cw.twin("the-robot-studio/so101")

# Set individual joints (radians by default)
robot.joints.set("shoulder_joint", math.pi / 4)

# Or use degrees
robot.joints.set("elbow_joint", 45, degrees=True)

# List controllable joint names
joint_names = robot.joints.list()

# Read positions (radians) — all joints by default
all_joints = robot.joints.get()

# Subset + multiple state kinds (PR3: live MQTT cache; PR1: local cache)
subset = robot.joints.get(what_joints=["shoulder_joint"], what_data=["position"])
states = robot.joints.get(what_data=["position", "velocity"])

# Set one joint or many (radians by default)
robot.joints.set("shoulder_joint", math.pi / 4)
robot.joints.set({"shoulder_joint": math.pi / 4, "elbow_joint": 0.5})

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

### Saved Movements and Poses

Replay saved poses and movements on a twin without thinking about which scope owns them. `run_movement` sends the movement action contract, `move_to_pose` snaps the twin to a saved pose, and `list_movements` enumerates everything available.

```python
robot = cw.twin("the-robot-studio/so101")

robot.list_movements()              # twin, asset, and environment scopes
robot.run_movement("Wave")          # plays whichever scope owns "Wave"
robot.move_to_pose("Stand")         # snaps to the saved pose by name
```

> Note: `run_movement`, `move_to_pose`, and `list_movements` default to `scope="auto"`, so the backend resolves the name across the twin/asset/environment scopes. Pass `scope="twin"`, `scope="asset"`, or `scope="environment"` to pin the lookup to a specific scope.

### Simulation vs. Live

> **Getting started:** see [Runtime mode (`affect`)](#runtime-mode-affect) and [examples/runtime_mode.py](examples/runtime_mode.py).

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

This same mode selection is also used by ``twin.get_frame(source='cloud')``, so frame retrieval stays consistent with the mode your script is affecting.

### Agent SDK

The SDK exposes typed agent namespaces under `cw.agents`. Use direct resource APIs for deterministic commands, and agent APIs when you want backend planning, previews, setup guidance, or explicit dispatch.

- `cw.agents.environment`: environment editor agent messages and agent-created environments.
- `cw.agents.workflow`: workflow planning, preview, setup-and-draft, and constrained workflow edits.
- `cw.agents.control`: control surfaces, route/action planning, route resolution, and explicit dispatch.
- `cw.agents.embodiment`: server-built embodiment context for an environment or twin.

`cw.control` is a convenience alias for `cw.agents.control`.

```python
cw = Cyberwave(mode="simulation")

surfaces = cw.agents.control.surfaces("environment-uuid")
print(surfaces[0]["capabilities"])

plan = cw.agents.control.plan(
    "environment-uuid",
    "Move the Go2 to Waypoint A",
    twin_uuid="twin-uuid",
    mode="simulation",
)

response = cw.control.dispatch(
    "environment-uuid",
    plan["dispatchable_actions"][0],
    confirmed=True,
)

status = cw.actions.wait(
    response["action_id"],
    twin_uuid="twin-uuid",
    timeout=60,
)
```

Workflow and environment agents follow the same plan/preview/apply shape:

```python
draft = cw.agents.workflow.plan(
    "environment-uuid",
    "inspect every pallet and alert if damage is detected",
)

preview = cw.agents.workflow.preview(
    "environment-uuid",
    "inspect every pallet and alert if damage is detected",
)

context = cw.agents.embodiment.context(
    "environment-uuid",
    twin_uuid="twin-uuid",
)
```

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

### GPS Telemetry

Publish raw GNSS data for twins equipped with a GPS receiver. GPS data is
stored in the backend as `twin_gps_update` telemetry events and does **not**
affect the twin's rendered position (use `cw.mqtt.update_twin_position` for
that). The MQTT client rate-limits GPS publishes to **2 Hz per twin**.

```python
# Via the MQTT client directly
cw.mqtt.update_twin_gps(
    twin_uuid="your-twin-uuid",
    latitude=37.7749,
    longitude=-122.4194,
    altitude=10.5,
    satellite_count=12,
    signal_level=5,
    compass_heading=270.0,
)

# Via BaseEdgeNode helper
class MyGpsNode(BaseEdgeNode):
    async def _setup(self):
        pass

    async def _main_loop(self):
        while self.running:
            fix = read_gps_receiver()
            for twin_uuid in self._get_twin_uuids():
                self.publish_gps(
                    twin_uuid,
                    latitude=fix.lat,
                    longitude=fix.lon,
                    altitude=fix.alt,
                    satellite_count=fix.sats,
                )
            await asyncio.sleep(1.0)
```

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

Import, manage, and export robotics datasets from HuggingFace or local files.

```python
# Import a dataset from HuggingFace.
# Idempotent by default: if the same repo was already imported it is reused.
# Pass reuse_existing=False to force a fresh import.
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

# Get, visualize URL, delete
ds = cw.datasets.get("dataset-uuid")
url = cw.datasets.visualize(ds)   # returns frontend URL (does not print)
print(url)                         # https://cyberwave.com/acme/datasets/pusht
cw.datasets.delete("dataset-uuid")

# Wait for async HuggingFace import to complete.
# Prints one status line per poll by default; pass on_poll=None to silence.
ds = cw.datasets.wait_until_ready(ds)
# With custom callback:
ds = cw.datasets.wait_until_ready(
    ds,
    poll_interval=5.0,
    timeout=1800,
    on_poll=lambda d: print(f"{d.processing_status} {d.processed_episodes}/{d.total_episodes}"),
)
# Fully silent (for libraries/production):
ds = cw.datasets.wait_until_ready(ds, on_poll=None)
```

Available properties on `DatasetSchema`: `uuid`, `name`, `slug`, `processing_status`,
`is_ready`, `total_episodes`, `processed_episodes`, `failed_episodes`,
`failed_episode_uuids`, `source`, `source_format`, `visibility`, `metadata`.

**Supported import source formats:**
- **LeRobot v3** (`lerobot3`) — Latest LeRobot format with parquet episodes
- **LeRobot v2.1** (`lerobot21`) — Normalised to LeRobot v3 automatically
- **RLDS** — TensorFlow Datasets / Open-X-Embodiment (zip upload)
- **Cyberwave Parquet** — Native format for natively generated datasets

### Export / download a converted format

Both calls are idempotent — if a conversion artifact already exists it is returned immediately; otherwise conversion is kicked off automatically.

```python
# Block until backend conversion is done, return the signed URL.
# Default on_poll prints one status line per poll; pass on_poll=None to silence.
url = cw.datasets.convert(ds, "rlds")
print(url)   # signed URL valid for 24 h

# Convert AND stream the zip to disk in one call.
path = cw.datasets.download(ds, "rlds", dest="./data")
print(path)  # absolute path to saved file

# Silence both:
url = cw.datasets.convert(ds, "rlds", on_poll=None)
path = cw.datasets.download(ds, "rlds", dest="./data", on_poll=None)
```

**Supported output formats:**

| `format` | Description |
|---|---|
| `parquet` | Cyberwave joined-parquet zip (native) |
| `lerobot3` | LeRobot v3 — recommended for LeRobot training pipelines |
| `lerobot21` | LeRobot v2.1 |
| `rlds` | RLDS / TF-Record (Open-X-Embodiment) |
| `openvla` | Cyberwave OpenVLA TFDS bundle |
| `robodm` | Berkeley `.vla` format |

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

## ML Models (`cw.models`)

Unified catalog, edge/cloud runtime, optional cascade inference, and a typed playground client bound to **`POST /api/v1/mlmodels/{uuid}/run`**. Vision / Ultralytics backends need the **`[ml]`** extra: `pip install cyberwave[ml]`.

**Catalog** (authenticated client): `cw.models.list()` with optional **`deployment`** (server-side) and **`filters`** (client-side shorthands — `edge`, `cloud`, `image`, unknown strings match tags), **`get` / `get_by_uuid` / `get_by_slug`**, **`delete`**. **`cw.models.create()` / `cw.models.update()`** are still stubs — use REST or generated `cw.api.*` helpers.

**Runtime**: `cw.models.load(id_or_entry)` resolves **`sdk_load_id` → slug → UUID** when you pass an **`MLModelSchema`**. **`load([a, b, ...])`** returns a **`CascadeModel`**. **`predict()`** returns a concrete **`PredictionResult` subclass** — **`TextResult`**, **`DetectionResult`**, **`ImageResult`**, etc. — not a wrapper. Example: `txt = model.predict(prompt="…")` is a **`TextResult`**; use **`txt.text`**, **`txt.save("out.txt")`**, **`txt.describe()`**. Images: **`img.save_jpg("out.jpg")`**, **`img.to_ndarray()`**. Detection-shaped models return **`DetectionResult`** (or pose/segment subtypes); use **`result.detections`** and **`for det in result:`** only there — STT and other runtimes use their own fields (`.text`, `.data`, …).

Cloud models map playground **`output_format`** into these types directly. **`input_data`** is optional for prompt-only text/image runs.

**Playground**: `cw.models.playground(slug).run(...)` → raw **`MLModelRunResultSchema`**. Prefer **`cw.models.load(slug).predict(...)`** for typed results.

```python
cw = Cyberwave()
model = cw.models.load("yolov8n")
result = model.predict(frame)
print(result.describe())
for det in result:  # when output is DetectionResult-backed
    print(f"{det.label}: {det.confidence:.2f}")
```

YOLO ONNX postprocessing applies per-class non-max suppression with the same default IoU threshold (`0.7`) as Ultralytics' `YOLO.predict`, so swapping `yolov8s.pt` for `yolov8s.onnx` produces the same number of boxes per object instead of a cluster of overlapping anchors. Tune via `model.predict(frame, iou=0.5)` (stricter) or pass `iou=1.0` to disable NMS entirely (raw output for custom trackers).

NMS-free / end-to-end exports (YOLO26's default one-to-one head, YOLOv10) are detected automatically and routed through a separate decoder that parses the leading `[x1, y1, x2, y2, conf, class_id]` fields directly. Detection exports use `[max_det, 6]`; segmentation e2e appends mask coefficients (`[max_det, 38]` by default); pose and OBB use `[max_det, 57]` and `[max_det, 7]` respectively. Mask/keypoint/angle columns are not decoded yet — boxes and labels work. The `iou` knob is ignored on the e2e path — suppression is already applied inside the model graph.

### Hailo edge accelerator — *stub*

Hailo HEFs (`.hef`) are first-class catalog entries alongside their `.pt` / `.onnx` siblings. Edge Core picks the `cyberwaveos/edge-ml-worker-hailo` worker image when `/dev/hailo0` is present and passes the device through to the container; inside the container `hailo_platform` is preinstalled and matched to the host HailoRT driver. The SDK detects the `hailo` runtime from the `.hef` extension or the `_h8` / `_h8l` / `_hailo` slug suffix used in the catalog to distinguish hardware variants (e.g. `yolov8s_h8` vs `yolov8s_h8l`):

```python
model = cw.models.load("yolov8s_h8")
result = model.predict(frame, classes=["person"])
```

The `ml-hailo` extra is an opt-in marker only — `hailo_platform` is not on PyPI and is installed out-of-band by the worker image:

```bash
pip install "cyberwave[ml-hailo]"   # marker only; install HailoRT separately
```

### Edge speech-to-text — *stub*

The SDK includes a `whisper_cpp` runtime for local STT on devices like Raspberry Pi 4. Install the STT extra on the edge device, then load a Whisper.cpp GGML/GGUF checkpoint:

```bash
pip install "cyberwave[ml-stt]"
```

```python
model = cw.models.load(
    "models/whisper/ggml-tiny.en-q5_1.bin",
    runtime="whisper_cpp",
    download_url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin",
)
result = model.predict(audio, sample_rate_hz=16000, channels=1, language="en")
print(result.raw["text"])
```

Workflow-generated `Audio Track -> Call Model` edge workers pass the seeded `download_url` automatically, so the first run downloads the Whisper weights into the local Cyberwave model cache and later runs reuse the cached file.

Additional seeded whisper.cpp checkpoints include **Small EN** (`ggml-small.en-q5_1.bin`) and hybrid **multilingual** Tiny/Base (`ggml-tiny-q5_1.bin`, `ggml-base-q5_1.bin`) with cloud fallback via the whisper node.

Hybrid catalog entries can use the `faster_whisper` runtime (CTranslate2) for higher throughput on CPU/GPU edge nodes:

```bash
pip install "cyberwave[ml-stt-faster]"
```

```python
model = cw.models.load(
    "models/whisper/faster-whisper-tiny.en",
    runtime="faster_whisper",
    faster_whisper_model_id="tiny.en",
    compute_type="int8",
    device="cpu",
)
result = model.predict(audio, sample_rate_hz=16000, channels=1, language="en")
print(result.raw["text"])
```

### Model warm-up — *stub*

Eliminate cold-start latency by calling `warm_up()` after loading:

```python
model = cw.models.load("yolov8n")
cold_ms, warm_ms = model.warm_up()  # two dummy inferences
# cold_ms ≈ 150 ms (JIT/allocation), warm_ms ≈ 8 ms (steady-state)
```

The worker runtime calls `warm_up()` on all loaded models **before** audio/frame hooks are activated, and serializes `predict()` per model so whisper.cpp / faster-whisper never see concurrent `transcribe()` calls.

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

## Version 0.5.0 twin API changelog (in progress)

Migration reference for SDK **0.5.x**. For usage and examples, start with the [Twin control guide](#twin-control-guide) and [Examples](#examples).

### Prefer this (unified surface)

| Area | Use | Notes |
| --- | --- | --- |
| **Commands (catalog)** | `twin.commands.<name>(**kwargs)` | Names from seeded `metadata["mqtt"]["commands"]["supported"]` (driver `cw-driver.yml`). Per-command `commands.specs[name].continuous` triggers `publish_command_burst` (repeated MQTT + `stop`); otherwise delegates to capability handles or a single publish. |
| **Locomotion** | `twin.locomotion.move_forward(distance, duration=…)` or top-level `twin.move_forward(…)` | Repeated publish + `stop` for velocity watchdogs. Speed is **m/s** (forward/back) or **rad/s** (turns), not travel distance. |
| **Flight** | `twin.flight.takeoff(…)` / `drone.takeoff(…)` | Same MQTT envelope; catalog `twin.commands.takeoff` delegates when listed. |
| **Joints (read)** | `twin.get_joints()` / `twin.get_pose()` (manipulators) or `twin.joints.get()` | Same API. Default: positions only; pass `what_data=("position", "velocity", "acceleration", "effort")` for multi-field state (effort = torque). MQTT-backed; sim/live buckets via `runtime_mode` + inbound `source_type`. |
| **Joints (write)** | `twin.set_joints()` / `twin.set_pose()` or `twin.joints.set(...)` | Same API. `what_data` selects position / velocity / acceleration / effort on publish. |
| **Cartesian pose (read)** | `twin.pose.get()` / `twin.get_pose()` (locomote twins) | World pose from position/rotation/kinematics topics; not the same as joint-space `get_pose()` on arms. |
| **Policy / teleop** | `twin.policy.keyboard()` | `twin.controller` was removed. |
| **Inbound MQTT** | `twin.listen(filters=["pose", "joints", …])` | Catalog-driven topics from asset `cw-driver.yml`; returns a session with `.stop()`. Replaces ad-hoc `subscribe_*` helpers. |
| **Discovery** | `twin.describe()` | Handles, catalog methods, `command_routing` (`via`, `continuous`), `mqtt.specs`. |
| **Runtime** | `cw.affect("live")` / `cw.affect("simulation")` | Sets `config.runtime_mode` and default control `source_type`; drives which MQTT bucket `get()` reads. |
| **Frames** | `twin.get_frame()` / `twin.get_frames()` | `source=` cloud, local, remote_edge, zenoh. |
| **Catalog introspection** | `twin.commands.get_schema()` | Topics, `commands.supported`, `commands.specs` (`continuous`, `rate_hz`), optional `joint_control`. Re-seed assets after editing `cw-driver.yml` (`seed_asset_driver_config`). |
| **Register manifest** | `twin.commands.set_schema("cw-driver.yml")` | Compiles the YAML, persists `metadata["mqtt"]` on your twin, and re-binds catalog-derived `twin.commands.<name>` methods. |

```python
# Go2-style: catalog entry and locomotion share behavior when delegated
rover.commands.move_forward(linear_x=0.5, duration=0.2, rate_hz=10)
rover.locomotion.move_forward(0.5, duration=0.2, rate_hz=10)  # equivalent burst path

joint_names = arm.joints.list()
arm.set_joints({joint_names[0]: 0.5, joint_names[1]: -0.2})
arm.get_joints(what_data=("position", "velocity", "effort"))

session = arm.listen(filters=["joints", "pose"])
# ... inbound MQTT updates cached state ...
session.stop()
```

### Removed or renamed

These are **gone or renamed** in the current SDK. Calling the old surface fails or no longer matches intent.

| Old | Status | Use instead |
| --- | --- | --- |
| `twin.controller` | **Removed** (`AttributeError`) | `twin.policy` — `twin.policy.keyboard()`, `twin.policy.ensure_attached()`, etc. |
| Ad-hoc command publish without catalog | **Replaced** by contract | `twin.commands.<name>()` from asset `commands.supported`, or capability handles (`locomotion`, `flight`, …) |

**Stable shortcuts (not deprecated):** `twin.get_joints()`, `twin.get_pose()`, `twin.set_joints()`, `twin.set_pose()` on manipulator twins; on the handle use only `twin.joints.get()` / `twin.joints.set()`.

### Deprecated (still works — migrate before removal)

Still callable today; emits `DeprecationWarning` or a logged warning. Will be removed in a later release.

| Old | Warning | Use instead |
| --- | --- | --- |
| `LocomoteTwin.move(position)` | Log warning; **no-op** | `twin.move_forward()` / `twin.locomotion.*`, or `edit_position()` for scene layout |
| `LocomoteTwin.rotate(...)` | Log warning; delegates to `edit_rotation` | `edit_rotation()` for editor layout; locomotion via `turn_left` / `turn_right` |
| `twin.subscribe(on_update)` | `DeprecationWarning` | `twin.listen(filters=[…])` or `handlers={slug: fn}` |
| `twin.subscribe_position()` / `subscribe_rotation()` | `DeprecationWarning` | `twin.listen(filters=["pose"])` + `twin.pose.get()` (or poll) |
| `examples/go2_locomotion.py` | **Removed** (duplicate) | [locomotion.py](examples/locomotion.py) or [commands_catalog.py](examples/commands_catalog.py) |
| `twin.get_controllable_joint_names()` | `DeprecationWarning` | `twin.joints.list()` |
| `twin.get_calibration()` / `update_calibration()` / `delete_calibration()` | `DeprecationWarning` | `twin.joints.calibration.get()` / `.set()` / `.delete()` |
| `joints.get_all()` / `joints.set_joints()` | `DeprecationWarning` on **handle** | `joints.get()` / `joints.set(...)` (twin shortcuts above stay) |
| `twin.get_latest_frame()` | `DeprecationWarning` | `twin.get_frame(source="cloud")` |
| `twin.capture_frame()` / `twin.capture_frames()` | `DeprecationWarning` | `twin.get_frame()` / `twin.get_frames()` |
| `twin.camera.latest_frame()`, `.capture()`, `.read()`, `.snapshot()`, `.edge_photo()`, `.edge_photos()` | `DeprecationWarning` | `twin.get_frame(source=…)` / `twin.get_frames()` |
| `Cyberwave(token=…)` ctor kwarg | `DeprecationWarning` | `Cyberwave(api_key=…)` |
| `client.video_stream(...)` | Documented deprecated; **still present** | Prefer twin-centric streaming (`twin.stream_camera()` / WebRTC helpers on camera twins) where available |
| `client.controller(twin_uuid)` | Documented deprecated; **still present** | `twin.policy` + MQTT command handles — not `twin.controller` |

### Not fully implemented yet

| Surface | Status |
| --- | --- |
| `twin.pose.set()` | Raises `NotImplementedError` (use `edit_position()` / `edit_rotation()` for editor layout today). |
| `twin.camera.rotate()` | Stub |
| `twin.sensors` IMU / GPS / compass / LiDAR inbound | Listen hooks stub; MQTT decode not wired. |
| Catalog-only commands (e.g. Go2 `camera_up`, `sit_down`) | Single MQTT publish via `twin.commands.*` (no capability delegate until a handle method exists). |
| `cyberwave.zenoh_mqtt` bridge | Documented stub above. |
| Some `cw.models` CRUD helpers | `create` / `update` not implemented on manager yet. |

## Contributing

Contributions are welcome. If you have an idea, bug report, or improvement request, please open an issue or submit a pull request.

## Support

- **Documentation**: [docs.cyberwave.com](https://docs.cyberwave.com)
- **Issues**: [GitHub Issues](https://github.com/cyberwave-os/cyberwave-python/issues)
- **Community**: [Discord](https://discord.gg/dfGhNrawyF)
