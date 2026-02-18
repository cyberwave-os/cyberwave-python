# Cyberwave Python SDK

The official Python SDK for Cyberwave. Create, control, and simulate robotics with ease.

## Status

[![PyPI version](https://img.shields.io/pypi/v/cyberwave.svg)](https://pypi.org/project/cyberwave/)
[![PyPI Python versions](https://img.shields.io/pypi/pyversions/cyberwave.svg)](https://pypi.org/project/cyberwave/)
![Tests](https://github.com/cyberwave-os/cyberwave-python/actions/workflows/test.yml/badge.svg)
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

### 1. Get Your Token

Get your API token from the Cyberwave platform:

- Log in to your Cyberwave instance
- Navigate to [Profile](https://cyberwave.com/profile) → API Tokens
- Create a token and copy it

### 2. Create Your First Digital Twin

```python
from cyberwave import Cyberwave

# Configure with your token
cw = Cyberwave(
    token="your_token_here",
)

# Create a digital twin from an asset
robot = cw.twin("the-robot-studio/so101")

# Change position and rotation in the environemtn
robot.edit_positon(x=1.0, y=0.0, z=0.5)
robot.edit_rotation(yaw=90)  # degrees

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
    token="your_token_here"
)

# You can also set your token as an environment variable: export CYBERWAVE_TOKEN=your_token_here
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
robot = cw.twin(assets[0].registry_id) # the registry_id is the unique identifier for the asset in the catalog. in this case it's the-robot-studio/so101

# Edit the twin to a specific position
robot.edit_position([1.0, 0.5, 0.0])

# Update scale
robot.edit_scale(x=1.5, y=1.5, z=1.5)

# Move a joint to a specific position using radians
robot.joints.set("shoulder_joint", math.pi/4)

# You can also use degrees:
robot.joints.set("shoulder_joint", 45, degrees=True)

# You can also go a get_or_create for a specific twin an environment you created:
 robot = cw.twin("the-robot-studio/so101", environment_id="YOUR_ENVIRONMENT_ID")
```

### Environment Variables

If you are always using the same environment, you can set it as a default with the CYBERWAVE_ENVIRONMENT_ID environment variable:

```bash
export CYBERWAVE_ENVIRONMENT_ID="YOUR_ENVIRONMENT_ID"
export CYBERWAVE_TOKEN="YOUR_TOKEN"
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

> **📌 Note for ARM64/Raspberry Pi**: The `pip install cyberwave[realsense]` command installs the Python wrapper, but you'll still need the librealsense SDK installed on your system. On x86_64 systems, you can install it via `sudo apt install librealsense2` or use pre-built wheels. **On Raspberry Pi OS (ARM64), you must build librealsense from source** - see our [Raspberry Pi Installation Guide](install_realsense_raspian_os.md).

#### Quick Start

```python
import asyncio
import os
from cyberwave import Cyberwave
cw = Cyberwave()
camera = cw.twin("cyberwave/standard-cam")

try:
    print(f"Streaming to twin {camera.uuid}... (Ctrl+C to stop)")
    await camera.start_streaming()

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
    await camera.start_streaming()

    while True:
        await asyncio.sleep(1)
except (KeyboardInterrupt, asyncio.CancelledError):
    print("\nStopping...")
finally:
    await camera.stop_streaming()
    cw.disconnect()
```

## Examples

Check the [examples](examples) directory for complete examples:

- Basic twin control
- Multi-robot coordination
- Real-time synchronization
- Joint manipulation for robot arms

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

### Changing data source

By default, the SDK will send data marked as arriving from the real world. If you want to send data from a simulated environment using the SDK, you can initialize the SDK as follows:

```python
from cyberwave import Cyberwave

cw = Cyberwave(source_type="sim")
```

You can also use the SDK as a client of the Studio editor - making it appear as if it was just another editor on the web app. To do so, you can initialize it as follows:

```python
from cyberwave import Cyberwave

cw = Cyberwave(source_type="edit")
```

Lastly, if you want to have your SDK act as a remote teleoperator, sending commands to the actual device from the cloud, you can init the SDK as follows:

```python
from cyberwave import Cyberwave

cw = Cyberwave(source_type="tele")
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

> **⚠️ Raspberry Pi / ARM64 Users**: If you're running on Raspberry Pi OS or other ARM64 systems, you'll need to manually build librealsense from source, as pre-built packages aren't available. See our [Raspberry Pi Installation Guide](install_realsense_raspian_os.md) for detailed instructions.

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

## Testing

### Unit Tests

Run basic import tests:

```bash
poetry install
poetry run python tests/test_imports.py
```

## Support

- **Documentation**: [docs.cyberwave.com](https://docs.cyberwave.com)
- **Issues**: [GitHub Issues](https://github.com/cyberwave/cyberwave-python/issues)
- **Community**: [Discord](https://discord.gg/dfGhNrawyF)
