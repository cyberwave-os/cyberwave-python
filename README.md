# Cyberwave Python SDK

The official Python SDK for Cyberwave. Create, control, and simulate robotics with ease.

## Installation

```bash
pip install cyberwave
```

## Quick Start

### 1. Get Your Token

Get your API token from the Cyberwave platform:

- Log in to your Cyberwave instance
- Navigate to Settings → API Tokens
- Copy your token

### 2. Create Your First Digital Twin

```python
import cyberwave as cw

# Configure with your token
cw.configure(
    token="your_token_here",
)

# Create a digital twin from an asset
robot = cw.twin("the-robot-studio/so101")

# Control position and rotation
robot.move(x=1.0, y=0.0, z=0.5)
robot.rotate(yaw=90)  # degrees

# Control robot joints (for URDF assets)
robot.joints.shoulder_joint = 45  # degrees
robot.joints.elbow_joint = -30

# Get current joint positions
print(robot.joints.get_all())
```

## Core Features

### Working with Workspaces and Projects

```python
from cyberwave import Cyberwave

client = Cyberwave(
    token="your_token_here"
)

# List workspaces
workspaces = client.workspaces.list()
print(f"Found {len(workspaces)} workspaces")

# Create a project
project = client.projects.create(
    name="My Robotics Project",
    workspace_id=workspaces[0].uuid
)

# Create an environment
environment = client.environments.create(
    name="Development",
    project_id=project.uuid
)
```

### Managing Assets and Twins

```python
# Search for assets
assets = client.assets.search("robot")

# Create a twin from an asset
twin_data = client.twins.create(
    asset_id=assets[0].uuid,
    environment_id=environment.uuid,
    name="Robot-01"
)

# Use the high-level Twin API
from cyberwave import Twin
robot = Twin(client, twin_data)

# Move to a specific position
robot.move_to([1.0, 0.5, 0.0])

# Update scale
robot.scale(x=1.5, y=1.5, z=1.5)

# Delete when done
robot.delete()
```

### Real-time Updates with MQTT

```python
# Define callback for position updates
def on_position_change(data):
    print(f"Twin moved to: {data}")

# Subscribe to real-time updates
client.mqtt.subscribe_twin_position("twin_uuid", on_position_change)

# Publish position updates
client.mqtt.publish_twin_position(
    twin_id="twin_uuid",
    x=1.0, y=0.0, z=0.5
)

# Subscribe to joint states
def on_joint_update(data):
    print(f"Joint states: {data}")

client.mqtt.subscribe_joint_states("twin_uuid", on_joint_update)
```

### Video Streaming (WebRTC)

Stream camera feeds to your digital twins using WebRTC.

To stream you will need to install FFMPEG if you don't have it.

On Mac with brew:

```bash
brew install ffmpeg pkg-config
```

On Ubuntu:

```bash
sudo apt-get install ffmpeg
```

Then install the additional deps for camera streaming:

```bash
# Install with camera support
pip install cyberwave[camera]
```

```python
import asyncio
from cyberwave import Cyberwave

# Initialize client
client = Cyberwave(token="your_token_here")

# Create camera streamer - integrated into the Cyberwave client!
streamer = client.video_stream(
    twin_uuid="your_twin_uuid",
    camera_id=0,  # Default camera
    fps=10        # Frames per second
)

# Start streaming
async def stream_camera():
    await streamer.start()
    # Stream runs until stopped
    await asyncio.sleep(60)  # Stream for 60 seconds
    await streamer.stop()

# Run the async function
asyncio.run(stream_camera())
```

The `video_stream()` method:

- Automatically uses the client's MQTT connection
- Pre-configures the streamer with the twin UUID
- Handles WebRTC peer connection setup
- Manages ICE candidate gathering with STUN/TURN servers
- Handles video encoding and streaming

## Configuration Options

You can also set your token as environment variable:

```bash
export CYBERWAVE_TOKEN="your_token_here"
```

```python
import cyberwave as cw

# SDK will automatically load from environment variables
robot = cw.twin("the-robot-studio/so101")
```

### Programmatic Configuration

```python
import cyberwave as cw

cw.configure(
    token="your_token_here",              # Bearer token
    environment="env_uuid",                # Default environment
    workspace="workspace_uuid",            # Default workspace
)
```

## Advanced Usage

### Context Manager for Cleanup

```python
from cyberwave import Cyberwave

with Cyberwave(token="YOURTOKEN") as client:
    twins = client.twins.list()
    for twin in twins:
        print(twin.name)
# Automatically disconnects MQTT and cleans up resources
```

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

## API Reference

### Cyberwave Client

- `client.workspaces` - Workspace management
- `client.projects` - Project management
- `client.environments` - Environment management
- `client.assets` - Asset catalog operations
- `client.twins` - Digital twin CRUD operations
- `client.mqtt` - Real-time MQTT client

### Twin Class

- `twin.move(x, y, z)` - Move twin to position
- `twin.move_to([x, y, z])` - Move to position array
- `twin.rotate(yaw, pitch, roll)` - Rotate using euler angles
- `twin.rotate(quaternion=[x,y,z,w])` - Rotate using quaternion
- `twin.scale(x, y, z)` - Scale the twin
- `twin.joints` - Joint controller for robot manipulation
- `twin.delete()` - Delete the twin
- `twin.refresh()` - Reload twin data from server

## Examples

Check the SDK repository for complete examples:

- Basic twin control
- Multi-robot coordination
- Real-time synchronization
- Joint manipulation for robot arms

## Testing

### Unit Tests

Run basic import tests:

```bash
poetry install
poetry run python tests/test_imports.py
```

## Support

- **Documentation**: [cyberwave.com/docs](https://docs.cyberwave.com)
- **Issues**: [GitHub Issues](https://github.com/cyberwave/cyberwave-python/issues)
<!-- - **Community**: [Discord](https://discord.gg/cyberwave) -->

```

```
