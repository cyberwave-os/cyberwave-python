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

Then install the additional deps for camera streaming, you can select between standard cameras or intel realsense ones:

```bash
# Install with camera support
pip install cyberwave[camera]
pip install cyberwave[realsense]
```

```python
import asyncio
from cyberwave import Cyberwave

# Initialize client
cw = Cyberwave()

camera = cw.twin("cyberwave/standard-cam")

# Start streaming
async def stream_camera():
    await camera.start_streaming()
    # Stream runs until stopped
    await asyncio.sleep(60)  # Stream for 60 seconds
    await streamer.stop_streaming()

# Run the async function
asyncio.run(stream_camera())
```

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

## Examples

Check the [examples](examples) directory for complete examples:

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

- **Documentation**: [docs.cyberwave.com](https://docs.cyberwave.com)
- **Issues**: [GitHub Issues](https://github.com/cyberwave/cyberwave-python/issues)
- **Community**: [Discord](https://discord.gg/dfGhNrawyF)
