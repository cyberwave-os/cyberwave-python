# Cyberwave Python SDK

The official Python SDK for the Cyberwave Digital Twin Platform. Create, control, and simulate digital twins with ease.

## Installation

```bash
pip install cyberwave
```

For robotics features:
```bash
pip install cyberwave cyberwave-robotics-integrations
```

## Quick Start

### Compact API (Recommended)

The easiest way to get started with digital twins:

```python
import cyberwave as cw

# One-liner to create a twin
robot = cw.twin("cyberwave/so101")

# Move and rotate
robot.move(x=1, y=0, z=0.5)
robot.rotate(yaw=90)  # degrees

# Robot control (for URDF assets)
robot.joints.arm_joint = 45  # degrees
robot.move_to([1, 0, 0.5])  # target position

# Simulation control
cw.simulation.play()
cw.simulation.step()
cw.simulation.pause()
```

### Configuration

```python
import cyberwave as cw

# Configure the SDK (optional - auto-configured by default)
cw.configure(
    api_key="your_api_key",
    base_url="http://localhost:8000",  # Your Cyberwave backend
    environment="your_environment_id"
)

# Create twins
drone = cw.twin("dji/tello")
robot_arm = cw.twin("kuka/kr3")
humanoid = cw.twin("berkeley/berkeley_humanoid")
```

### Advanced Usage - Mission API

For complex scenarios and mission planning:

```python
from cyberwave import Cyberwave, Mission

# Create client
client = Cyberwave(base_url="http://localhost:8000", token="<TOKEN>")

# Create mission
mission = Mission(key="demo/pick_and_place", version=1, name="Pick and Place Demo")

# Build world
world = mission.world()
world.asset("cyberwave/so101", alias="robot_arm")
world.asset("props/table", alias="table") 
world.asset("props/box", alias="target")

# Define positions
world.place("table", [0, 0, 0, 1, 0, 0, 0])
world.place("target", [0.5, 0, 0.8, 1, 0, 0, 0])
world.place("robot_arm", [0, 0, 0, 1, 0, 0, 0])

# Set mission goals
mission.goal_object_in_zone("target", "success_zone", tolerance_m=0.05)

# Register and run
client.missions.register(mission)
run = client.runs.start(
    environment_uuid="<ENV_UUID>", 
    mission_key=mission.key,
    mode="virtual"
)
```

## Features

- **🤖 Digital Twin Control** - Create and manipulate digital twins
- **🎮 Compact API** - Simple, intuitive interface for quick development
- **🏗️ Mission System** - Complex scenario planning and execution
- **🔄 Real-time Sync** - Live updates between digital and physical twins
- **📊 Simulation** - Physics-based simulation with multiple backends
- **🔌 Hardware Integration** - Connect to real robots and devices
- **📈 Analytics** - Built-in logging and performance monitoring

## Documentation

- **Catalog Integration** - Browse assets at [your-cyberwave-instance]/catalog
- **API Reference** - Complete method documentation
- **Examples** - See `examples/` directory for complete use cases
- **Robot Drivers** - Install `cyberwave-robotics-integrations` for hardware support

## Examples

See the `examples/` directory for complete examples:
- `quickstart_compact.py` - Compact API usage
- `quickstart_mvp.py` - Mission-based usage  
- `robot_control.py` - Advanced robot control
- `simulation_demo.py` - Simulation examples

