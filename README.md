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

The compact API spins up a twin locally by default and syncs with the backend when credentials are provided:

```python
import cyberwave as cw

# Create a twin (local sim if not authenticated)
arm = cw.twin("cyberwave/so101", name="lab-arm")

# Pose helpers
arm.move(x=0.3, z=0.6)
arm.rotate(yaw=90)  # degrees
arm.move_to([0.3, 0.0, 0.6], orientation=[0, 0, 180])  # optional orientation

# Joint control (aliases depend on robot metadata)
joints = arm.joints
print(joints.aliases())          # ['base_yaw', 'shoulder_pitch', 'elbow_roll', ...]
print(joints.describe())         # structured joint metadata + cached positions

joints.set("shoulder_pitch", 42.5)
joints.apply({"elbow_roll": -15.0, 3: 8.0})  # mix alias + 1-based index
joints.register_alias("gripper", actual="finger_joint")
joints.set("gripper", 5.0)

# Simulation control (always available)
cw.simulation.play()
cw.simulation.step()
cw.simulation.pause()
```

#### Joint controller essentials

The `CompactTwin.joints` helper provides ergonomic access to joint state:

- `aliases()` / `indices()` enumerate the known joints discovered from kinematics and joint state metadata
- `describe()` or `snapshot()` returns metadata with positions
- `get(identifier)` reads a single joint (alias, backend name, or index)
- `set(identifier, value)` / attribute assignment updates a single joint
- `set_many({...})` or `apply({...})` batches updates in one backend call
- `register_alias("wrist", index=5)` creates friendly handles for dynamic setups
- `all()` returns the cached joint map

When the twin is authenticated, all setters transparently call `twins.set_joint`/`twins.set_joints` to keep the backend in sync. In offline mode the cache updates immediately so you can still prototype controllers.

### Configuration

```python
import cyberwave as cw
from cyberwave import AuthTrigger, CyberWaveEnvironment

# Configure the SDK (optional - compact API works without this)
cw.configure(
    api_key="your_api_key",  # or set CYBERWAVE_API_KEY
    backend_url="http://localhost:8000/api/v1",  # overrides environment lookup
    environment=CyberWaveEnvironment.LOCAL,
    project_name="Lab Robotics",          # auto-create if missing
    environment_name="Integration Floor",  # auto-create if missing
    auth_trigger=AuthTrigger.ON_PROTECTED_RESOURCE,
    protected_patterns=["prod/*"],
    fallback_to_local=True,
)

# Create twins once configured
drone = cw.twin("dji/tello")
robot_arm = cw.twin("kuka/kr3")
humanoid = cw.twin("berkeley/berkeley_humanoid")
```

### Advanced Usage - Mission API

For complex scenarios and mission planning:

```python
from cyberwave import Cyberwave, Mission

# Create client
client = Cyberwave(base_url="http://localhost:8000/api/v1", token="<TOKEN>")

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

# Example: nudge a joint on an existing twin while the mission runs
client.twins.set_joint("<TWIN_UUID>", "shoulder_pitch", 15.0)
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
