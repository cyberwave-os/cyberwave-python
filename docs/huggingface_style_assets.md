# HuggingFace-Style Asset System

Cyberwave SDK provides a HuggingFace-inspired asset registry system that makes working with robots and sensors as easy as using pre-trained models.

## Quick Start

```python
from cyberwave.assets import DjiTello, BostonDynamicsSpot, IntelRealSenseD435

# Just like HuggingFace - instantiate by class
drone = DjiTello(ip="192.168.10.1")

# Or by registry ID
from cyberwave.assets import Robot
robot = Robot("boston-dynamics/spot", hostname="192.168.1.100")

# Access built-in capabilities and specs
print(drone.capabilities)  # ['flight', 'camera', 'wifi_control']
print(drone.specs['max_altitude'])  # 10 meters
```

## Asset Registry

The asset registry works like HuggingFace's model hub, providing pre-configured assets with known capabilities.

### Available Assets

```python
from cyberwave.assets import AssetRegistry

# List all robots
for asset in AssetRegistry.list(asset_type="robot"):
    print(f"{asset.asset_id}: {asset.manufacturer} {asset.model}")

# Output:
# dji/tello: DJI Tello
# boston-dynamics/spot: Boston Dynamics Spot
# unitree/go1: Unitree Go1
# franka/panda: Franka Panda
```

### Asset Hierarchy

```
BaseAsset
├── Sensor
│   ├── Camera
│   │   ├── IntelRealSenseD435  ("intel/realsense-d435")
│   │   └── ZedCamera           ("stereolabs/zed2")
│   └── Lidar
│       └── VelodynePuck        ("velodyne/puck")
│
└── Robot
    ├── FlyingRobot
    │   ├── DjiTello            ("dji/tello")
    │   ├── DjiMavic3           ("dji/mavic3")
    │   └── Parrot4K            ("parrot/anafi-4k")
    │
    ├── GroundRobot
    │   ├── BostonDynamicsSpot  ("boston-dynamics/spot")
    │   ├── UnitreeGo1          ("unitree/go1")
    │   └── ClearPathJackal     ("clearpath/jackal")
    │
    └── Manipulator
        ├── FrankaPanda         ("franka/panda")
        └── UniversalUR5        ("universal-robots/ur5")
```

## Usage Patterns

### 1. Simple Usage (No Platform)

```python
# Create and use locally
drone = DjiTello(ip="192.168.10.1")
await drone.connect()
await drone.takeoff()
await drone.move(x=1, y=0, z=0.5)
await drone.land()
```

### 2. Platform-Integrated Usage

```python
from cyberwave import Client
from cyberwave.assets import CyberwaveDjiTello

# Connect to platform
client = Client("https://api.cyberwave.dev")
await client.authenticate("api-key")

# Create drone with platform integration
drone = CyberwaveDjiTello(ip="192.168.10.1")
await drone.setup_on_platform(
    client=client,
    project_uuid="my-project",
    mode="hybrid"  # Physical device + digital twin
)

# All operations now sync to platform
await drone.takeoff()  # Updates digital twin
```

### 3. Generic Asset Creation

```python
# Create any asset by registry ID
robot = Robot("boston-dynamics/spot", 
    hostname="192.168.1.100",
    settings={"auto_stand": True}
)

# Override default specs
custom_drone = FlyingRobot("dji/tello",
    ip="192.168.10.1",
    specs={"max_altitude": 20}  # Override default 10m
)
```

## Creating Custom Assets

### Register Your Own Asset

```python
from cyberwave.assets import register_asset, FlyingRobot

@register_asset(
    "mycompany/custom-drone",
    asset_type="robot",
    default_capabilities=["flight", "thermal_camera", "gps"],
    default_specs={
        "max_altitude": 120,
        "flight_time": 25,
        "camera_resolution": "4K",
    }
)
class MyCustomDrone(FlyingRobot):
    def __init__(self, serial: str, **kwargs):
        super().__init__("mycompany/custom-drone", **kwargs)
        self.serial = serial
    
    async def connect(self):
        # Your connection logic
        pass
```

### Extend Existing Assets

```python
class InspectionTello(DjiTello):
    """Tello configured for inspection tasks"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capabilities.append("inspection")
        self.inspection_points = []
    
    async def add_inspection_point(self, x, y, z):
        self.inspection_points.append((x, y, z))
    
    async def run_inspection(self):
        for point in self.inspection_points:
            await self.move(*point)
            await self.capture_inspection_data()
```

## Twin Modes

When setting up assets on the platform, choose the appropriate mode:

- **`virtual`**: Simulation only (no physical hardware)
- **`physical`**: Physical device only (no simulation)
- **`hybrid`**: True digital twin (both physical and simulated)

```python
# Virtual robot for testing
virtual_spot = BostonDynamicsSpot(hostname="sim-001")
await virtual_spot.setup_on_platform(client, project_uuid, mode="virtual")

# Physical robot without simulation
physical_spot = BostonDynamicsSpot(hostname="192.168.1.100")
await physical_spot.setup_on_platform(client, project_uuid, mode="physical")

# Digital twin (recommended)
twin_spot = BostonDynamicsSpot(hostname="192.168.1.100")
await twin_spot.setup_on_platform(client, project_uuid, mode="hybrid")
```

## Configuration-Based Setup

```python
# Define fleet configuration
fleet_config = [
    {
        "asset_id": "dji/tello",
        "name": "Scout Alpha",
        "parameters": {"ip": "192.168.10.1"},
        "mode": "hybrid"
    },
    {
        "asset_id": "boston-dynamics/spot",
        "name": "Ground Unit 1", 
        "parameters": {"hostname": "192.168.1.100"},
        "mode": "hybrid"
    },
    {
        "asset_id": "intel/realsense-d435",
        "name": "Base Camera",
        "parameters": {"serial_number": "123456"},
        "mode": "physical"
    }
]

# Create fleet from config
from cyberwave.assets import AssetFactory

fleet = []
for config in fleet_config:
    asset = await AssetFactory.create_from_config({
        **config,
        "client": client,
        "project_uuid": "mission-project"
    })
    fleet.append(asset)
```

## Discovering Capabilities

```python
# Check what an asset can do
drone = DjiTello()
if "thermal_camera" in drone.capabilities:
    print("This drone has thermal imaging")

# Check specifications
if drone.specs.get('max_altitude', 0) > 50:
    print("High-altitude capable")

# Find all assets with specific capability
thermal_drones = [
    asset for asset in AssetRegistry.list()
    if "thermal_camera" in asset.default_capabilities
]
```

## Best Practices

1. **Use Registry IDs**: Prefer `Robot("dji/tello")` over direct class instantiation for flexibility
2. **Set Appropriate Modes**: Use `hybrid` for production, `virtual` for testing
3. **Extend Don't Modify**: Create subclasses rather than modifying registry entries
4. **Sync State**: Call `sync_state()` after significant operations
5. **Handle Connections**: Always wrap hardware operations in try/except blocks

## Advanced Features

### Dynamic Asset Loading

```python
# Load asset class dynamically
asset_id = "boston-dynamics/spot"
AssetClass = AssetRegistry.get(asset_id).asset_class
robot = AssetClass(hostname="192.168.1.100")
```

### Capability-Based Selection

```python
# Find all assets that can fly and have cameras
flying_cameras = [
    asset for asset in AssetRegistry.list()
    if "flight" in asset.default_capabilities 
    and "camera" in asset.default_capabilities
]
```

### Multi-Robot Coordination

```python
# Create heterogeneous robot team
team = {
    "aerial": [DjiTello(ip=f"192.168.10.{i}") for i in range(1, 4)],
    "ground": [BostonDynamicsSpot(hostname="192.168.1.100")],
    "sensors": [IntelRealSenseD435(serial="123456")]
}

# Set up all on platform
for category, assets in team.items():
    for asset in assets:
        await asset.setup_on_platform(
            client, project_uuid,
            mode="hybrid" if category != "sensors" else "physical"
        )
```

This HuggingFace-style system makes it incredibly easy to work with various robots and sensors while maintaining consistency and reusability across projects! 