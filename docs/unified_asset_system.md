# Unified Asset System - From Robots to 3D Meshes

The Cyberwave asset system provides a coherent way to work with any type of asset - from complex robots with behaviors to simple 3D meshes for visualization. Everything uses the same HuggingFace-style registry pattern.

## Asset Hierarchy

```
BaseAsset (Abstract)
├── Robot (Behavioral)
│   ├── FlyingRobot
│   │   └── DjiTello ("dji/tello")
│   ├── GroundRobot  
│   │   └── BostonDynamicsSpot ("boston-dynamics/spot")
│   └── Manipulator
│       └── FrankaPanda ("franka/panda")
│
├── Sensor (Data-producing)
│   ├── Camera
│   │   └── IntelRealSenseD435 ("intel/realsense-d435")
│   └── Lidar
│       └── VelodynePuck ("velodyne/puck")
│
└── StaticAsset (Non-behavioral)
    ├── Prop
    │   ├── Box ("generic/box")
    │   ├── Sphere ("generic/sphere")
    │   ├── TrafficCone ("props/traffic-cone")
    │   └── Pallet ("props/pallet")
    ├── Landmark
    │   ├── ArucoMarker ("markers/aruco")
    │   └── QRCode ("markers/qr-code")
    └── Infrastructure
        ├── Wall ("environment/wall")
        └── ChargingPad ("infrastructure/charging-pad")
```

## Unified Usage Pattern

The beauty of this system is that **everything is an asset** and uses the same patterns:

```python
from cyberwave.assets import DjiTello, Box, ArucoMarker, ChargingPad

# Complex robot with behaviors
drone = DjiTello(ip="192.168.10.1")
await drone.connect()
await drone.takeoff()

# Simple 3D mesh for visualization
box = Box(size=1.0, color=[1, 0, 0])
await box.place_at(5, 5, 0)

# Functional infrastructure
charging = ChargingPad()
await charging.place_at(10, 10, 0)

# Visual marker for localization
marker = ArucoMarker(marker_id=1)
await marker.place_at(0, 0, 2)
```

## Creating Complete Scenes

### Example: Warehouse Environment

```python
from cyberwave.assets import (
    BostonDynamicsSpot, Wall, Pallet, Box, 
    ArucoMarker, ChargingPad
)

async def create_warehouse():
    assets = []
    
    # Build the warehouse structure
    for i in range(4):
        wall = Wall(width=20, height=5)
        positions = [(10, 0, 2.5), (20, 10, 2.5), 
                    (10, 20, 2.5), (0, 10, 2.5)]
        rotations = [[0, 0, 0], [0, 0, 90], 
                    [0, 0, 180], [0, 0, 270]]
        await wall.place_at(*positions[i], rotation=rotations[i])
        assets.append(wall)
    
    # Add storage areas with pallets
    for row in range(5):
        for col in range(8):
            pallet = Pallet()
            await pallet.place_at(2 + col*1.5, 2 + row*1.5, 0)
            assets.append(pallet)
            
            # Some pallets have boxes
            if (row + col) % 3 == 0:
                box = Box(size=0.8, color=[0.6, 0.4, 0.2])
                await box.place_at(2 + col*1.5, 2 + row*1.5, 0.2)
                assets.append(box)
    
    # Navigation markers
    for i in range(6):
        marker = ArucoMarker(marker_id=i)
        await marker.place_at(5 + i*3, 0.1, 3)
        assets.append(marker)
    
    # Charging station
    charging = ChargingPad()
    await charging.place_at(18, 18, 0)
    assets.append(charging)
    
    # Add the robot
    spot = BostonDynamicsSpot(hostname="192.168.1.100")
    await spot.connect()
    assets.append(spot)
    
    return assets
```

## Static Assets for Different Use Cases

### 1. Simple Visualization References

```python
from cyberwave.assets import Box, Sphere, Cylinder

# Create reference objects
origin_marker = Sphere(radius=0.1, color=[1, 0, 0])
await origin_marker.place_at(0, 0, 0)

# Axis indicators
x_axis = Cylinder(radius=0.05, height=2, color=[1, 0, 0])
await x_axis.place_at(1, 0, 0, rotation=[0, 90, 0])

y_axis = Cylinder(radius=0.05, height=2, color=[0, 1, 0])
await y_axis.place_at(0, 1, 0, rotation=[90, 0, 0])

z_axis = Cylinder(radius=0.05, height=2, color=[0, 0, 1])
await z_axis.place_at(0, 0, 1)
```

### 2. Custom 3D Meshes

```python
from cyberwave.assets import CustomMesh

# Load any 3D model
machine = CustomMesh(
    mesh_url="https://example.com/models/cnc-machine.glb",
    name="CNC Machine",
    scale=[1.5, 1.5, 1.5],
    collision=True
)
await machine.place_at(5, 5, 0)

# Load decorative elements
statue = CustomMesh(
    mesh_url="https://example.com/models/statue.obj",
    name="Decorative Statue",
    material="marble",
    collision=False  # Just visual
)
await statue.place_at(10, 10, 0)
```

### 3. Functional Infrastructure

```python
from cyberwave.assets import ChargingPad, Wall

# Functional charging station
charging = ChargingPad()
await charging.place_at(5, 5, 0)

# Check if robot is charging
robot_pos = {"x": 5.1, "y": 5.0, "z": 0.1}
if await charging.is_robot_on_pad(robot_pos):
    print("Robot is on charging pad!")

# Barriers and obstacles
wall = Wall(width=5, height=2)
await wall.place_at(0, 10, 1)
```

## Registry Pattern for All Assets

The same registry pattern works for everything:

```python
from cyberwave.assets import AssetRegistry, BaseAsset

# List all available assets
for asset_info in AssetRegistry.list():
    print(f"{asset_info.asset_id}: {asset_info.asset_type}")

# Filter by type
robots = AssetRegistry.list(asset_type="robot")
static_objects = AssetRegistry.list(asset_type="static")
sensors = AssetRegistry.list(asset_type="sensor")

# Create any asset by ID
asset = BaseAsset("generic/box")  # Works!
robot = BaseAsset("dji/tello")    # Also works!
```

## Platform Integration

All assets - whether complex or simple - integrate with the platform the same way:

```python
from cyberwave import Client
from cyberwave.assets import DjiTello, Box, ArucoMarker

client = Client("https://api.cyberwave.dev")
await client.authenticate("api-key")

# Robot with full digital twin
drone = DjiTello(ip="192.168.10.1")
await drone.setup_on_platform(
    client, 
    project_uuid="my-project",
    mode="hybrid"  # Physical + virtual
)

# Static objects are always virtual
box = Box(size=1.0)
await box.setup_on_platform(
    client,
    project_uuid="my-project"
    # mode is always "virtual" for static assets
)

# Markers for the environment
marker = ArucoMarker(marker_id=1)
await marker.setup_on_platform(
    client,
    project_uuid="my-project"
)
```

## Custom Static Assets

Create your own static asset types:

```python
from cyberwave.assets import StaticAsset, register_asset

@register_asset(
    "custom/safety-barrier",
    asset_type="static",
    default_capabilities=["barrier", "visual_warning"],
    default_specs={
        "height": 1.2,
        "sections": 5,
        "color": [1.0, 1.0, 0.0]  # Yellow
    }
)
class SafetyBarrier(StaticAsset):
    """Custom safety barrier for industrial environments"""
    
    def __init__(self, sections: int = 5, **kwargs):
        super().__init__("custom/safety-barrier", **kwargs)
        self.specs['sections'] = sections
        
    async def create_line(self, start, end, spacing=2.0):
        """Create a line of barriers"""
        # Implementation to create multiple barrier sections
        pass
```

## Mixed Reality Scenarios

The unified system shines in mixed reality scenarios:

```python
async def ar_training_scenario():
    """AR training with virtual and physical assets"""
    
    # Physical robot (real)
    real_spot = BostonDynamicsSpot(hostname="192.168.1.100")
    await real_spot.setup_on_platform(client, project, mode="hybrid")
    
    # Virtual hazards (AR overlay)
    hazards = []
    for i in range(5):
        spill = CustomMesh(
            mesh_url="https://example.com/models/chemical-spill.glb",
            name=f"Hazard {i}",
            material="translucent"
        )
        await spill.place_at(random.uniform(0, 10), 
                           random.uniform(0, 10), 0)
        await spill.setup_on_platform(client, project)
        hazards.append(spill)
    
    # Virtual goal markers
    goals = []
    for i in range(3):
        goal = Sphere(radius=0.5, color=[0, 1, 0], opacity=0.5)
        await goal.place_at(i*5, 10, 1)
        await goal.setup_on_platform(client, project)
        goals.append(goal)
    
    # Everything tracked in the same system!
```

## Benefits of Unified System

1. **Consistency**: Same API whether it's a robot or a box
2. **Discoverability**: All assets in one registry
3. **Flexibility**: Mix behavioral and static assets seamlessly
4. **Simplicity**: Don't need different systems for different asset types
5. **Extensibility**: Easy to add new asset types

This unified approach makes Cyberwave incredibly powerful - you can build complete digital twin environments with the same simple patterns, whether you're adding a complex robot or just a reference cube! 