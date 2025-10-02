# Cyberwave Asset + Twin Architecture

## Overview

The Cyberwave platform has been simplified to use a unified **Asset + Twin** model that provides clear separation between asset definitions (blueprints) and their instances (twins). This architecture explicitly supports digital twin use cases while simplifying the overall system.

## Core Concepts

### Assets
- **What**: Blueprints or templates that define the characteristics of any asset type
- **Purpose**: Reusable definitions for robots, sensors, infrastructure, and static objects
- **Key Features**:
  - HuggingFace-style registry with namespaced IDs (e.g., `"dji/tello"`, `"boston-dynamics/spot"`)
  - Pre-configured capabilities and specifications
  - Hierarchical class structure for different asset types
  - Support for custom asset definitions

### Twins
- **What**: Instances of assets that can exist in virtual, physical, or hybrid modes
- **Purpose**: Represent actual robots, sensors, or objects in your system
- **Modes**:
  - **Virtual**: Simulation only (no physical hardware)
  - **Physical**: Hardware only (no simulation)
  - **Hybrid**: True digital twin (both physical and simulated)

## Architecture Benefits

1. **Clarity**: Clear distinction between "what something is" (Asset) and "an instance of it" (Twin)
2. **Digital Twin Support**: Explicit modes make it easy to work with pure simulations, pure hardware, or true digital twins
3. **Simplification**: Removed redundant models and complex relationships
4. **Industry Alignment**: Uses familiar "Digital Twin" terminology
5. **Extensibility**: Easy to add new asset types and capabilities

## Usage Examples

### 1. Using the Asset Registry

```python
from cyberwave.assets import DjiTello, AssetRegistry, AssetFactory

# Direct instantiation
drone = DjiTello(ip="192.168.10.1", name="Training Drone")
print(f"Capabilities: {drone.capabilities}")
print(f"Max altitude: {drone.specs['max_altitude']}m")

# Registry lookup
SpotClass = AssetRegistry.get("boston-dynamics/spot")
spot = SpotClass(hostname="192.168.1.100")

# Factory creation
sensor = AssetFactory.create(
    "intel/realsense-d435",
    serial_number="123456"
)

# List available assets
for asset_id in AssetRegistry.list():
    print(f"Available: {asset_id}")
```

### 2. Creating Twins

```python
from cyberwave import Client
from cyberwave.assets import make_twin_enabled, DjiTello, TwinMode

# Create client
client = Client(base_url="https://api.cyberwave.app")
await client.login(username="user", password="pass")

# Make asset twin-enabled
TwinDrone = make_twin_enabled(DjiTello)
drone = TwinDrone(ip="192.168.10.1")

# Create virtual twin (simulation only)
virtual_twin = await drone.create_twin(
    client=client,
    project_id=123,
    mode=TwinMode.VIRTUAL,
    name="Simulation Drone"
)

# Create physical twin (hardware only)
physical_twin = await drone.create_twin(
    client=client,
    project_id=123,
    mode=TwinMode.PHYSICAL,
    hardware_id="TELLO-001"
)

# Create hybrid twin (true digital twin)
hybrid_twin = await drone.create_twin(
    client=client,
    project_id=123,
    mode=TwinMode.HYBRID,
    hardware_id="TELLO-002"
)
```

### 3. Working with Sensors

```python
from cyberwave.assets import BostonDynamicsSpot, IntelRealSenseD435

# Create robot with sensor
TwinSpot = make_twin_enabled(BostonDynamicsSpot)
spot = TwinSpot(hostname="192.168.1.100")

await spot.create_twin(client, project_id, TwinMode.HYBRID)

# Attach sensor
TwinCamera = make_twin_enabled(IntelRealSenseD435)
camera = TwinCamera(serial_number="CAM001")

sensor_data = await spot.attach_sensor(
    sensor=camera,
    mount_transform={
        "position": [0.3, 0, 0.5],
        "rotation": [1, 0, 0, 0]
    }
)

# Stream sensor data
await camera.stream_data({
    "depth_image": depth_data,
    "rgb_image": rgb_data
})
```

### 4. Static Assets

```python
from cyberwave.assets import TrafficCone, ArucoMarker, ChargingPad

# Create warehouse scene
cone = TrafficCone(name="Navigation Cone 1")
marker = ArucoMarker(marker_id=42, size=0.2)
charging = ChargingPad(
    charging_type="contact",
    compatible_robots=["boston-dynamics/spot"]
)

# Create twins for static assets
for asset in [cone, marker, charging]:
    TwinAsset = make_twin_enabled(asset.__class__)
    twin = TwinAsset(**asset._config)
    await twin.create_twin(client, project_id, TwinMode.VIRTUAL)
```

## Available Assets

### Robots
- `dji/tello` - DJI Tello educational drone
- `dji/mavic-3` - DJI Mavic 3 professional drone
- `parrot/anafi` - Parrot ANAFI compact drone
- `boston-dynamics/spot` - Boston Dynamics Spot quadruped
- `unitree/go1` - Unitree Go1 quadruped
- `clearpath/husky` - Clearpath Husky wheeled robot
- `franka/panda` - Franka Emika Panda manipulator

### Sensors
- `intel/realsense-d435` - Intel RealSense D435 depth camera
- `velodyne/puck` - Velodyne Puck VLP-16 LiDAR
- `zed/zed-2` - Stereolabs ZED 2 stereo camera

### Static Assets
- `props/box`, `props/sphere`, `props/cylinder` - Basic shapes
- `props/traffic-cone`, `props/pallet` - Common props
- `landmarks/aruco-marker`, `landmarks/qr-code`, `landmarks/april-tag` - Fiducial markers
- `infrastructure/wall`, `infrastructure/charging-pad`, `infrastructure/conveyor` - Infrastructure
- `custom/mesh` - Custom 3D models

## Creating Custom Assets

```python
from cyberwave.assets import register_asset, Robot

@register_asset("mycompany/custom-robot", {
    "manufacturer": "My Company",
    "category": "Research Robot"
})
class CustomRobot(Robot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._capabilities.extend(['custom_feature'])
        self._specs.update({
            'custom_spec': 42,
            'special_mode': True
        })
```

## Migration from Old System

| Old Model | New Model | Notes |
|-----------|-----------|-------|
| AssetCatalog | Asset | Now the blueprint/template |
| Asset (old) | Twin (virtual mode) | Virtual instances |
| Device | Twin (physical/hybrid mode) | Physical hardware |
| RunAssetParticipation | TwinParticipation | Unified tracking |
| RunDeviceParticipation | TwinParticipation | Unified tracking |
| Entity | TwinData | Simplified data model |

## Best Practices

1. **Use the Registry**: Leverage pre-configured assets when possible
2. **Choose the Right Mode**: Use virtual for simulation, hybrid for digital twins
3. **Consistent Naming**: Follow the namespace/name pattern for custom assets
4. **Capability-Based Design**: Define capabilities that match your use cases
5. **Telemetry Integration**: Use the built-in telemetry methods for data streaming

## API Integration

The SDK automatically handles:
- Asset creation on the platform when first used
- Twin state synchronization
- Telemetry data transmission
- Position and rotation updates
- Sensor data streaming

## Future Enhancements

- Asset marketplace for community contributions
- Automatic capability detection
- Enhanced simulation integration
- Real-time twin synchronization
- Asset version management 