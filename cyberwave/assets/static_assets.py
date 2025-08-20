# Static Assets - Simple 3D objects, props, and reference items
# Part of the HuggingFace-style asset system

from typing import Optional, Dict, Any, List
from .registry import BaseAsset, register_asset, AssetRegistry


# =============================================
# STATIC ASSET BASE CLASSES
# =============================================

class StaticAsset(BaseAsset):
    """Base class for static, non-behavioral assets like 3D meshes, props, landmarks"""

    def __init__(self, asset_id: Optional[str] = None, **kwargs):
        name = kwargs.pop("name", None)
        super().__init__(name=name, **kwargs)
        if asset_id:
            self._registry_id = asset_id
            self.asset_id = asset_id

        # Static assets have simpler properties
        self.is_static = True
        self.is_interactive = kwargs.get('interactive', False)
        self.collision_enabled = kwargs.get('collision', True)
        
        # Visual properties
        self.material = kwargs.get('material', 'default')
        self.color = kwargs.get('color', [0.5, 0.5, 0.5])
        self.opacity = kwargs.get('opacity', 1.0)
        
    async def place_at(self, x: float, y: float, z: float, rotation: List[float] = None):
        """Place the static asset at a specific location"""
        self.position = {"x": x, "y": y, "z": z}
        if rotation:
            self.rotation = rotation
        return self.position


class Prop(StaticAsset):
    """Props are decorative static assets"""

    def __init__(self, asset_id: Optional[str] = None, **kwargs):
        super().__init__(asset_id, **kwargs)
        self.prop_type = kwargs.get('prop_type', 'decoration')


class Landmark(StaticAsset):
    """Landmarks are reference points in the environment"""

    def __init__(self, asset_id: Optional[str] = None, **kwargs):
        super().__init__(asset_id, **kwargs)
        self.landmark_type = kwargs.get('landmark_type', 'marker')
        self.is_visible = kwargs.get('visible', True)


class Infrastructure(StaticAsset):
    """Infrastructure assets like walls, charging stations, etc."""

    def __init__(self, asset_id: Optional[str] = None, **kwargs):
        super().__init__(asset_id, **kwargs)
        self.infrastructure_type = kwargs.get('infrastructure_type', 'obstacle')
        self.is_functional = kwargs.get('functional', False)


# =============================================
# REGISTERED STATIC ASSETS
# =============================================

@register_asset(
    "generic/box",
    asset_type="static",
    default_capabilities=[],  # No capabilities - it's just a box
    default_specs={
        "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0},
        "weight": 1.0,
        "material": "plastic"
    }
)
class Box(Prop):
    """Simple box mesh"""
    
    def __init__(self, size: float = 1.0, **kwargs):
        super().__init__("generic/box", **kwargs)
        self.specs['dimensions'] = {
            "width": size, 
            "height": size, 
            "depth": size
        }


@register_asset(
    "generic/sphere",
    asset_type="static",
    default_capabilities=[],
    default_specs={
        "radius": 0.5,
        "segments": 32
    }
)
class Sphere(Prop):
    """Simple sphere mesh"""
    
    def __init__(self, radius: float = 0.5, **kwargs):
        super().__init__("generic/sphere", **kwargs)
        self.specs['radius'] = radius


@register_asset(
    "generic/cylinder",
    asset_type="static",
    default_capabilities=[],
    default_specs={
        "radius": 0.5,
        "height": 2.0
    }
)
class Cylinder(Prop):
    """Simple cylinder mesh"""
    
    def __init__(self, radius: float = 0.5, height: float = 2.0, **kwargs):
        super().__init__("generic/cylinder", **kwargs)
        self.specs.update({"radius": radius, "height": height})


@register_asset(
    "markers/aruco",
    asset_type="static",
    default_capabilities=["visual_marker"],
    default_specs={
        "marker_size": 0.2,  # 20cm
        "marker_id": 0,
        "dictionary": "DICT_4X4_50"
    },
    metadata={"name": "ArUco Marker"}
)
class ArucoMarker(Landmark):
    """ArUco marker for visual localization"""
    
    def __init__(self, marker_id: int = 0, size: float = 0.2, **kwargs):
        super().__init__("markers/aruco", **kwargs)
        self.specs.update({
            "marker_id": marker_id,
            "marker_size": size
        })
        self._capabilities.extend(["visual_marker", "localization_reference"])


@register_asset(
    "markers/qr-code",
    asset_type="static",
    default_capabilities=["visual_marker", "data_encoding"],
    default_specs={
        "size": 0.15,
        "data": "",
        "error_correction": "M"
    }
)
class QRCode(Landmark):
    """QR code marker"""
    
    def __init__(self, data: str = "", size: float = 0.15, **kwargs):
        super().__init__("markers/qr-code", **kwargs)
        self.specs.update({
            "data": data,
            "size": size
        })


@register_asset(
    "infrastructure/charging-pad",
    asset_type="static",
    default_capabilities=["charging"],
    default_specs={
        "dimensions": {"width": 1.0, "depth": 1.0, "height": 0.1},
        "charging_power": 100,  # Watts
        "compatible_robots": ["boston-dynamics/spot", "unitree/go1"]
    }
)
class ChargingPad(Infrastructure):
    """Wireless charging pad for robots"""
    
    def __init__(self, **kwargs):
        super().__init__("infrastructure/charging-pad", **kwargs)
        self.is_functional = True
        self.infrastructure_type = "charging"
        
    async def is_robot_on_pad(self, robot_position: Dict[str, float]) -> bool:
        """Check if a robot is positioned on the charging pad"""
        if not hasattr(self, 'position'):
            return False
            
        pad_dims = self.specs['dimensions']
        dx = abs(robot_position['x'] - self.position['x'])
        dy = abs(robot_position['y'] - self.position['y'])
        
        return dx < pad_dims['width']/2 and dy < pad_dims['depth']/2


@register_asset(
    "props/traffic-cone",
    asset_type="static",
    default_capabilities=[],
    default_specs={
        "height": 0.7,
        "base_radius": 0.35,
        "color": [1.0, 0.5, 0.0],  # Orange
        "reflective": True
    }
)
class TrafficCone(Prop):
    """Traffic cone for marking areas"""
    
    def __init__(self, **kwargs):
        super().__init__("props/traffic-cone", **kwargs)
        self.color = kwargs.get('color', [1.0, 0.5, 0.0])


@register_asset(
    "props/pallet",
    asset_type="static", 
    default_capabilities=["load_bearing"],
    default_specs={
        "dimensions": {"width": 1.2, "depth": 1.0, "height": 0.15},
        "material": "wood",
        "max_load": 1000  # kg
    }
)
class Pallet(Prop):
    """Standard warehouse pallet"""
    
    def __init__(self, **kwargs):
        super().__init__("props/pallet", **kwargs)
        self.prop_type = "industrial"


@register_asset(
    "environment/wall",
    asset_type="static",
    default_capabilities=["barrier"],
    default_specs={
        "dimensions": {"width": 4.0, "height": 3.0, "thickness": 0.2},
        "material": "concrete"
    }
)
class Wall(Infrastructure):
    """Wall segment for building environments"""
    
    def __init__(self, width: float = 4.0, height: float = 3.0, **kwargs):
        super().__init__("environment/wall", **kwargs)
        self.specs['dimensions'].update({
            "width": width,
            "height": height
        })
        self.infrastructure_type = "barrier"


@register_asset(
    "custom/mesh",
    asset_type="static",
    default_capabilities=[],
    default_specs={
        "file_format": "glb",
        "scale": [1.0, 1.0, 1.0]
    }
)
class CustomMesh(StaticAsset):
    """Custom 3D mesh file"""
    
    def __init__(self, mesh_url: str, name: str = "Custom Mesh", **kwargs):
        super().__init__(None, name=name, **kwargs)  # No registry ID
        self.mesh_url = mesh_url
        self.specs['mesh_url'] = mesh_url
        self.specs['scale'] = kwargs.get('scale', [1.0, 1.0, 1.0])


# =============================================
# USAGE EXAMPLES
# =============================================

async def static_asset_examples():
    """Examples of using static assets"""
    
    # Simple geometric shapes
    box = Box(size=0.5, color=[1, 0, 0])  # Red box
    sphere = Sphere(radius=0.3, material="metal")
    cylinder = Cylinder(radius=0.2, height=1.5)
    
    # Place them in the scene
    await box.place_at(0, 0, 0)
    await sphere.place_at(1, 0, 0.5)
    await cylinder.place_at(-1, 0, 0.75)
    
    # Markers for localization
    aruco = ArucoMarker(marker_id=42, size=0.3)
    await aruco.place_at(2, 2, 1, rotation=[0, 0, 90])
    
    # Functional infrastructure
    charging_pad = ChargingPad()
    await charging_pad.place_at(5, 5, 0)
    
    # Check if robot is on pad
    robot_pos = {"x": 5.1, "y": 5.0, "z": 0.1}
    if await charging_pad.is_robot_on_pad(robot_pos):
        print("Robot is on charging pad!")
    
    # Custom mesh from URL
    custom_object = CustomMesh(
        mesh_url="https://example.com/models/machine.glb",
        name="Industrial Machine",
        scale=[2.0, 2.0, 2.0]
    )
    await custom_object.place_at(10, 0, 0)
    
    # Create obstacle course
    cones = []
    for i in range(5):
        cone = TrafficCone()
        await cone.place_at(i * 2, 0, 0)
        cones.append(cone)
    
    # Industrial scene
    pallets = []
    for row in range(3):
        for col in range(4):
            pallet = Pallet()
            await pallet.place_at(col * 1.5, row * 1.2, 0)
            pallets.append(pallet)


async def mixed_scene_example():
    """Example mixing robots and static assets"""
    from .registry import DjiTello, BostonDynamicsSpot
    
    # Create environment
    walls = []
    for i in range(4):
        wall = Wall(width=10, height=4)
        # Create a square room
        positions = [(5, 0, 2), (10, 5, 2), (5, 10, 2), (0, 5, 2)]
        rotations = [[0, 0, 0], [0, 0, 90], [0, 0, 180], [0, 0, 270]]
        await wall.place_at(*positions[i], rotation=rotations[i])
        walls.append(wall)
    
    # Add markers for navigation
    markers = []
    for i in range(4):
        marker = ArucoMarker(marker_id=i)
        await marker.place_at(2.5 + i*2.5, 2.5, 1.5)
        markers.append(marker)
    
    # Add charging station
    charging = ChargingPad()
    await charging.place_at(8, 8, 0)
    
    # Add some obstacles
    boxes = []
    for i in range(3):
        box = Box(size=0.8, color=[0.5, 0.5, 0.5])
        await box.place_at(3 + i*2, 5, 0.4)
        boxes.append(box)
    
    # Now add robots
    drone = DjiTello()
    spot = BostonDynamicsSpot(hostname="192.168.1.100")
    
    # Everything uses the same asset system!
    all_assets = walls + markers + [charging] + boxes + [drone, spot]
    
    print(f"Scene contains {len(all_assets)} assets")
    print(f"Static objects: {len([a for a in all_assets if getattr(a, 'is_static', False)])}")
    print(f"Robots: {len([a for a in all_assets if a.asset_type == 'robot'])}")


# =============================================
# PLATFORM INTEGRATION
# =============================================

class CyberwaveStaticAsset(StaticAsset):
    """Static asset with Cyberwave platform integration"""
    
    async def setup_on_platform(self, client, project_uuid: str, level_uuid: str = None):
        """Create this static asset on the platform"""
        # Static assets are always virtual
        asset_data = await client.create_asset(
            name=self.name,
            asset_type="static",
            specs=self.specs,
            capabilities=self.capabilities
        )
        
        twin_data = await client.create_twin(
            asset_uuid=asset_data['uuid'],
            name=self.name,
            mode="virtual",  # Static assets are always virtual
            project_uuid=project_uuid,
            transform={
                "position": getattr(self, 'position', {"x": 0, "y": 0, "z": 0}),
                "rotation": getattr(self, 'rotation', [0, 0, 0])
            }
        )
        
        self.asset_uuid = asset_data['uuid']
        self.twin_uuid = twin_data['uuid']
        
        # If level specified, place in level
        if level_uuid and hasattr(self, 'position'):
            await client.update_twin_position(
                self.twin_uuid,
                level_uuid=level_uuid,
                position=self.position
            )


if __name__ == "__main__":
    import asyncio
    asyncio.run(static_asset_examples())
    asyncio.run(mixed_scene_example()) 