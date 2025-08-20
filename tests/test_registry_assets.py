import pytest
from cyberwave.assets.registry import (
    AssetRegistry, BaseAsset, Robot, FlyingRobot, 
    DjiTello, BostonDynamicsSpot, register_asset
)
from cyberwave.assets.static_assets import (
    Box, Sphere, ArucoMarker, ChargingPad, CustomMesh
)


class TestAssetRegistry:
    """Test the HuggingFace-style asset registry"""
    
    def test_registry_list(self):
        """Test listing assets in registry"""
        assets = AssetRegistry.list()
        assert len(assets) > 0
        
        # Check for known assets
        asset_ids = [a.asset_id for a in assets]
        assert "dji/tello" in asset_ids
        assert "boston-dynamics/spot" in asset_ids
        assert "generic/box" in asset_ids
    
    def test_registry_get(self):
        """Test getting asset by ID"""
        info = AssetRegistry.get("dji/tello")
        assert info.asset_id == "dji/tello"
        assert info.asset_type == "robot"
        assert "flight" in info.default_capabilities
    
    def test_registry_filter(self):
        """Test filtering assets"""
        robots = AssetRegistry.list(asset_type="robot")
        assert all(a.asset_type == "robot" for a in robots)
        
        static = AssetRegistry.list(asset_type="static")
        assert all(a.asset_type == "static" for a in static)
    
    def test_registry_create_by_id(self):
        """Test creating asset by registry ID"""
        drone = BaseAsset("dji/tello")
        assert drone.name == "DJI Tello"
        assert drone.asset_type == "robot"
        assert "flight" in drone.capabilities


class TestRobotAssets:
    """Test robot asset instantiation and behavior"""
    
    def test_dji_tello_creation(self):
        """Test DJI Tello drone creation"""
        # Direct instantiation
        drone = DjiTello(ip="192.168.10.1")
        assert drone.name == "DJI Tello"
        assert drone.asset_id == "dji/tello"
        assert drone.ip == "192.168.10.1"
        assert "flight" in drone.capabilities
        assert drone.specs['max_altitude'] == 10
    
    def test_boston_dynamics_spot(self):
        """Test Boston Dynamics Spot creation"""
        spot = BostonDynamicsSpot(hostname="192.168.1.100")
        assert spot.name == "Boston Dynamics Spot"
        assert spot.asset_id == "boston-dynamics/spot"
        assert spot.hostname == "192.168.1.100"
        assert "walking" in spot.capabilities
        assert spot.specs['max_speed'] == 1.6
    
    def test_inheritance_chain(self):
        """Test that inheritance works correctly"""
        drone = DjiTello()
        assert isinstance(drone, FlyingRobot)
        assert isinstance(drone, Robot)
        assert isinstance(drone, BaseAsset)
        
        # Check inherited properties
        assert hasattr(drone, 'capabilities')
        assert hasattr(drone, 'specs')
        assert hasattr(drone, 'state')


class TestStaticAssets:
    """Test static asset creation and behavior"""
    
    def test_box_creation(self):
        """Test Box static asset"""
        box = Box(size=2.0, color=[1, 0, 0])
        assert box.name == "Box"
        assert box.asset_id == "generic/box"
        assert box.is_static is True
        assert box.specs['dimensions']['width'] == 2.0
        assert box.color == [1, 0, 0]
    
    def test_aruco_marker(self):
        """Test ArUco marker creation"""
        marker = ArucoMarker(marker_id=42, size=0.3)
        assert marker.name == "ArUco Marker"
        assert marker.asset_id == "markers/aruco"
        assert marker.specs['marker_id'] == 42
        assert marker.specs['marker_size'] == 0.3
        assert "visual_marker" in marker.capabilities
    
    def test_charging_pad(self):
        """Test ChargingPad functionality"""
        pad = ChargingPad()
        assert pad.name == "Charging Pad"
        assert pad.asset_id == "infrastructure/charging-pad"
        assert pad.is_functional is True
        assert "charging" in pad.capabilities
    
    def test_custom_mesh(self):
        """Test custom mesh loading"""
        mesh = CustomMesh(
            mesh_url="https://example.com/model.glb",
            name="Test Model",
            scale=[2, 2, 2]
        )
        assert mesh.name == "Test Model"
        assert mesh.mesh_url == "https://example.com/model.glb"
        assert mesh.specs['scale'] == [2, 2, 2]
    
    @pytest.mark.asyncio
    async def test_static_asset_placement(self):
        """Test placing static assets"""
        box = Box()
        position = await box.place_at(5, 5, 1, rotation=[0, 0, 90])
        assert position == {"x": 5, "y": 5, "z": 1}
        assert box.position == {"x": 5, "y": 5, "z": 1}
        assert box.rotation == [0, 0, 90]
    
    @pytest.mark.asyncio
    async def test_charging_pad_detection(self):
        """Test charging pad robot detection"""
        pad = ChargingPad()
        await pad.place_at(5, 5, 0)
        
        # Robot on pad
        robot_pos = {"x": 5.1, "y": 5.0, "z": 0.1}
        assert await pad.is_robot_on_pad(robot_pos) is True
        
        # Robot off pad
        robot_pos = {"x": 10, "y": 10, "z": 0}
        assert await pad.is_robot_on_pad(robot_pos) is False


class TestCustomAssetRegistration:
    """Test custom asset registration"""
    
    def test_register_custom_asset(self):
        """Test registering a custom asset"""
        @register_asset(
            "test/custom-robot",
            asset_type="robot",
            default_capabilities=["test"],
            default_specs={"test_spec": 123}
        )
        class TestRobot(Robot):
            pass
        
        # Check it's in registry
        info = AssetRegistry.get("test/custom-robot")
        assert info is not None
        assert info.asset_type == "robot"
        assert "test" in info.default_capabilities
        
        # Test instantiation
        robot = TestRobot()
        assert robot.asset_id == "test/custom-robot"
        assert robot.specs['test_spec'] == 123
    
    def test_registry_id_validation(self):
        """Test that registry IDs are validated"""
        with pytest.raises(ValueError):
            @register_asset(
                "invalid id with spaces",
                asset_type="robot"
            )
            class InvalidRobot(Robot):
                pass


class TestMixedScenes:
    """Test mixing robots and static assets"""
    
    @pytest.mark.asyncio
    async def test_mixed_asset_scene(self):
        """Test creating a scene with both robots and static assets"""
        # Create robots
        drone = DjiTello()
        spot = BostonDynamicsSpot()
        
        # Create static assets
        box1 = Box(size=1.0)
        box2 = Box(size=0.5)
        marker = ArucoMarker(marker_id=1)
        charging = ChargingPad()
        
        # Place everything
        await box1.place_at(0, 0, 0)
        await box2.place_at(2, 0, 0)
        await marker.place_at(0, 2, 1)
        await charging.place_at(5, 5, 0)
        
        # Collect all assets
        all_assets = [drone, spot, box1, box2, marker, charging]
        
        # Check types
        robots = [a for a in all_assets if isinstance(a, Robot)]
        static_assets = [a for a in all_assets if getattr(a, 'is_static', False)]
        
        assert len(robots) == 2
        assert len(static_assets) == 4
        
        # All should be BaseAsset
        assert all(isinstance(a, BaseAsset) for a in all_assets)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 