"""
Tests for the Cyberwave Asset System
"""

import pytest
from cyberwave.assets import (
    AssetRegistry,
    AssetFactory,
    BaseAsset,
    Robot,
    FlyingRobot,
    register_asset,
    DjiTello,
    BostonDynamicsSpot,
    IntelRealSenseD435,
    Box,
    ArucoMarker,
    make_twin_enabled,
    TwinMode,
)


class TestAssetRegistry:
    """Test the asset registry functionality"""
    
    def test_registry_singleton(self):
        """Test that AssetRegistry is a singleton"""
        registry1 = AssetRegistry()
        registry2 = AssetRegistry()
        assert registry1 is registry2
    
    def test_register_and_get(self):
        """Test registering and retrieving assets"""
        @register_asset("test/robot")
        class TestRobot(Robot):
            pass
        
        # Should be able to get by full ID
        assert AssetRegistry.get("test/robot") == TestRobot
        
        # Should also be registered without namespace if unique
        assert AssetRegistry.get("robot") == TestRobot
    
    def test_list_assets(self):
        """Test listing registered assets"""
        all_assets = AssetRegistry.list()
        assert "dji/tello" in all_assets
        assert "boston-dynamics/spot" in all_assets
        
        # Test namespace filtering
        dji_assets = AssetRegistry.list(namespace="dji")
        assert all(asset.startswith("dji/") for asset in dji_assets)
    
    def test_metadata(self):
        """Test asset metadata"""
        metadata = AssetRegistry.get_metadata("dji/tello")
        assert metadata.get("manufacturer") == "DJI"
        assert metadata.get("model") == "Tello"


class TestAssetFactory:
    """Test the asset factory"""
    
    def test_create_from_registry(self):
        """Test creating assets from registry IDs"""
        drone = AssetFactory.create("dji/tello", ip="192.168.10.1")
        assert isinstance(drone, DjiTello)
        assert drone.ip == "192.168.10.1"
    
    def test_create_with_invalid_params(self):
        """Test that invalid parameters are filtered out"""
        # Should not raise error even with invalid params
        drone = AssetFactory.create(
            "dji/tello",
            ip="192.168.10.1",
            invalid_param="should_be_ignored"
        )
        assert isinstance(drone, DjiTello)
    
    def test_create_from_config(self):
        """Test creating from configuration dict"""
        config = {
            "type": "props/box",
            "size": [2, 2, 2],
            "name": "Large Box"
        }
        box = AssetFactory.create_from_config(config)
        assert isinstance(box, Box)
        assert box.specs["dimensions"] == [2, 2, 2]


class TestBaseAsset:
    """Test base asset functionality"""
    
    def test_asset_initialization(self):
        """Test basic asset initialization"""
        asset = BaseAsset(name="Test Asset")
        assert asset.name == "Test Asset"
        assert isinstance(asset.capabilities, list)
        assert isinstance(asset.specs, dict)
    
    def test_to_dict(self):
        """Test asset serialization"""
        drone = DjiTello(name="Test Drone")
        data = drone.to_dict()
        assert data["type"] == "dji/tello"
        assert data["name"] == "Test Drone"
        assert "capabilities" in data
        assert "specs" in data
    
    def test_from_dict(self):
        """Test asset deserialization"""
        data = {
            "type": "dji/tello",
            "name": "Restored Drone",
            "config": {"ip": "192.168.10.2"}
        }
        drone = BaseAsset.from_dict(data)
        assert isinstance(drone, DjiTello)
        assert drone.name == "Restored Drone"


class TestRobotAssets:
    """Test robot asset classes"""
    
    def test_flying_robot(self):
        """Test flying robot capabilities"""
        drone = DjiTello()
        assert "fly" in drone.capabilities
        assert "hover" in drone.capabilities
        assert drone.specs["max_altitude"] == 10
    
    def test_ground_robot(self):
        """Test ground robot capabilities"""
        spot = BostonDynamicsSpot()
        assert "drive" in spot.capabilities
        assert "climbing_stairs" in spot.capabilities
        assert spot.specs["max_speed"] == 1.6
    
    def test_robot_with_registry_id(self):
        """Test creating robot with registry reference"""
        robot = Robot("boston-dynamics/spot", hostname="192.168.1.100")
        # Should create the actual Spot class
        assert "climbing_stairs" in robot.capabilities


class TestSensorAssets:
    """Test sensor asset classes"""
    
    def test_depth_sensor(self):
        """Test depth sensor capabilities"""
        sensor = IntelRealSenseD435()
        assert "measure_depth" in sensor.capabilities
        assert "capture_image" in sensor.capabilities
        assert sensor.specs["depth_range"] == [0.2, 10]
    
    def test_sensor_initialization(self):
        """Test sensor with serial number"""
        sensor = IntelRealSenseD435(serial_number="12345")
        assert sensor.serial_number == "12345"


class TestStaticAssets:
    """Test static asset classes"""
    
    def test_props(self):
        """Test prop assets"""
        box = Box(size=[2, 3, 4])
        assert box.specs["dimensions"] == [2, 3, 4]
        assert box.specs["shape"] == "box"
    
    def test_landmarks(self):
        """Test landmark assets"""
        marker = ArucoMarker(marker_id=42, size=0.15)
        assert "localization_reference" in marker.capabilities
        assert marker.specs["marker_id"] == 42
        assert marker.specs["size"] == 0.15


class TestTwinIntegration:
    """Test twin integration features"""
    
    def test_make_twin_enabled(self):
        """Test making assets twin-enabled"""
        TwinDrone = make_twin_enabled(DjiTello)
        drone = TwinDrone()
        
        # Should have twin methods
        assert hasattr(drone, 'create_twin')
        assert hasattr(drone, 'update_position')
        assert hasattr(drone, 'send_telemetry')
    
    def test_twin_enabled_preserves_registry(self):
        """Test that twin-enabled assets preserve registry info"""
        TwinSpot = make_twin_enabled(BostonDynamicsSpot)
        assert TwinSpot._registry_id == "boston-dynamics/spot"
    
    def test_twin_modes(self):
        """Test twin mode enum"""
        assert TwinMode.VIRTUAL.value == "virtual"
        assert TwinMode.PHYSICAL.value == "physical"
        assert TwinMode.HYBRID.value == "hybrid"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 