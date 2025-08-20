"""
Pre-configured asset implementations for common hardware

This module provides ready-to-use implementations of popular robots,
sensors, and static assets that can be used directly or via the registry.
"""

from typing import Optional, Dict, Any, List
from .registry import (
    register_asset,
    FlyingRobot,
    GroundRobot,
    CameraSensor,
    DepthSensor,
    StaticAsset,
    Prop,
    Landmark,
    Infrastructure,
)


# =============================================
# FLYING ROBOTS (DRONES)
# =============================================

@register_asset("dji/tello", {
    "manufacturer": "DJI",
    "model": "Tello",
    "category": "Educational Drone",
    "price_usd": 99,
})
class DjiTello(FlyingRobot):
    """DJI Tello educational drone"""
    
    def __init__(self, ip: str = "192.168.10.1", **kwargs):
        super().__init__(**kwargs)
        self.ip = ip
        self._capabilities.extend(['flip', 'stream_video'])
        self._specs.update({
            'max_altitude': 10,  # meters
            'max_speed': 8,  # m/s
            'flight_time': 13,  # minutes
            'weight': 0.08,  # kg
            'camera_resolution': [1280, 720],
        })


@register_asset("dji/mavic-3", {
    "manufacturer": "DJI",
    "model": "Mavic 3",
    "category": "Professional Drone",
})
class DjiMavic3(FlyingRobot):
    """DJI Mavic 3 professional drone"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._capabilities.extend(['obstacle_avoidance', 'gps_navigation', '4k_video'])
        self._specs.update({
            'max_altitude': 6000,  # meters
            'max_speed': 21,  # m/s
            'flight_time': 46,  # minutes
            'weight': 0.895,  # kg
            'camera_resolution': [5120, 2880],  # 5.1K
            'hasselblad_camera': True,
        })


@register_asset("parrot/anafi", {
    "manufacturer": "Parrot",
    "model": "ANAFI",
    "category": "Compact Drone",
})
class ParrotAnafi(FlyingRobot):
    """Parrot ANAFI compact drone"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._capabilities.extend(['180_degree_tilt', 'hdr_video', 'usb_charging'])
        self._specs.update({
            'max_altitude': 4500,  # meters
            'max_speed': 15,  # m/s
            'flight_time': 25,  # minutes
            'weight': 0.32,  # kg
            'camera_resolution': [3840, 2160],  # 4K
        })


# =============================================
# GROUND ROBOTS
# =============================================

@register_asset("boston-dynamics/spot", {
    "manufacturer": "Boston Dynamics",
    "model": "Spot",
    "category": "Quadruped Robot",
})
class BostonDynamicsSpot(GroundRobot):
    """Boston Dynamics Spot quadruped robot"""
    
    def __init__(self, hostname: str = "192.168.80.3", **kwargs):
        super().__init__(**kwargs)
        self.hostname = hostname
        self._capabilities.extend([
            'climbing_stairs', 'self_righting', 'manipulation_arm',
            'autonomous_navigation', 'inspection', 'walking'
        ])
        self._specs.update({
            'max_speed': 1.6,  # m/s
            'runtime': 90,  # minutes
            'payload_capacity': 14,  # kg
            'degrees_of_freedom': 12,
            'cameras': 5,
            'height': 0.84,  # meters
            'weight': 32.5,  # kg
        })


@register_asset("unitree/go1", {
    "manufacturer": "Unitree Robotics",
    "model": "Go1",
    "category": "Quadruped Robot",
})
class UnitreeGo1(GroundRobot):
    """Unitree Go1 quadruped robot"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._capabilities.extend(['walking', 'running', 'side_stepping'])
        self._specs.update({
            'max_speed': 3.7,  # m/s
            'runtime': 120,  # minutes
            'payload_capacity': 5,  # kg
            'height': 0.4,  # meters
            'weight': 12,  # kg
        })


@register_asset("clearpath/husky", {
    "manufacturer": "Clearpath Robotics",
    "model": "Husky",
    "category": "Wheeled Robot",
})
class ClearpathHusky(GroundRobot):
    """Clearpath Husky wheeled robot"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._capabilities.extend(['all_terrain', 'ros_compatible', 'modular_payload'])
        self._specs.update({
            'max_speed': 1.0,  # m/s
            'runtime': 180,  # minutes
            'payload_capacity': 75,  # kg
            'ground_clearance': 0.13,  # meters
            'weight': 50,  # kg
        })


@register_asset("franka/panda", {
    "manufacturer": "Franka Emika",
    "model": "Panda",
    "category": "Manipulator Arm",
})
class FrankaPanda(GroundRobot):  # Could be a Manipulator class
    """Franka Emika Panda collaborative robot arm"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._capabilities.extend(['manipulation', 'force_sensing', 'collision_detection'])
        self._specs.update({
            'degrees_of_freedom': 7,
            'reach': 0.855,  # meters
            'payload': 3,  # kg
            'repeatability': 0.1,  # mm
            'weight': 18,  # kg
        })


# =============================================
# SENSORS
# =============================================

@register_asset("intel/realsense-d435", {
    "manufacturer": "Intel",
    "model": "RealSense D435",
    "category": "Depth Camera",
})
class IntelRealSenseD435(DepthSensor):
    """Intel RealSense D435 depth camera"""
    
    def __init__(self, serial_number: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.serial_number = serial_number
        self._capabilities.extend(['stereo_vision', 'infrared'])
        self._specs.update({
            'depth_range': [0.2, 10],  # meters
            'depth_resolution': [1280, 720],
            'rgb_resolution': [1920, 1080],
            'fov': 87,  # degrees
            'fps': 30,
        })


@register_asset("velodyne/puck", {
    "manufacturer": "Velodyne",
    "model": "Puck (VLP-16)",
    "category": "LiDAR",
})
class VelodynePuck(DepthSensor):
    """Velodyne Puck VLP-16 LiDAR"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._capabilities.extend(['360_degree_scanning', 'outdoor_capable'])
        self._specs.update({
            'channels': 16,
            'range': 100,  # meters
            'accuracy': 0.03,  # meters
            'rotation_rate': 20,  # Hz
            'points_per_second': 300000,
            'fov_vertical': 30,  # degrees
        })


@register_asset("zed/zed-2", {
    "manufacturer": "Stereolabs",
    "model": "ZED 2",
    "category": "Stereo Camera",
})
class ZED2(DepthSensor):
    """Stereolabs ZED 2 stereo camera"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._capabilities.extend(['ai_detection', 'positional_tracking', 'spatial_mapping'])
        self._specs.update({
            'depth_range': [0.2, 20],  # meters
            'resolution': [3840, 1080],  # dual 1920x1080
            'fov': 110,  # degrees
            'fps': 100,
            'baseline': 0.12,  # meters
        })


# =============================================
# STATIC ASSETS - PROPS
# =============================================

@register_asset("props/box", {
    "category": "Basic Shape",
})
class Box(Prop):
    """Simple box prop"""
    
    def __init__(self, size: List[float] = None, **kwargs):
        super().__init__(**kwargs)
        self._specs['dimensions'] = size or [1, 1, 1]
        self._specs['shape'] = 'box'


@register_asset("props/sphere", {
    "category": "Basic Shape",
})
class Sphere(Prop):
    """Simple sphere prop"""
    
    def __init__(self, radius: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self._specs['radius'] = radius
        self._specs['shape'] = 'sphere'


@register_asset("props/cylinder", {
    "category": "Basic Shape",
})
class Cylinder(Prop):
    """Simple cylinder prop"""
    
    def __init__(self, radius: float = 0.5, height: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._specs['radius'] = radius
        self._specs['height'] = height
        self._specs['shape'] = 'cylinder'


@register_asset("props/traffic-cone", {
    "category": "Traffic Equipment",
})
class TrafficCone(Prop):
    """Traffic cone prop"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._specs.update({
            'dimensions': [0.3, 0.3, 0.7],  # base width x base depth x height
            'color': [1.0, 0.5, 0.0],  # Orange
            'reflective_strips': True,
        })


@register_asset("props/pallet", {
    "category": "Warehouse Equipment",
})
class Pallet(Prop):
    """Warehouse pallet prop"""
    
    def __init__(self, pallet_type: str = "euro", **kwargs):
        super().__init__(**kwargs)
        dimensions = {
            'euro': [1.2, 0.8, 0.144],
            'us': [1.219, 1.016, 0.153],
        }
        self._specs.update({
            'dimensions': dimensions.get(pallet_type, dimensions['euro']),
            'material': 'wood',
            'load_capacity': 1500,  # kg
        })


# =============================================
# STATIC ASSETS - LANDMARKS
# =============================================

@register_asset("landmarks/aruco-marker", {
    "category": "Fiducial Marker",
})
class ArucoMarker(Landmark):
    """ArUco fiducial marker for localization"""
    
    def __init__(self, marker_id: int = 0, size: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self._capabilities.append('visual_localization')
        self._specs.update({
            'marker_id': marker_id,
            'size': size,  # meters
            'dictionary': 'DICT_4X4_50',
        })


@register_asset("landmarks/qr-code", {
    "category": "Data Marker",
})
class QRCode(Landmark):
    """QR code marker"""
    
    def __init__(self, data: str = "", size: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self._capabilities.extend(['data_encoding', 'visual_localization'])
        self._specs.update({
            'data': data,
            'size': size,  # meters
            'error_correction': 'M',
        })


@register_asset("landmarks/april-tag", {
    "category": "Fiducial Marker",
})
class AprilTag(Landmark):
    """AprilTag fiducial marker"""
    
    def __init__(self, tag_id: int = 0, family: str = "tag36h11", size: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self._capabilities.append('visual_localization')
        self._specs.update({
            'tag_id': tag_id,
            'family': family,
            'size': size,  # meters
        })


# =============================================
# STATIC ASSETS - INFRASTRUCTURE
# =============================================

@register_asset("infrastructure/wall", {
    "category": "Building Element",
})
class Wall(Infrastructure):
    """Wall infrastructure element"""
    
    def __init__(self, length: float = 5.0, height: float = 3.0, thickness: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self._specs.update({
            'dimensions': [length, thickness, height],
            'material': kwargs.get('material', 'concrete'),
        })


@register_asset("infrastructure/charging-pad", {
    "category": "Robot Infrastructure",
})
class ChargingPad(Infrastructure):
    """Charging pad for robots"""
    
    def __init__(self, charging_type: str = "contact", **kwargs):
        super().__init__(**kwargs)
        self._capabilities.extend(['robot_charging', 'docking_guidance'])
        self._specs.update({
            'dimensions': [1.0, 1.0, 0.05],
            'charging_type': charging_type,  # 'contact' or 'wireless'
            'max_power': 500,  # watts
            'compatible_robots': kwargs.get('compatible_robots', []),
        })


@register_asset("infrastructure/conveyor", {
    "category": "Industrial Equipment",
})
class Conveyor(Infrastructure):
    """Conveyor belt infrastructure"""
    
    def __init__(self, length: float = 5.0, width: float = 0.6, **kwargs):
        super().__init__(**kwargs)
        self._capabilities.extend(['material_transport', 'speed_control'])
        self._specs.update({
            'dimensions': [length, width, 0.8],  # including support structure
            'belt_speed': kwargs.get('belt_speed', 0.5),  # m/s
            'max_load': kwargs.get('max_load', 100),  # kg
        })


# Custom mesh support
@register_asset("custom/mesh", {
    "category": "Custom 3D Model",
})
class CustomMesh(StaticAsset):
    """Custom 3D mesh asset"""
    
    def __init__(self, mesh_file: str, **kwargs):
        super().__init__(**kwargs)
        self._specs.update({
            'mesh_file': mesh_file,
            'file_format': mesh_file.split('.')[-1].lower(),
        })


# Export commonly used classes
__all__ = [
    # Drones
    'DjiTello',
    'DjiMavic3',
    'ParrotAnafi',
    # Ground Robots
    'BostonDynamicsSpot',
    'UnitreeGo1',
    'ClearpathHusky',
    'FrankaPanda',
    # Sensors
    'IntelRealSenseD435',
    'VelodynePuck',
    'ZED2',
    # Props
    'Box',
    'Sphere',
    'Cylinder',
    'TrafficCone',
    'Pallet',
    # Landmarks
    'ArucoMarker',
    'QRCode',
    'AprilTag',
    # Infrastructure
    'Wall',
    'ChargingPad',
    'Conveyor',
    'CustomMesh',
] 