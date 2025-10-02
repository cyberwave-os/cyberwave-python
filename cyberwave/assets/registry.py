"""
Asset Registry System - HuggingFace-style asset references for Cyberwave

This module provides a registry system that allows assets to be referenced
using HuggingFace-style identifiers (e.g., "dji/tello", "boston-dynamics/spot").
"""

import inspect
from dataclasses import dataclass, field
from typing import Dict, Type, Optional, Any, List, Callable
from functools import wraps
import json
import logging


@dataclass
class AssetInfo:
    """Metadata object returned by :meth:`AssetRegistry.list`."""

    asset_id: str
    asset_class: Type['BaseAsset']
    asset_type: Optional[str] = None
    default_capabilities: List[str] = field(default_factory=list)
    default_specs: Dict[str, Any] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, str):
            return self.asset_id == other
        if isinstance(other, type):
            return self.asset_class is other
        if isinstance(other, AssetInfo):
            return self.asset_id == other.asset_id
        return False

    def __hash__(self) -> int:  # Needed for set/list membership comparisons
        return hash(self.asset_id)

    def __repr__(self) -> str:
        return self.asset_id

    # Support old-style string API used in some tests
    def startswith(self, prefix: str) -> bool:
        return self.asset_id.startswith(prefix)

logger = logging.getLogger(__name__)


class AssetRegistry:
    """Central registry for all asset types"""
    
    _instance = None
    _registry: Dict[str, Type['BaseAsset']] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, 
                 asset_id: str, 
                 asset_class: Type['BaseAsset'],
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register an asset class with the registry.
        
        Args:
            asset_id: HuggingFace-style ID (e.g., "dji/tello")
            asset_class: The asset class to register
            metadata: Optional metadata about the asset
        """
        if asset_id in cls._registry:
            logger.warning(f"Overwriting existing asset registration: {asset_id}")
        
        cls._registry[asset_id] = asset_class
        cls._metadata[asset_id] = metadata or {}
        
        # Also register without namespace if it's the only one
        if "/" in asset_id:
            namespace, name = asset_id.split("/", 1)
            # Check if this name is unique across namespaces
            conflicting = [k for k in cls._registry.keys() 
                          if k.endswith(f"/{name}") and k != asset_id]
            if not conflicting:
                cls._registry[name] = asset_class
                cls._metadata[name] = metadata or {}
    
    @classmethod
    def get(cls, asset_id: str) -> Optional[Type['BaseAsset']]:
        """Get an asset class by ID"""
        return cls._registry.get(asset_id)
    
    @classmethod
    def list(
        cls,
        namespace: Optional[str] = None,
        asset_type: Optional[str] = None,
    ) -> List[AssetInfo]:
        """List registered assets with optional filtering."""
        results: List[AssetInfo] = []
        seen: set[str] = set()
        for asset_id, asset_cls in cls._registry.items():
            if '/' not in asset_id:
                # Skip shorthand aliases
                continue
            if asset_id in seen:
                continue
            seen.add(asset_id)
            if namespace and not asset_id.startswith(f"{namespace}/"):
                continue
            meta = cls._metadata.get(asset_id, {})
            atype = meta.get('asset_type') or getattr(asset_cls, 'asset_type', None)
            if asset_type and atype != asset_type:
                continue
            capabilities = meta.get('default_capabilities', getattr(asset_cls, 'default_capabilities', []))
            specs = meta.get('default_specs', getattr(asset_cls, 'default_specs', {}))
            results.append(
                AssetInfo(
                    asset_id=asset_id,
                    asset_class=asset_cls,
                    asset_type=atype,
                    default_capabilities=list(capabilities) if capabilities else [],
                    default_specs=dict(specs) if specs else {},
                )
            )
        return results
    
    @classmethod
    def get_metadata(cls, asset_id: str) -> Dict[str, Any]:
        """Get metadata for an asset"""
        return cls._metadata.get(asset_id, {})


def register_asset(
    asset_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    asset_type: Optional[str] = None,
    default_capabilities: Optional[List[str]] = None,
    default_specs: Optional[Dict[str, Any]] = None,
) -> Callable:
    """Decorator to register an asset class."""

    def decorator(cls: Type['BaseAsset']) -> Type['BaseAsset']:
        if '/' not in asset_id or any(c.isspace() for c in asset_id):
            raise ValueError(f"Invalid asset_id '{asset_id}'")
        meta = dict(metadata or {})
        inferred_type = asset_type
        if inferred_type is None:
            if issubclass(cls, Robot) or issubclass(cls, FlyingRobot) or issubclass(cls, GroundRobot):
                inferred_type = 'robot'
            elif issubclass(cls, Sensor):
                inferred_type = 'sensor'
            elif issubclass(cls, StaticAsset):
                inferred_type = 'static'
        if inferred_type is not None:
            meta.setdefault('asset_type', inferred_type)
        if default_capabilities is not None:
            meta.setdefault('default_capabilities', default_capabilities)
        else:
            try:
                meta.setdefault('default_capabilities', cls().capabilities)
            except Exception:
                meta.setdefault('default_capabilities', [])
        if default_specs is not None:
            meta.setdefault('default_specs', default_specs)
        else:
            try:
                meta.setdefault('default_specs', cls().specs)
            except Exception:
                meta.setdefault('default_specs', {})

        display_name = meta.get('name')
        if not display_name:
            manufacturer = meta.get('manufacturer', '')
            model = meta.get('model', '')
            display_name = f"{manufacturer} {model}".strip()
        if not display_name:
            import re
            display_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', cls.__name__).strip()
        cls.default_name = display_name

        AssetRegistry.register(asset_id, cls, meta)
        cls._registry_id = asset_id
        cls.asset_id = asset_id
        if inferred_type is not None:
            cls.asset_type = inferred_type
        if default_capabilities is not None:
            cls.default_capabilities = default_capabilities
        else:
            cls.default_capabilities = meta.get('default_capabilities', [])
        if default_specs is not None:
            cls.default_specs = default_specs
        else:
            cls.default_specs = meta.get('default_specs', {})
        return cls

    return decorator


class AssetFactory:
    """Factory for creating assets from configuration"""
    
    @staticmethod
    def create(asset_id: str, **kwargs) -> 'BaseAsset':
        """
        Create an asset instance from registry ID.
        
        Args:
            asset_id: Registry ID or class name
            **kwargs: Configuration parameters
            
        Returns:
            Configured asset instance
        """
        asset_class = AssetRegistry.get(asset_id)
        
        if not asset_class:
            raise ValueError(f"Unknown asset type: {asset_id}")
        
        # Filter kwargs to only include valid constructor parameters
        sig = inspect.signature(asset_class.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        
        return asset_class(**filtered_kwargs)
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> 'BaseAsset':
        """
        Create an asset from a configuration dictionary.
        
        Args:
            config: Configuration with 'type' and optional parameters
            
        Returns:
            Configured asset instance
        """
        asset_type = config.pop('type')
        return AssetFactory.create(asset_type, **config)


class BaseAsset:
    """Base class for all assets in the Cyberwave system"""
    
    _registry_id: Optional[str] = None

    def __init__(self,
                 name: Optional[str] = None,
                 **kwargs):
        """
        Initialize base asset.
        
        Args:
            name: Optional custom name for the asset instance
            **kwargs: Additional configuration
        """
        if name and '/' in name:
            asset_cls = AssetRegistry.get(name)
            if asset_cls and asset_cls is not self.__class__:
                self.__class__ = asset_cls
                asset_cls.__init__(self, **kwargs)
                return

        self.name = name or getattr(self.__class__, 'default_name', self.__class__.__name__)
        self._config = kwargs
        self._capabilities = []
        self._specs = {}
        self.state: Dict[str, Any] = {}
        if hasattr(self.__class__, 'default_capabilities'):
            self._capabilities.extend(list(getattr(self.__class__, 'default_capabilities')))
        if hasattr(self.__class__, 'default_specs'):
            self._specs.update(dict(getattr(self.__class__, 'default_specs')))

    @property
    def asset_id(self) -> Optional[str]:
        """Alias for :pyattr:`registry_id` for compatibility."""
        return self.registry_id
        
    def __init_subclass__(cls, **kwargs):
        """Called when a class is subclassed"""
        super().__init_subclass__(**kwargs)
        
        # Auto-register if running in a specific mode
        # This could be controlled by an environment variable
        # For now, we'll require explicit registration
        pass
    
    @property
    def registry_id(self) -> Optional[str]:
        """Get the registry ID for this asset type"""
        return self._registry_id
    
    @property
    def capabilities(self) -> List[str]:
        """Get asset capabilities"""
        return self._capabilities
    
    @property
    def specs(self) -> Dict[str, Any]:
        """Get asset specifications"""
        return self._specs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert asset to dictionary representation"""
        return {
            'type': self.registry_id or self.__class__.__name__,
            'name': self.name,
            'capabilities': self.capabilities,
            'specs': self.specs,
            'config': self._config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseAsset':
        """Create asset from dictionary representation"""
        asset_type = data.get('type', cls.__name__)
        
        # Try to get from registry first
        asset_class = AssetRegistry.get(asset_type)
        if asset_class:
            return asset_class(
                name=data.get('name'),
                **data.get('config', {})
            )
        
        # Fallback to current class
        return cls(
            name=data.get('name'),
            **data.get('config', {})
        )


class Robot(BaseAsset):
    """Base class for all robots"""
    
    def __init__(self, 
                 registry_id: Optional[str] = None,
                 **kwargs):
        """
        Initialize a robot, either from registry or direct instantiation.
        
        Args:
            registry_id: Optional registry ID to load pre-configured robot
            **kwargs: Configuration parameters
        """
        # If registry_id provided, create from registry
        if registry_id:
            robot_class = AssetRegistry.get(registry_id)
            if robot_class and robot_class != self.__class__:
                # Create instance of the registered class
                instance = robot_class(**kwargs)
                # Copy attributes to self
                self.__dict__.update(instance.__dict__)
                return
        
        super().__init__(**kwargs)
        self._capabilities.extend(['move', 'sense'])


class FlyingRobot(Robot):
    """Base class for flying robots (drones, quadcopters, etc.)"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._capabilities.extend(['fly', 'hover', 'land', 'takeoff', 'flight'])
        self._specs.update({
            'max_altitude': kwargs.get('max_altitude', 100),  # meters
            'max_speed': kwargs.get('max_speed', 10),  # m/s
            'battery_capacity': kwargs.get('battery_capacity', 2000),  # mAh
        })


class GroundRobot(Robot):
    """Base class for ground robots"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._capabilities.extend(['drive', 'turn'])
        self._specs.update({
            'max_speed': kwargs.get('max_speed', 5),  # m/s
            'turning_radius': kwargs.get('turning_radius', 0.5),  # meters
        })


class Sensor(BaseAsset):
    """Base class for all sensors"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._capabilities.append('measure')


class CameraSensor(Sensor):
    """Base class for camera sensors"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._capabilities.extend(['capture_image', 'stream_video'])
        self._specs.update({
            'resolution': kwargs.get('resolution', [1920, 1080]),
            'fps': kwargs.get('fps', 30),
            'fov': kwargs.get('fov', 60),  # degrees
        })


class DepthSensor(CameraSensor):
    """Base class for depth sensors"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._capabilities.append('measure_depth')
        self._specs.update({
            'depth_range': kwargs.get('depth_range', [0.1, 10]),  # meters
            'depth_resolution': kwargs.get('depth_resolution', 0.001),  # meters
        })


class StaticAsset(BaseAsset):
    """Base class for static assets (props, landmarks, infrastructure)"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._specs.update({
            'dimensions': kwargs.get('dimensions', [1, 1, 1]),  # x, y, z in meters
            'material': kwargs.get('material', 'default'),
            'color': kwargs.get('color', [0.5, 0.5, 0.5]),  # RGB
        })


class Prop(StaticAsset):
    """Static objects used as props in scenes"""
    pass


class Landmark(StaticAsset):
    """Static objects used as landmarks for navigation"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._capabilities.append('localization_reference')


class Infrastructure(StaticAsset):
    """Static infrastructure elements"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._capabilities.append('environment_element')


# Export main components
# Re-export common asset implementations so they can be imported directly from
# ``cyberwave.assets.registry``. This matches expectations in some tests and
# simplifies the public API.
from .implementations import (
    DjiTello,
    DjiMavic3,
    ParrotAnafi,
    BostonDynamicsSpot,
    UnitreeGo1,
    ClearpathHusky,
    FrankaPanda,
    IntelRealSenseD435,
    VelodynePuck,
    ZED2,
    Box,
    Sphere,
    Cylinder,
    TrafficCone,
    Pallet,
    ArucoMarker,
    QRCode,
    AprilTag,
    Wall,
    ChargingPad,
    Conveyor,
    CustomMesh,
)

__all__ = [
    'AssetRegistry',
    'register_asset',
    'AssetFactory',
    'BaseAsset',
    'Robot',
    'FlyingRobot',
    'GroundRobot',
    'Sensor',
    'CameraSensor',
    'DepthSensor',
    'StaticAsset',
    'Prop',
    'Landmark',
    'Infrastructure',
    # Implementations
    'DjiTello',
    'DjiMavic3',
    'ParrotAnafi',
    'BostonDynamicsSpot',
    'UnitreeGo1',
    'ClearpathHusky',
    'FrankaPanda',
    'IntelRealSenseD435',
    'VelodynePuck',
    'ZED2',
    'Box',
    'Sphere',
    'Cylinder',
    'TrafficCone',
    'Pallet',
    'ArucoMarker',
    'QRCode',
    'AprilTag',
    'Wall',
    'ChargingPad',
    'Conveyor',
    'CustomMesh',
]
