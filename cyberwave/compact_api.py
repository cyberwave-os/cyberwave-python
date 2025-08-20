"""
Compact API for Cyberwave SDK
Provides the simplified interface shown in the catalog
"""

from typing import Optional, List, Tuple, Union
from .client import Client
from .digital_twin import DigitalTwin
from .constants import DEFAULT_BACKEND_URL
import os

# Global client instance
_global_client: Optional[Client] = None

def configure(api_key: Optional[str] = None, base_url: str = DEFAULT_BACKEND_URL, environment: Optional[str] = None):
    """Configure the global Cyberwave client"""
    global _global_client
    
    if api_key is None:
        api_key = os.getenv('CYBERWAVE_API_KEY')
    
    _global_client = Client(base_url=base_url)
    
    # Store the API key for later use
    if api_key:
        _global_client._access_token = api_key
    
    if environment:
        _global_client.default_environment = environment

def _get_client() -> Client:
    """Get the global client, creating one if needed"""
    global _global_client
    
    if _global_client is None:
        configure()
    
    return _global_client

class CompactTwin:
    """
    Compact twin interface for easy digital twin control
    Implements the API shown in the catalog Python snippets
    """
    
    def __init__(self, registry_id: str, name: Optional[str] = None, environment_id: Optional[str] = None):
        self.registry_id = registry_id
        self.name = name or registry_id.split('/')[-1]
        self.environment_id = environment_id
        self._client = _get_client()
        self._twin_uuid: Optional[str] = None
        self._position = [0.0, 0.0, 0.0]
        self._rotation = [0.0, 0.0, 0.0]  # roll, pitch, yaw in degrees
        
    def _ensure_twin_exists(self):
        """Ensure the twin exists in the environment"""
        if self._twin_uuid is None:
            # TODO: Implement twin creation via API
            # For now, we'll simulate this
            self._twin_uuid = f"twin-{self.registry_id.replace('/', '-')}"
    
    @property
    def position(self) -> List[float]:
        """Get current position [x, y, z]"""
        return self._position.copy()
    
    @position.setter
    def position(self, pos: List[float]):
        """Set position [x, y, z]"""
        if len(pos) != 3:
            raise ValueError("Position must be [x, y, z]")
        self._position = pos
        self._update_backend_position()
    
    @property
    def rotation(self) -> List[float]:
        """Get current rotation [roll, pitch, yaw] in degrees"""
        return self._rotation.copy()
    
    @rotation.setter
    def rotation(self, rot: List[float]):
        """Set rotation [roll, pitch, yaw] in degrees"""
        if len(rot) != 3:
            raise ValueError("Rotation must be [roll, pitch, yaw] in degrees")
        self._rotation = rot
        self._update_backend_rotation()
    
    def move(self, x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None):
        """Move to position with optional coordinates"""
        if x is not None:
            self._position[0] = x
        if y is not None:
            self._position[1] = y
        if z is not None:
            self._position[2] = z
        
        self._update_backend_position()
    
    def rotate(self, roll: Optional[float] = None, pitch: Optional[float] = None, yaw: Optional[float] = None):
        """Rotate with optional angles in degrees"""
        if roll is not None:
            self._rotation[0] = roll
        if pitch is not None:
            self._rotation[1] = pitch
        if yaw is not None:
            self._rotation[2] = yaw
        
        self._update_backend_rotation()
    
    def move_to(self, position: List[float], orientation: Optional[List[float]] = None):
        """Move to target position with optional orientation"""
        self.position = position
        if orientation:
            self.rotation = orientation
    
    def _update_backend_position(self):
        """Update position in backend"""
        self._ensure_twin_exists()
        # TODO: Implement actual API call to update twin position
        print(f"[CompactTwin] {self.name} moved to position: {self._position}")
    
    def _update_backend_rotation(self):
        """Update rotation in backend"""
        self._ensure_twin_exists()
        # TODO: Implement actual API call to update twin rotation
        print(f"[CompactTwin] {self.name} rotated to: {self._rotation} degrees")
    
    @property
    def joints(self):
        """Access to joint control interface"""
        return JointController(self)
    
    @property
    def has_sensors(self) -> bool:
        """Check if twin has sensors"""
        # TODO: Implement sensor detection
        return False
    
    def delete(self):
        """Delete the twin from the environment"""
        # TODO: Implement twin deletion
        print(f"[CompactTwin] {self.name} deleted")

class JointController:
    """Joint control interface for robotic twins"""
    
    def __init__(self, twin: CompactTwin):
        self._twin = twin
        self._joint_states = {}
    
    def __setattr__(self, name: str, value: float):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            # This is a joint name
            self._joint_states[name] = value
            self._update_joint_backend(name, value)
    
    def __getattr__(self, name: str) -> float:
        return self._joint_states.get(name, 0.0)
    
    def _update_joint_backend(self, joint_name: str, value: float):
        """Update joint state in backend"""
        print(f"[CompactTwin] {self._twin.name} joint '{joint_name}' set to {value}")
    
    def all(self) -> dict:
        """Get all joint states"""
        return self._joint_states.copy()

class SimulationController:
    """Global simulation control"""
    
    @staticmethod
    def play():
        """Start/resume simulation"""
        print("[Simulation] Started")
    
    @staticmethod
    def pause():
        """Pause simulation"""
        print("[Simulation] Paused")
    
    @staticmethod
    def step():
        """Single simulation step"""
        print("[Simulation] Single step")
    
    @staticmethod
    def reset():
        """Reset simulation"""
        print("[Simulation] Reset")

# Global simulation instance
simulation = SimulationController()

def twin(registry_id: str, name: Optional[str] = None, environment_id: Optional[str] = None) -> CompactTwin:
    """
    Create a digital twin from a registry ID
    
    This is the compact API shown in the catalog:
    
    Example:
        so101 = cw.twin("cyberwave/so101")
        so101.move(x=1, y=0, z=0.5)
        so101.rotate(yaw=90)
        so101.joints.arm_joint = 45
    """
    return CompactTwin(registry_id, name, environment_id)
