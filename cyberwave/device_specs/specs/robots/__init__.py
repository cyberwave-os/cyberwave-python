"""
Robot Device Specifications

Contains specifications for robotic devices including drones, ground robots, and robotic arms.
"""

from .dji_tello import DjiTelloSpec
from .boston_dynamics_spot import BostonDynamicsSpotSpec  
from .so101 import So101Spec
from .kuka_kr3 import KukaKr3Spec
from .ur5e import UniversalRobotsUR5eSpec

__all__ = [
    "DjiTelloSpec",
    "BostonDynamicsSpotSpec",
    "So101Spec", 
    "KukaKr3Spec",
    "UniversalRobotsUR5eSpec",
]
