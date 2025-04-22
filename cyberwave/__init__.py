"""
CyberWave - A Python project for robot control and automation
"""

from .robot import Robot
from .trainer import VideoTrainer, perform_welding
from .client import Client, CyberWaveError, APIError, AuthenticationError

__version__ = "0.1.0" 

__all__ = [
    "Client",
    "CyberWaveError",
    "APIError",
    "AuthenticationError",
    # "Robot", 
] 