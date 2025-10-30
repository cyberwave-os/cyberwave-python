from enum import Enum
from typing import Sequence
from cyberwave.mqtt.entity import Entity



class Vector3(Entity):

    def __init__(
            self,
            x: float,
            y: float,
            z: float):
        self.x = x
        self.y = y
        self.z = z


