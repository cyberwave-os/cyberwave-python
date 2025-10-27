from enum import Enum
from typing import Sequence
from entity import Entity



class Quaternion(Entity):

    def __init__(
            self,
            w: float,
            x: float,
            y: float,
            z: float):
        self.w = w
        self.x = x
        self.y = y
        self.z = z


