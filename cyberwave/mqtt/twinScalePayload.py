from enum import Enum
from typing import Sequence
from entity import Entity



class TwinScalePayload(Entity):

    class Scale(Entity):

        def __init__(
                self,
                x: float,
                y: float,
                z: float):
            self.x = x
            self.y = y
            self.z = z


    def __init__(
            self,
            scale: Scale,
            timestamp: float):
        self.scale = scale
        self.timestamp = timestamp


