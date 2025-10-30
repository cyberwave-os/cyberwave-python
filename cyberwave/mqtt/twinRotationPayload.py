from enum import Enum
from typing import Sequence
from cyberwave.mqtt.entity import Entity



class TwinRotationPayload(Entity):

    class Rotation(Entity):

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


    def __init__(
            self,
            rotation: Rotation,
            timestamp: float):
        self.rotation = rotation
        self.timestamp = timestamp


