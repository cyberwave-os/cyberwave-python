from enum import Enum
from typing import Sequence
from cyberwave.mqtt.entity import Entity



class TwinPositionPayload(Entity):

    class Position(Entity):

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
            position: Position,
            timestamp: float):
        self.position = position
        self.timestamp = timestamp


