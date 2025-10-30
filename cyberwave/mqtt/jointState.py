from enum import Enum
from typing import Sequence
from cyberwave.mqtt.entity import Entity



class JointState(Entity):

    def __init__(
            self,
            position: float,
            velocity: float,
            effort: float):
        self.position = position
        self.velocity = velocity
        self.effort = effort


