from enum import Enum
from typing import Sequence
from cyberwave.mqtt.entity import Entity



class PingPayload(Entity):

    def __init__(
            self,
            type: str,
            timestamp: float):
        self.type = type
        self.timestamp = timestamp


