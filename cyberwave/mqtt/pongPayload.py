from enum import Enum
from typing import Sequence
from entity import Entity



class PongPayload(Entity):

    def __init__(
            self,
            type: str,
            timestamp: float):
        self.type = type
        self.timestamp = timestamp


