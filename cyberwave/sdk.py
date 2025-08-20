from __future__ import annotations

import time
from typing import Optional, Dict, Any, List
from .http import HttpClient
from .missions import MissionsAPI, Mission
from .runs import RunsAPI
from .twins import TwinsAPI
from .teleop import TeleopAPI
from .environments import EnvironmentsAPI
from .sensors import SensorsAPI
from .projects import ProjectsAPI


class Cyberwave:
    def __init__(self, base_url: str, token: str):
        http = HttpClient(base_url, token)
        self._http = http
        self.missions = MissionsAPI(http)
        self.runs = RunsAPI(http)
        self.environments = EnvironmentsAPI(http)
        self.projects = ProjectsAPI(http)
        self.twins = TwinsAPI(http)
        self.teleop = TeleopAPI(http)
        self.sensors = SensorsAPI(http)


