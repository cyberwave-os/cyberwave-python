from __future__ import annotations

import os
import pytest

from cyberwave.sdk import Cyberwave
from cyberwave.missions import Mission


@pytest.mark.skipif(
    not os.getenv("CYBERWAVE_BASE_URL") or not os.getenv("CYBERWAVE_TOKEN") or not os.getenv("CYBERWAVE_ENV_UUID"),
    reason="Requires running backend and env vars: CYBERWAVE_BASE_URL, CYBERWAVE_TOKEN, CYBERWAVE_ENV_UUID",
)
def test_mission_run_flow_smoke():
    base = os.getenv("CYBERWAVE_BASE_URL")
    token = os.getenv("CYBERWAVE_TOKEN")
    env = os.getenv("CYBERWAVE_ENV_UUID")

    cw = Cyberwave(base, token)

    mission = Mission(key="test/SmokeMission", version=1, name="Smoke Mission")
    (mission.world()
        .asset("props/box", alias="box1")
        .place("box1", [0, 0, 0, 1, 0, 0, 0])
    )

    cw.missions.register(mission)
    run = cw.runs.start(environment_uuid=env, mission_key=mission.key, mission_version=mission.version, mode="virtual")
    assert run.get("uuid")

    info = cw.runs.wait_until_complete(run["uuid"], timeout_s=5, poll_s=1.0)
    assert info.get("uuid") == run["uuid"]


