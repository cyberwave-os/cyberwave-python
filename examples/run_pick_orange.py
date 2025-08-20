from __future__ import annotations

import os
from cyberwave import Cyberwave, Mission


def main():
    base = os.getenv("CYBERWAVE_BASE_URL", "http://localhost:8000")
    token = os.getenv("CYBERWAVE_TOKEN", "")
    env_uuid = os.getenv("CYBERWAVE_ENV_UUID", "")
    crawler_uuid = os.getenv("CYBERWAVE_CRAWLER_TWIN_UUID", "")
    if not (token and env_uuid):
        raise SystemExit("Set CYBERWAVE_TOKEN and CYBERWAVE_ENV_UUID")

    cw = Cyberwave(base, token)
    env = cw.environments.get(env_uuid)

    # Define mission (setup + goals)
    mission = Mission(key="so101/PickOrange", version=1, name="Pick Orange into Bin")
    (mission.world()
        .asset("props/table-simple", alias="table")
        .asset("props/bin", alias="bin")
        .asset("props/orange", alias="orange1")
        .place("table",   [0,0,0, 1,0,0,0])
        .place("bin",     [0.6,0,0.8, 1,0,0,0])
        .place("orange1", [0.1,0,0.8, 1,0,0,0])
    )
    mission.parameters["seed"] = 42
    mission.goal_object_in_zone("orange1", "bin", tolerance_m=0.05, hold_s=2.0)

    cw.missions.register(mission)
    run = cw.runs.start(environment_uuid=env.uuid, mission_key=mission.key, mission_version=mission.version, parameters=mission.parameters, mode="virtual")
    run_id = run["uuid"]
    print("Run started:", run_id)

    # Pick crawler twin: prefer env var; otherwise try by name "crawler"
    if not crawler_uuid:
        t = env.find_twin_by_name("crawler")
        if not t:
            raise SystemExit("Set CYBERWAVE_CRAWLER_TWIN_UUID or name a twin 'crawler' in the environment")
        crawler_uuid = t["uuid"]

    # Teleop session (logs events) and issue high-level commands
    with cw.teleop.session(crawler_uuid, sensors=["front_cam"]):
        cw.twins.command(crawler_uuid, "navigate", {"target": [0.6, 0.0, 0.8]})
        cw.twins.command(crawler_uuid, "manipulator.pick", {"object": "orange1"})
        cw.twins.command(crawler_uuid, "manipulator.place", {"target": "bin"})
        cw.teleop.mark_outcome(crawler_uuid, "success")

    info = cw.runs.wait_until_complete(run_id, timeout_s=30)
    print("Run status:", info.get("status"))
    print("Metrics:", info.get("metrics"))
    if (info.get("resolved_world") or {}).get("xml"):
        print("MuJoCo XML present in run.resolved_world.xml")


if __name__ == "__main__":
    main()


