from __future__ import annotations

import os
from cyberwave import Cyberwave, Mission


def main():
    base = os.getenv("CYBERWAVE_BASE_URL", "http://localhost:8000")
    token = os.getenv("CYBERWAVE_TOKEN", "")
    env_uuid = os.getenv("CYBERWAVE_ENV_UUID", "")
    project_uuid = os.getenv("CYBERWAVE_PROJECT_UUID", "")
    crawler_uuid = os.getenv("CYBERWAVE_CRAWLER_TWIN_UUID", "")
    if not token:
        raise SystemExit("Set CYBERWAVE_TOKEN")

    cw = Cyberwave(base, token)

    # Optionally create an environment if only project UUID was provided
    if not env_uuid and project_uuid:
        env = cw.environments.create(project_uuid, name="SDK Demo Env", description="End-to-end demo")
        env_uuid = env["uuid"]

    if not env_uuid:
        raise SystemExit("Set CYBERWAVE_ENV_UUID or CYBERWAVE_PROJECT_UUID to create an environment")

    # Define a mission (setup + goals)
    mission = Mission(key="so101/PickOrange", version=1, name="E2E Pick Orange")
    (mission.world()
        .asset("props/table-simple", alias="table")
        .asset("props/bin", alias="bin")
        .asset("props/orange", alias="orange1")
        .place("table",   [0,0,0, 1,0,0,0])
        .place("bin",     [0.6,0,0.8, 1,0,0,0])
        .place("orange1", [0.1,0,0.8, 1,0,0,0])
    )
    mission.parameters["seed"] = 7
    mission.goal_object_in_zone("orange1", "bin", tolerance_m=0.05, hold_s=2.0)

    cw.missions.register(mission)
    run = cw.runs.start(environment_uuid=env_uuid, mission_key=mission.key, mission_version=mission.version, parameters=mission.parameters, mode="virtual")
    run_id = run["uuid"]
    print("Run:", run_id)

    # Optional: teleop the crawler if provided
    if crawler_uuid:
        with cw.teleop.session(crawler_uuid, sensors=["front_cam"]):
            cw.twins.command(crawler_uuid, "navigate", {"target": [0.6, 0.0, 0.8]})
            cw.twins.command(crawler_uuid, "manipulator.pick", {"object": "orange1"})
            cw.twins.command(crawler_uuid, "manipulator.place", {"target": "bin"})
            cw.teleop.mark_outcome(crawler_uuid, "success")

    info = cw.runs.wait_until_complete(run_id, timeout_s=30)
    print("Status:", info.get("status"))
    print("Metrics:", info.get("metrics"))
    if (info.get("resolved_world") or {}).get("xml"):
        print("MuJoCo XML present in run.resolved_world.xml")


if __name__ == "__main__":
    main()
