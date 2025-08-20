## Cyberwave Developer Value Proposition — MVP Tracker

Legend:
- ✅ Implemented
- 🟡 Limited (MVP stub or partial)
- ❌ Not implemented

### Feature Matrix

| Area | Feature | Status | Notes / Next Steps |
|---|---|---|---|
| Backend | Missions registry (list/get/register) | ✅ | In-memory registry (`/api/v1/missions`); consider persistence later |
| Backend | Runs (start/list/get/stop) | ✅ | Build returns MuJoCo XML in `resolved_world`; session runtime TBD |
| Backend | Teleop session + commands | ✅ | `/twins/{uuid}/teleop/*`, `/twins/{uuid}/commands` exist |
| Backend | Run evaluator (success/KPIs) | 🟡 | Add run-scoped outcome + minimal KPIs (`time_to_complete`, `outcome`) |
| Backend | Link teleop outcome → Run | 🟡 | Add `POST /runs/{id}/outcome` or map existing teleop mark to active run |
| Backend | MuJoCo build overlays (mission world_setup) | 🟡 | Currently ignored; build returns baseline env XML; add non-persistent overlays |
| Backend | Streaming (WS/SSE) | ❌ | Add simple WS broadcast for run status / twin pose |
| SDK | Pythonic facade (`Cyberwave`) | ✅ | Missions, Runs, Environments, Twins, Teleop |
| SDK | Mission builder (setup + goals) | ✅ | `Mission.world().asset().place().goal_*()` |
| SDK | Environments helpers | 🟡 | `get`, `list_for_project`, `create`, `EnvironmentHandle.find_twin_by_name()` added; alias helper TBD |
| SDK | Run outcome API | ❌ | Add `runs.outcome(run_id, "success"|"failure")` when backend ready |
| SDK | Viewer/stream sample | ❌ | Add example for WS/SSE updates |
| SDK | CLI quick commands | ❌ | `cyberwave runs start`, `runs stop`, `missions register` |
| Examples | Quickstart Mission/Run + teleop | ✅ | `examples/quickstart_mvp.py`, `run_pick_orange.py` |
| Examples | Tello teleop | ✅ | Updated to use facade + context manager |
| Examples | Arm commands | ✅ | Updated SO100 example to unified `twins.command` |
| Examples | End-to-end Mission/Run | ✅ | Creates env if project UUID set; starts run; optional teleop |
| Examples | Centralized schema demo | ✅ | Uses minimal dict; no deprecated LevelDefinition |
| Examples | Streaming/live status | ❌ | Add example once WS endpoint is in place |
| Planning | Coverage planner (2D/3D) | ❌ | Start with 2D strip planner; export poses |
| KPI | Coating KPIs (coverage/DFT) | ❌ | Simple estimator + run metrics/dashboard hooks |
| NLU | Natural-language → Mission | ❌ | Slot-fill prototype or LLM-backed helper |

### User Stories (checklist)

- ✅ As a developer, I can define a Mission (world setup + goals) programmatically and register it.
- ✅ As a developer, I can start a Run for an Environment from a Mission and poll its status.
- ✅ As a developer, I can open a teleop session and send high-level commands to a twin during a Run.
- ✅ As a developer, I can fetch the MuJoCo scene XML generated for the Run to inspect or export.
- 🟡 As a developer, I can mark a Run as success/failure and see basic KPIs (time_to_complete, outcome).
- 🟡 As a developer, I can find twins in an environment conveniently (by name/alias) without digging for UUIDs.
- ❌ As a developer, I can see live status/telemetry updates during a Run via a simple streaming API.
- ❌ As a developer, I can generate a Mission from a natural-language goal.
- ❌ As a developer, I can plan coverage paths and execute them in sim with basic KPI reporting.
- ❌ As a developer, I can export a Run (events + metadata) for training a policy or VLM/VLA scorer.

### Near-Term TODOs (high impact, low risk)

1) Backend
   - 🟡 Add `POST /api/v1/runs/{uuid}/outcome` and write `outcome`, `time_to_complete` to `Run.metrics`.
   - 🟡 Map existing teleop outcome to active Run if present (fallback to run endpoint).
   - 🟡 Extend MuJoCo build to accept mission overlays (non-persistent asset/placement injection).

2) SDK
   - 🟡 `runs.outcome(run_id, outcome)` wrapper.
   - 🟡 `environments.list()` and `twins.by_alias(env, alias)` helper.
   - ❌ Streaming helper once WS endpoint exists (subscribe, print status/pose).

3) Examples/Docs
   - 🟡 `examples/first_run_setup.py`: create env + add sample twins; print UUIDs for quickstarts.
   - ❌ `examples/live_status.py`: connect to WS, display status updates.
   - 🟡 README: Examples index with required env vars, expected outputs, and troubleshooting.

### Notes
- Current registry and runs store are in-memory for speed. Plan persistence as adoption grows.
- Keep environment mutations explicit; Runs should remain ephemeral by default.


