"""
Mission Examples — Cyberwave SDK

Three layers of progressive disclosure:

  Level 1 — VERB CALLS: one line, the skill handles everything.
  Level 2 — STEP SEQUENCES: multi-step, you control the order.
  Level 3 — OVERRIDES: tune AI models, retries, assessment when needed.

The intelligence (perception, planning, control, assessment, retry, fallback)
lives INSIDE skills — not in the mission definition. Mission authors declare
WHAT, not HOW.

Requirements:
    pip install cyberwave
"""

from cyberwave import Cyberwave

cw = Cyberwave()


# ═════════════════════════════════════════════════════════════════════════
# LEVEL 1 — VERB CALLS
#
# One line per mission. The system resolves which robot does what
# (based on capabilities), picks the right models, handles retries.
# This covers 80% of real missions.
# ═════════════════════════════════════════════════════════════════════════


# ── Warehouse: AGV + arm pick and deliver ────────────────────────────────

fleet = cw.fleet("warehouse_alpha", members=["agv-01", "arm-station-03"])
fleet.pick_and_deliver("SKU-4421", from_loc="shelf_B3", to_loc="packing_station_7")


# ── Retail: quadruped shelf audit ────────────────────────────────────────

spot = cw.twin("spot-01")
spot.inspect_shelves(aisles=["aisle_1", "aisle_2", "aisle_3", "aisle_4"],
                     report_to="store_manager@company.com")


# ── Defense: drone + ground robot perimeter patrol ───────────────────────

fleet = cw.fleet("perimeter_alpha", members=["drone-recon-01", "ugv-patrol-01"])
fleet.patrol(area="perimeter_zone", escalate_to="ops_center", repeat=True)


# ── Manufacturing: mobile inspector checks weld quality ──────────────────

inspector = cw.twin("mobile-inspector-01")
inspector.inspect_welds(
    stations=["W1", "W2", "W3"],
    spec="AWS D1.1 Class A",
    report_to="qa_dashboard",
)


# ── Physical security: humanoid + drone night shift ──────────────────────

fleet = cw.fleet("night_security", members=["humanoid-sec-01", "drone-sec-01"])
fleet.patrol(
    areas={"indoor": ["lobby", "floor_1", "floor_2"], "outdoor": ["parking", "perimeter"]},
    schedule="22:00-06:00",
    escalate_to="security_ops",
)


# ── Manipulation: arm sorts parts from bin ───────────────────────────────

arm = cw.twin("ur10e-station-02")
arm.pick_and_sort(
    from_bin="parts_bin",
    trays={"bolts": "tray_bolts", "nuts": "tray_nuts", "washers": "tray_washers"},
    repeat_until="bin_empty",
)


# ── Aerial: drone inspects roof for defects ──────────────────────────────

drone = cw.twin("drone-survey-01")
drone.survey_and_report(area="building_A_roof", report_to="manager@property.com")


# ── Delivery: rover brings package to customer ──────────────────────────

rover = cw.twin("rover-delivery-14")
rover.deliver(package="ORD-7823", to={"address": "145 Main St", "apt": "3B"})


# ── Construction: drone + rover daily progress ───────────────────────────

fleet = cw.fleet("site_monitor", members=["drone-survey-02", "rover-measure-01"])
fleet.monitor_progress(
    area="construction_site_7",
    compare_to="bim_model",
    report_to="pm@construction.com",
    schedule="0 8 * * 1-5",
)


# ── Agriculture: drone + rover vineyard health ───────────────────────────

fleet = cw.fleet("vineyard_team", members=["ag-drone-01", "ag-rover-01"])
fleet.inspect_crops(area="block_C", concern="disease", report_to="agronomist@vineyard.com")


# ── Port: drone + AGVs + crane unload vessel ─────────────────────────────

fleet = cw.fleet("port_ops", members=["port-drone-01", "port-agv-01", "port-agv-02", "gantry-crane-01"])
fleet.unload_vessel(berth="5")


# ── Search & rescue: swarm responds to building collapse ─────────────────

fleet = cw.fleet("sar_bravo", members=["sar-drone-01", "sar-drone-02", "sar-spot-01", "sar-humanoid-01"])
fleet.search_and_rescue(area="collapse_zone_east", priority="critical")


# ═════════════════════════════════════════════════════════════════════════
# LEVEL 2 — STEP SEQUENCES
#
# When you need to control the order, compose multiple skills,
# or add logic between steps. Still no DAG wiring — just steps.
# ═════════════════════════════════════════════════════════════════════════


# ── Warehouse: custom pick-and-deliver with inspection step ──────────────

fleet = cw.fleet("warehouse_alpha", members=["agv-01", "arm-station-03"])

with fleet.mission("pick_inspect_deliver") as m:
    m.navigate("shelf_B3", who="agv-01")
    scan = m.scan("shelf_B3", looking_for="SKU-4421", who="arm-station-03")
    m.pick(scan.target, who="arm-station-03")
    m.inspect(scan.target, check="no visible damage", who="arm-station-03")
    m.place_on("agv_tray", who="arm-station-03")
    m.navigate("packing_station_7", who="agv-01")

m.execute()


# ── Bin picking: explicit pick-classify-place loop ───────────────────────

arm = cw.twin("ur10e-station-02")

with arm.mission("bin_picking_custom") as m:
    while m.repeat_until("bin_empty"):
        scene = m.scan("parts_bin")
        target = m.plan_grasp(scene, strategy="most_accessible")
        m.pick(target)
        part = m.classify(camera="wrist")
        m.place(tray=part.label)

m.execute()


# ── Defense patrol with investigation branch ─────────────────────────────

fleet = cw.fleet("perimeter_alpha", members=["drone-recon-01", "ugv-patrol-01"])

with fleet.mission("patrol_with_investigation") as m:
    while m.repeat():
        anomaly = m.patrol(area="perimeter_zone")
        if anomaly:
            threat = m.investigate(anomaly.location)
            if threat.confirmed:
                m.escalate(threat, to="ops_center")

m.execute()


# ── Cloth folding: garment-aware multi-step ──────────────────────────────

humanoid = cw.twin("humanoid-fold-01")

with humanoid.mission("fold_shirt") as m:
    garment = m.scan("work_surface", looking_for="garment")
    if garment.state == "crumpled":
        m.flatten(garment)
    m.fold(garment, style="retail")
    quality = m.inspect(garment, check="symmetric fold, no wrinkles")
    if not quality.acceptable:
        m.unfold(garment)
        m.fold(garment, style="retail")
    m.place(garment, on="folded_stack")

m.execute()


# ── Aerial inspection with follow-up on defects ─────────────────────────

drone = cw.twin("drone-survey-01")

with drone.mission("roof_with_followup") as m:
    survey = m.survey("building_A_roof")
    defects = m.analyze(survey, looking_for="damage")
    for defect in defects:
        m.fly_to(defect.location, altitude=5)
        m.photograph(defect, detail="close_up")
    m.generate_report(defects, to="manager@property.com")

m.execute()


# ── Delivery with real-world contingencies ───────────────────────────────

rover = cw.twin("rover-delivery-14")

with rover.mission("delivery_ORD_7823") as m:
    m.load_package("ORD-7823", at="depot_loading_bay")
    m.navigate(to={"lat": 37.7849, "lon": -122.4094}, mode="sidewalk")
    m.notify("customer_7823", message="Your delivery is arriving")
    entry = m.request_access("building_entrance")
    if not entry.granted:
        m.wait(timeout=180)
        m.alert("dispatch", message="Cannot access building")
        m.return_to("depot")
        return
    m.navigate(to="floor_3_apt_3B")
    m.photograph("package_at_door")
    m.notify("customer_7823", message="Delivered to your door")
    m.return_to("depot")

m.execute()


# ── SAR with phased approach ─────────────────────────────────────────────

fleet = cw.fleet("sar_bravo",
    members=["sar-drone-01", "sar-drone-02", "sar-spot-01", "sar-humanoid-01"])

with fleet.mission("building_collapse_sar") as m:
    m.priority("critical")

    # Phase 1: aerial sweep
    detections = m.sweep(
        areas={"north": "sar-drone-01", "south": "sar-drone-02"},
        looking_for="survivors",
    )

    for detection in detections:
        # Phase 2: ground confirmation
        confirmed = m.confirm(
            detection,
            who="sar-spot-01",
            sensors=["thermal", "rgb"],
        )
        if not confirmed:
            continue

        # Phase 3: human approval + extraction
        approved = m.request_approval(
            f"Survivor confirmed at {detection.location}. Approve extraction?",
        )
        if approved:
            m.extract(detection, who="sar-humanoid-01")
            m.overwatch(detection.location, who="sar-drone-01")
        else:
            m.mark_for_human_team(detection.location)

m.execute()


# ═════════════════════════════════════════════════════════════════════════
# LEVEL 3 — OVERRIDES
#
# When you need to control the AI: which model, how many retries,
# whether to assess visually, custom prompts. These are kwargs on
# the same skill calls — not a separate API.
# ═════════════════════════════════════════════════════════════════════════


# ── Override the perception model ────────────────────────────────────────

arm = cw.twin("ur10e-station-02")
arm.pick_and_sort(
    from_bin="parts_bin",
    trays={"bolts": "tray_bolts", "nuts": "tray_nuts"},
    perception="custom-part-detector-v1",       # use a custom detection model
    repeat_until="bin_empty",
)


# ── Override the control strategy ────────────────────────────────────────

arm.pick("bolt_m6", from_bin="parts_bin",
    control="code_policy",                      # prefer code-as-policy over default VLA
    control_model="claude-opus",                # which LLM writes the code
    max_retries=5,
)


# ── Force VLA with specific checkpoint ───────────────────────────────────

arm.pick("bolt_m6", from_bin="parts_bin",
    control="vla",
    control_model="openvla-oft",
    control_checkpoint="bin-pick-v2.3",
    control_device="cloud",                     # run on cloud GPU
    control_frequency_hz=10,
)


# ── Force diffusion policy for dexterous tasks ───────────────────────────

humanoid = cw.twin("humanoid-fold-01")
humanoid.fold("shirt",
    control="diffusion",
    control_model="pi0.5-cloth-fold",
    cameras=["overhead_rgb", "wrist_left", "wrist_right"],
)


# ── Enable visual assessment with custom prompt ──────────────────────────

arm.pick("bolt_m6", from_bin="parts_bin",
    assess=True,                                # enable visual verification
    assess_prompt="Is the bolt held firmly without slipping?",  # custom check
    assess_model="gpt-5-vision",                # which vision model judges
)


# ── Custom LLM analysis in inspection ────────────────────────────────────

drone = cw.twin("drone-survey-01")
drone.survey_and_report("building_A_roof",
    report_to="manager@property.com",
    focus="drainage issues and membrane integrity",   # injected into LLM context
    compare_to="last_inspection_2025_09",             # diff against previous
    report_model="gpt-5",                             # which LLM generates report
)


# ── Multi-model ensemble for threat classification ───────────────────────

fleet = cw.fleet("perimeter_alpha", members=["drone-01", "ugv-01"])
fleet.patrol(area="perimeter_zone",
    threat_models=["threat-classifier-rgb-v3", "threat-classifier-thermal-v2", "gpt-5-vision"],
    consensus="majority",                       # 2/3 must agree
    confidence_threshold=0.7,                   # minimum to escalate
    escalate_to="ops_center",
    repeat=True,
)


# ── Skill synthesis from successful executions ───────────────────────────

arm.pick_and_sort(
    from_bin="parts_bin",
    trays={"bolts": "tray_bolts"},
    learn=True,                                 # synthesize reusable skill from successes
    promote_after=10,                           # add to skill library after 10 successes
)


# ═════════════════════════════════════════════════════════════════════════
# SKILL CONFIGURATION — PLATFORM SIDE
#
# The intelligence lives in skill definitions. Mission authors never
# see this. It's configured once by the platform team or auto-discovered
# from the robot's capabilities.
# ═════════════════════════════════════════════════════════════════════════


# Skills are pre-configured per robot type. The system picks the right
# strategy based on the robot's cameras, actuators, and compute budget.

cw.skills.configure("pick",
    # Perception: how the robot sees
    perception_chain=["sam3", "megapose"],       # segment, then estimate pose
    perception_fallback="graspnet",              # if megapose fails, use grasp heuristics

    # Planning: how the robot decides
    planner="llm",                               # LLM selects grasp strategy
    planner_model="gpt-5",
    planner_context=["scene", "robot_state", "skill_library"],

    # Control: how the robot moves
    control_priority=["code_policy", "vla", "diffusion"],   # try in order
    control_fallback_on="assessment_failure",                # switch to next on fail

    # Assessment: how the robot checks its work
    assess_model="gpt-5-vision",
    assess_prompt="Is the object securely grasped? Check for partial grasp or slip.",

    # Retries: how the robot recovers
    max_retries=3,
    retry_strategy="reperceive_and_replan",      # re-scan scene before retry
)

cw.skills.configure("patrol",
    # Perception
    anomaly_detector="anomaly-detector-edge-v1",  # fast edge model
    anomaly_classifier="threat-classifier-v3",    # detailed cloud model

    # Behavior
    on_anomaly="investigate_then_classify",
    consensus_models=3,
    consensus_threshold=0.66,

    # Escalation
    escalation_channels=["mqtt", "sms", "api"],
)

cw.skills.configure("survey_and_report",
    # Capture
    capture_mode="grid",
    overlap=0.7,
    edge_prefilter="anomaly-prefilter-edge",     # tag interesting frames on-device

    # Analysis
    defect_model="roof-defect-segmentation-v3",
    orthomosaic_engine="opendronemap",

    # Reporting
    report_model="gpt-5",
    report_format="pdf",
    report_template="inspection_professional",
)
