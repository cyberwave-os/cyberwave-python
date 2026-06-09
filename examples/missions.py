"""
Missions (Aspirational API) — declarative mission examples.

NOTE: This file demonstrates a future mission API that is not yet implemented.
These examples illustrate the intended developer experience.

Requirements:
    pip install cyberwave
"""

from cyberwave import Cyberwave

cw = Cyberwave()


# --- Level 1: Verb Calls (one line per mission) ---

# Warehouse pick-and-deliver
fleet = cw.fleet("warehouse_alpha", members=["agv-01", "arm-station-03"])
fleet.pick_and_deliver("SKU-4421", from_loc="shelf_B3", to_loc="packing_station_7")

# Quadruped shelf inspection
spot = cw.twin("spot-01")
spot.inspect_shelves(aisles=["aisle_1", "aisle_2"], report_to="manager@company.com")

# Drone + ground robot patrol
fleet = cw.fleet("perimeter_alpha", members=["drone-recon-01", "ugv-patrol-01"])
fleet.patrol(area="perimeter_zone", escalate_to="ops_center", repeat=True)


# --- Level 2: Step Sequences ---

fleet = cw.fleet("warehouse_alpha", members=["agv-01", "arm-station-03"])

with fleet.mission("pick_inspect_deliver") as m:
    m.navigate("shelf_B3", who="agv-01")
    scan = m.scan("shelf_B3", looking_for="SKU-4421", who="arm-station-03")
    m.pick(scan.target, who="arm-station-03")
    m.inspect(scan.target, check="no visible damage", who="arm-station-03")
    m.place_on("agv_tray", who="arm-station-03")
    m.navigate("packing_station_7", who="agv-01")

m.execute()


# --- Level 3: Overrides (control the AI) ---

arm = cw.twin("ur10e-station-02")
arm.pick_and_sort(
    from_bin="parts_bin",
    trays={"bolts": "tray_bolts", "nuts": "tray_nuts"},
    perception="custom-part-detector-v1",
    repeat_until="bin_empty",
)
