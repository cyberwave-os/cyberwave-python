"""
UR7 Santa's Little Helper - MQTT coordinated multi-joint control example

Configuration:
    Required: CYBERWAVE_TWIN_UUID
    Optional: CYBERWAVE_TOKEN, CYBERWAVE_API_KEY, CYBERWAVE_MQTT_HOST, CYBERWAVE_MQTT_PORT
    
    Set via .env file in SDK root or export as environment variables.
"""

import os
import time
import threading
from pathlib import Path
from typing import Dict, Optional

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass

from cyberwave import Cyberwave, SOURCE_TYPE_TELE, SOURCE_TYPE_EDGE

# ============================================================================
# Configuration Constants
# ============================================================================

LOOP_COUNT = 1
POSITION_TOLERANCE = 0.02  # rad
VERIFICATION_TIMEOUT = 15.0  # seconds
VACUUM_ON = 0.5
VACUUM_OFF = -0.5
VACUUM_RELEASE_DELAY = 2.0  # seconds

# Predefined joint positions
ZERO_POSITION = {
    "elbow_joint": -1.5751792192459106,
    "shoulder_lift_joint": -1.5719558201231898,
    "shoulder_pan_joint": 1.562535285949707,
    "wrist_1_joint": -1.5689985008542244,
    "wrist_2_joint": 1.5630512237548828,
    "wrist_3_joint": -0.006603542958394826,
    "ee_fixed_joint": VACUUM_OFF
}

FIRST_POSITION = {
    "elbow_joint": -2.4102394580841064,
    "shoulder_lift_joint": -1.3169731956771393,
    "shoulder_pan_joint": 0.8636429905891418,
    "wrist_1_joint": -0.9836568397334595,
    "wrist_2_joint": 1.5628662109375,
    "wrist_3_joint": -0.7050760428058069,
    "ee_fixed_joint": VACUUM_OFF
}

SECOND_POSITION = {
    "elbow_joint": -2.4945693016052246,
    "shoulder_lift_joint": -1.4424329561046143,
    "shoulder_pan_joint": 0.8638515472412109,
    "wrist_1_joint": -0.7738697093776246,
    "wrist_2_joint": 1.562424898147583,
    "wrist_3_joint": -0.7051599661456507,
    "ee_fixed_joint": VACUUM_OFF
}

THIRD_POSITION = {
    "elbow_joint": -2.4102394580841064,
    "shoulder_lift_joint": -1.3169731956771393,
    "shoulder_pan_joint": 0.8636429905891418,
    "wrist_1_joint": -0.9836568397334595,
    "wrist_2_joint": 1.5628662109375,
    "wrist_3_joint": -0.7050760428058069,
    "ee_fixed_joint": VACUUM_ON
}

FOURTH_POSITION = {
    "elbow_joint": -1.3520451784133911,
    "shoulder_lift_joint": -2.462132593194479,
    "shoulder_pan_joint": -1.293713394795553,
    "wrist_1_joint": -0.8949940961650391,
    "wrist_2_joint": 1.5753843784332275,
    "wrist_3_joint": -2.9583125750171106,
    "ee_fixed_joint": VACUUM_ON,
}

FIFTH_POSITION = {
    "elbow_joint": -1.0901590585708618,
    "shoulder_lift_joint": -3.0497638187804164,
    "shoulder_pan_joint": -1.3203261534320276,
    "wrist_1_joint": -0.5512225192836304,
    "wrist_2_joint": 1.6471755504608154,
    "wrist_3_joint": -2.984924379979269,
    "ee_fixed_joint": VACUUM_ON,
}

SIXTH_POSITION = {
    "elbow_joint": -1.3520451784133911,
    "shoulder_lift_joint": -2.462132593194479,
    "shoulder_pan_joint": -1.293713394795553,
    "wrist_1_joint": -0.8949940961650391,
    "wrist_2_joint": 1.5753843784332275,
    "wrist_3_joint": -2.9583125750171106,
    "ee_fixed_joint": VACUUM_OFF,
}

SEVENTH_POSITION = {
    "elbow_joint": -2.0459234714508057,
    "shoulder_lift_joint": -1.705637594262594,
    "shoulder_pan_joint": 1.238063931465149,
    "wrist_1_joint": -0.9621141713908692,
    "wrist_2_joint": 1.5617938041687012,
    "wrist_3_joint": -0.33134586015810186,
    "ee_fixed_joint": VACUUM_OFF,
}

EIGHTH_POSITION = {
    "elbow_joint": -2.09897518157959,
    "shoulder_lift_joint": -1.7802230320372523,
    "shoulder_pan_joint": 1.2381253242492676,
    "wrist_1_joint": -0.834440605049469,
    "wrist_2_joint": 1.5615777969360352,
    "wrist_3_joint": -0.3314622084247034,
    "ee_fixed_joint": VACUUM_OFF,
}

NINTH_POSITION = {
    "elbow_joint": -2.0580971240997314,
    "shoulder_lift_joint": -1.720102926293844,
    "shoulder_pan_joint": 1.237954020500183,
    "wrist_1_joint": -0.9354360860637208,
    "wrist_2_joint": 1.5617492198944092,
    "wrist_3_joint": -0.7050760428058069,
    "ee_fixed_joint": VACUUM_ON,
}

TENTH_POSITION = {
    "elbow_joint": -1.2298262119293213,
    "shoulder_lift_joint": -2.6546231708922328,
    "shoulder_pan_joint": -1.1697538534747522,
    "wrist_1_joint": -0.7942519348910828,
    "wrist_2_joint": 1.623485803604126,
    "wrist_3_joint": -2.81087834039797,
    "ee_fixed_joint": VACUUM_ON,
}

ELEVENTH_POSITION = {
    "elbow_joint": -1.0116291046142578,
    "shoulder_lift_joint": -3.0800992451109828,
    "shoulder_pan_joint": -1.1684759298907679,
    "wrist_1_joint": -0.5699064296535035,
    "wrist_2_joint": 1.6640836000442505,
    "wrist_3_joint": -2.808795754109518,
    "ee_fixed_joint": VACUUM_ON,
}

TWELFTH_POSITION = {
    "elbow_joint": -1.2298262119293213,
    "shoulder_lift_joint": -2.6546231708922328,
    "shoulder_pan_joint": -1.1697538534747522,
    "wrist_1_joint": -0.7942519348910828,
    "wrist_2_joint": 1.623485803604126,
    "wrist_3_joint": -2.81087834039797,
    "ee_fixed_joint": VACUUM_OFF,
}

THIRTEENTH_POSITION = ZERO_POSITION

POSITIONS = [
    ("zero_position", ZERO_POSITION),
    ("first_position", FIRST_POSITION),
    ("second_position", SECOND_POSITION),
    ("third_position", THIRD_POSITION),
    ("fourth_position", FOURTH_POSITION),
    ("fifth_position", FIFTH_POSITION),
    ("sixth_position", SIXTH_POSITION),
    ("seventh_position", SEVENTH_POSITION),
    ("eighth_position", EIGHTH_POSITION),
    ("ninth_position", NINTH_POSITION),
    ("tenth_position", TENTH_POSITION),
    ("eleventh_position", ELEVENTH_POSITION),
    ("twelfth_position", TWELFTH_POSITION),
    ("thirteenth_position", THIRTEENTH_POSITION)
]


class PositionObserver:
    """Monitor robot position from edge messages"""
    
    def __init__(self, client: Cyberwave, twin_uuid: str):
        self.client = client
        self.twin_uuid = twin_uuid
        self.lock = threading.Lock()
        self.current_positions: Dict[str, float] = {}
        self.last_update_time: Optional[float] = None
        self.target_positions: Optional[Dict[str, float]] = None
        self.position_reached = threading.Event()
        self.subscribed = False
        
    def _on_joint_update(self, data: Dict):
        source_type = data.get("source_type", "unknown")
        
        if source_type != SOURCE_TYPE_EDGE:
            return
        
        positions = data.get("positions", {})
        if not positions:
            return
        
        with self.lock:
            self.current_positions = positions.copy()
            self.last_update_time = time.time()
            
            if self.target_positions is not None:
                if self._positions_match(self.current_positions, self.target_positions):
                    self.position_reached.set()
                else:
                    if self.position_reached.is_set():
                        self.position_reached.clear()
    
    def _positions_match(self, current: Dict[str, float], target: Dict[str, float]) -> bool:
        target_joints = [jn for jn in target.keys() if jn != 'ee_fixed_joint']
        
        if not target_joints:
            return True
        
        for joint_name in target_joints:
            if joint_name not in current:
                return False
        
        for joint_name in target_joints:
            target_pos = target[joint_name]
            current_pos = current[joint_name]
            difference = abs(current_pos - target_pos)
            
            if difference > POSITION_TOLERANCE:
                return False
        
        return True
    
    def subscribe(self):
        if self.subscribed:
            return
        
        topic = f"{self.client.mqtt.topic_prefix}cyberwave/joint/{self.twin_uuid}/update"
        self.client.mqtt.subscribe(topic, self._on_joint_update)
        self.subscribed = True
    
    def wait_for_position(self, target_positions: Dict[str, float], timeout: float = VERIFICATION_TIMEOUT) -> bool:
        with self.lock:
            self.target_positions = target_positions.copy()
            self.position_reached.clear()
            initial_time = time.time()
            initial_positions = self.current_positions.copy() if self.current_positions else {}
            last_check_time = initial_time
            last_check_positions = initial_positions.copy()
        
        with self.lock:
            if self.current_positions and self._positions_match(self.current_positions, target_positions):
                return True
        
        check_interval = 0.1
        elapsed = 0.0
        
        while elapsed < timeout:
            time.sleep(check_interval)
            elapsed += check_interval
            
            with self.lock:
                if self.position_reached.is_set():
                    if self.current_positions and self._positions_match(self.current_positions, target_positions):
                        return True
                    else:
                        self.position_reached.clear()
                
                current_positions = self.current_positions.copy()
                if current_positions:
                    positions_changed = False
                    for joint_name in set(list(last_check_positions.keys()) + list(current_positions.keys())):
                        last_pos = last_check_positions.get(joint_name, 0.0)
                        curr_pos = current_positions.get(joint_name, 0.0)
                        if abs(curr_pos - last_pos) > 0.001:
                            positions_changed = True
                            break
                    
                    if positions_changed:
                        last_check_time = time.time()
                        last_check_positions = current_positions.copy()
                    else:
                        time_since_change = time.time() - last_check_time
                        if time_since_change >= timeout:
                            print("ERROR: Robot not responding")
                            return False
        
        with self.lock:
            if self.current_positions and self._positions_match(self.current_positions, target_positions):
                return True
            
            if not initial_positions or not self.current_positions:
                print("ERROR: No edge position updates received")
                return False
            
            any_movement = False
            for joint_name in set(list(initial_positions.keys()) + list(self.current_positions.keys())):
                initial_pos = initial_positions.get(joint_name, 0.0)
                current_pos = self.current_positions.get(joint_name, 0.0)
                if abs(current_pos - initial_pos) > 0.001:
                    any_movement = True
                    break
            
            if not any_movement:
                print("ERROR: Robot did not move")
                return False
            else:
                return True


def main():
    print("=" * 60)
    print("UR7 Santa's Little Helper - MQTT Motion Control")
    print("=" * 60)
    print()
    
    api_key = os.getenv("CYBERWAVE_API_KEY")
    token = os.getenv("CYBERWAVE_TOKEN")
    mqtt_host = os.getenv("CYBERWAVE_MQTT_HOST", "mqtt.cyberwave.com")
    mqtt_port = int(os.getenv("CYBERWAVE_MQTT_PORT", "1883"))
    twin_uuid = os.getenv("CYBERWAVE_TWIN_UUID")
    
    if not twin_uuid:
        print("ERROR: CYBERWAVE_TWIN_UUID is required")
        return
    
    if not token:
        print("WARNING: CYBERWAVE_TOKEN not set")
    
    print(f"MQTT: {mqtt_host}:{mqtt_port}")
    print(f"Twin: {twin_uuid}")
    print(f"Loops: {LOOP_COUNT}")
    print()
    
    client = Cyberwave(
        api_key=api_key,
        token=token,
        mqtt_host=mqtt_host,
        mqtt_port=mqtt_port,
    )
    
    try:
        client.mqtt.connect()
        time.sleep(0.5)
        
        if not client.mqtt.connected:
            print("ERROR: MQTT connection failed")
            return
        
        print("Connected to MQTT")
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    observer = PositionObserver(client, twin_uuid)
    observer.subscribe()
    
    initial_wait = 0.0
    max_initial_wait = 3.0
    while initial_wait < max_initial_wait:
        time.sleep(0.1)
        initial_wait += 0.1
        with observer.lock:
            if observer.current_positions:
                print(f"Received initial position ({len(observer.current_positions)} joints)")
                break
    
    print()
    print(f"Starting motion sequence ({len(POSITIONS)} positions)...")
    print()
    
    try:
        for loop_iteration in range(LOOP_COUNT):
            print(f"Loop {loop_iteration + 1}/{LOOP_COUNT}")
            
            for idx, (position_name, joint_positions) in enumerate(POSITIONS):
                print(f"  → {position_name}")
                
                client.mqtt.update_joints_state(
                    twin_uuid=twin_uuid,
                    joint_positions=joint_positions,
                    source_type=SOURCE_TYPE_TELE
                )
                
                position_for_verification = {k: v for k, v in joint_positions.items() if k != 'ee_fixed_joint'}
                position_reached = observer.wait_for_position(position_for_verification, timeout=VERIFICATION_TIMEOUT)
                
                if not position_reached:
                    print(f"ERROR: Failed to reach {position_name}")
                    import sys
                    sys.exit(1)
                
                with observer.lock:
                    if not observer._positions_match(observer.current_positions, position_for_verification):
                        print(f"ERROR: Verification failed for {position_name}")
                        import sys
                        sys.exit(1)
                
                current_vacuum = joint_positions.get('ee_fixed_joint', None)
                next_vacuum = None
                if idx + 1 < len(POSITIONS):
                    next_position_name, next_joint_positions = POSITIONS[idx + 1]
                    next_vacuum = next_joint_positions.get('ee_fixed_joint', None)
                
                if current_vacuum is not None and next_vacuum is not None:
                    if current_vacuum >= 0 and next_vacuum < 0:
                        client.mqtt.update_joints_state(
                            twin_uuid=twin_uuid,
                            joint_positions={"ee_fixed_joint": VACUUM_OFF},
                            source_type=SOURCE_TYPE_TELE
                        )
                        time.sleep(VACUUM_RELEASE_DELAY)
            
            print(f"Loop {loop_iteration + 1} completed\n")
            
            if loop_iteration < LOOP_COUNT - 1:
                time.sleep(1.0)
        
        print("=" * 60)
        print("Motion sequence completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    except SystemExit:
        raise
    finally:
        try:
            client.mqtt.disconnect()
        except Exception as e:
            print(f"Disconnect error: {e}")

if __name__ == "__main__":
    main()
