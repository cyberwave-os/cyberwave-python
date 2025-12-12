"""
UR7 Santa's Little Helper - Example using MQTT with velocity and acceleration control

This example demonstrates:
- MQTT connection setup
- Using update_joint_state with position, velocity, and effort
- Safe motion control for UR7 robot
- Looping through predefined joint positions

Configuration:
    You can set environment variables in a .env file in the SDK root directory
    (cyberwave-sdks/cyberwave-python/.env) or export them manually.
    
    Required variables:
        CYBERWAVE_TWIN_UUID - Your robot's twin UUID
    
    Optional variables:
        CYBERWAVE_TOKEN - Bearer token for authentication
        CYBERWAVE_API_KEY - API key (alternative to token)
        CYBERWAVE_MQTT_HOST - MQTT broker host (default: mqtt.cyberwave.com)
        CYBERWAVE_MQTT_PORT - MQTT broker port (default: 1883)
    
    Example .env file:
        CYBERWAVE_TWIN_UUID=your-twin-uuid-here
        CYBERWAVE_TOKEN=your-token-here
        CYBERWAVE_MQTT_HOST=mqtt.cyberwave.com
        CYBERWAVE_MQTT_PORT=1883
    
    Note: If python-dotenv is installed, the .env file will be loaded automatically.
"""

import os
import time
import threading
from pathlib import Path
from typing import Dict, Optional

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load .env file from the SDK root directory (parent of examples/)
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Also try loading from current directory
        load_dotenv()
except ImportError:
    # python-dotenv not installed, skip .env loading
    # Environment variables must be set manually or via export
    pass

from cyberwave import Cyberwave, SOURCE_TYPE_TELE, SOURCE_TYPE_EDGE

# ============================================================================
# Configuration Constants
# ============================================================================

# Number of times to loop through positions (default: 1)
LOOP_COUNT = 1

# Safe motion parameters for UR7 (medium-safe speed)
# These values match the robot_constants in the mqtt_bridge config
SAFE_VELOCITY = 0.5  # rad/s - medium-safe speed
SAFE_EFFORT = 0.0    # Nm - typically not used for position control

# Position verification parameters
POSITION_TOLERANCE = 0.02  # rad - tolerance for position matching (reduced for stricter verification)
VERIFICATION_TIMEOUT = 15.0  # seconds - timeout for position verification (increased for slower robot movement)

# Vacuum control parameters
VACUUM_ON = 0.5   # Value >= 0 activates vacuum
VACUUM_OFF = -0.5  # Value < 0 deactivates vacuum

# Predefined joint positions
# Joint names in order: elbow_joint, shoulder_lift_joint, shoulder_pan_joint,
#                       wrist_1_joint, wrist_2_joint, wrist_3_joint

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
    "elbow_joint": -2.4820642471313477,
    "shoulder_lift_joint": -1.4188166570714493,
    "shoulder_pan_joint": 0.8638309836387634,
    "wrist_1_joint": -0.8100251716426392,
    "wrist_2_joint": 1.5624916553497314,
    "wrist_3_joint": -0.7050936857806605,
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
    "elbow_joint": -1.1822501420974731,
    "shoulder_lift_joint": -2.932572980920309,
    "shoulder_pan_joint": -1.293023411427633,
    "wrist_1_joint": -0.5943545264056702,
    "wrist_2_joint": 1.5755789279937744,
    #"wrist_3_joint": -1.3828824202166956,
    "wrist_3_joint": -2.7355805079089563,
    "ee_fixed_joint": VACUUM_ON,
}

FIFTH_POSITION = {
    "elbow_joint": -1.0535956621170044,
    "shoulder_lift_joint": -3.087613721887106,
    "shoulder_pan_joint": -1.2927468458758753,
    "wrist_1_joint": -0.5679686826518555,
    "wrist_2_joint": 1.5758286714553833,
    #"wrist_3_joint": -1.3828743139850062,
    "wrist_3_joint": -2.7355805079089563,
    "ee_fixed_joint": VACUUM_ON,
}

SIXTH_POSITION = {
    "elbow_joint": -2.0459234714508057,
    "shoulder_lift_joint": -1.705637594262594,
    "shoulder_pan_joint": 1.238063931465149,
    "wrist_1_joint": -0.9621141713908692,
    "wrist_2_joint": 1.5617938041687012,
    "wrist_3_joint": -0.33134586015810186,
    "ee_fixed_joint": VACUUM_OFF,
}

SEVENTH_POSITION = {
    "elbow_joint": -2.0959010124206543,
    "shoulder_lift_joint": -1.7749349079527796,
    "shoulder_pan_joint": 1.2380353212356567,
    "wrist_1_joint": -0.8428585690310975,
    "wrist_2_joint": 1.5616052150726318,
    "wrist_3_joint": -0.3314107100116175,
    "ee_fixed_joint": VACUUM_OFF,
}

EIGHTH_POSITION = {
    "elbow_joint": -2.0459234714508057,
    "shoulder_lift_joint": -1.705637594262594,
    "shoulder_pan_joint": 1.238063931465149,
    "wrist_1_joint": -0.9621141713908692,
    "wrist_2_joint": 1.5617938041687012,
    "wrist_3_joint": -0.33134586015810186,
    "ee_fixed_joint": VACUUM_ON,
}

NINTH_POSITION = {
    "elbow_joint": -1.0238438844680786,
    "shoulder_lift_joint": -2.9887944660582484,
    "shoulder_pan_joint": -1.0157249609576624,
    "wrist_1_joint": -0.7091726821712037,
    "wrist_2_joint": 1.5751378536224365,
    "wrist_3_joint": -2.7355805079089563,
    "ee_fixed_joint": VACUUM_ON,
}


TENTH_POSITION = {
    "elbow_joint": -0.9081596732139587,
    "shoulder_lift_joint": -3.120554586450094,
    "shoulder_pan_joint": -1.0154646078692835,
    "wrist_1_joint": -0.693094329243042,
    "wrist_2_joint": 1.5753759145736694,
    "wrist_3_joint": -2.6834617296801966,
    "ee_fixed_joint": VACUUM_ON,
}

ELEVENTH_POSITION = ZERO_POSITION

# List of positions to cycle through
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
    ("eleventh_position", ELEVENTH_POSITION)
]


class PositionObserver:
    """Observer class to monitor robot position from edge messages"""
    
    def __init__(self, client: Cyberwave, twin_uuid: str):
        self.client = client
        self.twin_uuid = twin_uuid
        self.lock = threading.Lock()
        self.current_positions: Dict[str, float] = {}
        self.last_update_time: Optional[float] = None
        self.target_positions: Optional[Dict[str, float]] = None
        self.position_reached = threading.Event()
        self.subscribed = False
        self.message_count = 0  # Track total messages received
        self.edge_message_count = 0  # Track edge messages received
        
    def _on_joint_update(self, data: Dict):
        """Callback for joint state updates - filters for edge messages"""
        # Debug: Log all received messages to understand format
        source_type = data.get("source_type", "unknown")
        
        # Only process messages with source_type='edge'
        if source_type != SOURCE_TYPE_EDGE:
            # Debug: Log non-edge messages (but don't spam)
            if hasattr(self, '_debug_count'):
                self._debug_count += 1
                if self._debug_count <= 3:  # Log first 3 non-edge messages
                    print(f"  [DEBUG] Received non-edge message: source_type='{source_type}'")
            else:
                self._debug_count = 1
                print(f"  [DEBUG] Received non-edge message: source_type='{source_type}'")
            return
        
        # Extract positions from the message
        positions = data.get("positions", {})
        if not positions:
            print(f"  [DEBUG] Edge message received but no 'positions' field found")
            return
        
        with self.lock:
            self.current_positions = positions.copy()
            self.last_update_time = time.time()
            print(f"  [DEBUG] Edge position update received: {len(positions)} joints")
            
            # Check if the target position is reached (ALL joints must match)
            if self.target_positions is not None:
                if self._positions_match(self.current_positions, self.target_positions):
                    # All joints match - set the event
                    self.position_reached.set()
                    print(f"  [DEBUG] Position match detected - ALL joints verified!")
                else:
                    # Clear event if positions don't match (in case it was set incorrectly)
                    if self.position_reached.is_set():
                        self.position_reached.clear()
    
    def _positions_match(self, current: Dict[str, float], target: Dict[str, float]) -> bool:
        """
        Check if ALL current positions match target within tolerance.
        
        This is strict: ALL joints in target must be present in current and match within tolerance.
        Missing joints or joints outside tolerance will cause this to return False.
        """
        # First, ensure all target joints (except ee_fixed_joint) are present in current
        target_joints = [jn for jn in target.keys() if jn != 'ee_fixed_joint']
        
        if not target_joints:
            # No joints to check (only ee_fixed_joint)
            return True
        
        # Check that all target joints exist in current
        for joint_name in target_joints:
            if joint_name not in current:
                return False
        
        # Now verify ALL joints match within tolerance
        for joint_name in target_joints:
            target_pos = target[joint_name]
            current_pos = current[joint_name]
            difference = abs(current_pos - target_pos)
            
            if difference > POSITION_TOLERANCE:
                # Log which joint doesn't match for debugging
                if hasattr(self, '_last_mismatch_log') and time.time() - self._last_mismatch_log > 1.0:
                    print(f"  [DEBUG] Joint {joint_name} mismatch: current={current_pos:.4f}, target={target_pos:.4f}, diff={difference:.4f} (tolerance={POSITION_TOLERANCE})")
                    self._last_mismatch_log = time.time()
                elif not hasattr(self, '_last_mismatch_log'):
                    print(f"  [DEBUG] Joint {joint_name} mismatch: current={current_pos:.4f}, target={target_pos:.4f}, diff={difference:.4f} (tolerance={POSITION_TOLERANCE})")
                    self._last_mismatch_log = time.time()
                return False
        
        # All joints match!
        return True
    
    def subscribe(self):
        """Subscribe to joint state updates"""
        if self.subscribed:
            return
        
        topic = f"{self.client.mqtt.topic_prefix}cyberwave/joint/{self.twin_uuid}/update"
        self.client.mqtt.subscribe(topic, self._on_joint_update)
        self.subscribed = True
        print(f"  ✓ Subscribed to edge position updates: {topic}")
        print(f"  [DEBUG] Waiting for edge messages (source_type='{SOURCE_TYPE_EDGE}')...")
    
    def wait_for_position(self, target_positions: Dict[str, float], timeout: float = VERIFICATION_TIMEOUT) -> bool:
        """
        Wait for robot to reach target position.
        
        Returns:
            True if position reached, False if timeout or no change detected
        """
        with self.lock:
            self.target_positions = target_positions.copy()
            self.position_reached.clear()
            # Capture initial state when command is sent
            initial_time = time.time()
            initial_positions = self.current_positions.copy() if self.current_positions else {}
            last_check_time = initial_time
            last_check_positions = initial_positions.copy()
        
        print(f"  Waiting for position verification (timeout: {timeout}s, tolerance: {POSITION_TOLERANCE} rad)...")
        
        # Check if robot is at target position (before sending command)
        with self.lock:
            if self.current_positions and self._positions_match(self.current_positions, target_positions):
                print(f"  ✓ Robot already at target position (all joints verified)")
                return True
        
        # Poll for position updates and check for movement
        check_interval = 0.1  # Check every 100ms
        elapsed = 0.0
        
        while elapsed < timeout:
            time.sleep(check_interval)
            elapsed += check_interval
            
            with self.lock:
                # Check if position reached (verify all joints match)
                if self.position_reached.is_set():
                    # Double-check all joints match before confirming
                    if self.current_positions and self._positions_match(self.current_positions, target_positions):
                        print(f"  ✓ Position verified - ALL joints reached target (tolerance: {POSITION_TOLERANCE} rad)")
                        return True
                    else:
                        # Position event was set but joints don't actually match - reset and continue
                        self.position_reached.clear()
                
                # Check if positions have changed (robot is moving)
                current_positions = self.current_positions.copy()
                if current_positions:
                    # Compare with last check positions
                    positions_changed = False
                    for joint_name in set(list(last_check_positions.keys()) + list(current_positions.keys())):
                        last_pos = last_check_positions.get(joint_name, 0.0)
                        curr_pos = current_positions.get(joint_name, 0.0)
                        if abs(curr_pos - last_pos) > 0.001:  # Detect any movement (> 0.001 rad)
                            positions_changed = True
                            break
                    
                    if positions_changed:
                        # Robot is moving, update last check
                        last_check_time = time.time()
                        last_check_positions = current_positions.copy()
                    else:
                        # No change since last check - check how long it's been
                        time_since_change = time.time() - last_check_time
                        if time_since_change >= timeout:
                            # No change for full timeout period
                            print(f"  ✗ ERROR: No position change detected for {timeout}s")
                            print(f"  Robot may not be responding - exiting program")
                            return False
        
        # Timeout reached - do final verification
        with self.lock:
            # Final check: verify ALL joints match target position
            if self.current_positions and self._positions_match(self.current_positions, target_positions):
                print(f"  ✓ Position verified - ALL joints reached target (tolerance: {POSITION_TOLERANCE} rad)")
                return True
            
            # Check if robot was moving at all
            if not initial_positions or not self.current_positions:
                print(f"  ✗ ERROR: No edge position updates received")
                print(f"  [DEBUG] Total messages received: {self.message_count}")
                print(f"  [DEBUG] Edge messages received: {self.edge_message_count}")
                print(f"  [DEBUG] initial_positions: {initial_positions}")
                print(f"  [DEBUG] current_positions: {self.current_positions}")
                print(f"  [DEBUG] last_update_time: {self.last_update_time}")
                if self.message_count == 0:
                    print(f"  [DEBUG] No messages received at all - subscription may not be working")
                elif self.edge_message_count == 0:
                    print(f"  [DEBUG] Messages received but none with source_type='edge'")
                return False
            
            # Check if any movement occurred
            any_movement = False
            for joint_name in set(list(initial_positions.keys()) + list(self.current_positions.keys())):
                initial_pos = initial_positions.get(joint_name, 0.0)
                current_pos = self.current_positions.get(joint_name, 0.0)
                if abs(current_pos - initial_pos) > 0.001:
                    any_movement = True
                    break
            
            if not any_movement:
                print(f"  ✗ ERROR: No position change detected for {timeout}s")
                print(f"  Robot did not move - exiting program")
                return False
            else:
                print(f"  ⚠ Warning: Position not reached within tolerance after {timeout}s")
                print(f"  Robot moved but may not have reached target - continuing")
                return True  # Continue if robot moved, even if not at exact target


def main():
    """Main function to run the UR7 motion example"""
    
    print("=" * 60)
    print("UR7 Santa's Little Helper - MQTT Motion Control Example")
    print("=" * 60)
    print()
    
    # Get configuration from environment variables
    api_key = os.getenv("CYBERWAVE_API_KEY")
    token = os.getenv("CYBERWAVE_TOKEN")
    mqtt_host = os.getenv("CYBERWAVE_MQTT_HOST", "mqtt.cyberwave.com")
    mqtt_port = int(os.getenv("CYBERWAVE_MQTT_PORT", "1883"))
    twin_uuid = os.getenv("CYBERWAVE_TWIN_UUID")
    
    if not twin_uuid:
        print("ERROR: CYBERWAVE_TWIN_UUID environment variable is required")
        print("Please set it to your robot's twin UUID")
        return
    
    if not token:
        print("WARNING: CYBERWAVE_TOKEN not set, MQTT connection may fail")
    
    print("Configuration:")
    print(f"  MQTT Host: {mqtt_host}")
    print(f"  MQTT Port: {mqtt_port}")
    print(f"  Twin UUID: {twin_uuid}")
    print(f"  Loop Count: {LOOP_COUNT}")
    print(f"  Safe Velocity: {SAFE_VELOCITY} rad/s")
    print(f"  Safe Effort: {SAFE_EFFORT} Nm")
    print()
    
    # Initialize Cyberwave client
    print("Initializing Cyberwave client...")
    client = Cyberwave(
        api_key=api_key,
        token=token,
        mqtt_host=mqtt_host,
        mqtt_port=mqtt_port,
    )
    
    # Connect to MQTT
    print("Connecting to MQTT broker...")
    try:
        client.mqtt.connect()
        
        # Wait a moment for connection to establish
        time.sleep(0.5)
        
        if not client.mqtt.connected:
            print("ERROR: Failed to connect to MQTT broker")
            return
        
        print("Connected to MQTT broker")
        print()
    except Exception as e:
        print(f"ERROR: Failed to connect to MQTT: {e}")
        return
    
    # Initialize position observer
    print("Initializing position observer...")
    observer = PositionObserver(client, twin_uuid)
    observer.subscribe()
    
    # Give subscription time to establish and wait for initial edge message
    print("  Waiting for initial edge position update...")
    initial_wait = 0.0
    max_initial_wait = 3.0
    while initial_wait < max_initial_wait:
        time.sleep(0.1)
        initial_wait += 0.1
        with observer.lock:
            if observer.current_positions:
                joint_names = list(observer.current_positions.keys())[:3]
                print(f"  ✓ Received initial edge position update ({len(observer.current_positions)} joints: {joint_names}...)")
                break
        # Show progress every second
        if int(initial_wait) != int(initial_wait - 0.1):
            print(f"  [DEBUG] Still waiting for edge messages... ({initial_wait:.1f}s)")
    else:
        print(f"  ⚠ WARNING: No initial edge position update received after {max_initial_wait}s")
        print(f"  [DEBUG] Total messages received: {observer.message_count}")
        print(f"  [DEBUG] Edge messages received: {observer.edge_message_count}")
        print(f"  [DEBUG] This may be normal if robot hasn't published edge updates yet")
        print(f"  [DEBUG] Will continue anyway and check during position verification")
    print()
    
    # Execute motion sequence
    print("Starting motion sequence...")
    print(f"Will loop through {len(POSITIONS)} positions, {LOOP_COUNT} time(s)")
    print()
    
    try:
        for loop_iteration in range(LOOP_COUNT):
            print(f"--- Loop Iteration {loop_iteration + 1}/{LOOP_COUNT} ---")
            
            for position_name, joint_positions in POSITIONS:
                print(f"\nMoving to {position_name}...")
                
                # Send all joints at once using multi-joint format
                # This creates a single trajectory instead of multiple conflicting ones
                # Format: {"joint_name_1": position1, "joint_name_2": position2, ...}
                # The bridge recognizes this format and creates one trajectory with all joints
                multi_joint_message = {}
                for joint_name, position in joint_positions.items():
                    if joint_name == 'ee_fixed_joint':
                        # Special handling for vacuum control
                        vacuum_status = "ON" if position >= 0 else "OFF"
                        print(f"  Setting {joint_name} to {position:.4f} (vacuum {vacuum_status})")
                    else:
                        print(f"  Setting {joint_name} to {position:.4f} rad (velocity: {SAFE_VELOCITY} rad/s)")
                    multi_joint_message[joint_name] = position
                
                # Publish all joints in one message using multi-joint format
                # This avoids the issue where single-joint updates create multiple conflicting trajectories
                topic = f"{client.mqtt.topic_prefix}cyberwave/joint/{twin_uuid}/update"
                message = {
                    "source_type": SOURCE_TYPE_TELE,
                    **multi_joint_message  # Joint positions as keys (e.g., {"elbow_joint": 1.5, ...})
                }
                client.mqtt.publish(topic, message)
                
                print(f"✓ {position_name} command sent (all {len(joint_positions)} joints in one message)")
                
                # Wait for robot to reach target position
                # Filter out ee_fixed_joint for position verification (it's for tool control, not position)
                position_for_verification = {k: v for k, v in joint_positions.items() if k != 'ee_fixed_joint'}
                
                print(f"  Verifying ALL {len(position_for_verification)} joints reach target (tolerance: {POSITION_TOLERANCE} rad)...")
                position_reached = observer.wait_for_position(position_for_verification, timeout=VERIFICATION_TIMEOUT)
                
                if not position_reached:
                    print(f"\n✗ ERROR: Robot did not reach {position_name}")
                    print(f"  Not all joints matched target position within tolerance ({POSITION_TOLERANCE} rad)")
                    print("  Exiting program due to position verification failure")
                    import sys
                    sys.exit(1)
                
                # Final verification before proceeding to next position
                with observer.lock:
                    if not observer._positions_match(observer.current_positions, position_for_verification):
                        print(f"\n✗ ERROR: Final verification failed for {position_name}")
                        print("  Joints did not match after wait - exiting program")
                        import sys
                        sys.exit(1)
                
                print(f"  ✓ ALL joints verified for {position_name} - proceeding to next position")
            
            print(f"\n✓ Loop {loop_iteration + 1} completed")
            
            # Small pause between loops
            if loop_iteration < LOOP_COUNT - 1:
                print("Pausing before next loop...")
                time.sleep(1.0)
        
        print()
        print("=" * 60)
        print("Motion sequence completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nMotion sequence interrupted by user")
    except Exception as e:
        print(f"\nERROR during motion sequence: {e}")
        import traceback
        traceback.print_exc()
    except SystemExit:
        # Position verification timeout - exit gracefully
        print("\nExiting due to position verification timeout")
        raise
    finally:
        # Cleanup
        print("\nDisconnecting from MQTT...")
        try:
            client.mqtt.disconnect()
            print("✓ Disconnected")
        except Exception as e:
            print(f"Error during disconnect: {e}")

if __name__ == "__main__":
    main()
