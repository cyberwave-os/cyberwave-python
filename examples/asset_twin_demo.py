"""Minimal Asset registry demo (updated).

This example prints a few registry interactions without requiring async client.
"""

from cyberwave.assets import (
    AssetRegistry,
    AssetFactory,
    DjiTello,
    BostonDynamicsSpot,
    IntelRealSenseD435,
    TrafficCone,
    ArucoMarker,
    ChargingPad,
)


def demo_asset_registry():
    """Demonstrate the asset registry system"""
    print("=== Asset Registry Demo ===\n")
    
    # Method 1: Direct instantiation
    print("1. Direct instantiation:")
    drone = DjiTello(ip="192.168.10.1", name="Training Drone")
    print(f"   Created: {drone.name}")
    print(f"   Type: {drone.__class__.__name__}")
    print(f"   Capabilities: {drone.capabilities}")
    print(f"   Specs: {drone.specs}")
    
    # Method 2: Registry lookup
    print("\n2. Registry lookup:")
    spot_class = AssetRegistry.get("boston-dynamics/spot")
    if spot_class:
        spot = spot_class(hostname="192.168.1.100", name="Inspection Bot")
        print(f"   Created: {spot.name} from registry")
        print(f"   Capabilities: {spot.capabilities}")
    
    # Method 3: Factory creation
    print("\n3. Factory creation:")
    sensor = AssetFactory.create(
        "intel/realsense-d435",
        serial_number="123456",
        name="Front Camera"
    )
    print(f"   Created: {sensor.name}")
    print(f"   Specs: {sensor.specs}")
    
    # List all registered assets
    print("\n4. All registered assets:")
    for asset_id in AssetRegistry.list():
        if "/" in asset_id:  # Only show namespaced ones
            metadata = AssetRegistry.get_metadata(asset_id)
            print(f"   - {asset_id}: {metadata.get('category', 'N/A')}")


def main():
    print("Cyberwave Asset Registry Demo")
    print("============================\n")
    demo_asset_registry()
    print("\nTip: Use the Cyberwave facade (cyberwave.sdk) for Missions/Runs.")


if __name__ == "__main__":
    main()