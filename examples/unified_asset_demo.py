"""Unified Asset System demo (sync)."""
from cyberwave.assets import (
    # Robots
    DjiTello, BostonDynamicsSpot,
    # Static assets
    Box, ArucoMarker, ChargingPad,
    # Registry
    AssetRegistry
)


def main():
    print("=== Cyberwave Unified Asset System Demo ===\n")
    
    # 1. List available assets
    print("Available assets in registry:")
    for info in AssetRegistry.list():
        print(f"  - {info.asset_id}: {info.name} ({info.asset_type})")
    print()
    
    # 2. Create robots using HuggingFace-style instantiation
    print("Creating robots:")
    drone = DjiTello(ip="192.168.10.1")
    print(f"  ✓ Created {drone.name} with capabilities: {drone.capabilities}")
    
    spot = BostonDynamicsSpot()  # Uses default hostname
    print(f"  ✓ Created {spot.name} with capabilities: {spot.capabilities}")
    print()
    
    # 3. Create static assets
    print("Creating static assets:")
    
    # Simple 3D shapes
    box1 = Box(size=1.0, color=[1, 0, 0])  # Red box
    box2 = Box(size=0.5, color=[0, 1, 0])  # Green box
    print(f"  ✓ Created {box1.name} (size={box1.specs['dimensions']['width']}m)")
    print(f"  ✓ Created {box2.name} (size={box2.specs['dimensions']['width']}m)")
    
    # Functional infrastructure
    charging = ChargingPad()
    print(f"  ✓ Created {charging.name} with capabilities: {charging.capabilities}")
    
    # Visual markers
    markers = []
    for i in range(3):
        marker = ArucoMarker(marker_id=i, size=0.2)
        markers.append(marker)
        print(f"  ✓ Created {marker.name} with ID {marker.specs['marker_id']}")
    print()
    
    # 4. Build a scene
    print("Building warehouse scene:")
    
    # Place static objects
    # Note: place_at() is async in some implementations; here we just show intent.
    # Replace with platform-backed placement via Missions/Runs as needed.
    
    for i, marker in enumerate(markers):
        pass
    
    print("  ✓ Placed all static assets")
    
    # Simulate robot behavior
    print("\nSimulating robot behavior:")
    
    # Drone takes off and hovers
    print(f"  - {drone.name} taking off...")
    # Replace with Cyberwave.twins.command() when using platform twins
    print(f"    ✓ {drone.name} is flying at altitude: {drone.altitude}m")
    
    # Check if Spot is on charging pad
    spot_pos = {"x": 5.1, "y": 5.0, "z": 0}
    # Placeholder check
    if abs(spot_pos["x"] - 5.0) < 0.5 and abs(spot_pos["y"] - 5.0) < 0.5:
        print(f"  - {spot.name} is on the charging pad!")
    else:
        print(f"  - {spot.name} is not on the charging pad")
    
    # 5. Show unified nature
    print("\nUnified asset system:")
    all_assets = [drone, spot, box1, box2, charging] + markers
    
    robots = [a for a in all_assets if hasattr(a, 'connected')]
    static = [a for a in all_assets if getattr(a, 'is_static', False)]
    
    print(f"  Total assets: {len(all_assets)}")
    print(f"  - Robots: {len(robots)}")
    print(f"  - Static assets: {len(static)}")
    print(f"  All are instances of BaseAsset!")
    
    # 6. Platform integration (when connected)
    print("\nPlatform integration example:")
    print("  # When connected to Cyberwave platform:")
    print("  await drone.setup_on_platform(client, project_uuid, mode='hybrid')")
    print("  await box1.setup_on_platform(client, project_uuid)  # Always virtual")
    print("  # Everything uses the same API!")


if __name__ == "__main__":
    main()