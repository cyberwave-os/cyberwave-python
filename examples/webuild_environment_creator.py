#!/usr/bin/env python3
"""
WeBuild Environment Creator

Creates real environments with actual assets and twins for the WeBuild showcase.
This script sets up environments that can be viewed in the integrated components.
"""

import asyncio
import sys
import os
from typing import Dict, Any, List

# Add SDK to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cyberwave import Client


async def create_webuild_environments(client: Client) -> List[Dict[str, Any]]:
    """Create all WeBuild showcase environments with real assets"""
    
    print("🏗️ Creating WeBuild Showcase Environments")
    print("=" * 50)
    
    # Create or get WeBuild project
    project = await client.projects.get_or_create_by_name(
        name="WeBuild Construction Showcase",
        description="Comprehensive showcase of Cyberwave capabilities for construction industry"
    )
    print(f"✅ Project: {project['name']} ({project['uuid'][:8]}...)")
    
    environments = []
    
    # 1. Control Tower Environment
    print("\n🏢 Creating Control Tower Environment...")
    control_env = await client.environments.create(
        project_uuid=project["uuid"],
        name="WeBuild Safety Control Tower",
        description="Central command hub for construction site safety monitoring",
        settings={
            "environment_type": "control_center",
            "safety_protocols": "construction_site",
            "monitoring_radius": 2000,
            "webuild_showcase": True
        }
    )
    
    # Add control tower assets
    control_handle = await client.environments.get(control_env["uuid"])
    
    # Add security cameras
    camera_positions = [
        {"name": "Main Gate Security", "pos": [20, 0, 8]},
        {"name": "Perimeter North", "pos": [0, 30, 12]},
        {"name": "Work Zone Monitor", "pos": [-15, 15, 10]},
        {"name": "Equipment Yard Cam", "pos": [-25, -10, 8]}
    ]
    
    for i, cam_data in enumerate(camera_positions):
        await control_handle.create_twin(
            name=cam_data["name"],
            asset_registry_id="generic/security_camera",
            position=cam_data["pos"],
            rotation=[1, 0, 0, 0],
            metadata={
                "device_type": "security_camera",
                "ai_analytics": True,
                "webuild_demo": True,
                "monitoring_zone": f"zone_{i+1}"
            }
        )
    
    # Add security drones
    await control_handle.create_twin(
        name="Security Patrol Drone",
        asset_registry_id="generic/security_drone", 
        position=[0, 0, 40],
        rotation=[1, 0, 0, 0],
        metadata={
            "device_type": "security_drone",
            "patrol_active": True,
            "webuild_demo": True
        }
    )
    
    environments.append(control_env)
    print(f"✅ Control Tower Environment: {control_env['uuid'][:8]}...")
    
    # 2. Excavation Environment
    print("\n🚜 Creating Excavation Environment...")
    excavation_env = await client.environments.create(
        project_uuid=project["uuid"],
        name="WeBuild Excavation Operations",
        description="End-to-end excavation process with heavy machinery monitoring",
        settings={
            "environment_type": "active_construction",
            "work_phases": ["excavation", "material_handling"],
            "safety_zones": {"exclusion_radius": 20},
            "webuild_showcase": True
        }
    )
    
    excavation_handle = await client.environments.get(excavation_env["uuid"])
    
    # Add excavators
    excavator_positions = [
        {"name": "CAT-320-Alpha", "pos": [0, 0, 0]},
        {"name": "CAT-320-Beta", "pos": [25, 15, 0]},
        {"name": "CAT-320-Gamma", "pos": [-20, 10, 0]}
    ]
    
    for exc_data in excavator_positions:
        await excavation_handle.create_twin(
            name=exc_data["name"],
            asset_registry_id="caterpillar/320",
            position=exc_data["pos"],
            rotation=[1, 0, 0, 0],
            metadata={
                "device_type": "construction_equipment",
                "operator_certified": True,
                "webuild_demo": True,
                "telemetry_active": True
            }
        )
    
    environments.append(excavation_env)
    print(f"✅ Excavation Environment: {excavation_env['uuid'][:8]}...")
    
    # 3. Computer Vision Environment  
    print("\n👁️ Creating Computer Vision Environment...")
    cv_env = await client.environments.create(
        project_uuid=project["uuid"],
        name="WeBuild AI Vision Analytics",
        description="Computer vision analytics from video capture to AI suggestions",
        settings={
            "environment_type": "ai_analytics_lab",
            "ai_models": ["safety_compliance", "ppe_detection", "behavior_analysis"],
            "real_time_processing": True,
            "webuild_showcase": True
        }
    )
    
    cv_handle = await client.environments.get(cv_env["uuid"])
    
    # Add AI-enabled cameras
    ai_cameras = [
        {"name": "PPE Compliance Camera", "pos": [10, 10, 8], "spec": "ppe_detection"},
        {"name": "Behavior Analysis Camera", "pos": [-10, 15, 10], "spec": "behavior_analysis"},
        {"name": "Equipment Tracking Camera", "pos": [20, -10, 12], "spec": "equipment_tracking"},
        {"name": "Hazard Detection Camera", "pos": [0, 25, 15], "spec": "hazard_detection"}
    ]
    
    for cam_data in ai_cameras:
        await cv_handle.create_twin(
            name=cam_data["name"],
            asset_registry_id="generic/security_camera",
            position=cam_data["pos"],
            rotation=[1, 0, 0, 0],
            metadata={
                "device_type": "security_camera",
                "ai_specialization": cam_data["spec"],
                "ai_model": f"construction_{cam_data['spec']}_v2",
                "webuild_demo": True,
                "real_time_analytics": True
            }
        )
    
    environments.append(cv_env)
    print(f"✅ Computer Vision Environment: {cv_env['uuid'][:8]}...")
    
    # 4. Drone Operations Environment
    print("\n🚁 Creating Drone Operations Environment...")
    drone_env = await client.environments.create(
        project_uuid=project["uuid"],
        name="WeBuild Drone Command & Control",
        description="Digital twin command and control for construction drone operations",
        settings={
            "environment_type": "drone_operations",
            "airspace_management": True,
            "flight_restrictions": {"max_altitude": 120},
            "webuild_showcase": True
        }
    )
    
    drone_handle = await client.environments.get(drone_env["uuid"])
    
    # Add drone fleet
    drone_fleet = [
        {"name": "Site Survey Drone", "pos": [0, 0, 30], "role": "site_survey"},
        {"name": "Security Patrol Drone", "pos": [50, 0, 40], "role": "security_patrol"},
        {"name": "Safety Inspection Drone", "pos": [0, 50, 35], "role": "safety_inspection"}
    ]
    
    for drone_data in drone_fleet:
        await drone_handle.create_twin(
            name=drone_data["name"],
            asset_registry_id="generic/security_drone",
            position=drone_data["pos"],
            rotation=[1, 0, 0, 0],
            metadata={
                "device_type": "security_drone",
                "operational_role": drone_data["role"],
                "flight_status": "ready",
                "webuild_demo": True,
                "autonomous_capable": True
            }
        )
    
    # Add ground control station
    await drone_handle.create_twin(
        name="Ground Control Station",
        asset_registry_id="generic/control_tower",
        position=[0, -30, 0],
        rotation=[1, 0, 0, 0],
        metadata={
            "device_type": "control_infrastructure",
            "station_type": "drone_operations",
            "webuild_demo": True
        }
    )
    
    environments.append(drone_env)
    print(f"✅ Drone Operations Environment: {drone_env['uuid'][:8]}...")
    
    return environments


async def create_webuild_sensors_and_devices(environments: List[Dict[str, Any]], client: Client):
    """Add sensors and devices to environments for enhanced functionality"""
    
    print("\n📡 Adding Sensors and Devices...")
    
    for env in environments:
        env_handle = await client.environments.get(env["uuid"])
        
        # Add sensors based on environment type
        if "control" in env["name"].lower():
            # Add communication and monitoring sensors
            await env_handle.create_sensor(
                name="Emergency Communication Hub",
                sensor_type="communication",
                position=[0, 0, 15],
                metadata={
                    "communication_channels": ["radio", "cellular", "satellite"],
                    "emergency_protocols": True,
                    "webuild_demo": True
                }
            )
            
        elif "excavation" in env["name"].lower():
            # Add safety and monitoring sensors
            await env_handle.create_sensor(
                name="Work Zone Safety Monitor",
                sensor_type="safety_sensor",
                position=[0, 0, 5],
                metadata={
                    "proximity_detection": True,
                    "personnel_tracking": True,
                    "webuild_demo": True
                }
            )
            
        elif "vision" in env["name"].lower():
            # Add edge processing units
            await env_handle.create_sensor(
                name="AI Processing Unit",
                sensor_type="edge_processor",
                position=[0, 0, 2],
                metadata={
                    "ai_acceleration": True,
                    "real_time_inference": True,
                    "webuild_demo": True
                }
            )
    
    print("✅ Sensors and devices added to all environments")


async def main():
    """Main function to create WeBuild showcase environments"""
    
    print("🏗️ WeBuild Cyberwave Environment Creator")
    print("=" * 50)
    print("Creating real environments with actual platform integration")
    print()
    
    # Create client
    client = Client(base_url="http://localhost:8000")
    
    try:
        # Note: Authentication would be required for real usage
        print("⚠️  Authentication required - please ensure you're logged in")
        print("   Use: await client.login('username', 'password')")
        print()
        
        # Create environments
        environments = await create_webuild_environments(client)
        
        # Add enhanced sensors and devices
        await create_webuild_sensors_and_devices(environments, client)
        
        print("\n🎉 WeBuild Showcase Environments Created!")
        print("=" * 50)
        print(f"✅ Created {len(environments)} environments with enhanced features")
        
        print("\n📋 Environment UUIDs (for frontend integration):")
        for env in environments:
            print(f"  • {env['name']}: {env['uuid']}")
        
        print("\n🔗 Next Steps:")
        print("1. Update WEBUILD_ENVIRONMENTS in the frontend page with these UUIDs")
        print("2. Start the frontend: cd cyberwave-frontend && npm run dev")
        print("3. Navigate to: http://localhost:3000/webuild")
        print("4. Interact with the integrated platform features")
        print("5. Replace placeholder GLB models with real construction assets")
        
        print("\n✨ Features Available:")
        print("• Real environment viewer with 3D twins")
        print("• Actual device controls and telemetry")
        print("• Video stream integration for cameras")
        print("• Joint controls for robotic equipment")
        print("• Enhanced construction-specific analytics")
        print("• Emergency protocols and safety monitoring")
        
        return environments
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure backend is running: docker-compose up")
        print("2. Check authentication credentials")
        print("3. Verify database is seeded with base assets")
        raise
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
