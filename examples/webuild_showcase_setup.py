#!/usr/bin/env python3
"""
WeBuild Showcase Environment Setup

Creates comprehensive digital environments to showcase Cyberwave capabilities for WeBuild:
1. Control Tower for Safety - security monitoring dashboard
2. Digital Environment for excavation process with 3D models
3. Computer Vision use-cases - video capture to AI suggestions
4. Drone Digital Twin command and control
"""

import asyncio
import sys
import os
from typing import Dict, Any, List

# Add SDK to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cyberwave as cw
from cyberwave import Client


async def create_webuild_control_tower_environment(client: Client) -> Dict[str, Any]:
    """Create Control Tower for Safety environment"""
    print("🏗️ Creating WeBuild Control Tower Environment...")
    
    # Create project for WeBuild showcase
    project = await client.projects.get_or_create_by_name(
        name="WeBuild Construction Showcase",
        description="Comprehensive showcase of Cyberwave capabilities for construction industry"
    )
    
    # Create Control Tower environment
    environment = await client.environments.create(
        project_uuid=project["uuid"],
        name="Safety Control Tower",
        description="Central command and control hub for construction site safety monitoring",
        settings={
            "environment_type": "control_center",
            "safety_protocols": "construction_site",
            "monitoring_radius": 2000,  # meters
            "alert_systems": ["visual", "audio", "sms", "email"],
            "emergency_procedures": "webuild_standard"
        }
    )
    
    env_handle = await client.environments.get(environment["uuid"])
    
    # Add Control Tower Hub
    control_tower_twin = await env_handle.create_twin(
        name="Main Control Tower",
        asset_registry_id="generic/control_tower",
        position=[0, 0, 0],
        rotation=[1, 0, 0, 0],
        metadata={
            "role": "command_center",
            "operators": ["safety_officer", "site_manager", "security_chief"],
            "monitoring_zones": ["zone_a", "zone_b", "zone_c", "perimeter"],
            "communication_channels": ["radio", "cellular", "satellite"]
        }
    )
    
    # Add Security Cameras around the perimeter
    camera_positions = [
        {"name": "Main Gate Camera", "pos": [50, 0, 8], "zone": "entrance"},
        {"name": "North Perimeter Camera", "pos": [0, 100, 12], "zone": "perimeter_north"},
        {"name": "South Perimeter Camera", "pos": [0, -100, 12], "zone": "perimeter_south"},
        {"name": "Equipment Yard Camera", "pos": [-80, 0, 10], "zone": "equipment_storage"},
        {"name": "Work Zone Alpha Camera", "pos": [30, 30, 15], "zone": "work_zone_a"},
        {"name": "Work Zone Beta Camera", "pos": [-30, 30, 15], "zone": "work_zone_b"}
    ]
    
    for cam_data in camera_positions:
        camera_twin = await env_handle.create_twin(
            name=cam_data["name"],
            asset_registry_id="generic/ptz_security_camera",
            position=cam_data["pos"],
            rotation=[1, 0, 0, 0],
            metadata={
                "monitoring_zone": cam_data["zone"],
                "ai_analytics_enabled": True,
                "motion_detection": True,
                "night_vision": True,
                "recording_schedule": "24/7",
                "alert_triggers": ["intrusion", "ppe_violation", "unsafe_behavior"]
            }
        )
    
    # Add Security Drones for patrol
    drone_positions = [
        {"name": "Patrol Drone Alpha", "pos": [0, 0, 50], "route": "perimeter_patrol"},
        {"name": "Patrol Drone Beta", "pos": [0, 0, 50], "route": "work_zone_patrol"}
    ]
    
    for drone_data in drone_positions:
        drone_twin = await env_handle.create_twin(
            name=drone_data["name"],
            asset_registry_id="generic/security_drone",
            position=drone_data["pos"],
            rotation=[1, 0, 0, 0],
            metadata={
                "patrol_route": drone_data["route"],
                "thermal_imaging": True,
                "autonomous_patrol": True,
                "emergency_response": True,
                "flight_restrictions": {
                    "max_altitude": 120,
                    "no_fly_zones": ["crane_operation_area"],
                    "weather_limits": {"wind": 12, "rain": False}
                }
            }
        )
    
    print(f"✅ Control Tower Environment created: {environment['uuid']}")
    return environment


async def create_webuild_excavation_environment(client: Client) -> Dict[str, Any]:
    """Create Digital Environment for excavation process"""
    print("🚜 Creating WeBuild Excavation Process Environment...")
    
    # Get the WeBuild project
    projects = await client.projects.list()
    project = next((p for p in projects if "WeBuild" in p.get("name", "")), None)
    
    if not project:
        project = await client.projects.create(
            name="WeBuild Construction Showcase",
            description="Comprehensive showcase of Cyberwave capabilities"
        )
    
    # Create Excavation environment
    environment = await client.environments.create(
        project_uuid=project["uuid"],
        name="Excavation Operations",
        description="End-to-end excavation process with heavy machinery and safety monitoring",
        settings={
            "environment_type": "active_construction",
            "work_phases": ["site_prep", "excavation", "material_handling", "cleanup"],
            "safety_zones": {
                "exclusion_radius": 20,  # meters around equipment
                "personnel_zones": ["safe_observation", "equipment_operation"],
                "emergency_assembly": [100, 100, 0]
            },
            "soil_conditions": "mixed_clay_rock",
            "weather_monitoring": True
        }
    )
    
    env_handle = await client.environments.get(environment["uuid"])
    
    # Add Excavators
    excavator_positions = [
        {"name": "CAT-320-Alpha", "pos": [0, 0, 0], "zone": "primary_dig"},
        {"name": "CAT-320-Beta", "pos": [25, 15, 0], "zone": "secondary_dig"},
        {"name": "CAT-320-Gamma", "pos": [-20, 10, 0], "zone": "material_loading"}
    ]
    
    for exc_data in excavator_positions:
        excavator_twin = await env_handle.create_twin(
            name=exc_data["name"],
            asset_registry_id="caterpillar/320",
            position=exc_data["pos"],
            rotation=[1, 0, 0, 0],
            metadata={
                "work_zone": exc_data["zone"],
                "operator_certified": True,
                "maintenance_status": "operational",
                "fuel_level": 85,
                "operating_hours": 1247,
                "safety_systems": {
                    "proximity_sensors": True,
                    "backup_camera": True,
                    "work_zone_monitoring": True,
                    "emergency_stop": True
                },
                "work_schedule": {
                    "shift": "day_shift",
                    "start_time": "07:00",
                    "end_time": "17:00"
                }
            }
        )
    
    # Add monitoring cameras for excavation
    excavation_cameras = [
        {"name": "Excavation Overview Cam", "pos": [50, 50, 20], "target": "excavation_site"},
        {"name": "Equipment Safety Cam", "pos": [-30, 40, 15], "target": "equipment_operations"},
        {"name": "Material Loading Cam", "pos": [40, -20, 12], "target": "loading_zone"}
    ]
    
    for cam_data in excavation_cameras:
        camera_twin = await env_handle.create_twin(
            name=cam_data["name"],
            asset_registry_id="generic/security_camera",
            position=cam_data["pos"],
            rotation=[1, 0, 0, 0],
            metadata={
                "monitoring_target": cam_data["target"],
                "ai_analytics": {
                    "equipment_tracking": True,
                    "safety_compliance": True,
                    "productivity_analysis": True,
                    "hazard_detection": True
                },
                "recording_quality": "4K",
                "storage_retention": "30_days"
            }
        )
    
    print(f"✅ Excavation Environment created: {environment['uuid']}")
    return environment


async def create_webuild_computer_vision_showcase(client: Client) -> Dict[str, Any]:
    """Create Computer Vision showcase environment"""
    print("👁️ Creating WeBuild Computer Vision Showcase...")
    
    # Get the WeBuild project
    projects = await client.projects.list()
    project = next((p for p in projects if "WeBuild" in p.get("name", "")), None)
    
    # Create Computer Vision environment
    environment = await client.environments.create(
        project_uuid=project["uuid"],
        name="AI Vision Analytics",
        description="Computer vision use-cases from video capture to AI agent suggestions",
        settings={
            "environment_type": "ai_analytics_lab",
            "ai_models": ["yolo_v8", "safety_compliance", "ppe_detection", "behavior_analysis"],
            "processing_pipeline": ["capture", "preprocess", "inference", "post_process", "alert"],
            "real_time_processing": True,
            "edge_computing": True
        }
    )
    
    env_handle = await client.environments.get(environment["uuid"])
    
    # Add AI-enabled cameras with different specializations
    ai_cameras = [
        {
            "name": "PPE Compliance Camera",
            "pos": [10, 10, 8],
            "specialization": "ppe_detection",
            "ai_model": "safety_compliance_v2"
        },
        {
            "name": "Behavior Analysis Camera", 
            "pos": [-10, 15, 10],
            "specialization": "behavior_analysis",
            "ai_model": "construction_behavior_v1"
        },
        {
            "name": "Equipment Tracking Camera",
            "pos": [20, -10, 12],
            "specialization": "equipment_tracking",
            "ai_model": "heavy_machinery_v3"
        },
        {
            "name": "Hazard Detection Camera",
            "pos": [0, 25, 15],
            "specialization": "hazard_detection", 
            "ai_model": "construction_hazards_v2"
        }
    ]
    
    for cam_data in ai_cameras:
        ai_camera_twin = await env_handle.create_twin(
            name=cam_data["name"],
            asset_registry_id="generic/security_camera",
            position=cam_data["pos"],
            rotation=[1, 0, 0, 0],
            metadata={
                "ai_specialization": cam_data["specialization"],
                "ai_model": cam_data["ai_model"],
                "inference_frequency": "real_time",
                "confidence_threshold": 0.85,
                "alert_integration": True,
                "data_pipeline": {
                    "preprocessing": ["noise_reduction", "contrast_enhancement"],
                    "inference": ["object_detection", "classification", "tracking"],
                    "postprocessing": ["alert_generation", "report_creation"]
                },
                "ai_capabilities": {
                    "real_time_inference": True,
                    "edge_processing": True,
                    "cloud_backup": True,
                    "model_updates": "automatic"
                }
            }
        )
    
    print(f"✅ Computer Vision Environment created: {environment['uuid']}")
    return environment


async def create_webuild_drone_command_control(client: Client) -> Dict[str, Any]:
    """Create Drone Digital Twin command and control environment"""
    print("🚁 Creating WeBuild Drone Command & Control...")
    
    # Get the WeBuild project
    projects = await client.projects.list()
    project = next((p for p in projects if "WeBuild" in p.get("name", "")), None)
    
    # Create Drone C2 environment
    environment = await client.environments.create(
        project_uuid=project["uuid"],
        name="Drone Command & Control",
        description="Digital twin command and control for construction site drone operations",
        settings={
            "environment_type": "drone_operations",
            "airspace_management": True,
            "flight_restrictions": {
                "max_altitude": 120,  # meters (legal limit)
                "no_fly_zones": ["crane_operations", "helicopter_landing"],
                "weather_limits": {"wind": 12, "visibility": 1000}
            },
            "mission_types": ["patrol", "inspection", "survey", "emergency_response"],
            "coordination_required": True
        }
    )
    
    env_handle = await client.environments.get(environment["uuid"])
    
    # Add multiple drones with different roles
    drone_fleet = [
        {
            "name": "Site Survey Drone",
            "pos": [0, 0, 30],
            "role": "site_survey",
            "mission": "daily_progress_mapping"
        },
        {
            "name": "Security Patrol Drone",
            "pos": [100, 0, 40], 
            "role": "security_patrol",
            "mission": "perimeter_monitoring"
        },
        {
            "name": "Safety Inspection Drone",
            "pos": [0, 100, 35],
            "role": "safety_inspection",
            "mission": "equipment_inspection"
        },
        {
            "name": "Emergency Response Drone",
            "pos": [50, 50, 25],
            "role": "emergency_response",
            "mission": "standby"
        }
    ]
    
    for drone_data in drone_fleet:
        drone_twin = await env_handle.create_twin(
            name=drone_data["name"],
            asset_registry_id="generic/security_drone",
            position=drone_data["pos"],
            rotation=[1, 0, 0, 0],
            metadata={
                "operational_role": drone_data["role"],
                "current_mission": drone_data["mission"],
                "flight_status": "ready",
                "battery_level": 95,
                "last_maintenance": "2024-01-15",
                "certification": "commercial_operations",
                "pilot_in_command": "drone_operator_001",
                "equipment": {
                    "visual_camera": "4K_stabilized",
                    "thermal_camera": "640x512_FLIR",
                    "lidar": "velodyne_puck_lite",
                    "communication": "encrypted_radio"
                },
                "mission_capabilities": {
                    "autonomous_flight": True,
                    "obstacle_avoidance": True,
                    "return_to_home": True,
                    "emergency_landing": True,
                    "real_time_streaming": True
                },
                "flight_envelope": {
                    "max_altitude": 120,
                    "max_range": 2000,
                    "max_wind_speed": 12,
                    "min_visibility": 1000
                }
            }
        )
    
    # Add Ground Control Station
    gcs_twin = await env_handle.create_twin(
        name="Ground Control Station",
        asset_registry_id="generic/control_tower",
        position=[0, -50, 0],
        rotation=[1, 0, 0, 0],
        metadata={
            "station_type": "drone_operations",
            "operator_stations": 4,
            "simultaneous_drones": 8,
            "communication_range": 5000,
            "backup_systems": ["secondary_radio", "satellite_uplink"],
            "software_systems": [
                "flight_planning",
                "mission_control", 
                "video_analytics",
                "airspace_management"
            ]
        }
    )
    
    print(f"✅ Drone C2 Environment created: {environment['uuid']}")
    return environment


async def setup_mock_telemetry_streams(environments: List[Dict[str, Any]]):
    """Set up mock telemetry streams for all environments"""
    print("📊 Setting up mock telemetry streams...")
    
    # This would typically connect to real telemetry endpoints
    # For the showcase, we'll create mock data generators
    
    mock_telemetry_config = {
        "update_frequency": 5,  # seconds
        "data_retention": "7_days",
        "metrics": {
            "excavators": [
                "engine_rpm", "hydraulic_pressure", "fuel_consumption",
                "operating_temperature", "location_gps", "work_efficiency"
            ],
            "security_cameras": [
                "video_quality", "storage_usage", "network_latency",
                "detection_count", "alert_frequency"
            ],
            "security_drones": [
                "battery_level", "gps_coordinates", "altitude", "speed",
                "camera_status", "mission_progress", "weather_conditions"
            ]
        },
        "alert_thresholds": {
            "excavator_fuel": {"critical": 10, "warning": 25},
            "drone_battery": {"critical": 15, "warning": 30},
            "camera_storage": {"critical": 90, "warning": 75}
        }
    }
    
    print("✅ Mock telemetry configuration prepared")
    return mock_telemetry_config


async def create_gaussian_splatting_environments(client: Client) -> List[Dict[str, Any]]:
    """Create environments with Gaussian Splatting for realistic visualization"""
    print("🌟 Creating Gaussian Splatting Environments...")
    
    # Get the WeBuild project
    projects = await client.projects.list()
    project = next((p for p in projects if "WeBuild" in p.get("name", "")), None)
    
    # Create realistic construction site environment
    realistic_env = await client.environments.create(
        project_uuid=project["uuid"],
        name="Realistic Construction Site",
        description="Photorealistic construction site using Gaussian Splatting technology",
        settings={
            "environment_type": "photorealistic_3d",
            "rendering_method": "gaussian_splatting",
            "data_source": "photogrammetry_scan",
            "update_frequency": "weekly",
            "quality_level": "high_fidelity",
            "lighting": "dynamic_hdri",
            "weather_simulation": True
        }
    )
    
    print(f"✅ Gaussian Splatting Environment created: {realistic_env['uuid']}")
    return [realistic_env]


async def main():
    """Main setup function for WeBuild showcase"""
    print("🏗️ WeBuild Cyberwave Showcase Setup")
    print("=" * 50)
    
    # Configure SDK for local development
    cw.configure(environment=cw.CyberWaveEnvironment.LOCAL)
    
    # Create client
    client = Client(base_url="http://localhost:8000")
    
    try:
        # Authenticate (you'll need to provide credentials)
        print("🔐 Authenticating...")
        # await client.login("your_username", "your_password")
        print("⚠️  Please authenticate manually or provide credentials")
        print("   You can use: await client.login('username', 'password')")
        print()
        
        # Create all showcase environments
        environments = []
        
        # 1. Control Tower for Safety
        control_tower_env = await create_webuild_control_tower_environment(client)
        environments.append(control_tower_env)
        
        # 2. Excavation Process Environment  
        excavation_env = await create_webuild_excavation_environment(client)
        environments.append(excavation_env)
        
        # 3. Computer Vision Showcase
        cv_env = await create_webuild_computer_vision_showcase(client)
        environments.append(cv_env)
        
        # 4. Drone Command & Control
        drone_env = await create_webuild_drone_command_control(client)
        environments.append(drone_env)
        
        # 5. Gaussian Splatting Environments
        gs_envs = await create_gaussian_splatting_environments(client)
        environments.extend(gs_envs)
        
        # Set up telemetry
        telemetry_config = await setup_mock_telemetry_streams(environments)
        
        print("\n🎉 WeBuild Showcase Setup Complete!")
        print("=" * 50)
        print(f"Created {len(environments)} environments:")
        for env in environments:
            print(f"  • {env['name']} ({env['uuid'][:8]}...)")
        
        print("\n🔗 Access URLs:")
        print("Frontend: http://localhost:3000")
        print("Backend API: http://localhost:8000")
        print("\n📋 Next Steps:")
        print("1. Start the frontend and backend services")
        print("2. Navigate to the environments in the web UI")
        print("3. Interact with the digital twins and controls")
        print("4. Replace placeholder GLB models with real assets")
        
        return environments
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        raise
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
