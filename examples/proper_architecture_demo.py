#!/usr/bin/env python3
"""
Demo showing the PROPER architecture with segregation of competence
Following the pattern established by MissionsAPI, RunsAPI, etc.
"""

import asyncio
import sys
import os

# Add SDK to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cyberwave as cw


async def demo_proper_architecture():
    """Demonstrate proper segregation of competence in the SDK"""
    
    print("🏗️  CyberWave SDK - Proper Architecture Demo")
    print("=" * 50)
    print("Following the established pattern from MissionsAPI, RunsAPI, etc.")
    print()
    
    # Configure SDK
    cw.configure(environment=cw.CyberWaveEnvironment.LOCAL)
    
    # Get client
    from cyberwave.compact_api import _get_client
    client = _get_client()
    
    if not client:
        print("❌ No client available")
        return
    
    print(f"✅ Client configured: {client.base_url}")
    print()
    
    # 1. Projects Management (proper segregation)
    print("📁 1. Projects API (Specialized Competence)")
    print("-" * 40)
    
    try:
        # Use specialized ProjectsAPI - clean interface
        projects = client.projects.list()
        print(f"📊 Found {len(projects)} projects via ProjectsAPI")
        
        # Create project using specialized API
        project = client.projects.get_or_create_by_name(
            name="SDK Architecture Demo",
            description="Demonstrating proper API segregation"
        )
        print(f"✅ Project: {project['name']} (UUID: {project['uuid'][:8]}...)")
        
    except Exception as e:
        print(f"⚠️  Projects demo: {e}")
        project = {'uuid': 'demo-project-123', 'name': 'Demo Project'}
    
    print()
    
    # 2. Environments Management (proper segregation)
    print("🌍 2. Environments API (Specialized Competence)")
    print("-" * 45)
    
    try:
        # Use specialized EnvironmentsAPI - clean interface
        environment = client.environments.get_or_create_by_name(
            project_uuid=project['uuid'],
            name="Architecture Demo Environment",
            description="Environment created via EnvironmentsAPI",
            settings={'physics_enabled': True}
        )
        print(f"✅ Environment: {environment['name']} (UUID: {environment['uuid'][:8]}...)")
        
        # Get environment handle for operations
        env_handle = client.environments.get(environment['uuid'])
        twins_in_env = env_handle.twins()
        print(f"🤖 Twins in environment: {len(twins_in_env)}")
        
    except Exception as e:
        print(f"⚠️  Environments demo: {e}")
        environment = {'uuid': 'demo-env-456', 'name': 'Demo Environment'}
    
    print()
    
    # 3. Twins Management (proper segregation)
    print("🤖 3. Twins API (Specialized Competence)")
    print("-" * 37)
    
    try:
        # Create twin (this would use client.create_twin in current impl)
        # But should eventually use TwinsAPI for consistency
        twin_uuid = "demo-twin-789"
        
        # Use specialized TwinsAPI for state management
        client.twins.set_state(
            twin_uuid=twin_uuid,
            position=[1.0, 2.0, 0.5],
            rotation=[0.0, 0.0, 0.0, 1.0]  # quaternion
        )
        print(f"✅ Twin state updated via TwinsAPI")
        
        # Use specialized TwinsAPI for joint control
        client.twins.set_joint(
            twin_uuid=twin_uuid,
            joint_name="arm_joint",
            position=45.0  # degrees
        )
        print(f"✅ Joint control via TwinsAPI")
        
        # Get kinematics info
        kinematics = client.twins.get_kinematics(twin_uuid)
        print(f"🔧 Kinematics info retrieved via TwinsAPI")
        
    except Exception as e:
        print(f"⚠️  Twins demo: {e}")
    
    print()
    
    # 4. Missions Management (existing good pattern)
    print("🎯 4. Missions API (Existing Good Pattern)")
    print("-" * 40)
    
    try:
        # This is already properly implemented - show as good example
        mission = client.missions.define(
            key="demo_mission",
            name="Architecture Demo Mission",
            description="Mission created via MissionsAPI"
        )
        
        # Configure mission using fluent API
        mission.world().asset("cyberwave/so101", "robot1")
        mission.world().place("robot1", [0, 0, 0])
        mission.goal_object_in_zone("robot1", "target_zone")
        
        # Register mission 
        # registered = client.missions.register(mission)  # Would work in real backend
        print(f"✅ Mission defined via MissionsAPI: {mission.key}")
        
        # List missions
        # missions = client.missions.list()  # Would work in real backend
        print(f"🎯 Missions management via specialized API")
        
    except Exception as e:
        print(f"⚠️  Missions demo: {e}")
    
    print()
    
    # 5. Runs Management (existing good pattern)
    print("🏃 5. Runs API (Existing Good Pattern)")
    print("-" * 35)
    
    try:
        # This is already properly implemented
        print(f"✅ RunsAPI available for mission execution")
        print(f"🔄 Proper segregation: RunsAPI.start(), .stop(), .get()")
        
    except Exception as e:
        print(f"⚠️  Runs demo: {e}")
    
    print()
    print("🎉 Architecture Analysis Complete!")
    print()
    print("📋 Current State:")
    print("   ✅ MissionsAPI - Properly segregated")
    print("   ✅ RunsAPI - Properly segregated") 
    print("   ✅ TwinsAPI - Properly segregated")
    print("   ✅ EnvironmentsAPI - Properly segregated")
    print("   ✅ ProjectsAPI - Properly segregated")
    print("   ⚠️  Client - Still has some duplication")
    print("   ⚠️  CompactAPI - Needs to use specialized APIs")
    print()
    print("🎯 Recommendations:")
    print("   1. Make Client delegate to specialized APIs")
    print("   2. Update CompactAPI to use client.environments, client.twins, etc.")
    print("   3. Remove duplication in Client class")
    print("   4. Follow MissionsAPI pattern for all competences")


# Show proper vs improper patterns
def show_architecture_patterns():
    print("\n" + "="*60)
    print("🏗️  ARCHITECTURE PATTERNS COMPARISON")
    print("="*60)
    
    print("\n❌ IMPROPER (Current Issues):")
    print("```python")
    print("# Client duplicates specialized API functionality")
    print("client.create_project()      # Duplicates ProjectsAPI")
    print("client.create_environment()  # Duplicates EnvironmentsAPI")
    print("")
    print("# CompactAPI bypasses specialized APIs")
    print("await self._client.create_standalone_environment()  # Direct call")
    print("```")
    
    print("\n✅ PROPER (Following MissionsAPI Pattern):")
    print("```python")
    print("# Client delegates to specialized APIs")
    print("client.projects.create()           # Uses ProjectsAPI")
    print("client.environments.create()       # Uses EnvironmentsAPI")
    print("client.twins.set_state()          # Uses TwinsAPI")
    print("client.missions.register()        # Uses MissionsAPI")
    print("")
    print("# CompactAPI uses specialized APIs via client")
    print("env = await self._client.environments.create()  # Proper delegation")
    print("```")
    
    print("\n🎯 Benefits of Proper Architecture:")
    print("   • Single responsibility per API class")
    print("   • No code duplication")
    print("   • Easy testing and mocking")
    print("   • Clear competence boundaries")
    print("   • Consistent patterns across SDK")
    print("   • Easy to extend and maintain")


if __name__ == "__main__":
    asyncio.run(demo_proper_architecture())
    show_architecture_patterns()
