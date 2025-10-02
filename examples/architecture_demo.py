#!/usr/bin/env python3
"""
Demo of the new CyberWave SDK architecture with utilities
Shows how backend environment creation with initial assets creates synergies
"""

import asyncio
import sys
import os

# Add SDK to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cyberwave as cw


async def demo_utilities_architecture():
    """Demonstrate the new utilities-based architecture"""
    
    print("🏗️  CyberWave SDK Architecture Demo")
    print("=" * 50)
    
    # Configure SDK
    cw.configure(
        environment=cw.CyberWaveEnvironment.LOCAL,
        auth_trigger=cw.AuthTrigger.AUTO
    )
    
    # Get client for direct utility usage
    from cyberwave.compact_api import _get_client
    client = _get_client()
    
    if not client:
        print("❌ No client available - running in demo mode")
        return
    
    print(f"✅ Client configured for: {client.base_url}")
    print()
    
    # 1. Demonstrate Environment Utils
    print("🌍 1. Environment Utilities Demo")
    print("-" * 30)
    
    env_utils = cw.EnvironmentUtils(client)
    
    try:
        # Create environment with multiple initial assets
        environment_data = await env_utils.create_environment_with_assets(
            name="Multi-Robot Laboratory",
            description="Environment with multiple robotic assets",
            asset_registry_ids=[
                "cyberwave/so101",    # Robotic arm
                "cyberwave/drone",    # Drone (if available)
                "cyberwave/sensor"    # Sensor (if available)
            ],
            settings={
                "physics_enabled": True,
                "gravity": [0, 0, -9.81],
                "lighting": "outdoor"
            }
        )
        
        environment_uuid = environment_data['uuid']
        print(f"✅ Environment created: {environment_uuid}")
        print(f"📦 Initial assets: 3 digital twins added by backend")
        
        # Create public link
        link_data = await env_utils.create_public_environment_link(environment_uuid)
        print(f"🔑 Public token: {link_data['token'][:12]}...")
        
    except Exception as e:
        print(f"⚠️  Environment creation demo: {e}")
        environment_uuid = "demo-env-12345"
    
    print()
    
    # 2. Demonstrate Twin Utils  
    print("🤖 2. Twin Utilities Demo")
    print("-" * 25)
    
    twin_utils = cw.TwinUtils(client)
    
    try:
        # Add additional twin to existing environment
        twin_data = await twin_utils.create_twin_in_environment(
            registry_id="cyberwave/conveyor",
            environment_uuid=environment_uuid,
            name="Production Conveyor",
            position=[5.0, 0.0, 0.0],
            rotation=[0.0, 0.0, 45.0],
            metadata={
                "speed": 2.5,
                "capacity": 100,
                "production_line": "A"
            }
        )
        
        print(f"✅ Additional twin created: {twin_data['name']}")
        print(f"📍 Position: {twin_data['position']}")
        
    except Exception as e:
        print(f"⚠️  Twin creation demo: {e}")
    
    print()
    
    # 3. Demonstrate URL Utils
    print("🔗 3. URL Utilities Demo") 
    print("-" * 23)
    
    frontend_base = cw.URLUtils.get_frontend_base_url(client.base_url)
    print(f"🌐 Frontend base: {frontend_base}")
    
    # Generate URLs with different configurations
    env_url_public = cw.URLUtils.generate_environment_url(
        frontend_base, environment_uuid, "demo-public-token"
    )
    env_url_private = cw.URLUtils.generate_environment_url(
        frontend_base, environment_uuid
    )
    
    print(f"🔓 Public URL:  {env_url_public}")
    print(f"🔒 Private URL: {env_url_private}")
    
    print()
    
    # 4. Demonstrate Compact API Utils (High-level orchestration)
    print("🎯 4. Compact API Utilities Demo")
    print("-" * 33)
    
    compact_utils = cw.CompactAPIUtils(client)
    
    try:
        # Create complete environment with orchestrated setup
        complete_env = await compact_utils.create_complete_environment(
            name="Automated Factory Floor",
            description="Complete robotic production environment",
            asset_registry_ids=[
                "cyberwave/so101",      # Assembly robot
                "cyberwave/agv",        # Automated guided vehicle  
                "cyberwave/quality_scanner"  # Quality control scanner
            ],
            twin_positions={
                "cyberwave/so101": [0.0, 0.0, 0.0],
                "cyberwave/agv": [10.0, 5.0, 0.0], 
                "cyberwave/quality_scanner": [5.0, 10.0, 2.0]
            },
            public_access=True
        )
        
        print(f"✅ Complete environment created!")
        print(f"🏭 Environment: {complete_env['environment']['name']}")
        print(f"🤖 Assets: {complete_env['asset_count']} digital twins")
        print(f"🌐 Public URL: {complete_env['environment_url']}")
        print(f"🔑 Token: {complete_env['public_token'][:12]}...")
        
    except Exception as e:
        print(f"⚠️  Complete environment demo: {e}")
    
    print()
    print("🎉 Architecture Demo Complete!")
    print()
    print("💡 Key Benefits:")
    print("   • Backend handles initial asset placement")
    print("   • Utilities create clean separation of concerns") 
    print("   • Compact API delegates to specialized utils")
    print("   • Easy to extend and maintain")
    print("   • Reusable components across different APIs")


if __name__ == "__main__":
    asyncio.run(demo_utilities_architecture())
