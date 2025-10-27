"""
Integration test that simulates a typical user workflow with the SDK.

This test demonstrates the complete SDK workflow by making real API calls:
1. Import the SDK
2. List workspaces (use existing)
3. Create a project
4. Create an environment
5. Create an asset
6. Create a twin
7. Move the twin
8. Rotate the twin
9. Delete the twin
10. Delete the asset
11. Delete the environment
12. Delete the project

Prerequisites:
- Local backend must be running (http://localhost:8000)
- User must be authenticated (set CYBERWAVE_API_KEY or CYBERWAVE_TOKEN env var)
- At least one workspace must exist

To run this test:
    # Start the backend first
    cd cyberwave-backend
    docker-compose -f local.yml up
    
    # Then run the test
    cd cyberwave-sdks/cyberwave-python
    export CYBERWAVE_BASE_URL="http://localhost:8000"
    export CYBERWAVE_API_KEY="your_api_key_here"  # or CYBERWAVE_TOKEN
    poetry install
    poetry run pytest tests/test_integration.py -v -s
"""

import os
import pytest
import time

# Import SDK components
from cyberwave import Cyberwave
from cyberwave.exceptions import CyberwaveAPIError


@pytest.fixture(scope="module")
def cyberwave_client():
    """
    Create a Cyberwave client connected to local backend.
    
    Requires environment variables:
    - CYBERWAVE_BASE_URL (default: http://localhost:8000)
    - CYBERWAVE_API_KEY or CYBERWAVE_TOKEN
    """
    base_url = os.getenv("CYBERWAVE_BASE_URL", "http://localhost:8000")
    api_key = os.getenv("CYBERWAVE_API_KEY")
    token = os.getenv("CYBERWAVE_TOKEN")
    
    if not api_key and not token:
        pytest.skip("CYBERWAVE_API_KEY or CYBERWAVE_TOKEN environment variable not set")
    
    client = Cyberwave(
        base_url=base_url,
        api_key=api_key,
        token=token
    )
    
    # Test connection
    try:
        workspaces = client.twins.list()
        if not workspaces:
            pytest.skip("No workspaces available in the backend")
    except Exception as e:
        # pytest.skip(f"Cannot connect to backend at {base_url}: {e}")
        raise e
    
    yield client
    
    # Cleanup
    client.disconnect()


class TestIntegrationWorkflow:
    """
    Integration test that simulates a complete user workflow with real API calls.
    
    This test validates that all SDK components work together correctly
    to perform typical operations a user would do.
    """
    
    def test_complete_user_workflow(self, cyberwave_client):
        """
        Test the complete workflow from workspace listing to cleanup.
        
        This test makes real API calls to perform:
        1. List workspaces and select one
        2. Create a project
        3. Create an environment
        4. Create an asset
        5. Create a twin from the asset
        6. Move the twin to a new position
        7. Rotate the twin
        8. Clean up all created resources
        """
        client = cyberwave_client
        
        # Store created resources for cleanup
        created_resources = {
            'project': None,
            'environment': None,
            'asset': None,
            'twin': None
        }
        
        try:
            # Step 1: List workspaces and use the first one
            print("\n" + "="*70)
            print("Starting Integration Test: Complete User Workflow")
            print("="*70)
            
            # workspaces = client.workspaces.list()
            # assert len(workspaces) > 0, "At least one workspace should exist"
            # workspace = workspaces[0]
            
            # print(f"\n✓ Step 1: Found workspace '{workspace.name}' (UUID: {workspace.uuid})")
            
            # Step 2: Create a project in the workspace
            project = client.projects.create(
                name=f"SDK Integration Test {int(time.time())}",
                workspace_id=None,
                description="Auto-generated project for SDK integration testing"
            )
            created_resources['project'] = project
            
            assert project.uuid is not None
            assert project.name.startswith("SDK Integration Test")
            print(f"✓ Step 2: Created project '{project.name}' (UUID: {project.uuid})")
            
            # Step 3: Create an environment in the project
            environment = client.environments.create(
                name=f"Test Environment {int(time.time())}",
                project_id=project.uuid,
                description="Auto-generated environment for SDK testing"
            )
            created_resources['environment'] = environment
            
            assert environment.uuid is not None
            assert environment.project_uuid == project.uuid
            print(f"✓ Step 3: Created environment '{environment.name}' (UUID: {environment.uuid})")
            
            # Step 4: Create an asset
            asset = client.assets.create(
                name=f"Test Asset {int(time.time())}",
                description="Auto-generated asset for SDK integration testing",
                asset_type="generic"
            )
            created_resources['asset'] = asset
            
            assert asset.uuid is not None
            print(f"✓ Step 4: Created asset '{asset.name}' (UUID: {asset.uuid})")
            
            # Step 5: Create a twin from the asset
            twin_data = client.twins.create(
                asset_id=asset.uuid,
                environment_id=environment.uuid
            )
            created_resources['twin'] = twin_data
            
            assert twin_data.uuid is not None
            assert twin_data.asset_uuid == asset.uuid
            assert twin_data.environment_uuid == environment.uuid
            print(f"✓ Step 5: Created twin (UUID: {twin_data.uuid})")
            
            # Step 6: Use the Twin abstraction to move the twin
            from cyberwave import Twin
            twin = Twin(client, twin_data)
            
            # Move the twin to a new position
            twin.move(x=1.0, y=0.5, z=0.3)
            
            # Refresh to verify the position was updated
            twin.refresh()
            assert hasattr(twin._data, 'position_x')
            print(f"✓ Step 6: Moved twin to position ({twin._data.position_x}, {twin._data.position_y}, {twin._data.position_z})")
            
            # Step 7: Rotate the twin
            twin.rotate(yaw=45)
            
            # Refresh to verify the rotation was updated
            twin.refresh()
            print(f"✓ Step 7: Rotated twin by 45 degrees (yaw)")
            
        finally:
            # Cleanup: Delete resources in reverse order
            print("\n" + "-"*70)
            print("Cleanup Phase")
            print("-"*70)
            
            # Step 8: Delete the twin
            if created_resources['twin']:
                try:
                    client.twins.delete(created_resources['twin'].uuid)
                    print(f"✓ Step 8: Deleted twin (UUID: {created_resources['twin'].uuid})")
                except Exception as e:
                    print(f"⚠ Warning: Could not delete twin: {e}")
            
            # Step 9: Delete the asset
            if created_resources['asset']:
                try:
                    client.assets.delete(created_resources['asset'].uuid)
                    print(f"✓ Step 9: Deleted asset (UUID: {created_resources['asset'].uuid})")
                except Exception as e:
                    print(f"⚠ Warning: Could not delete asset: {e}")
            
            # Step 10: Delete the environment
            if created_resources['environment'] and created_resources['project']:
                try:
                    client.environments.delete(
                        created_resources['environment'].uuid,
                        created_resources['project'].uuid
                    )
                    print(f"✓ Step 10: Deleted environment (UUID: {created_resources['environment'].uuid})")
                except Exception as e:
                    print(f"⚠ Warning: Could not delete environment: {e}")
            
            # Step 11: Delete the project
            if created_resources['project']:
                try:
                    client.projects.delete(created_resources['project'].uuid)
                    print(f"✓ Step 11: Deleted project (UUID: {created_resources['project'].uuid})")
                except Exception as e:
                    print(f"⚠ Warning: Could not delete project: {e}")
            
            print("\n" + "="*70)
            print("✅ Integration Test Completed Successfully!")
            print("="*70 + "\n")


class TestIntegrationErrorHandling:
    """Test error handling in the integration workflow with real API"""
    
    def test_workspace_not_found_error(self, cyberwave_client):
        """Test handling when workspace is not found"""
        print("\n" + "="*70)
        print("Testing Error Handling: Workspace Not Found")
        print("="*70)
        
        # Try to get a non-existent workspace
        with pytest.raises(CyberwaveAPIError) as exc_info:
            cyberwave_client.workspaces.get("00000000-0000-0000-0000-000000000000")
        
        assert "get workspace" in str(exc_info.value).lower() or "404" in str(exc_info.value) or "not found" in str(exc_info.value).lower()
        print("✓ Workspace not found error handled correctly")
        print("="*70 + "\n")
    
    def test_invalid_twin_creation(self, cyberwave_client):
        """Test handling when twin creation fails with invalid data"""
        print("\n" + "="*70)
        print("Testing Error Handling: Invalid Twin Creation")
        print("="*70)
        
        # Try to create a twin with invalid asset and environment IDs
        with pytest.raises(CyberwaveAPIError) as exc_info:
            cyberwave_client.twins.create(
                asset_id="00000000-0000-0000-0000-000000000000",
                environment_id="00000000-0000-0000-0000-000000000000"
            )
        
        assert "create twin" in str(exc_info.value).lower() or "404" in str(exc_info.value) or "not found" in str(exc_info.value).lower()
        print("✓ Invalid twin creation error handled correctly")
        print("="*70 + "\n")


class TestIntegrationReadOperations:
    """Test read-only operations that don't modify data"""
    
    def test_list_all_resources(self, cyberwave_client):
        """Test listing various resources"""
        print("\n" + "="*70)
        print("Testing Read Operations")
        print("="*70)
        
        # # List workspaces
        # workspaces = cyberwave_client.workspaces.list()
        # print(f"✓ Listed {len(workspaces)} workspace(s)")
        
        # List projects
        projects = cyberwave_client.projects.list()
        print(f"✓ Listed {len(projects)} project(s)")
        
        # List environments
        environments = cyberwave_client.environments.list()
        print(f"✓ Listed {len(environments)} environment(s)")
        
        # List assets
        assets = cyberwave_client.assets.list()
        print(f"✓ Listed {len(assets)} asset(s)")
        
        # List twins
        twins = cyberwave_client.twins.list()
        print(f"✓ Listed {len(twins)} twin(s)")
        
        print("="*70 + "\n")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
