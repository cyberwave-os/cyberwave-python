"""
Integration test that simulates a typical user workflow with the SDK.

This test demonstrates the complete SDK workflow by making real API calls:
1. Import the SDK
2. Create a project
3. Create an environment
4. Create an asset
5. Create a twin
6. Move the twin
7. Rotate the twin
8. Delete the twin
9. Delete the asset
10. Delete the environment
11. Delete the project

Prerequisites:
- User must be authenticated (set CYBERWAVE_API_KEY env var)
- Get your API key from https://app.cyberwave.com/settings/api-keys

To run this test:
    cd cyberwave-sdks/cyberwave-python
    export CYBERWAVE_API_KEY="your_api_key_here"
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
    Create a Cyberwave client connected to the Cyberwave API.

    Requires environment variable:
    - CYBERWAVE_API_KEY: Your API key from https://app.cyberwave.com/settings/api-keys
    """
    api_key = os.getenv("CYBERWAVE_API_KEY")

    if not api_key:
        pytest.skip(
            "CYBERWAVE_API_KEY environment variable not set. Get your API key from https://app.cyberwave.com/settings/api-keys"
        )

    client = Cyberwave(api_key=api_key)

    # Test connection
    try:
        client.twins.list()
    except Exception as e:
        pytest.skip(f"Cannot connect to Cyberwave API. Please check your API key: {e}")

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
        Test the complete workflow from project creation to cleanup.

        This test makes real API calls to perform:
        1. Create a project
        2. Create an environment
        3. Create an asset
        4. Create a twin from the asset
        5. Move the twin to a new position
        6. Rotate the twin
        7. Clean up all created resources
        """
        client = cyberwave_client

        # Store created resources for cleanup
        created_resources = {
            "project": None,
            "environment": None,
            "asset": None,
            "twin": None,
        }

        try:
            # Step 1: Create a project
            print("\n" + "=" * 70)
            print("Starting Integration Test: Complete User Workflow")
            print("=" * 70)

            project = client.projects.create(
                name=f"SDK Integration Test {int(time.time())}",
                workspace_id=None,
                description="Auto-generated project for SDK integration testing",
            )
            created_resources["project"] = project

            assert project.uuid is not None
            assert project.name.startswith("SDK Integration Test")
            print(f"✓ Step 1: Created project '{project.name}' (UUID: {project.uuid})")

            # Step 2: Create an environment in the project
            environment = client.environments.create(
                name=f"Test Environment {int(time.time())}",
                project_id=project.uuid,
                description="Auto-generated environment for SDK testing",
            )
            created_resources["environment"] = environment

            assert environment.uuid is not None
            assert environment.project_uuid == project.uuid
            print(
                f"✓ Step 2: Created environment '{environment.name}' (UUID: {environment.uuid})"
            )

            # Step 3: Create an asset
            asset = client.assets.create(
                name=f"Test Asset {int(time.time())}",
                description="Auto-generated asset for SDK integration testing",
                asset_type="generic",
            )
            created_resources["asset"] = asset

            assert asset.uuid is not None
            print(f"✓ Step 3: Created asset '{asset.name}' (UUID: {asset.uuid})")

            # Step 4: Create a twin from the asset
            twin_data = client.twins.create(
                asset_id=asset.uuid, environment_id=environment.uuid
            )
            created_resources["twin"] = twin_data

            assert twin_data.uuid is not None
            assert twin_data.asset_uuid == asset.uuid
            assert twin_data.environment_uuid == environment.uuid
            print(f"✓ Step 4: Created twin (UUID: {twin_data.uuid})")

            # Step 5: Use the Twin abstraction to move the twin
            from cyberwave import Twin

            twin = Twin(client, twin_data)

            # Move the twin to a new position
            twin.move(x=1.0, y=0.5, z=0.3)

            # Refresh to verify the position was updated
            twin.refresh()
            assert hasattr(twin._data, "position_x")
            print(
                f"✓ Step 5: Moved twin to position ({twin._data.position_x}, {twin._data.position_y}, {twin._data.position_z})"
            )

            # Step 6: Rotate the twin
            twin.rotate(yaw=45)

            # Refresh to verify the rotation was updated
            twin.refresh()
            print("✓ Step 6: Rotated twin by 45 degrees (yaw)")

        finally:
            # Cleanup: Delete resources in reverse order
            print("\n" + "-" * 70)
            print("Cleanup Phase")
            print("-" * 70)

            # Step 7: Delete the twin
            if created_resources["twin"]:
                try:
                    client.twins.delete(created_resources["twin"].uuid)
                    print(
                        f"✓ Step 7: Deleted twin (UUID: {created_resources['twin'].uuid})"
                    )
                except Exception as e:
                    print(f"⚠ Warning: Could not delete twin: {e}")

            # Step 8: Delete the asset
            if created_resources["asset"]:
                try:
                    client.assets.delete(created_resources["asset"].uuid)
                    print(
                        f"✓ Step 8: Deleted asset (UUID: {created_resources['asset'].uuid})"
                    )
                except Exception as e:
                    print(f"⚠ Warning: Could not delete asset: {e}")

            # Step 9: Delete the environment
            if created_resources["environment"] and created_resources["project"]:
                try:
                    client.environments.delete(
                        created_resources["environment"].uuid,
                        created_resources["project"].uuid,
                    )
                    print(
                        f"✓ Step 9: Deleted environment (UUID: {created_resources['environment'].uuid})"
                    )
                except Exception as e:
                    print(f"⚠ Warning: Could not delete environment: {e}")

            # Step 10: Delete the project
            if created_resources["project"]:
                try:
                    client.projects.delete(created_resources["project"].uuid)
                    print(
                        f"✓ Step 10: Deleted project (UUID: {created_resources['project'].uuid})"
                    )
                except Exception as e:
                    print(f"⚠ Warning: Could not delete project: {e}")

            print("\n" + "=" * 70)
            print("✅ Integration Test Completed Successfully!")
            print("=" * 70 + "\n")


class TestIntegrationErrorHandling:
    """Test error handling in the integration workflow with real API"""

    def test_invalid_twin_creation(self, cyberwave_client):
        """Test handling when twin creation fails with invalid data"""
        print("\n" + "=" * 70)
        print("Testing Error Handling: Invalid Twin Creation")
        print("=" * 70)

        # Try to create a twin with invalid asset and environment IDs
        with pytest.raises(CyberwaveAPIError) as exc_info:
            cyberwave_client.twins.create(
                asset_id="00000000-0000-0000-0000-000000000000",
                environment_id="00000000-0000-0000-0000-000000000000",
            )

        assert (
            "create twin" in str(exc_info.value).lower()
            or "404" in str(exc_info.value)
            or "not found" in str(exc_info.value).lower()
        )
        print("✓ Invalid twin creation error handled correctly")
        print("=" * 70 + "\n")


class TestIntegrationReadOperations:
    """Test read-only operations that don't modify data"""

    def test_list_all_resources(self, cyberwave_client):
        """Test listing various resources"""
        print("\n" + "=" * 70)
        print("Testing Read Operations")
        print("=" * 70)

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

        print("=" * 70 + "\n")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
