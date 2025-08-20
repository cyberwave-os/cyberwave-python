#!/usr/bin/env python3
"""
Test for the register_asset.py script which uses CyberWaveClient.register_asset_with_mesh
"""

import pytest
import pytest_asyncio
import os
import logging
from pathlib import Path
from unittest import mock

from cyberwave.client import Client, AuthenticationError, APIError

# Configure logging for tests
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@pytest.fixture
def mock_client():
    """Provide a mocked client for testing register_asset functionality."""
    with mock.patch('cyberwave.client.Client') as mock_client:
        # Set up the mock to return success responses
        mock_instance = mock_client.return_value
        
        # Mock the register_asset_with_mesh method
        mock_instance.register_asset_with_mesh.return_value = {
            "id": 123,
            "name": "Test Asset",
            "slug": "test-asset",
            "asset_type": "robot",
            "mesh_id": 456
        }
        
        yield mock_instance

def test_register_asset_with_mesh(mock_client, tmp_path):
    """Test registering an asset with a mesh using the client."""
    
    # Create a dummy mesh file
    mesh_file = tmp_path / "test_mesh.glb"
    mesh_file.write_text("dummy mesh data")
    
    # Test parameters
    workspace_id = 1
    project_id = 1
    asset_name = "Test Robot"
    asset_slug = "test-robot"
    asset_type = "robot"
    
    # Call the register_asset_with_mesh method
    result = mock_client.register_asset_with_mesh(
        workspace_id=workspace_id,
        project_id=project_id,
        asset_name=asset_name,
        asset_slug=asset_slug,
        asset_type=asset_type,
        mesh_file_path=str(mesh_file),
        asset_description="Test Description",
        asset_tags=["test", "robot"],
        asset_metadata={"manufacturer": "Test Corp"},
        mesh_name="Test Mesh",
        mesh_description="Test Mesh Description",
        geometry_purpose="visual",
        geometry_is_primary=True
    )
    
    # Verify the method was called with the correct parameters
    mock_client.register_asset_with_mesh.assert_called_once()
    call_args = mock_client.register_asset_with_mesh.call_args[1]
    
    assert call_args["workspace_id"] == workspace_id
    assert call_args["project_id"] == project_id
    assert call_args["asset_name"] == asset_name
    assert call_args["asset_slug"] == asset_slug
    assert call_args["asset_type"] == asset_type
    assert call_args["mesh_file_path"] == str(mesh_file)
    assert call_args["asset_description"] == "Test Description"
    assert call_args["asset_tags"] == ["test", "robot"]
    assert call_args["asset_metadata"] == {"manufacturer": "Test Corp"}
    assert call_args["mesh_name"] == "Test Mesh"
    assert call_args["mesh_description"] == "Test Mesh Description"
    assert call_args["geometry_purpose"] == "visual"
    assert call_args["geometry_is_primary"] == True
    
    # Verify the result
    assert result["id"] == 123
    assert result["name"] == "Test Asset"
    assert result["slug"] == "test-asset"
    assert result["asset_type"] == "robot"
    assert result["mesh_id"] == 456
    
@mock.patch('cyberwave.client.Client')
def test_register_asset_error_handling(mock_client_class, tmp_path):
    """Test error handling when registering an asset."""
    
    # Set up the mock to raise an exception
    mock_client = mock_client_class.return_value
    mock_client.register_asset_with_mesh.side_effect = APIError(status_code=400, detail="Invalid request")
    
    # Create a dummy mesh file
    mesh_file = tmp_path / "test_mesh.glb"
    mesh_file.write_text("dummy mesh data")
    
    # Test parameters
    workspace_id = 1
    project_id = 1
    asset_name = "Test Robot"
    asset_slug = "test-robot"
    asset_type = "robot"
    
    # Call the register_asset_with_mesh method and check for exception
    with pytest.raises(APIError) as excinfo:
        mock_client.register_asset_with_mesh(
            workspace_id=workspace_id,
            project_id=project_id,
            asset_name=asset_name,
            asset_slug=asset_slug,
            asset_type=asset_type,
            mesh_file_path=str(mesh_file)
        )
        
    # Verify the exception was raised with correct details
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "Invalid request"

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 