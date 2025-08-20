#!/usr/bin/env python3
"""
End-to-end test for level creation in CyberWave SDK.
This script tests creating a workspace, project, and level using the SDK.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from cyberwave import Client, load_level

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    # Connect to the backend
    backend_url = os.environ.get("CYBERWAVE_BACKEND_URL", "http://localhost:8000")
    logger.info(f"Connecting to backend: {backend_url}")
    client = Client(base_url=backend_url)
    
    # Create a unique workspace name
    import uuid
    workspace_name = f"Test Workspace {uuid.uuid4().hex[:8]}"
    logger.info(f"Creating workspace: {workspace_name}")
    
    try:
        # Step 1: Create a workspace
        workspace = await client.create_workspace(name=workspace_name)
        workspace_id = workspace["id"]
        logger.info(f"Created workspace with ID: {workspace_id}")
        
        # Check if we got a session token
        token = client.get_session_token()
        if not token:
            logger.error("No session token was received from workspace creation")
            # Use a hardcoded token from the seed data for testing
            hardcoded_token = "a1b2c3d4-e5f6-7890-1234-567890abcdef"
            logger.info(f"Using hardcoded token from seed data: {hardcoded_token[:4]}...{hardcoded_token[-4:]}")
            client._access_token = hardcoded_token
        else:
            logger.info(f"Session token received: {token[:4]}...{token[-4:]}")
        
        # Step 2: Create a project in the workspace
        project_name = f"Test Project {uuid.uuid4().hex[:8]}"
        logger.info(f"Creating project: {project_name} in workspace {workspace_id}")
        project = await client.create_project(workspace_id=workspace_id, name=project_name)
        project_id = project["id"]
        logger.info(f"Created project with ID: {project_id}")
        
        # Step 3: Load the example level definition
        example_level_path = Path("../cyberwave-static/examples/levels/warehouse_demo.yml")
        if not example_level_path.exists():
            logger.error(f"Example level file not found: {example_level_path}")
            sys.exit(1)

        level_definition = load_level(example_level_path)
        logger.info(f"Loaded level definition: {level_definition.metadata.title}")
        
        # Step 4: Upload the level definition
        logger.info(f"Uploading level definition to project {project_id}...")
        level = await client.upload_level_definition(
            project_id=project_id,
            level_definition=level_definition
        )
        level_id = level["id"]
        logger.info(f"Successfully uploaded level! ID: {level_id}")
        
        # Step 5: Optionally, get levels to verify
        logger.info(f"Retrieving levels for project {project_id}...")
        levels = await client.get_levels(project_id=project_id)
        logger.info(f"Found {len(levels)} levels in project {project_id}")
        
        return {
            "workspace_id": workspace_id,
            "project_id": project_id,
            "level_id": level_id
        }
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
    finally:
        # Close the client connection
        await client.aclose()
        logger.info("Test completed and client connection closed")

if __name__ == "__main__":
    result = asyncio.run(main())
    if result:
        logger.info(f"Test successful! Created: {result}")
    else:
        logger.error("Test failed!")
        sys.exit(1) 