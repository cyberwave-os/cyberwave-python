#!/usr/bin/env python3
"""
Deprecated test for legacy 'level' APIs. Environments replace levels.
"""

import asyncio
import logging
import os
from pathlib import Path

from cyberwave import Client

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    # Connect to the backend
    client = Client(use_token_cache=False)
    
    # Create a workspace
    workspace_name = "Test Workspace"
    logger.info(f"Creating workspace: {workspace_name}")
    workspace = await client.create_workspace(name=workspace_name)
    workspace_id = workspace["id"]
    logger.info(f"Created workspace with ID: {workspace_id}")
    
    # Create a project
    project_name = "Test Project"
    try:
        logger.info(f"Creating project: {project_name}")
        project = await client.create_project(workspace_id=workspace_id, name=project_name)
        project_id = project["id"]
        logger.info(f"Created project with ID: {project_id}")
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        logger.info("Proceeding with test anyway...")
        # For testing, use a hardcoded project ID if the above fails
        project_id = 1
    
    logger.warning("Level tests deprecated. Use environment APIs instead.")
    
    # Close the client connection
    await client.aclose()

if __name__ == "__main__":
    asyncio.run(main()) 