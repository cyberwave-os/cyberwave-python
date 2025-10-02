#!/usr/bin/env python3
"""
Debug script for testing client authentication.
"""

import asyncio
import logging
import os
from pathlib import Path
import sys

# Add the parent directory to the path so we can import the cyberwave module
sys.path.insert(0, str(Path(__file__).parent.parent))

from cyberwave.client import Client, AuthenticationError

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def debug_auth():
    """Debug the authentication process."""
    logger.info("--- Starting Authentication Debug ---")
    
    # Initialize client with cache disabled for clean test
    client = Client(use_token_cache=False)
    logger.info("Client initialized.")
    
    try:
        # 1. Create Workspace (Should get a token from this)
        workspace_name = f"Debug-WS-{os.urandom(4).hex()}"
        logger.info(f"Creating workspace: {workspace_name}")
        workspace = await client.create_workspace(name=workspace_name)
        
        logger.info(f"Workspace created: {workspace}")
        logger.info(f"Has active session: {client.has_active_session()}")
        logger.info(f"Session token: {client.get_session_token()}")
        
        # 2. Try creating a project
        if client.has_active_session():
            logger.info("Attempting to create project...")
            project_name = f"Debug-Proj-{os.urandom(4).hex()}"
            ws_id = workspace.get("id")
            
            try:
                project = await client.create_project(workspace_id=ws_id, name=project_name)
                logger.info(f"Project created: {project}")
            except AuthenticationError as e:
                logger.error(f"Authentication error creating project: {e}")
        else:
            logger.warning("No active session, skipping project creation.")
            
    except Exception as e:
        logger.error(f"Error during debug: {e}", exc_info=True)
    finally:
        await client.aclose()
        logger.info("--- Authentication Debug Complete ---")

if __name__ == "__main__":
    asyncio.run(debug_auth()) 