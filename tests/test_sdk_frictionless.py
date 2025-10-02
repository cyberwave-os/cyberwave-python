# cyberwave/test_sdk_frictionless.py

import asyncio
import cyberwave # Import the SDK
import sys
import os

# Configure logging for the test script
import logging
logger = logging.getLogger("test_sdk")
logging.basicConfig(level=logging.INFO)

# Ensure the backend server is running at http://localhost:8000

async def run_test():
    logger.info("--- Starting CyberWave SDK Frictionless Flow Test ---")
    
    # Clear any previous cached token for a clean test run
    # Initialize client temporarily just for cache clearing
    try:
        client_for_setup = cyberwave.Client(use_token_cache=True) 
        client_for_setup._clear_token_cache() # Use internal method for test setup
        await client_for_setup.aclose()
        logger.info("Cleared token cache.")
    except Exception as e:
         logger.warning(f"Could not clear token cache (maybe first run): {e}")

    # 1. Initialize Client (no token initially)
    logger.info("[Step 1] Initializing client...")
    client = cyberwave.Client(use_token_cache=True) # Ensure caching is enabled
    assert not client.has_active_session(), "Client should not have active session initially"
    logger.info("Client initialized successfully.")

    shared_token_from_run = None
    robot_1_id = None
    robot_2_id = None

    try:
        # 2. Add first robot (triggers temp session)
        logger.info("[Step 2] Adding Robot 'SdkBot-1' (no token, no level_id)...")
        robot_1_info = await client.add_robot(name="SdkBot-1", robot_type="sdk_tester")
        logger.info(f"add_robot response: {robot_1_info}")

        assert "id" in robot_1_info
        robot_1_id = robot_1_info["id"]
        assert "share_token" in robot_1_info and robot_1_info["share_token"], \
               "Response should include a share_token"
        assert "share_url" in robot_1_info and robot_1_info["share_url"], \
               "Response should include a share_url"
        
        assert client.has_active_session(), "Client should now have an active session"
        shared_token_from_run = client.get_session_token()
        assert shared_token_from_run == robot_1_info["share_token"], \
               "Client token should match returned token"
        logger.info(f"[Step 2] Success. Got token: {shared_token_from_run[:4]}... Got Robot ID: {robot_1_id}")
        session_info = client.get_session_info()
        logger.info(f"Session Info: {session_info}")

        # 3. Add second robot (uses cached token)
        logger.info("[Step 3] Adding Robot 'SdkBot-2' (using session token)...")
        robot_2_info = await client.add_robot(name="SdkBot-2", robot_type="sdk_tester_v2")
        logger.info(f"add_robot response: {robot_2_info}")
        
        assert "id" in robot_2_info
        robot_2_id = robot_2_info["id"]
        # Should NOT get a new token
        assert robot_2_info.get("share_token") is None, "Should not get share_token on subsequent calls"
        assert robot_2_info.get("share_url") is None
        logger.info(f"[Step 3] Success. Got Robot ID: {robot_2_id}")

        # 4. (Optional) Add verification step - e.g., get robots for the session
        # Needs get_robots implemented in SDK and backend filtering by token
        # logger.info("[Step 4] Verifying robots list for session...")
        # robots = await client.get_robots()
        # assert len(robots) == 2
        # assert {r['id'] for r in robots} == {robot_1_id, robot_2_id}
        # logger.info("[Step 4] Success. Found correct robots in session.")

        logger.info("--- Frictionless Flow Test PASSED ---")

    except cyberwave.APIError as e:
        logger.error(f"Test FAILED due to API Error: Status={e.status_code}, Detail={e.detail}")
        sys.exit(1)
    except cyberwave.CyberWaveError as e:
        logger.error(f"Test FAILED due to SDK Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test FAILED due to unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Cleanup phase (manual for now)...")
        if shared_token_from_run:
             logger.info(f"Temporary session token was: {shared_token_from_run}")
             logger.info("Robots created might need manual cleanup via API/DB or wait for expiration.")
        # Ensure client is closed even if initialization failed partially
        if 'client' in locals() and client is not None:
            await client.aclose()

if __name__ == "__main__":
    asyncio.run(run_test()) 