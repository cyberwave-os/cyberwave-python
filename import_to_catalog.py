import argparse
import logging
import os
import sys
import getpass
from typing import Tuple, List, Optional
from cyberwave.sdk import Client
from cyberwave.sdk.exceptions import APIError
from cyberwave.sdk.utils import create_slug

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Import MuJoCo Menagerie robots to CyberWave catalog")
    parser.add_argument("--images-only", action="store_true", help="Only copy images, don't update catalog data")
    parser.add_argument("--workspace-slug", default="default", help="Workspace slug to use")
    parser.add_argument("--backend-url", help="CyberWave backend URL")
    parser.add_argument("--username", help="CyberWave username")
    parser.add_argument("--password", help="CyberWave password")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

async def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging to be more verbose if debug flag is set
    if args.debug:
        logging.getLogger("cyberwave.sdk").setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        # Add a console handler with detailed formatting if not already present
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(console_handler)
        logger.debug("Debug mode enabled, verbose logging activated")

    # Get backend URL from args or environment
    backend_url = args.backend_url or os.environ.get("CYBERWAVE_BACKEND_URL")
    
    # Initialize the client
    client = Client(
        base_url=backend_url,
        debug=args.debug  # Pass debug flag to client
    )
    
    # Try to get credentials from args, env vars, or prompt
    username = args.username or os.environ.get("CYBERWAVE_USERNAME")
    password = args.password or os.environ.get("CYBERWAVE_PASSWORD")
    
    # If interactive and credentials not provided, prompt for them
    if not username and sys.stdin.isatty():
        username = input("CyberWave Username: ")
    if not password and sys.stdin.isatty():
        password = getpass("CyberWave Password: ")
    
    # Log in if we have credentials
    if username and password:
        try:
            await client.login(username, password)
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return
    
    # Only copy images and exit if --images-only flag is set
    if args.images_only:
        logger.info("Images-only mode active. Only copying robot images to frontend.")
        robot_folders = [p for p in MENAGERIE_PATH.iterdir() 
                        if p.is_dir() and not p.name.startswith(('.', '_')) and p.name != 'assets']
        
        for robot_folder in robot_folders:
            metadata = extract_robot_metadata(robot_folder)
            # Use a placeholder ID for image paths since we're just copying
            copy_robot_image(metadata, f"placeholder-{robot_folder.name}")
            
        logger.info(f"Copied available images for {len(robot_folders)} robots to frontend public directory.")
        await client.aclose()
        return
    
    # Find workspace by slug
    workspace_slug = args.workspace_slug
    try:
        workspace = await client.get_workspace_by_slug(workspace_slug)
        if not workspace:
            logger.info(f"Workspace with slug '{workspace_slug}' not found. Creating it...")
            workspace = await client.create_workspace(name=workspace_slug.title(), slug=workspace_slug)
            logger.info(f"Created workspace '{workspace['name']}' with ID {workspace['id']}")
        else:
            logger.info(f"Found workspace '{workspace['name']}' with ID {workspace['id']}")
    except Exception as e:
        logger.error(f"Error finding/creating workspace: {e}")
        await client.aclose()
        return
    
    # Import robots
    try:
        created, updated, failed, skipped = await import_robots_to_catalog(client, workspace["id"])
        
        # Print summary
        logger.info("\n--- Import Summary ---")
        logger.info(f"Successfully Created ({len(created)}):")
        if created:
            for robot in created:
                logger.info(f"  - {robot}")
        else:
            logger.info("  (None)")
            
        logger.info(f"Successfully Updated ({len(updated)}):")
        if updated:
            for robot in updated:
                logger.info(f"  - {robot}")
        else:
            logger.info("  (None)")
            
        logger.info(f"Skipped Update (existing, due to backend issue) ({len(skipped)}):")
        if skipped:
            for robot in skipped:
                logger.info(f"  - {robot}")
        else:
            logger.info("  (None)")
            
        logger.info(f"Failed to Process ({len(failed)}):")
        if failed:
            for robot in failed:
                logger.info(f"  - {robot}")
        else:
            logger.info("  (None)")
        
        logger.info("--- End of Summary ---")
    except Exception as e:
        logger.error(f"Error during import: {e}")
    
    # Close client
    logger.info("Closing CyberWave client connection.")
    await client.aclose()

async def add_robot_to_catalog(client: Client, workspace_id: int, robot_metadata: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Add or update a robot in the catalog as an asset definition.
    First checks if the asset exists by slug. If it doesn't exist, creates it.
    If it exists, updates its metadata and image_url.
    Returns a tuple: (asset_definition_dict_or_None, status_string)
    status_string can be 'created', 'updated', or 'failed'.
    """
    slug = create_slug(robot_metadata["name"])
    asset_def: Optional[Dict[str, Any]] = None
    created_new = False # Flag to track if the asset was newly created in this run

    # First check if the asset definition already exists by slug
    try:
        logger.info(f"Checking if asset definition '{slug}' exists in workspace {workspace_id}...")
        existing_asset = await client.get_asset_definition_by_slug(workspace_id, slug)
        
        if existing_asset:
            logger.info(f"Found existing asset definition '{slug}' (ID: {existing_asset['id']}).")
            asset_def = existing_asset
        else:
            # Asset doesn't exist, create it
            logger.info(f"Asset definition '{slug}' not found. Creating new asset...")
            asset_def = await client.create_asset_definition(
                workspace_id=workspace_id,
                name=robot_metadata["display_name"],
                slug=slug,
                definition_type="robot",
                description=robot_metadata["description"],
                tags=robot_metadata["capabilities"],
                metadata={
                    "mesh_count": robot_metadata["mesh_count"],
                    "joint_count": robot_metadata["joint_count"],
                    "body_count": robot_metadata["body_count"],
                    "actuator_count": robot_metadata["actuator_count"],
                    "xml_file": robot_metadata["xml_file"],
                    "source": "mujoco_menagerie",
                }
            )
            created_new = True
            logger.info(f"Successfully CREATED new asset definition '{slug}' (ID: {asset_def['id']}).")
    except APIError as err:
        if isinstance(err, APIError) and err.status_code == 404:
            # Asset not found, which is expected if it doesn't exist
            # Try to create it
            try:
                logger.info(f"Asset definition '{slug}' not found. Creating new asset...")
                asset_def = await client.create_asset_definition(
                    workspace_id=workspace_id,
                    name=robot_metadata["display_name"],
                    slug=slug,
                    definition_type="robot",
                    description=robot_metadata["description"],
                    tags=robot_metadata["capabilities"],
                    metadata={
                        "mesh_count": robot_metadata["mesh_count"],
                        "joint_count": robot_metadata["joint_count"],
                        "body_count": robot_metadata["body_count"],
                        "actuator_count": robot_metadata["actuator_count"],
                        "xml_file": robot_metadata["xml_file"],
                        "source": "mujoco_menagerie",
                    }
                )
                created_new = True
                logger.info(f"Successfully CREATED new asset definition '{slug}' (ID: {asset_def['id']}).")
            except APIError as create_err:
                logger.error(f"API error during CREATE for slug '{slug}': {create_err}. Cannot proceed with this item.")
                return None, "failed"
            except Exception as unexp_create_err:
                logger.error(f"Unexpected error during CREATE for slug '{slug}': {unexp_create_err}. Cannot proceed with this item.")
                return None, "failed"
        else:
            # Other API error during initial check
            logger.error(f"API error checking if asset '{slug}' exists: {err}. Cannot proceed with this item.")
            return None, "failed"
    except Exception as e:
        logger.error(f"Unexpected error checking if asset '{slug}' exists: {e}. Cannot proceed with this item.")
        return None, "failed"

    if not asset_def:
        logger.error(f"Failed to obtain asset definition for slug '{slug}' after create/fetch attempts. Aborting for this item.")
        return None, "failed"

    # Skip update if there are known backend serialization issues
    if not created_new:
        try:
            # --- Update Core Metadata for existing assets ---
            logger.info(f"Updating core metadata for existing asset definition '{slug}' (ID: {asset_def['id']}).")
            update_succeeded = False # Flag to track if the update was successful for status reporting
            
            core_metadata_payload = {
                "mesh_count": robot_metadata["mesh_count"],
                "joint_count": robot_metadata["joint_count"],
                "body_count": robot_metadata["body_count"],
                "actuator_count": robot_metadata["actuator_count"],
                "xml_file": robot_metadata["xml_file"],
                "source": "mujoco_menagerie",
            }
            if "metadata" in asset_def and "image_url" in asset_def["metadata"]:
                core_metadata_payload["image_url"] = asset_def["metadata"]["image_url"]

            updated_asset_def_for_metadata = await client.update_asset_definition(
                workspace_id=workspace_id,
                definition_id_or_slug=asset_def["id"],
                name=robot_metadata["display_name"],
                description=robot_metadata["description"],
                tags=robot_metadata["capabilities"],
                metadata=core_metadata_payload
            )
            asset_def = updated_asset_def_for_metadata # Update our reference
            logger.info(f"Core metadata update for '{slug}' (ID: {asset_def['id']}) successful.")
            update_succeeded = True # Mark update as successful
        except APIError as core_update_err:
            logger.error(f"API error updating core metadata for '{slug}' (ID: {asset_def['id']}): {core_update_err}. Image update will be attempted with potentially stale core metadata.")
            update_succeeded = False # Mark update as failed
        except Exception as unexp_core_update_err:
            logger.error(f"Unexpected error updating core metadata for '{slug}' (ID: {asset_def['id']}): {unexp_core_update_err}. Image update will be attempted with potentially stale core metadata.")
            update_succeeded = False # Mark update as failed
    else:
        # For newly created assets, we consider the update "succeeded" as we just created it
        update_succeeded = True

    # --- Copy image and update/set the image_url (always, for both new and existing) ---
    image_update_succeeded = False # Track image update success separately
    try:
        if not asset_def.get("id"): # Should have ID by now
            logger.error(f"Cannot copy/update image for '{slug}' because asset definition ID is missing. Skipping image step.")
        else:
            image_path = copy_robot_image(robot_metadata, asset_def["id"])
            current_asset_metadata_for_image = asset_def.get("metadata", {}).copy()

            if image_path:
                if current_asset_metadata_for_image.get("image_url") != image_path:
                    logger.info(f"Updating image_url for asset definition '{slug}' (ID: {asset_def['id']}) to '{image_path}'.")
                    current_asset_metadata_for_image["image_url"] = image_path

                    final_updated_asset_def = await client.update_asset_definition(
                        workspace_id=workspace_id,
                        definition_id_or_slug=asset_def["id"],
                        metadata=current_asset_metadata_for_image
                    )
                    asset_def = final_updated_asset_def
                    logger.info(f"Successfully SET/UPDATED image_url for asset definition '{slug}' (ID: {asset_def['id']}).")
                    image_update_succeeded = True
                else:
                    logger.info(f"Image_url '{image_path}' already correctly set for '{slug}' (ID: {asset_def['id']}). No image_url update needed.")
                    image_update_succeeded = True # Considered success as it's already correct
            else:
                logger.info(f"No image found for '{robot_metadata['name']}'. Skipping image_url update/creation for '{slug}' (ID: {asset_def['id']}).")
                # If no image exists, we consider the 'image update' step 'successful' in the sense that it doesn't need doing.
                image_update_succeeded = True 

    except APIError as img_update_e:
        logger.error(f"API error during final image_url update for '{slug}' (ID: {asset_def.get('id', 'N/A')}): {img_update_e}. Asset def state may be partially updated.")
        image_update_succeeded = False # Image update definitely failed
    except Exception as img_update_unexp_e:
        logger.error(f"Unexpected error during final image_url update for '{slug}' (ID: {asset_def.get('id', 'N/A')}): {img_update_unexp_e}. Asset def state may be partially updated.")
        image_update_succeeded = False # Image update definitely failed

    # Determine final status for return based on overall success
    final_status = "failed" # Default to failed
    if update_succeeded and image_update_succeeded: # Only succeed if both core and image updates (or skips) were okay
        final_status = "created" if created_new else "updated"
    
    if final_status != "failed":
         logger.info(f"Successfully {final_status} robot '{robot_metadata['display_name']}' (Slug: {slug}, ID: {asset_def.get('id', 'N/A')}).")
    else:
         # Log failure more explicitly if we reach here but status is failed
         logger.error(f"Failed to fully process robot '{robot_metadata['display_name']}' (Slug: {slug}, ID: {asset_def.get('id', 'N/A')}). See previous errors for details.")

    return asset_def, final_status 

async def import_robots_to_catalog(client: Client, workspace_id: int) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Import all robots from MuJoCo Menagerie into the catalog.
    Returns four lists: names of created, updated, failed, and skipped_for_update robots.
    """
    created_robot_names: List[str] = []
    updated_robot_names: List[str] = []
    failed_robot_names: List[str] = []
    skipped_update_robot_names: List[str] = [] # New list for skipped updates
    
    robot_folders = [p for p in MENAGERIE_PATH.iterdir() 
                    if p.is_dir() and not p.name.startswith(('.', '_')) and p.name != 'assets']
    
    logger.info(f"Found {len(robot_folders)} potential robot folders in MuJoCo Menagerie")
    
    for robot_folder in robot_folders:
        metadata = extract_robot_metadata(robot_folder)
        robot_name_for_summary = metadata.get("display_name", metadata.get("name", robot_folder.name)) # Use display_name if available

        asset_def, status = await add_robot_to_catalog(client, workspace_id, metadata)
        
        if status == "created":
            created_robot_names.append(robot_name_for_summary)
        elif status == "updated":
            updated_robot_names.append(robot_name_for_summary)
        elif status == "skipped_update_due_to_backend_issue": # Handle new status
            skipped_update_robot_names.append(robot_name_for_summary)
        else: # status == "failed" or asset_def is None
            failed_robot_names.append(robot_name_for_summary)
            
    return created_robot_names, updated_robot_names, failed_robot_names, skipped_update_robot_names 