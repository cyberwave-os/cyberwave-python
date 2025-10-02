"""
Python Wrapper for Centralized Schema System in SDK

This module provides Python access to the centralized TypeScript schema system
for mission and level YAML generation and validation in the SDK.
"""

import os
import sys
import subprocess
import json
import tempfile
import yaml
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

# Centralized API version constant
CYBERWAVE_API_VERSION = "cyberwave.com/v1"
CYBERWAVE_LEVEL_API_VERSION = "cyberwave.com/v1"


class CentralizedSchemaError(Exception):
    """Exception raised when centralized schema operations fail"""
    pass


class SDKSchemaAdapter:
    """Adapter for SDK to use centralized schema system"""
    
    def __init__(self):
        """Initialize the schema adapter"""
        # Try to find centralized schema directory
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "cyberwave-schemas"),
            os.path.join(os.path.dirname(__file__), "..", "..", "cyberwave-schemas"),
            os.environ.get("CENTRALIZED_SCHEMA_PATH", ""),
            "/cyberwave-schemas"
        ]
        
        self.schema_dir = None
        for path in possible_paths:
            if path and os.path.exists(path):
                self.schema_dir = os.path.abspath(path)
                break
    
    def convert_sdk_to_centralized(self, sdk_level: Dict[str, Any]) -> Dict[str, Any]:
        """Convert SDK level format to centralized format"""
        try:
            # Create centralized format structure
            centralized = {
                "apiVersion": CYBERWAVE_LEVEL_API_VERSION,
                "kind": "Level",
                "metadata": self._convert_metadata(sdk_level.get("metadata", {})),
                "coordinateSystem": self._convert_coordinate_system(sdk_level.get("metadata", {})),
                "scene": self._convert_scene(sdk_level)
            }
            
            return centralized
            
        except Exception as e:
            raise CentralizedSchemaError(f"Failed to convert SDK level to centralized format: {e}")
    
    def _convert_metadata(self, sdk_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert SDK metadata to centralized format"""
        now = datetime.utcnow().isoformat() + "Z"
        
        # Use the actual title from SDK, not "Unnamed Level"
        title = sdk_metadata.get("title", "Unnamed Level") 
        description = sdk_metadata.get("description", "No description available")
        level_id = sdk_metadata.get("id", "unnamed-level")
        
        return {
            "uuid": level_id,
            "name": title,  # Use the actual title as the name
            "version": "1.0.0", 
            "description": description,
            "author": "SDK Generated",
            "created": now,
            "modified": now,
            "tags": [],
            "environment_type": "industrial",
            "complexity_level": "moderate",
            "safety_level": "medium"
        }
    
    def _convert_coordinate_system(self, sdk_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert SDK coordinate system to centralized format"""
        coord_system = sdk_metadata.get("coordinate_system", "ENU")
        units = sdk_metadata.get("units", "meters")
        
        # Standard ENU system
        return {
            "up": "z",
            "forward": "y", 
            "handedness": "right",
            "units": units,
            "origin": [0, 0, 0]
        }
    
    def _convert_scene(self, sdk_level: Dict[str, Any]) -> Dict[str, Any]:
        """Convert SDK scene to centralized format"""
        # Handle None values safely
        environment = sdk_level.get("environment") or {}
        lighting = environment.get("lighting") or {}
        entities = sdk_level.get("entities") or []
        
        scene = {
            "environment": [],
            "entities": [],
            "lighting": self._convert_lighting(lighting),
            "physics": {
                "gravity": [0, 0, -9.81],
                "enable_collisions": True,
                "simulation_step": 0.016,
                "solver_iterations": 6
            }
        }
        
        # Add ground plane
        scene["environment"].append({
            "id": "ground_plane",
            "type": "mesh",
            "archetype": "ground",
            "transform": {
                "position": [0, 0, -0.5],
                "rotation": [0, 0, 0],
                "scale": [100, 100, 1]
            },
            "visible": True,
            "name": "Ground Plane",
            "userData": {
                "color": [0.3, 0.4, 0.3],
                "material": "ground",
                "physics": True,
                "collider": "box"
            }
        })
        
        # Convert entities safely
        for entity in entities:
            if entity:  # Check entity is not None
                scene["entities"].append(self._convert_entity(entity))
        
        return scene
    
    def _convert_entity(self, sdk_entity: Dict[str, Any]) -> Dict[str, Any]:
        """Convert SDK entity to centralized format"""
        # Handle None values safely
        transform = sdk_entity.get("transform") or {}
        entity_id = sdk_entity.get("id", "unknown_entity")
        archetype = sdk_entity.get("archetype", "unknown")
        
        entity = {
            "id": entity_id,
            "type": "robot" if archetype == "robot" else "mesh",
            "archetype": self._map_archetype(archetype),
            "transform": {
                "position": transform.get("position", [0, 0, 0]),
                "rotation": transform.get("rotation", [0, 0, 0]),
                "scale": transform.get("scale", [1, 1, 1])
            },
            "visible": True,
            "name": sdk_entity.get("id", "Unknown Entity"),
            "userData": self._convert_user_data(sdk_entity)
        }
        
        return entity
    
    def _map_archetype(self, sdk_archetype: str) -> str:
        """Map SDK archetype to centralized archetype"""
        mapping = {
            "robot": "aerial_drone",
            "fixed_asset": "ground",
            "sensor_array": "sensor"
        }
        return mapping.get(sdk_archetype, "ground")
    
    def _convert_user_data(self, sdk_entity: Dict[str, Any]) -> Dict[str, Any]:
        """Convert SDK entity properties to centralized userData"""
        user_data = {
            "color": [0.5, 0.5, 0.5]
        }
        
        # Add capabilities if robot
        if sdk_entity.get("archetype") == "robot":
            user_data.update({
                "capabilities": sdk_entity.get("capabilities", ["basic"]),
                "safety_certified": True,
                "color": [0, 0.5, 1]
            })
        
        # Add properties safely (handle None values)
        properties = sdk_entity.get("properties")
        if properties and isinstance(properties, dict):
            user_data.update(properties)
        
        return user_data
    
    def _convert_lighting(self, sdk_lighting: Dict[str, Any]) -> Dict[str, Any]:
        """Convert SDK lighting to centralized format"""
        lighting = {
            "ambient": {
                "color": [0.4, 0.4, 0.4],
                "intensity": sdk_lighting.get("ambient", 0.3)
            },
            "directional": {
                "direction": [-1, -1, -1],
                "color": [1, 1, 1],
                "intensity": 0.8,
                "shadows": True
            }
        }
        
        # Use first directional light if available
        directional_lights = sdk_lighting.get("directional", [])
        if directional_lights:
            first_light = directional_lights[0]
            lighting["directional"]["direction"] = first_light.get("direction", [-1, -1, -1])
            lighting["directional"]["intensity"] = first_light.get("intensity", 0.8)
        
        return lighting
    
    def generate_level_yaml(self, sdk_level: Union[Dict[str, Any], str, Path]) -> str:
        """Generate centralized level YAML from SDK level"""
        try:
            # Handle different input types
            if isinstance(sdk_level, (str, Path)):
                with open(sdk_level, 'r') as f:
                    sdk_level = yaml.safe_load(f)
            
            # Convert to centralized format
            centralized_level = self.convert_sdk_to_centralized(sdk_level)
            
            # For now, use Python fallback to ensure proper conversion
            # TODO: Fix TypeScript generator integration later
            return self._generate_fallback_yaml(centralized_level)
                
        except Exception as e:
            raise CentralizedSchemaError(f"Failed to generate level YAML: {e}")
    
    def _generate_with_typescript(self, level_data: Dict[str, Any]) -> str:
        """Generate YAML using TypeScript centralized system"""
        try:
            # Create temporary file for level data
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(level_data, f)
                temp_file = f.name
            
            # Run TypeScript generator
            cmd = [
                "npx", "ts-node", "-e",
                f"""
                import {{ generateLevelYAML }} from './generators';
                import fs from 'fs';
                const data = JSON.parse(fs.readFileSync('{temp_file}', 'utf8'));
                console.log(generateLevelYAML(data));
                """
            ]
            
            result = subprocess.run(
                cmd, 
                cwd=self.schema_dir,
                capture_output=True, 
                text=True,
                timeout=30
            )
            
            # Clean up
            os.unlink(temp_file)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                raise CentralizedSchemaError(f"TypeScript generator failed: {result.stderr}")
                
        except Exception as e:
            # Fall back to Python generation
            return self._generate_fallback_yaml(level_data)
    
    def _generate_fallback_yaml(self, level_data: Dict[str, Any]) -> str:
        """Generate YAML using Python fallback"""
        return yaml.dump(level_data, default_flow_style=False, sort_keys=False)
    
    def validate_level(self, level_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate level using centralized system"""
        errors = []
        
        # Basic validation
        if not level_data.get("apiVersion"):
            errors.append("Missing apiVersion")
        elif level_data["apiVersion"] != CYBERWAVE_LEVEL_API_VERSION:
            errors.append(f"Invalid apiVersion: {level_data['apiVersion']}")
        
        if not level_data.get("kind"):
            errors.append("Missing kind")
        elif level_data["kind"] != "Level":
            errors.append(f"Invalid kind: {level_data['kind']}")
        
        if not level_data.get("metadata"):
            errors.append("Missing metadata")
        elif not level_data["metadata"].get("name"):
            errors.append("Missing metadata.name")
        
        if not level_data.get("coordinateSystem"):
            errors.append("Missing coordinateSystem")
        
        if not level_data.get("scene"):
            errors.append("Missing scene")
        
        return len(errors) == 0, errors


# Global instance
_sdk_adapter = SDKSchemaAdapter()

# Public API functions
def convert_sdk_to_centralized(sdk_level: Dict[str, Any]) -> Dict[str, Any]:
    """Convert SDK level format to centralized format"""
    return _sdk_adapter.convert_sdk_to_centralized(sdk_level)

def generate_centralized_level_yaml(sdk_level: Union[Dict[str, Any], str, Path]) -> str:
    """Generate centralized level YAML from SDK level"""
    return _sdk_adapter.generate_level_yaml(sdk_level)

def validate_centralized_level(level_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate level using centralized system"""
    return _sdk_adapter.validate_level(level_data) 