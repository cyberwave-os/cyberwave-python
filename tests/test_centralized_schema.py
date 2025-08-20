#!/usr/bin/env python3
"""
Tests for centralized schema integration in SDK
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from cyberwave import (
    convert_sdk_to_centralized,
    generate_centralized_level_yaml,
    validate_centralized_level,
    CYBERWAVE_LEVEL_API_VERSION,
    CentralizedSchemaError
)
from cyberwave.level.schema import LevelDefinition, Metadata, Entity, Transform


class TestCentralizedSchemaIntegration:
    """Test centralized schema system integration"""
    
    def test_sdk_to_centralized_conversion(self):
        """Test converting SDK format to centralized format"""
        # Create a simple SDK level
        sdk_level = {
            "version": "1.0",
            "metadata": {
                "id": "test-level-1",
                "title": "Test Level",
                "description": "Test description",
                "floor_number": 1
            },
            "entities": [
                {
                    "id": "test_robot",
                    "archetype": "robot",
                    "transform": {
                        "position": [1, 2, 3],
                        "rotation": [0, 0, 0],
                        "scale": [1, 1, 1]
                    },
                    "capabilities": ["navigate_2d"],
                    "properties": {"battery": 100}
                }
            ]
        }
        
        # Convert to centralized format
        centralized = convert_sdk_to_centralized(sdk_level)
        
        # Verify structure
        assert centralized["apiVersion"] == CYBERWAVE_LEVEL_API_VERSION
        assert centralized["kind"] == "Level"
        assert centralized["metadata"]["name"] == "Test Level"
        assert centralized["metadata"]["uuid"] == "test-level-1"
        assert len(centralized["scene"]["entities"]) == 1
        
        # Verify entity conversion
        entity = centralized["scene"]["entities"][0]
        assert entity["id"] == "test_robot"
        assert entity["type"] == "robot"
        assert entity["transform"]["position"] == [1, 2, 3]
        assert entity["userData"]["capabilities"] == ["navigate_2d"]
    
    def test_centralized_yaml_generation(self):
        """Test generating centralized YAML from SDK level"""
        sdk_level = {
            "version": "1.0",
            "metadata": {
                "id": "yaml-test",
                "title": "YAML Test Level",
                "description": "Testing YAML generation"
            },
            "entities": []
        }
        
        # Generate YAML
        yaml_output = generate_centralized_level_yaml(sdk_level)
        
        # Verify YAML is valid and contains expected structure
        parsed = yaml.safe_load(yaml_output)
        assert parsed["apiVersion"] == CYBERWAVE_LEVEL_API_VERSION
        assert parsed["kind"] == "Level"
        assert parsed["metadata"]["name"] == "YAML Test Level"
        assert "coordinateSystem" in parsed
        assert "scene" in parsed
    
    def test_centralized_validation(self):
        """Test validation of centralized format"""
        # Valid centralized level
        valid_level = {
            "apiVersion": CYBERWAVE_LEVEL_API_VERSION,
            "kind": "Level",
            "metadata": {"name": "Valid Level"},
            "coordinateSystem": {"up": "z"},
            "scene": {"entities": []}
        }
        
        is_valid, errors = validate_centralized_level(valid_level)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid level (missing required fields)
        invalid_level = {
            "apiVersion": "wrong/version",
            "kind": "Mission"  # Wrong kind
        }
        
        is_valid, errors = validate_centralized_level(invalid_level)
        assert not is_valid
        assert len(errors) > 0
    
    def test_pydantic_model_conversion(self):
        """Test converting Pydantic models to centralized format"""
        # Create using Pydantic models
        metadata = Metadata(
            title="Pydantic Test",
            id="pydantic-test",
            description="Testing Pydantic conversion",
            floor_number=1
        )
        
        entity = Entity(
            id="test_entity",
            archetype="robot",
            transform=Transform(
                position=[5, 10, 15],
                rotation=[0, 0, 0],
                scale=[1, 1, 1]
            ),
            capabilities=["lidar", "camera"]
        )
        
        level = LevelDefinition(
            version="1.0",
            metadata=metadata,
            entities=[entity]
        )
        
        # Convert to dict and then to centralized
        sdk_dict = level.model_dump()
        centralized = convert_sdk_to_centralized(sdk_dict)
        
        # Verify conversion
        assert centralized["metadata"]["name"] == "Pydantic Test"
        assert len(centralized["scene"]["entities"]) == 1
        
        entity_data = centralized["scene"]["entities"][0]
        assert entity_data["id"] == "test_entity"
        assert entity_data["transform"]["position"] == [5, 10, 15]
    
    def test_error_handling(self):
        """Test error handling in centralized schema system"""
        # Test with invalid input
        with pytest.raises(CentralizedSchemaError):
            convert_sdk_to_centralized(None)
        
        with pytest.raises(CentralizedSchemaError):
            generate_centralized_level_yaml(None)
    
    def test_ground_plane_generation(self):
        """Test that ground plane is automatically added"""
        sdk_level = {
            "version": "1.0", 
            "metadata": {"title": "Ground Test"},
            "entities": []
        }
        
        centralized = convert_sdk_to_centralized(sdk_level)
        
        # Should have ground plane in environment
        assert len(centralized["scene"]["environment"]) > 0
        ground = centralized["scene"]["environment"][0]
        assert ground["id"] == "ground_plane"
        assert ground["archetype"] == "ground"
    
    def test_lighting_conversion(self):
        """Test lighting system conversion"""
        sdk_level = {
            "version": "1.0",
            "metadata": {"title": "Lighting Test"},
            "environment": {
                "lighting": {
                    "ambient": 0.5,
                    "directional": [
                        {
                            "direction": [0, -1, 0],
                            "intensity": 1.0
                        }
                    ]
                }
            }
        }
        
        centralized = convert_sdk_to_centralized(sdk_level)
        
        lighting = centralized["scene"]["lighting"]
        assert lighting["ambient"]["intensity"] == 0.5
        assert lighting["directional"]["direction"] == [0, -1, 0]
        assert lighting["directional"]["intensity"] == 1.0
    
    def test_coordinate_system_generation(self):
        """Test coordinate system generation"""
        sdk_level = {
            "version": "1.0",
            "metadata": {
                "title": "Coordinate Test",
                "units": "millimeters"
            }
        }
        
        centralized = convert_sdk_to_centralized(sdk_level)
        
        coord_sys = centralized["coordinateSystem"]
        assert coord_sys["up"] == "z"
        assert coord_sys["forward"] == "y"
        assert coord_sys["handedness"] == "right"
        assert coord_sys["units"] == "millimeters" 