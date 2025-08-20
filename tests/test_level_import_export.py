#!/usr/bin/env python3
"""
Tests for level import and export functionality.

This test verifies:
1. Loading level definitions from YAML files
2. Saving level definitions to YAML/JSON
3. Validation of level structures
4. Converting between different formats
"""

import os
import json
import tempfile
import unittest
from pathlib import Path

from cyberwave import load_level, save_level
from cyberwave.level.schema import LevelDefinition, Entity, Zone, Metadata

# Path to example file
EXAMPLE_LEVEL_PATH = Path("examples/levels/warehouse_demo.yml")

class TestLevelImportExport(unittest.TestCase):
    """Test suite for level import/export functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        # Check if the example file exists
        if not EXAMPLE_LEVEL_PATH.exists():
            self.skipTest(f"Example level file not found: {EXAMPLE_LEVEL_PATH}")
            
        # Load the example level
        self.example_level = load_level(EXAMPLE_LEVEL_PATH)
    
    def test_load_level(self):
        """Test loading a level from YAML."""
        level = self.example_level
        
        # Verify basic metadata
        self.assertEqual(level.metadata.title, "Smart Warehouse Demo")
        self.assertEqual(level.metadata.id, "warehouse_demo_1")
        self.assertEqual(level.metadata.floor_number, 1)
        
        # Verify entities
        self.assertIsNotNone(level.entities)
        self.assertTrue(len(level.entities) > 0)
        
        # Verify zones
        self.assertIsNotNone(level.zones)
        self.assertTrue(len(level.zones) > 0)
        
    def test_save_level_yaml(self):
        """Test saving a level to YAML."""
        level = self.example_level
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.yml', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            # Save the level
            save_level(level, tmp_path, format='yaml')
            
            # Verify the file exists
            self.assertTrue(tmp_path.exists())
            
            # Reload the level
            reloaded = load_level(tmp_path)
            
            # Verify it's the same
            self.assertEqual(reloaded.metadata.title, level.metadata.title)
            self.assertEqual(reloaded.metadata.id, level.metadata.id)
            self.assertEqual(len(reloaded.entities or []), len(level.entities or []))
            self.assertEqual(len(reloaded.zones or []), len(level.zones or []))
        finally:
            # Clean up
            if tmp_path.exists():
                os.unlink(tmp_path)
    
    def test_save_level_json(self):
        """Test saving a level to JSON."""
        level = self.example_level
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            # Save the level
            save_level(level, tmp_path, format='json')
            
            # Verify the file exists
            self.assertTrue(tmp_path.exists())
            
            # Verify it's valid JSON
            with open(tmp_path, 'r') as f:
                json_data = json.load(f)
            
            # Check basic structure
            self.assertIn('metadata', json_data)
            self.assertIn('title', json_data['metadata'])
            self.assertEqual(json_data['metadata']['title'], level.metadata.title)
        finally:
            # Clean up
            if tmp_path.exists():
                os.unlink(tmp_path)
    
    def test_create_level_programmatically(self):
        """Test creating a level programmatically with the schema."""
        # Create a simple level
        level = LevelDefinition(
            version="1.0",
            metadata=Metadata(
                title="Test Level",
                id="test_level_1",
                floor_number=1,
                description="A test level created programmatically",
            ),
            entities=[
                Entity(
                    id="robot_1",
                    archetype="robot",
                    reference="test_robot",
                    capabilities=["navigate_2d"],
                    status="idle",
                    battery_percentage=100.0
                )
            ],
            zones=[
                Zone(
                    id="test_zone",
                    name="Test Zone",
                    type="OPERATIONAL",
                    geometry={
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]
                    }
                )
            ]
        )
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.yml', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            # Save the level
            save_level(level, tmp_path)
            
            # Reload the level
            reloaded = load_level(tmp_path)
            
            # Verify it's correct
            self.assertEqual(reloaded.metadata.title, "Test Level")
            self.assertEqual(reloaded.metadata.id, "test_level_1")
            self.assertEqual(len(reloaded.entities), 1)
            self.assertEqual(reloaded.entities[0].id, "robot_1")
            self.assertEqual(len(reloaded.zones), 1)
            self.assertEqual(reloaded.zones[0].id, "test_zone")
        finally:
            # Clean up
            if tmp_path.exists():
                os.unlink(tmp_path)

if __name__ == "__main__":
    unittest.main() 