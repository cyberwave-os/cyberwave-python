"""Test loading the example warehouse level."""

import os
import unittest
from pathlib import Path

from cyberwave.level import LevelDefinition, load_level


class TestExampleLevel(unittest.TestCase):
    """Test loading the example warehouse level."""
    
    def test_load_warehouse_example(self):
        """Test that we can load the warehouse example level."""
        # Find the warehouse example file
        example_path = Path("../cyberwave-static/examples/levels/warehouse_demo.yml")
        
        if not example_path.exists():
            self.skipTest(f"Example file not found at {example_path}")
        
        # Load the level
        level = load_level(example_path)
        
        # Check basic properties
        self.assertEqual(level.version, "1.0")
        self.assertEqual(level.metadata.title, "Smart Warehouse Demo")
        self.assertEqual(level.metadata.id, "warehouse_demo_1")
        self.assertEqual(level.metadata.floor_number, 1)
        
        # Check settings
        self.assertEqual(level.settings.export_mode, "hybrid")
        
        # Check entities
        self.assertGreaterEqual(len(level.entities), 2)  # At least two entities in the example
        
        # Find the charging dock
        charging_dock = next((e for e in level.entities if e.id == "charging_dock_a"), None)
        self.assertIsNotNone(charging_dock)
        self.assertEqual(charging_dock.archetype, "fixed_asset")
        self.assertEqual(charging_dock.reference, "charging_station")
        
        # Check zones
        self.assertGreaterEqual(len(level.zones), 1)  # At least one zone in the example
        
        # Find the shipping zone
        shipping_zone = next((z for z in level.zones if z.id == "shipping_zone"), None)
        self.assertIsNotNone(shipping_zone)
        self.assertEqual(shipping_zone.name, "Shipping Bay Zone")
        self.assertEqual(shipping_zone.type, "OPERATIONAL")


if __name__ == "__main__":
    unittest.main() 