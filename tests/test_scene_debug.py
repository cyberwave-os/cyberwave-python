import unittest
from unittest.mock import MagicMock
from cyberwave.schema import GeometryType
from cyberwave.scene import Scene


class TestScene(unittest.TestCase):
    def test_add_primitive(self):
        # Mock client
        client = MagicMock()
        client.environments.get.return_value = MagicMock(
            universal_schema=None, name="test", description="test"
        )

        scene = Scene(client, "env_id")

        # Add primitive
        link = scene.add_primitive(GeometryType.BOX, size=[1, 1, 1], name="box")

        # Check schema
        self.assertEqual(len(scene.schema.links), 1)
        self.assertEqual(scene.schema.links[0].name, "box")

        # Check to_dict
        data = scene.schema.to_dict()
        print("Schema dict keys:", data.keys())
        if "links" in data:
            print("Links in dict:", [l["name"] for l in data["links"]])
        else:
            print("Links MISSING in dict")

        self.assertIn("links", data)
        self.assertEqual(len(data["links"]), 1)
        self.assertEqual(data["links"][0]["name"], "box")


if __name__ == "__main__":
    unittest.main()
