"""Tests for the level definition module."""
import os
import json
import yaml
import tempfile
from pathlib import Path
import unittest

# Direct imports to avoid circular imports
from cyberwave.level.schema import LevelDefinition
from cyberwave.level.loader import load_level, save_level
