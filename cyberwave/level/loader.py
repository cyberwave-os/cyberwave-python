"""Utilities for loading and saving CyberWave level definitions.

This module provides functions for working with level definition files,
including loading from YAML/JSON files and saving to disk.
"""

import os
import json
import yaml
from enum import Enum
from pathlib import Path
from typing import Dict, Union, Optional, Any, TextIO

import jsonschema
from pydantic import ValidationError

from .schema import LevelDefinition

# Custom YAML representer for Enum values
def _enum_representer(dumper, data):
    """Custom representer for Enum values in YAML serialization."""
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data.value))

# Register the representer with PyYAML
yaml.add_representer(Enum, _enum_representer)

# Custom encoder for JSON serialization
class EnumJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles Enum values."""
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


def load_level(source: Union[str, Path, TextIO]) -> LevelDefinition:
    """Load a level definition from a file or stream.
    
    Args:
        source: Either a file path or an open file-like object containing
               YAML or JSON level definition.
               
    Returns:
        A validated LevelDefinition object.
        
    Raises:
        ValueError: If the level definition is invalid.
        FileNotFoundError: If the source file doesn't exist.
    """
    if isinstance(source, (str, Path)):
        with open(source, 'r') as f:
            return _load_from_stream(f)
    else:
        return _load_from_stream(source)


def _load_from_stream(stream: TextIO) -> LevelDefinition:
    """Load level definition from an open file stream."""
    try:
        content = yaml.safe_load(stream)
        return LevelDefinition.model_validate(content)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format: {e}")
    except ValidationError as e:
        raise ValueError(f"Invalid level definition: {e}")


def save_level(level: LevelDefinition, target: Union[str, Path, TextIO], 
               format: str = 'yaml') -> None:
    """Save a level definition to a file or stream.
    
    Args:
        level: The LevelDefinition object to save.
        target: Either a file path or an open file-like object.
        format: The format to save in ('yaml' or 'json').
        
    Raises:
        ValueError: If the format is invalid.
        IOError: If there's an error writing to the file.
    """
    if format.lower() not in ('yaml', 'json'):
        raise ValueError(f"Invalid format: {format}. Must be 'yaml' or 'json'.")
    
    level_dict = level.model_dump(exclude_none=True, mode="json")
    
    if isinstance(target, (str, Path)):
        with open(target, 'w') as f:
            _save_to_stream(level_dict, f, format.lower())
    else:
        _save_to_stream(level_dict, target, format.lower())


def _save_to_stream(level_dict: Dict[str, Any], stream: TextIO, format: str) -> None:
    """Save level definition to an open file stream."""
    if format == 'yaml':
        yaml.dump(level_dict, stream, default_flow_style=False, sort_keys=False)
    else:  # json
        json.dump(level_dict, stream, indent=2, cls=EnumJSONEncoder)


def validate_against_schema(level_dict: Dict[str, Any], schema_path: Optional[str] = None) -> bool:
    """Validate a level definition against the JSON schema.
    
    Args:
        level_dict: The level definition as a dictionary.
        schema_path: Path to the JSON schema file. If None, will use the built-in schema.
        
    Returns:
        True if validation passes.
        
    Raises:
        jsonschema.exceptions.ValidationError: If validation fails.
        FileNotFoundError: If the schema file doesn't exist.
    """
    if schema_path is None:
        # Try to find the schema in the standard locations
        candidates = [
            os.path.join(os.path.dirname(__file__), "../../../cyberwave-static/schemas/level.json"),
            os.path.join(os.path.dirname(__file__), "../../static/schemas/level.json"),
            "/etc/cyberwave/schemas/level.json",
        ]
        
        for path in candidates:
            if os.path.exists(path):
                schema_path = path
                break
        
        if schema_path is None:
            raise FileNotFoundError("Cannot find level schema file")
    
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    jsonschema.validate(level_dict, schema)
    return True 