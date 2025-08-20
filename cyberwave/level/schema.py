"""Schema definitions for CyberWave level files.

This module contains Pydantic models for validating and working with level
definition files in the CyberWave platform.
"""

from enum import Enum
from typing import List, Dict, Optional, Union, Any, Literal
from pydantic import BaseModel, Field, model_validator


class ExportMode(str, Enum):
    """Controls how the level handles remote dependencies when exported."""
    PORTABLE = "portable"  # Embed all assets locally
    CONNECTED = "connected"  # Keep platform references
    HYBRID = "hybrid"  # Mix of local and platform assets


class SyncMode(str, Enum):
    """Controls how instances sync with the platform."""
    LIVE = "live"  # Always get latest state
    SNAPSHOT = "snapshot"  # Capture current state
    TEMPLATE = "template"  # Starting point with local overrides


class CoordinateSystem(str, Enum):
    """Coordinate system used for the level."""
    ENU = "ENU"  # East-North-Up
    NED = "NED"  # North-East-Down
    ECEF = "ECEF"  # Earth-Centered, Earth-Fixed


class UnitSystem(str, Enum):
    """Unit system used for measurements."""
    METERS = "meters"
    FEET = "feet"


class Settings(BaseModel):
    """Level export settings."""
    export_mode: ExportMode = Field(default=ExportMode.HYBRID)


class Metadata(BaseModel):
    """Level metadata information."""
    title: str
    id: str
    floor_number: Optional[int] = None
    description: Optional[str] = None
    coordinate_system: CoordinateSystem = Field(default=CoordinateSystem.ENU)
    units: UnitSystem = Field(default=UnitSystem.METERS)
    project_id: Optional[str] = None


class CatalogReference(BaseModel):
    """Reference information for an asset hosted in the catalog."""
    workspace: str
    asset_slug: str
    version: Optional[str] = None
    policy: Optional[str] = None


class OfflineFallback(BaseModel):
    """Paths to local assets used when the catalog asset is unavailable."""
    render: Optional[str] = None
    physics: Optional[str] = None
    thumb: Optional[str] = None


class AssetEntry(BaseModel):
    """Represents a single asset used in the level."""
    id: str
    source: Literal["local", "catalog"] = Field(default="local")
    type: str
    src: Optional[str] = None
    catalog_ref: Optional[CatalogReference] = None
    offline_fallback: Optional[OfflineFallback] = None

    @model_validator(mode="after")
    def _validate(cls, model: "AssetEntry") -> "AssetEntry":
        if model.source == "local":
            if not model.src:
                raise ValueError("Local asset must define 'src'")
        elif model.source == "catalog":
            if model.catalog_ref is None:
                raise ValueError("Catalog asset must define 'catalog_ref'")
        return model


class InstanceAsset(BaseModel):
    """Reference to an existing asset instance in the CyberWave platform."""
    id: str
    instance_id: str
    type: str
    sync_mode: SyncMode = Field(default=SyncMode.LIVE)




class DirectionalLight(BaseModel):
    """Directional light source in the environment."""
    direction: List[float] = Field(..., min_length=3, max_length=3)
    intensity: float


class Lighting(BaseModel):
    """Environment lighting settings."""
    ambient: Optional[float] = None
    directional: Optional[List[DirectionalLight]] = None


class Environment(BaseModel):
    """Environment settings for the level."""
    map_id: Optional[str] = None
    lighting: Optional[Lighting] = None


class Transform(BaseModel):
    """3D transform information for an entity."""
    position: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    rotation: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    scale: Optional[List[float]] = Field(None, min_length=3, max_length=3)


class Entity(BaseModel):
    """An entity in the level (robot, fixed asset, etc.)."""
    id: str
    archetype: str
    reference: Optional[str] = None
    transform: Optional[Transform] = None
    properties: Optional[Dict[str, Any]] = None
    capabilities: Optional[List[str]] = None
    status: Optional[str] = None
    battery_percentage: Optional[float] = None


class GeoGeometry(BaseModel):
    """Geometry for a zone using GeoJSON-like format."""
    type: str
    coordinates: List[Any]


class Zone(BaseModel):
    """A zone in the level (operational area, storage, etc.)."""
    id: str
    name: str
    type: str
    geometry: GeoGeometry
    properties: Optional[Dict[str, Any]] = None


class LevelDefinition(BaseModel):
    """Complete level definition for a CyberWave environment."""
    version: str
    settings: Optional[Settings] = None
    metadata: Metadata
    assets: Optional[List[AssetEntry]] = None
    instances: Optional[List[InstanceAsset]] = None
    environment: Optional[Environment] = None
    entities: Optional[List[Entity]] = None
    zones: Optional[List[Zone]] = None
