"""Cyberwave SDK data layer — transport-agnostic pub/sub for edge data.

Public API::

    from cyberwave.data import DataBus, get_backend

    backend = get_backend()
    bus = DataBus(backend, twin_uuid="...")
    bus.publish("frames", frame_array)
    latest = bus.latest("depth", max_age_ms=50)
    sub = bus.subscribe("joint_states", on_joints)

Or via the high-level client::

    cw = Cyberwave(api_key="...")
    cw.data.publish("frames", frame_array)

Debug / replay::

    from cyberwave.data.recording import record, replay

    with record(backend, ["frames/default"], "/tmp/session"):
        ...  # live samples captured to disk

    replay(backend, "/tmp/session", speed=1.0)

Time-aware fusion (Phases 3–4)::

    from cyberwave.data import FusionLayer

    fusion = FusionLayer()
    fusion.ingest("joint_states", ts=1.0, value=[0.1, 0.2, 0.3])
    joints = fusion.at("joint_states", t=1.5, interpolation="linear")
    window = fusion.window("imu", from_t=0.0, to_t=2.0)
"""

from .api import DataBus
from .backend import DataBackend, Sample, Subscription
from .config import (
    BackendConfig,
    get_backend,
    is_mqtt_publish_enabled,
    is_zenoh_publish_enabled,
)
from .exceptions import (
    BackendConfigError,
    BackendUnavailableError,
    ChannelError,
    DataBackendError,
    PublishError,
    RecordingError,
    SubscriptionError,
    WireFormatError,
)
from .fusion import (
    ChannelBuffer,
    FusionLayer,
    Quaternion,
    WindowResult,
    interpolate_linear,
    interpolate_nearest,
    interpolate_slerp,
)
from .header import (
    CONTENT_TYPE_BYTES,
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_NUMPY,
    HeaderMeta,
    HeaderTemplate,
    decode,
    encode,
)
from .keys import (
    COMMAND_CHANNELS,
    FILTERED_FRAME_CHANNEL,
    FRAME_OVERLAY_CHANNEL,
    LATEST_VALUE_CHANNELS,
    STREAM_CHANNELS,
    WELL_KNOWN_CHANNELS,
    KeyExpression,
    build_key,
    build_keys,
    build_wildcard,
    channel_from_key,
    is_valid_key,
    parse_key,
)
from .recording import RecordingSession, ReplayResult, record, replay
from .recording_format import RecordingManifest
from .ring_buffer import BracketResult, TimeIndexedRingBuffer, TimestampedSample

__all__ = [
    # Public facade
    "DataBus",
    # Backend
    "DataBackend",
    "Sample",
    "Subscription",
    "BackendConfig",
    "get_backend",
    "is_zenoh_publish_enabled",
    "is_mqtt_publish_enabled",
    # Exceptions
    "DataBackendError",
    "BackendUnavailableError",
    "BackendConfigError",
    "ChannelError",
    "PublishError",
    "RecordingError",
    "SubscriptionError",
    "WireFormatError",
    # Wire format / header
    "CONTENT_TYPE_NUMPY",
    "CONTENT_TYPE_JSON",
    "CONTENT_TYPE_BYTES",
    "HeaderMeta",
    "HeaderTemplate",
    "encode",
    "decode",
    # Key expressions
    "KeyExpression",
    "build_key",
    "build_keys",
    "build_wildcard",
    "parse_key",
    "channel_from_key",
    "is_valid_key",
    "COMMAND_CHANNELS",
    "STREAM_CHANNELS",
    "LATEST_VALUE_CHANNELS",
    "WELL_KNOWN_CHANNELS",
    "FILTERED_FRAME_CHANNEL",
    "FRAME_OVERLAY_CHANNEL",
    # Recording / replay
    "RecordingSession",
    "ReplayResult",
    "RecordingManifest",
    "record",
    "replay",
    # Ring buffer (CYB-1584)
    "TimeIndexedRingBuffer",
    "TimestampedSample",
    "BracketResult",
    # Fusion (CYB-1584)
    "FusionLayer",
    "ChannelBuffer",
    "Quaternion",
    "WindowResult",
    "interpolate_linear",
    "interpolate_slerp",
    "interpolate_nearest",
]
