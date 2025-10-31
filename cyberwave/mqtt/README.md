# Cyberwave MQTT Client

## Overview

This module provides a high-level MQTT client for real-time communication with the Cyberwave platform. It uses `paho-mqtt` (2.1.0+) for reliable MQTT connectivity.

## Features

- **Digital Twin Updates**: Subscribe and publish position, rotation, and scale updates
- **Joint States**: Real-time joint state updates for robotic twins
- **Sensor Streams**: Video, depth, and point cloud data streaming
- **WebRTC Signaling**: WebRTC connection setup via MQTT
- **Rate Limiting**: Built-in rate limiting to prevent message flooding
- **Duplicate Prevention**: Automatic filtering of duplicate position/rotation updates
- **Reconnection**: Automatic reconnection handling

## Installation

The MQTT client is included in the main Cyberwave SDK:

```bash
pip install cyberwave
```

## Usage

### Basic Connection

```python
from cyberwave.mqtt import CyberwaveMQTTClient

# Create and connect to MQTT broker
client = CyberwaveMQTTClient(
    mqtt_broker="mqtt.cyberwave.com",
    mqtt_port=1883,
    api_token="your_api_token"
)

# The client auto-connects by default
# To prevent auto-connect, pass auto_connect=False
```

### Subscribe to Twin Updates

```python
def on_position_update(data):
    position = data["position"]
    print(f"Position: x={position['x']}, y={position['y']}, z={position['z']}")

# Subscribe to twin position updates
client.subscribe_twin("twin_uuid", on_update=on_position_update)
```

### Publish Twin Position

```python
# Update twin position
client.update_twin_position(
    twin_uuid="twin_uuid",
    position={"x": 1.0, "y": 2.0, "z": 3.0}
)
```

### Update Joint States

```python
# Update a single joint
client.update_joint_state(
    twin_uuid="twin_uuid",
    joint_name="shoulder_joint",
    position=1.57,  # radians
    velocity=0.1,
    effort=5.0
)
```

### Subscribe to Joint State Updates

```python
def on_joint_update(data):
    joint_name = data["joint_name"]
    joint_state = data["joint_state"]
    print(f"Joint {joint_name}: {joint_state}")

client.subscribe_twin_joint_states(
    twin_uuid="twin_uuid",
    on_update=on_joint_update
)
```

### Custom Topic Prefix

For custom deployments with topic prefixes:

```python
client = CyberwaveMQTTClient(
    mqtt_broker="mqtt.cyberwave.com",
    api_token="your_api_token",
    topic_prefix="custom/prefix/"
)
```

### Disconnect

```python
# Clean disconnect
client.disconnect()
```

## Configuration Options

| Parameter       | Type   | Default        | Description                                                     |
| --------------- | ------ | -------------- | --------------------------------------------------------------- |
| `mqtt_broker`   | `str`  | Required       | MQTT broker hostname or IP                                      |
| `mqtt_port`     | `int`  | `1883`         | MQTT broker port                                                |
| `mqtt_username` | `str`  | `"cyberwave"`  | MQTT username                                                   |
| `mqtt_password` | `str`  | `None`         | MQTT password                                                   |
| `api_token`     | `str`  | `None`         | Cyberwave API token (used as password if mqtt_password not set) |
| `client_id`     | `str`  | Auto-generated | Custom MQTT client ID                                           |
| `topic_prefix`  | `str`  | `""`           | Prefix for all MQTT topics                                      |
| `auto_connect`  | `bool` | `True`         | Automatically connect on initialization                         |

## Advanced Features

### Rate Limiting

The client automatically rate-limits updates to 40 Hz (25ms interval) per resource to prevent message flooding:

```python
# These will be rate-limited automatically
for i in range(1000):
    client.update_twin_position(
        twin_uuid="twin_uuid",
        position={"x": i, "y": 0, "z": 0}
    )
    # Only ~40 messages per second will be sent
```

### Duplicate Prevention

Position and rotation updates that haven't changed are automatically filtered:

```python
# First update - sent
client.update_twin_position("twin_uuid", {"x": 1.0, "y": 0, "z": 0})

# Second update with same position - skipped
client.update_twin_position("twin_uuid", {"x": 1.0, "y": 0, "z": 0})

# Third update with different position - sent
client.update_twin_position("twin_uuid", {"x": 2.0, "y": 0, "z": 0})
```

## Topic Structure

The client uses the following topic structure:

- Twin position: `{prefix}cyberwave/twin/{twin_uuid}/position`
- Twin rotation: `{prefix}cyberwave/twin/{twin_uuid}/rotation`
- Twin scale: `{prefix}cyberwave/twin/{twin_uuid}/scale`
- Joint states: `{prefix}cyberwave/joint/{twin_uuid}/update`
- Video stream: `{prefix}cyberwave/twin/{twin_uuid}/video`
- Depth stream: `{prefix}cyberwave/twin/{twin_uuid}/depth`
- Point cloud: `{prefix}cyberwave/twin/{twin_uuid}/pointcloud`
- WebRTC: `{prefix}cyberwave/twin/{twin_uuid}/webrtc`
- Ping/Pong: `{prefix}cyberwave/ping|pong/{resource_uuid}/request|response`

## Compatibility

- Python 3.9+
- paho-mqtt 2.1.0+ (with backward compatibility for 1.x)
