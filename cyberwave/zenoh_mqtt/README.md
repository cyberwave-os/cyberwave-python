# Zenoh ↔ MQTT topic mapping

The bridge in this package forwards messages between Zenoh key expressions and
MQTT topics. `topic_mapping.py` holds the pure mapping logic — no broker, no
network, no robot — which makes it a good place to explore the SDK. Every
example below can be verified against `tests/test_bridge_topic_mapping.py`
without starting anything:

```bash
poetry run pytest tests/test_bridge_topic_mapping.py -v
```

## Vocabulary

- **Outbound** — Zenoh → MQTT. The bridge subscribes on Zenoh and publishes to MQTT.
- **Inbound** — MQTT → Zenoh. The bridge subscribes on MQTT and publishes to Zenoh.
- **Channel** — the Zenoh side's single segment after `data/` (e.g. `model_output`).
- **Sensor segment** — an optional extra Zenoh segment after the channel
  (e.g. `default` in `cw/{twin}/data/model_output/default`).
- **Suffix** — everything after the twin UUID on the MQTT side; it may contain
  slashes (e.g. `commands/sync_workflows`).
- **`mqtt_prefix`** — optional string *prepended* to the MQTT topic (no separator added).
- **`zenoh_prefix`** — first segment of the Zenoh key; defaults to `cw`.
- **Twin UUID** — the identity segment; both directions validate it against the
  canonical UUID format and reject anything else.

## Outbound: Zenoh → MQTT

The canonical Zenoh key has the shape `cw/{twin}/data/{channel}`:

```python
from cyberwave.zenoh_mqtt.topic_mapping import zenoh_to_mqtt

result = zenoh_to_mqtt("cw/12345678-1234-1234-1234-123456789abc/data/model_output")
# OutboundMapping(mqtt_topic="cyberwave/twin/12345678-1234-1234-1234-123456789abc/model_output",
#                 twin_uuid="12345678-1234-1234-1234-123456789abc",
#                 channel="model_output")
```

Sensor-qualified keys are accepted but the sensor segment is **dropped** — the
MQTT topic keeps only the channel:

```python
zenoh_to_mqtt("cw/12345678-1234-1234-1234-123456789abc/data/model_output/default")
# → cyberwave/twin/12345678-1234-1234-1234-123456789abc/model_output   (no `/default`)
```

Keys missing the `data` segment, carrying a non-UUID twin, or with an empty
channel are not canonical and return `None`:

```python
zenoh_to_mqtt("cw/not-a-uuid/data/event")   # None — twin is not a UUID
zenoh_to_mqtt("cw/12345678-1234-1234-1234-123456789abc/stream/event")  # None — no data segment
zenoh_to_mqtt("cw/12345678-1234-1234-1234-123456789abc/data/")         # None — empty channel
```

### MQTT prefix

`mqtt_prefix` is concatenated **without a separator**. `staging` becomes part of
the first segment:

```python
zenoh_to_mqtt("cw/12345678-1234-1234-1234-123456789abc/data/event", mqtt_prefix="staging")
# → stagingcyberwave/twin/12345678-1234-1234-1234-123456789abc/event
```

If your broker separates namespaces with a slash, pass it yourself
(`mqtt_prefix="staging/"`).

## Inbound: MQTT → Zenoh

The canonical MQTT topic has the shape `cyberwave/twin/{twin}/{suffix}`:

```python
from cyberwave.zenoh_mqtt.topic_mapping import mqtt_to_zenoh

result = mqtt_to_zenoh("cyberwave/twin/12345678-1234-1234-1234-123456789abc/commands/sync_workflows")
# InboundMapping(zenoh_key="cw/12345678-1234-1234-1234-123456789abc/data/commands_sync_workflows",
#                twin_uuid="12345678-1234-1234-1234-123456789abc",
#                channel="commands_sync_workflows")
```

Slashes in the suffix are flattened to underscores so the result is a valid
single-segment Zenoh channel:

```text
commands/sync_workflows  →  commands_sync_workflows
```

Single-segment suffixes pass through unchanged:

```python
mqtt_to_zenoh("cyberwave/twin/12345678-1234-1234-1234-123456789abc/event")
# → cw/12345678-1234-1234-1234-123456789abc/data/event
```

Both prefixes apply here too — `mqtt_prefix` is stripped when present, and the
Zenoh key starts with the configurable `zenoh_prefix` (default `cw`):

```python
mqtt_to_zenoh("stagingcyberwave/twin/12345678-1234-1234-1234-123456789abc/event",
              mqtt_prefix="staging")
# → cw/12345678-1234-1234-1234-123456789abc/data/event

mqtt_to_zenoh("cyberwave/twin/12345678-1234-1234-1234-123456789abc/event", zenoh_prefix="myprefix")
# → myprefix/12345678-1234-1234-1234-123456789abc/data/event
```

Topics with a wrong root, a non-UUID twin, or an empty suffix return `None`:

```python
mqtt_to_zenoh("other/twin/12345678-1234-1234-1234-123456789abc/event")  # None — wrong root
mqtt_to_zenoh("cyberwave/twin/bad-uuid/event")                          # None — twin is not a UUID
mqtt_to_zenoh("cyberwave/twin/12345678-1234-1234-1234-123456789abc/")    # None — empty suffix
```

## Why mapping is not a lossless round trip

Round-tripping one key through both directions loses information — by design:

1. **Outbound drops the sensor segment.** `cw/{twin}/data/model_output/default`
   publishes to `cyberwave/twin/{twin}/model_output`. Mapping that back inbound
   yields `cw/{twin}/data/model_output` — the sensor identity is gone.
2. **Inbound flattens nested suffixes.** `commands/sync_workflows` becomes the
   single channel `commands_sync_workflows`. The bridge cannot know whether an
   underscore in a channel name was originally a slash, so distinct MQTT topics
   that differ only in slash-vs-underscore collapse to the same Zenoh channel.

So `zenoh_to_mqtt` → `mqtt_to_zenoh` reproduces the channel for simple keys,
but neither direction preserves everything the other side expressed.

## Subscription builders

The module also offers two pure helpers that compute *what to subscribe to*,
using the same conventions:

```python
from cyberwave.zenoh_mqtt.topic_mapping import build_outbound_subscriptions, build_inbound_subscriptions

# Outbound: one exact key plus a wildcard for sensor-qualified keys, per channel
build_outbound_subscriptions(["12345678-1234-1234-1234-123456789abc"], ["model_output", "event"])
# ['cw/{twin}/data/model_output', 'cw/{twin}/data/model_output/*',
#  'cw/{twin}/data/event', 'cw/{twin}/data/event/*']

# Inbound: one MQTT topic filter per suffix
build_inbound_subscriptions(["12345678-1234-1234-1234-123456789abc"], ["commands/sync_workflows"])
# ['cyberwave/twin/{twin}/commands/sync_workflows']
```

Both honor `zenoh_prefix` / `mqtt_prefix` exactly like the mapping functions
above. See the docstrings in `topic_mapping.py` for the full parameter list.

## Where to go next

- `cyberwave/zenoh_mqtt/topic_mapping.py` — the implementation (~140 lines, no broker required to read).
- `tests/test_bridge_topic_mapping.py` — every behavior above, as executable spec.
- `cyberwave/zenoh_mqtt/bridge.py` — how the bridge wires these mappings to live Zenoh/MQTT sessions.
