# Cyberwave SDK Examples

Runnable scripts demonstrating the Cyberwave Python SDK. Most require an API key:

```bash
pip install cyberwave
export CYBERWAVE_API_KEY="your_api_key_here"
python examples/quickstart.py
```

Adjust asset slugs (`the-robot-studio/so101`, `unitree/go2`, …) to match your workspace
catalog. Some examples need optional extras (e.g. `cyberwave[camera]`, `cyberwave[ml]`,
`cyberwave[zenoh]`) — see the [installation docs](https://docs.cyberwave.com/overview).

## Getting started

| Script | Description |
| --- | --- |
| [quickstart.py](quickstart.py) | Create a twin, scene layout, joints, and locomotion in one script |
| [get_started.py](get_started.py) | Minimal "create a twin" starter |
| [compact.py](compact.py) | The compact module-level API (one-liners) |

## Twins · joints · locomotion

| Script | Description |
| --- | --- |
| [joints.py](joints.py) | Read/write joint positions by name |
| [actuate_arm.py](actuate_arm.py) | Drive an arm through joint targets |
| [locomotion.py](locomotion.py) | Velocity-style locomotion (burst + stop) |
| [commands_catalog.py](commands_catalog.py) | Asset-declared MQTT command catalog |
| [runtime_mode.py](runtime_mode.py) | Switch between simulation and live with `cw.affect()` |
| [policy_assign.py](policy_assign.py) | Attach a teleop controller policy |
| [get_pose.py](get_pose.py) | Read joint-space and Cartesian pose |
| [robot_startup_and_readiness.py](robot_startup_and_readiness.py) | Startup and readiness checks |
| [missions.py](missions.py) | Run saved missions |

## Frames & cameras

| Script | Description |
| --- | --- |
| [capture_frame.py](capture_frame.py) | Grab a single frame from a twin |
| [camera_stream.py](camera_stream.py) | Stream a USB/webcam feed over WebRTC |
| [realsense_stream.py](realsense_stream.py) | Intel RealSense RGB + depth streaming |
| [webcam_pose_anonymize.py](webcam_pose_anonymize.py) | Local webcam with frame filters |

## Streaming & audio

| Script | Description |
| --- | --- |
| [audio_stream.py](audio_stream.py) | Stream microphone audio to a twin |
| [multimedia_stream.py](multimedia_stream.py) | Combined video + audio streaming |
| [listen_zenoh_audio.py](listen_zenoh_audio.py) ([.sh](listen_zenoh_audio.sh)) | Subscribe to a twin's audio over Zenoh |

## Drones & flight

| Script | Description |
| --- | --- |
| [drone_hovering.py](drone_hovering.py) | Takeoff, hover, and land |
| [drone_dji_mini.py](drone_dji_mini.py) | DJI Mini 4 Pro flight + gimbal |
| [flying.py](flying.py) | General flight helpers |

## Inbound MQTT & receivers

| Script | Description |
| --- | --- |
| [listen_mqtt.py](listen_mqtt.py) | Subscribe to inbound twin telemetry |
| [command_receiver_simple.py](command_receiver_simple.py) | Receive commands on the edge (simple) |
| [command_receiver_advanced.py](command_receiver_advanced.py) | Receive commands on the edge (advanced) |

## Edge · Zenoh · workers

| Script | Description |
| --- | --- |
| [edge_worker_detect_people.py](edge_worker_detect_people.py) | Edge ML worker: person detection |
| [edge_worker_hooks.py](edge_worker_hooks.py) | Edge worker lifecycle hooks |
| [edge_worker_multi_camera.py](edge_worker_multi_camera.py) | Multi-camera edge worker |
| [zenoh_triad.py](zenoh_triad.py) | Zenoh pub/sub triad |
| [zenoh_fanout.py](zenoh_fanout.py) | Zenoh fan-out |
| [zenoh_bench.py](zenoh_bench.py) | Zenoh throughput benchmark |
| [zenoh_data_fusion.py](zenoh_data_fusion.py) | Time-aware multi-sensor fusion |
| [zenoh_data_recording.py](zenoh_data_recording.py) | Record Zenoh channels to disk |
| [inference_bench.py](inference_bench.py) | Inference throughput benchmark |

## Workflows · datasets · alerts

| Script | Description |
| --- | --- |
| [workflows.py](workflows.py) | List, trigger, and poll workflow runs |
| [datasets.py](datasets.py) | Import, convert, and export robotics datasets |
| [alerts.py](alerts.py) | Create and manage alerts |
| [alerts_example.py](alerts_example.py) | End-to-end alert lifecycle |

## Machine learning (Colab notebooks)

| Notebook / script | Description |
| --- | --- |
| [ai/yolo.ipynb](ai/yolo.ipynb) | YOLO tasks: detection, segmentation, pose, classification |
| [ai/yolo_minimal.ipynb](ai/yolo_minimal.ipynb) | Minimal YOLO inference |
| [ai/yolo_minimal_on_twin.ipynb](ai/yolo_minimal_on_twin.ipynb) | Live twin frame → predict |
| [ai/models.py](ai/models.py) | Load and run catalog models |
| [ai/model_predict_on_twin.py](ai/model_predict_on_twin.py) | Predict on a live twin frame |

## End-to-end demos

| Demo | Description |
| --- | --- |
| [nl_arm_controller/](nl_arm_controller/) | Natural-language voice control of a robot arm |
| [so101_mujoco_demo/](so101_mujoco_demo/) | SO-101 arm in MuJoCo simulation |
| [ur7-santas-little-helper.py](ur7-santas-little-helper.py) | UR-style arm demo |
