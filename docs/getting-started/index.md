# Quickstart

This page explains how to get up and running with the VBOGS pipeline

## Prerequisites

- NVIDIA GPU and working NVIDIA Container Toolkit for Docker GPU access.

## Build the Stack
By default, ```scripts/build_stack_serial.sh``` will compile against the host machine's CUDA architecture. 
```bash
bash scripts/build_stack_serial.sh
```

## Start the Stack

The dev overlay bind-mounts the local VBOGS code checkout.

**Start the stack:**
```bash
./dc_up.sh
```
**Stop the stack:**
```bash
./dc_down.sh
```
**Enter a container:**
```bash
./dc_bash.sh # Default entrypoint is vbogs-pipeline container

# To enter another container pass it as an argument
# Example:
# ./dc_bash.sh vbogs-vbgs-render
```

## Downloading the Datasets

VBOGS currently supports KITTI-360, NVIDIA NCore, and DJI Osmo 360 inputs. Follow the instructions [here](data.md).

## NVIDIA NCore Test Run

From inside `vbogs-pipeline`, use the downloaded clip UUID as `--scene-id` and
select the NCore dataset adapter:

List downloaded clips first if you do not remember which UUIDs are available:

```bash
python scripts/list_dataset_clips.py --dataset-name nvidia_ncore --ready-only --commands
```

```bash
scripts/run_pipeline.sh \
  --config configs/pipeline/nvidia_ncore_dev.yaml \
  --dataset-name nvidia_ncore \
  --scene-id 00b769dd-b4fa-4d88-ba4e-e6a230ff0c66 \
  --gpu 0 \
  --jax-device 0 \
  --start-at prepare \
  --stop-after render \
  --frame-step 2 \
  --max-frames 30 \
  --resolution 4 \
  --iterations 7000 \
  --max-points-per-frame 50000 \
  --render-max-views 2
```

For a full NCore run:

```bash
scripts/run_pipeline.sh \
  --config configs/pipeline/nvidia_ncore_dev.yaml \
  --dataset-name nvidia_ncore \
  --scene-id 00b769dd-b4fa-4d88-ba4e-e6a230ff0c66 \
  --gpu 0 \
  --jax-device 0 \
  --start-at prepare \
  --stop-after bundle
```

`--drive` by itself is the KITTI-360 path. If the run header says
`Dataset: kitti360`, stop and rerun with `--dataset-name nvidia_ncore`.

## KITTI-360 Test Run

From inside `vbogs-pipeline`, run a quick end-to-end check:

```bash
scripts/run_pipeline.sh \
  --drive 2013_05_28_drive_0004_sync \
  --gpu 0 \
  --jax-device 0 \
  --start-at prepare \
  --stop-after render \
  --frame-step 20 \
  --max-frames 30 \
  --resolution 4 \
  --iterations 7000 \
  --max-points-per-frame 50000 \
  --render-max-views 2
```

For a full run:

```bash
scripts/run_pipeline.sh \
  --drive 2013_05_28_drive_0004_sync \
  --gpu 0 \
  --jax-device 0 \
  --start-at prepare \
  --stop-after bundle
```

## Experiment Guides

Experiment-specific runbooks live under the docs' Experiments section:

- [Experiment 02: 3D Baseline Drive Sweep](../experiments/experiment02.md)
- [Experiment 03: Stereo Training Comparison](../experiments/experiment03.md)

## Realtime Navigation and Visualization

Navigate the Octree-AnyGS representation and visualize the Uncertainty Anchor Map

For a step-by-step workflow that downloads a completed server run, extracts the
portable viewer package locally, starts the render container, and queries
uncertainty scores, use [Download and View Artifacts](local-artifact-viewing.md).

Enter the render server container
```bash
# Enter the container
./dc_bash.sh vbogs-vbgs-render

# Start the render server for a scene available in the stack volumes
python scripts/view_octree_anygs.py \
  --drive 2013_05_28_drive_0004_sync \
  --resolution 1
```

In a browser visit:

```text
http://localhost:8071
```

When a pipeline run reaches `bundle`, it also writes a portable local viewer
package inside the scene bundle:

```text
outputs/v1_0/<scene-id>/local_viewer/
outputs/v1_0/<scene-id>/<scene-id>.zip
```

Download the scene zip from File Browser, extract it on a machine with a
checked-out VBOGS repo, then follow the generated `VIEWER_COMMANDS.md` inside
the extracted `<scene-id>/local_viewer` folder. That package contains the
Octree-AnyGS checkpoint, prepared camera metadata, and `uncertainty/U.npy`
needed to render locally and query uncertainty through the viewer API.

### Query the render server API

The same server exposes REST endpoints for programmatic rendering. From the
host machine, use `http://localhost:8071`; from inside the container, use
`http://localhost:8070`.

Inspect the loaded scene and available camera IDs:

```bash
curl http://localhost:8071/api/metadata
curl http://localhost:8071/api/cameras
```

Render an RGB JPEG for a camera already known to the viewer:

```bash
curl -X POST http://localhost:8071/api/render \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id": "rgb-example",
    "camera_id": "test:0",
    "layer": "rgb",
    "quality": 90
  }'
```

Render an uncertainty heatmap JPEG from the same camera:

```bash
curl -X POST http://localhost:8071/api/render \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id": "uncertainty-example",
    "camera_id": "test:0",
    "layer": "uncertainty",
    "quality": 90
  }'
```

`/api/render` returns JSON with `metadata` and a `jpeg_base64` field. Decode
that field to bytes to save the image as a `.jpg`.

Query the uncertainty score for a camera pose:

```bash
curl -X POST http://localhost:8071/api/rendered-anchors \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id": "score-example",
    "camera_id": "test:0",
    "max_anchors": 25
  }'
```

The score response includes `uncertainty_image_sum`, `alpha_sum`, and
`alpha_normalized_uncertainty`, where the normalized score is the value used for
NBV ranking. Add `"pose": "x y z yaw pitch roll"` to either POST body to query
a custom pose instead of the saved camera pose.

See the full [render server API reference](../running/realtime-viewer.md) for
all layers, pose formats, response fields, and capture helpers.


## Browse Files and Artifacts

Open the web-based file browser

```text
http://localhost:8088
```

From ```vbogs-pipeline``` get the filebrowser credentials, you may need to refresh the page after running the following command

```bash
python scripts/get_filebrowser_login.py --reset-password
```
