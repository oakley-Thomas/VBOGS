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

VBOGS currently supports both the KITTI-360 dataset and the NVIDIA-NCore dataset. Follow the instructions [here](data.md).

## NVIDIA NCore Test Run

From inside `vbogs-pipeline`, use the downloaded clip UUID as `--scene-id` and
select the NCore dataset adapter:

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

Enter the render server container
```bash
# Enter the container
./dc_bash.sh vbogs-vbgs-render

# Start the render server
python scripts/view_octree_anygs.py \
  --drive 2013_05_28_drive_0004_sync \
  --resolution 1

# Start the render server (server trained scene)
python /workspace/VBOGS/scripts/view_octree_anygs.py \
  --model-path /workspace/VBOGS/outputs/2013_05_28_drive_0004_sync/model \
  --u-path /workspace/VBOGS/outputs/2013_05_28_drive_0004_sync/uncertainty/U.npy \
  --iteration 90000 \
  --resolution 2 \
  --port 8070 \
  --octree-root /workspace/VBOGS/Octree-AnyGS
```

In a browser visit: 
```http://localhost:8071```


## Browse Files and Artifacts

Open the web-based file browser

```text
http://localhost:8088
```

From ```vbogs-pipeline``` get the filebrowser credentials, you may need to refresh the page after running the following command

```bash
python scripts/get_filebrowser_login.py --reset-password
```


