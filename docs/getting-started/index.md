# Quickstart

This page explains how to get up and running with the VBOGS pipeline

## Prerequisites

- NVIDIA GPU and working NVIDIA Container Toolkit for Docker GPU access.

## Build
```bash
bash scripts/build_stack_serial.sh
```

**IMPORTANT NOTE:** by default, ```scripts/build_stack_serial.sh``` will compile ```gsplat``` against the CUDA architecture on the machine that builds the images. If you intend to deploy on a different CUDA architecture, you need to specify the supported versions using ```--cuda-arch-list```.

```bash
# Example - supports RTX 5080 (sm_12.0) and RTX Quadro 8000 (sm_7.5)
bash scripts/build_stack_serial.sh --cuda-arch-list '7.5;12.0'
```

To rebuild one service:
```bash
bash scripts/build_stack_serial.sh vbogs-torch
bash scripts/build_stack_serial.sh vbogs-jax
bash scripts/build_stack_serial.sh vbogs-vbgs-render
bash scripts/build_stack_serial.sh vbogs-pipeline
```
Use `--no-cache` to rebuild from scratch

### Publish to Dockerhub (optional)
To publish the built images to Docker Hub:
```bash
docker login
bash scripts/push_stack_images.sh <dockerhub-username> <version>
```

If Docker Hub returns a transient registry or auth `500` during a push, rerun
the failed service and any remaining services:

```bash
bash scripts/push_stack_images.sh <dockerhub-username> <version> vbogs-vbgs-render vbogs-pipeline
```

## Running Local Compose Stacks

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
# ./dc_bash.sh vbogs-jax
```

## Running Remote Compose Stack
Coming Soon!

## Downloading the Datasets

VBOGS currently supports both the KITTI-360 dataset and the NVIDIA-NCore dataset. Follow the instructions [here](data.md).



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

## Realtime Navigation and Visualization

Navigate the Octree-AnyGS representation and visualize the Uncertainty Anchor Map

Enter the render server container
```bash
# Enter the container
./dc_bash vbogs-vbgs-render

# Start the render server
python scripts/view_octree_anygs.py \
  --drive 2013_05_28_drive_0004_sync \
  --resolution 1
```

In a browser visit: 
```http://localhost:8071```


## Browse Files and Artifacts

Open the web-based file browser

```text
http://localhost:8088
```

From ```vbogs-pipeline``` get the filebrowser credentials

```bash
python scripts/get_filebrowser_login.py --reset-password
```
