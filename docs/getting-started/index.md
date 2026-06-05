# Quickstart

This page explains how to get up and running with the VBOGS pipeline

## Prerequisites

- NVIDIA GPU and working NVIDIA Container Toolkit for Docker GPU access.

## Build
```bash
bash scripts/build_stack_serial.sh
```

To rebuild one service:
```bash
bash scripts/build_stack_serial.sh vbogs-torch
bash scripts/build_stack_serial.sh vbogs-jax
bash scripts/build_stack_serial.sh vbogs-vbgs-render
bash scripts/build_stack_serial.sh vbogs-pipeline
```
Use `--no-cache` to rebuild from scratch

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


















## Test Runs

Print the planned stage commands:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0004_sync \
  --dry-run \
  --compose-file docker/compose/compose.yml \
  --compose-file docker/compose/dev.yml \
  --compose-project-directory .
```

Quick end-to-end check:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
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
  --render-max-views 2 \
  --compose-file docker/compose/compose.yml \
  --compose-file docker/compose/dev.yml \
  --compose-project-directory .
```

## Realtime Visualization

After the smoke run has produced an Octree-AnyGS scene and `U.npy`, enter the
render container from the host:

```bash
docker compose --project-directory . \
  -f docker/compose/compose.yml \
  -f docker/compose/dev.yml \
  exec vbogs-vbgs-render bash
```

Then start the browser viewer from inside `vbogs-vbgs-render`:

```bash
python scripts/view_octree_anygs.py \
  --drive 2013_05_28_drive_0004_sync \
  --resolution 4
```

Open the viewer:

```text
http://localhost:8071
```

The compose stack maps
`${VBOGS_RENDER_VIEWER_HOST_BIND:-0.0.0.0}:${VBOGS_RENDER_VIEWER_HOST_PORT:-8071}`
on the host to port `8070` in `vbogs-vbgs-render`. Use `--rgb-only` when you
only want to inspect the trained Octree-AnyGS scene before uncertainty artifacts
exist:

```bash
python scripts/view_octree_anygs.py \
  --drive 2013_05_28_drive_0004_sync \
  --resolution 1
```

For more options, including explicit model paths, pose teleport, REST API
usage, rendered-anchor uncertainty queries, and capture scripts, see
[Realtime Viewer](../running/realtime-viewer.md).


## Browse Files and Artifacts

The stack starts a read-only File Browser sidecar for project files and
artifacts:

```text
http://localhost:8088
```

On first boot, get the generated `admin` password from the service logs:

```bash
docker compose --project-directory . \
  -f docker/compose/compose.yml \
  -f docker/compose/dev.yml \
  logs vbogs-filebrowser
```

Confirm GPU visibility from a stack container:

```bash
docker compose --project-directory . \
  -f docker/compose/compose.yml \
  -f docker/compose/dev.yml \
  exec vbogs-pipeline nvidia-smi
```