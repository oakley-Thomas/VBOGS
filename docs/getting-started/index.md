# Quickstart

This page is the shortest path from a fresh checkout to a runnable VBOGS
pipeline. Use it first, then jump into the detailed references when a stage
needs tuning.

## Prerequisites

- NVIDIA GPU and working NVIDIA Container Toolkit for Docker GPU access.
- Docker Compose v2.
- KITTI-360 perspective stereo images, poses, and calibration files available
  under `data/KITTI-360/` or mounted into the same path in the containers.
- The two submodules checked out: `Octree-AnyGS/` and `vbgs/`.

## Build

The CUDA images are intentionally built serially. This avoids overlapping
large CUDA wheel downloads and compiles on smaller local machines.

```bash
bash scripts/build_stack_serial.sh
```

To rebuild one service:

```bash
bash scripts/build_stack_serial.sh vbogs-torch
bash scripts/build_stack_serial.sh vbogs-jax
bash scripts/build_stack_serial.sh vbogs-pipeline
```

## Start Local Containers

Use the base compose file plus the dev overlay. The dev overlay bind-mounts
this checkout and maps local `outputs/` into the containers.

```bash
docker compose --project-directory . \
  -f docker/compose/compose.yml \
  -f docker/compose/dev.yml \
  up -d --no-build
```

Then enter the pipeline container:

```bash
docker compose --project-directory . \
  -f docker/compose/compose.yml \
  -f docker/compose/dev.yml \
  exec vbogs-pipeline bash
```

Inside the container, confirm GPU visibility:

```bash
nvidia-smi
```

## Dry Run

Before running expensive work, print the planned stage commands:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --use-service-labels \
  --dry-run
```

## Small Smoke Run

This keeps training, stereo, fitting, and rendering small enough for a quick
end-to-end check:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
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
  --use-service-labels
```

## Build This Documentation Site

Install MkDocs in whichever Python environment you use for docs:

```bash
python -m pip install -r docs/requirements.txt
```

Serve the site locally:

```bash
python -m mkdocs serve
```

Build the static site:

```bash
python -m mkdocs build
```
