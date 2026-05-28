# Quickstart

This page is the shortest path from a fresh checkout to a runnable VBOGS
pipeline. Use it first, then jump into the detailed references when a stage
needs tuning.

## Prerequisites

- NVIDIA GPU and working NVIDIA Container Toolkit for Docker GPU access.
- Docker Compose v2.
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
this checkout; generated artifacts stay in Docker volumes.

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

## Download KITTI-360

VBOGS expects the KITTI-360 perspective stereo images, camera poses, and
calibration files in the container at `/workspace/VBOGS/data/KITTI-360`. In the
compose stack, that path is backed by the `KITTI-360` external Docker volume.

The quickstart smoke run below uses:

```text
2013_05_28_drive_0007_sync
```

First, get the official **KITTI-360 download links** for calibration and poses from
the [KITTI-360 download page](https://www.cvlibs.net/datasets/kitti-360/download.php)
after accepting the dataset terms. These links may be account- or
session-specific, so keep them out of committed files.

Run the downloader from the interactive `vbogs-pipeline` container shell:

```bash
export VBOGS_DRIVE=2013_05_28_drive_0007_sync
export KITTI_CALIBRATION_LINK='https://.../calibration.zip'
export KITTI_POSES_LINK='https://.../data_poses.zip'

bash data/download_kitti_360.sh
```

The helper writes into `/workspace/VBOGS/data/KITTI-360`, downloads the selected
drive's left and right perspective image archives when `KITTI_IMAGES` is not
set, extracts all archives, and normalizes them into the layout VBOGS expects:

```text
/workspace/VBOGS/data/KITTI-360/
  calibration/
    perspective.txt
  data_poses/
    2013_05_28_drive_0007_sync/
      cam0_to_world.txt
  images/
    2013_05_28_drive_0007_sync/
      image_00/data_rect/
      image_01/data_rect/
```

To use a different drive, set `VBOGS_DRIVE` to that drive id and pass the same
id to `scripts/run_drive_pipeline.py --drive`. See [Data Setup](data.md) for
the full layout, Docker volume notes, and alternate download helper.

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
