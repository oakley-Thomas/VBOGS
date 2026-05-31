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

From [KITTI-360 download page](https://www.cvlibs.net/datasets/kitti-360/download.php)
accept the dataset terms and get the official **KITTI-360 download links** for 

1.) Calibrations

2.) Vehicle Poses 

3.) Left/Right perspective images ("Test SLAM" recommended)


Run the downloader from the interactive `vbogs-pipeline` container shell:

```bash
export KITTI_CALIBRATION_LINK='https://.../calibration.zip'
export KITTI_POSES_LINK='https://.../data_poses.zip'
export KITTI_IMAGES_LINK='https://.../data_2d_test_slam.zip'

bash scripts/download_kitti_360.sh
```

Use the `scripts/` path inside Docker so the helper comes from this checkout;
`/workspace/VBOGS/data` is a persistent volume.

The helper writes into `/workspace/VBOGS/data/KITTI-360`, downloads the linked
left and right perspective image archive, extracts all archives, and
normalizes them into the layout VBOGS expects:

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

## Test Runs

Print the planned stage commands:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0004_sync \
  --use-service-labels \
  --dry-run
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
  --use-service-labels
```

## Realtime Visualization

After the smoke run has produced an Octree-AnyGS scene and `U.npy`, enter the
Torch container from the host:

```bash
docker compose --project-directory . \
  -f docker/compose/compose.yml \
  -f docker/compose/dev.yml \
  exec vbogs-torch bash
```

Then start the browser viewer from inside `vbogs-torch`:

```bash
python scripts/view_octree_anygs.py \
  --drive 2013_05_28_drive_0004_sync \
  --resolution 4
```

Open the viewer:

```text
http://localhost:8070
```

The dev compose overlay maps `${VBOGS_VIEWER_PORT:-8070}` on the host to port
`8070` in `vbogs-torch`. Use `--rgb-only` when you only want to inspect the
trained Octree-AnyGS scene before uncertainty artifacts exist:

```bash
python scripts/view_octree_anygs.py \
  --drive 2013_05_28_drive_0004_sync \
  --resolution 1
```

For more options, including explicit model paths, pose teleport, REST API
usage, rendered-anchor uncertainty queries, and capture scripts, see
[Realtime Viewer](../running/realtime-viewer.md).
