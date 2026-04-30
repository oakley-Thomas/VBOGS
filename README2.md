# VBOGS
Combining Octree-GS's scene scalability with Variational Bayes GS uncertainty for better autonomous vehicle mapping

## Environment Notes

VBOGS uses two separate docker images, each with a corresponding conda environment

- `vbogs-torch` for Octree-AnyGS, stereo, point bucketing, and rendering
- `vbogs-jax` for VBGS fitting and posterior computations

Build the images locally
```bash
docker compose build vbogs-torch vbogs-jax
```
or pull them from DockerHub
```bash
oakleyth/vbogs-vbogs-torch:latest
oakleyth/vbogs-vbogs-jax:latest
```

Run the docker compose
```bash
export VBOGS_TORCH_IMAGE=<name-of-torch-image>
export VBOGS_JAX_IMAGE=<name-of-jax-image>

# Example (if pulled from DockerHub)
# export VBOGS_TORCH_IMAGE=oakleyth/vbogs-vbogs-torch:latest
# export VBOGS_JAX_IMAGE=oakleyth/vbogs-vbogs-jax:latest

# Add external volumes (if they do not exist)
# docker volume create KITTI-360
# docker volume create COLMAP
# docker volume create OCTREE-ANYGS

# Start stack
docker compose up -d

# Stop stack
docker compose down
```

## Usage

### 1. Clone this repo (it is already git checked out in the images mentioned above)

### 2. Download the dataset:
```bash
export KITTI_CALIBRATION_LINK=<link-to-kitti-360-calibration>
export KITTI_POSES_LINK=<link-to-kitti-360-poses>
export KITTI_IMAGES=<link-to-kitti-360-images>

# data will be downloaded to VBOGS/data/KITTI-360
# archives are normalized to:
#   data/KITTI-360/calibration/
#   data/KITTI-360/data_poses/
#   data/KITTI-360/images/
cd data/
./download_kitti_360.sh
```

### 3. Run Octree-AnyGS Training

**Local conda workflow:** Octree-AnyGS training runs are written to `data/OCTREE-ANYGS/$DRIVE/<timestamp>/`.

```bash
DRIVE=2013_05_28_drive_0009_sync

python scripts/prepare_kitti360_colmap.py \
  --drive "$DRIVE" \
  --frame-step 10 \
  --max-frames 160 \
  --output-root data/COLMAP

python scripts/train_octree_anygs.py \
  --dataset-path "data/COLMAP/$DRIVE" \
  --output-root data/OCTREE-ANYGS \
  --gpu 0
```

**Docker workflow:** Octree-AnyGS training runs are written to `/data/OCTREE-ANYGS/$DRIVE/<timestamp>/`.
```bash
DRIVE=2013_05_28_drive_0009_sync

python scripts/check_torch_stack.py --repo-root /workspace/VBOGS

python scripts/prepare_kitti360_colmap.py \
  --drive "$DRIVE" \
  --frame-step 10 \
  --max-frames 160

python scripts/train_octree_anygs.py \
  --dataset-path "/data/COLMAP/$DRIVE" \
  --gpu 0
```

### 4. Create the Colored Point Cloud

This step creates the M3 stereo point-cloud artifact used by anchor bucketing.
It runs in the torch environment and writes:

- `data/points_world/$DRIVE/points_world.npz` with `xyz`, `rgb`, and `frame_id`
- `data/points_world/$DRIVE/points_world_metadata.json`
- `data/points_world/$DRIVE/points_world.ply` when `--write-ply` is passed

Full run, with a colored PLY sidecar for viewer inspection:

```bash
DRIVE=2013_05_28_drive_0009_sync

python scripts/stereo_to_pointcloud.py \
  --drive "$DRIVE" \
  --selection-metadata "data/COLMAP/$DRIVE/metadata.json" \
  --write-ply
```

Dev-machine friendly run:

```bash
DRIVE=2013_05_28_drive_0009_sync

python scripts/stereo_to_pointcloud.py \
  --drive "$DRIVE" \
  --selection-metadata "data/COLMAP/$DRIVE/metadata.json" \
  --max-frames 40 \
  --pixel-step 2 \
  --max-points-per-frame 50000 \
  --max-depth-m 60
```

Useful size/runtime controls:

- `--max-frames N` limits how many selected frames are processed.
- `--pixel-step N` keeps every Nth valid pixel in x and y.
- `--max-points-per-frame N` caps retained points per frame after filtering.
- `--max-depth-m M` removes far stereo points before export.
- `--write-ply` writes a colored `.ply`; omit it on memory-limited runs.

For a quick visual check on a dev machine, use the friendly command first, then
add `--write-ply` only after the point count looks reasonable.
