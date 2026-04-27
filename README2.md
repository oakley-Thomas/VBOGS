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

0. Clone this repo (it is already git checked out in the images mentioned above)

1. Download the dataset:
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

2. Run Octree-AnyGS Training
```bash
DRIVE=2013_05_28_drive_0018_sync

# COLMAP SFM - outputs written to VBOGS/data/COLMAP/<$DRIVE>
python scripts/prepare_kitti360_colmap.py \
  --drive "$DRIVE" \
  --frame-step 10 \
  --max-frames 160

python scripts/train_octree_anygs.py \
  --dataset-path "data/COLMAP/$DRIVE" \
  --gpu 0
```

Octree-AnyGS training runs are written to `/data/OCTREE-ANYGS/$DRIVE/<timestamp>/`
by default.

3.
