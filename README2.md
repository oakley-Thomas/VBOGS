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

### 5. Assign Colored Points to Octree-AnyGS Buckets

This step assigns the verified colored point cloud to the trained
Octree-AnyGS anchors at every LOD level, then writes the normalized inputs used
by the VBGS per-anchor fitting step.

**Local conda workflow:**

```bash
DRIVE=2013_05_28_drive_0009_sync
MODEL_PATH="data/OCTREE-ANYGS/$DRIVE/<timestamp>"

python scripts/bucket_points.py \
  --drive "$DRIVE" \
  --model-path "$MODEL_PATH" \
  --points-world "data/points_world/$DRIVE/points_world.npz" \
  --output-root "data/m4/$DRIVE"
```

**Docker workflow:**

```bash
DRIVE=2013_05_28_drive_0009_sync
MODEL_PATH="/data/OCTREE-ANYGS/$DRIVE/<timestamp>"

python scripts/bucket_points.py \
  --drive "$DRIVE" \
  --model-path "$MODEL_PATH" \
  --points-world "data/points_world/$DRIVE/points_world.npz" \
  --output-root "data/m4/$DRIVE"
```

Replace `<timestamp>` with the Octree-AnyGS run directory from step 3. If you
omit `--model-path`, the script uses the latest run under
`/data/OCTREE-ANYGS/$DRIVE`.

This writes:

- `data/m4/$DRIVE/points_norm.npz`
- `data/m4/$DRIVE/pts_by_anchor.npz`
- `data/m4/$DRIVE/norm_params.json`
- `data/m4/$DRIVE/bucket_metadata.json`

Check the printed summary before continuing. You want a non-trivial number of
anchors with at least `20` points; those are the anchors M4b will fit.

### 6. Fit Per-Anchor VBGS Posteriors

This step is M4b. It runs in the JAX environment and consumes the M4a bucket
artifacts from `data/m4/$DRIVE/`.

Start with a checkpointed partial run:

```bash
DRIVE=2013_05_28_drive_0009_sync

python scripts/fit_anchors.py \
  --drive "$DRIVE" \
  --max-observed-anchors 500 \
  --log-every 25 \
  --checkpoint-every 100 \
  --k-growth-min-points 512 \
  --device 0
```

Resume the same run after an interruption:

```bash
python scripts/fit_anchors.py \
  --drive "$DRIVE" \
  --max-observed-anchors 500 \
  --log-every 25 \
  --checkpoint-every 100 \
  --k-growth-min-points 512 \
  --resume \
  --device 0
```

For a full-scene run, omit `--max-observed-anchors` but keep checkpointing on:

```bash
python scripts/fit_anchors.py \
  --drive "$DRIVE" \
  --log-every 100 \
  --checkpoint-every 500 \
  --k-growth-min-points 512 \
  --resume \
  --device 0
```

Checkpoint files are written beside the final posterior:

- `anchor_posterior.checkpoint.npz`
- `fit_metadata.checkpoint.json`

Smoke/partial runs use the `.smoke` names. A completed run writes
`anchor_posterior.npz` and `fit_metadata.json`.
