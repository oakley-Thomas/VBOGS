# Overview

VBOGS supports KITTI-360 perspective stereo and NVIDIA PhysicalAI Autonomous
Vehicles clips converted to NCore V4.

## KITTI-360
From [KITTI-360 download page](https://www.cvlibs.net/datasets/kitti-360/download.php)
accept the dataset terms and get the official **KITTI-360 download links** for 

- Calibrations
- Vehicle Poses 
- Left/Right perspective images ("Test SLAM" recommended)

Run the downloader from the interactive `vbogs-pipeline` container shell:

```bash
docker exec -it vbogs-vbogs-pipeline-1 /bin/bash

export KITTI_CALIBRATION_LINK='https://.../calibration.zip'
export KITTI_POSES_LINK='https://.../data_poses.zip'
export KITTI_IMAGES_LINK='https://.../data_2d_test_slam.zip'
bash scripts/download_kitti_360.sh
```

### Layout

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

### Drive IDs
Most examples use KITTI-360 drive ids such as:

```text
2013_05_28_drive_0002_sync
2013_05_28_drive_0008_sync
```

## NVIDIA NCore
NVIDIA PhysicalAI Autonomous Vehicles NCore clips are hosted on Hugging Face at
[`nvidia/PhysicalAI-Autonomous-Vehicles-NCore`](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore).
The repository is gated: sign in to Hugging Face, accept the NVIDIA Autonomous
Vehicle Dataset License Agreement on the dataset page, then create a Hugging
Face user access token with read access.

### Web Console download helper

For the Web Experiment Console, set `VBOGS_NCORE_HF_TOKEN` as a deployment or
Portainer secret on the `vbogs-web` service. The console never sends this token
to browsers or stores it in download records/logs. Operators can then search the
authorized catalog and download one pipeline-ready full clip at a time from the
NVIDIA NCore panel. Do not add the token to this repository or to
`configs/docker/stack.env`.

Run the downloader from the interactive `vbogs-pipeline` container shell. Keep
the token in your shell or deployment secrets; do not commit it to the repo.

```bash
# Enter pipeline container
./dc_bash.sh

export HF_TOKEN='hf_...'
python scripts/download_nvidia_ncore_dataset.py \
  --scene-id <clip_uuid> \
  --mode full \
  --skip-existing
```

For a quick first clip, let the script discover the repository index and take
the first available scene:

```bash
python scripts/download_nvidia_ncore_dataset.py \
  --all \
  --max-scenes 1 \
  --mode full \
  --skip-existing
```

Use `--mode full` for pipeline-ready clips. It downloads the core NCore file,
metadata, camera sensors, and LiDAR component needed by VBOGS. The default
`core-only` mode is useful for lightweight checks, but it is not enough for
`prepare`, LiDAR seeding, or point export.

### Layout

```text
/workspace/VBOGS/data/NVIDIA-PhysicalAI-AV-NCore/
  <clip_uuid>/
    pai_<clip_uuid>.json
    pai_<clip_uuid>.ncore4.zarr.itar
    pai_<clip_uuid>.ncore4-camera_front_wide_120fov.zarr.itar
    pai_<clip_uuid>.ncore4-camera_front_tele_30fov.zarr.itar
    pai_<clip_uuid>.ncore4-lidar_top_360fov.zarr.itar
    ...
```

### Clip IDs

Clip IDs are the UUID-like directory names under the dataset's `clips/` folder,
for example:

```text
000da9de-0ee5-465a-9a2d-e7e91d3016bb
```

You can preview the first few downloadable clips without writing files:

```bash
python scripts/download_nvidia_ncore_dataset.py \
  --all \
  --max-scenes 5 \
  --mode full \
  --dry-run
```

To see which clips have already been downloaded locally and which ones are
ready for the default NCore pipeline profile:

```bash
python scripts/list_dataset_clips.py --dataset-name nvidia_ncore --ready-only --commands
```

Drop `--ready-only` to include partial downloads and see which components are
missing:

```bash
python scripts/list_dataset_clips.py --dataset-name nvidia_ncore
```

After download, run the NCore pipeline profile by passing the same clip id:

```bash
scripts/run_pipeline.sh \
  --config configs/pipeline/nvidia_ncore_dev.yaml \
  --dataset-name nvidia_ncore \
  --scene-id <clip_uuid> \
  --gpu 0 \
  --jax-device 0
```

Do not pass an NCore clip UUID as `--drive` unless you also set
`--dataset-name nvidia_ncore`. If the pipeline prints `Dataset: kitti360`, it
will call the KITTI-360 adapter and look for `data/KITTI-360/images/<clip_uuid>`.

## How Dataset Processing Differs

KITTI-360 and NVIDIA NCore enter the pipeline through different dataset
adapters, but they are normalized into the same downstream artifact contracts.
The main differences are in dataset selection, `prepare`, and point-cloud
export.

| Pipeline part | KITTI-360 | NVIDIA NCore |
| --- | --- | --- |
| Dataset selector | Default adapter: `--dataset-name kitti360`. `--drive` is the drive id. | Must set `--dataset-name nvidia_ncore`. Use `--scene-id` for the clip UUID. |
| Source data | Rectified stereo images, camera poses, and calibration under `data/KITTI-360/`. | NCore V4 `.zarr.itar` clip components under `data/NVIDIA-PhysicalAI-AV-NCore/`. |
| `prepare` script | `scripts/prepare_kitti360_colmap.py` | `scripts/prepare_nvidia_ncore_colmap.py` |
| Training images | Assigns selected frames to deterministic timeline-uniform `70% train / 20% test / 10% validation` splits and writes only train frames to Octree-AnyGS `images.txt`. Defaults to left `image_00`; `--training-cameras stereo` also materializes `image_01` as a second posed camera for train frames. | Assigns selected primary-camera frames to the same `70/20/10` split and writes only train frames to Octree-AnyGS `images.txt`. The NCore dev profile uses `camera_front_wide_120fov` plus `camera_front_tele_30fov`. |
| Sparse training seed | Bootstraps `sparse/0/points3D.ply` from train-split lightweight stereo depth, unless `--seed-mode random` is used. | Bootstraps `sparse/0/points3D.ply` from train-split NCore LiDAR by default. If `--seed-mode stereo` is passed through the generic pipeline, it maps to LiDAR for NCore. |
| Point-cloud export stage | The stage name is `stereo`, and it exports train-split world points from rectified stereo disparity by default. KITTI only supports `--point-source stereo`. | The stage name is still `stereo` for pipeline compatibility, but the default point source is `lidar`. The NCore dev profile also exports `--frame-split train` so VBGS fitting does not use held-out test/validation observations. |
| Point colors | RGB comes from the KITTI left image. | LiDAR points are projected into selected camera views for RGB coloring, or camera-depth points use the configured camera pair. |

Held-out `test` and `validation` views remain in `/data/COLMAP/<scene-id>/metadata.json`.
Render and NBV stages can use those metadata-backed camera poses with
`--render-split test`, `--render-split validation`, `--nbv-candidate-source test`,
or `--nbv-candidate-source validation`.

Both adapters write the prepared Octree-AnyGS dataset to:

```text
/data/COLMAP/<scene-id>/
```

Both point exporters write the shared Stage 2 artifact:

```text
data/points_world/<scene-id>/points_world.npz
```

After `points_world.npz` exists, the rest of the pipeline is dataset-agnostic:
bucket points into Octree-AnyGS anchors, fit per-anchor VBGS posteriors,
compute scalar uncertainty, render uncertainty views, score NBV candidates,
and bundle outputs. Those later stages key off the shared scene id and artifact
layout rather than the original dataset format.

## Docker Mounts

The compose stack mounts source datasets at:

```text
/workspace/VBOGS/data/KITTI-360
/workspace/VBOGS/data/NVIDIA-PhysicalAI-AV-NCore
```

KITTI-360 is backed by `VBOGS_DATASETS_VOLUME`. NVIDIA NCore is backed by
`VBOGS_NVIDIA_NCORE_VOLUME`.
