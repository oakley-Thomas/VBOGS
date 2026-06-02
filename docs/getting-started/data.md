# Data Setup

VBOGS supports KITTI-360 perspective stereo and NVIDIA PhysicalAI Autonomous
Vehicles clips converted to NCore V4. KITTI remains the default dataset;
NVIDIA NCore is selected with `dataset.name: nvidia_ncore` or
`--dataset-name nvidia_ncore`.

## Download Helper

```bash
# Download KITTI-360 Dataset
bash scripts/download_kitti_360.sh
```

## NVIDIA NCore Downloader

Use the repo downloader to pull converted NVIDIA NCore clips from Hugging Face:

```bash
# Download a small set of scenes
python scripts/download_nvidia_ncore_dataset.py \
  --scene-id pai_00000000_00000000_0000,pai_11111111_11111111_1111

# Download first 10 scenes from the repo (recommended only if you have enough space)
python scripts/download_nvidia_ncore_dataset.py \
  --all \
  --max-scenes 10

# Download everything for a scene (optional; includes all sensor files)
python scripts/download_nvidia_ncore_dataset.py \
  --scene-id-file /path/to/scene_ids.txt \
  --mode full
```

Notes:

- This script reads your Hugging Face token from `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`/`HUGGING_FACE_TOKEN` if `--token` is not passed.
- Default destination is `/workspace/VBOGS/data/NVIDIA-PhysicalAI-AV-NCore/<scene-id>/`, matching VBOGS adapter resolution.
- Use `--dry-run` to preview planned downloads before transfer.
- Use `--force` to overwrite existing clip files.

## KITTI-360 Expected Layout

The layout should be:

```text
data/KITTI-360/
  calibration/
    perspective.txt
  data_poses/
    <drive>/
      cam0_to_world.txt
  images/
    <drive>/
      image_00/
        data_rect/
          *.png
      image_01/
        data_rect/
          *.png
```



## Docker Mounts

The compose stack mounts source datasets at:

```text
/workspace/VBOGS/data/KITTI-360
/workspace/VBOGS/data/NVIDIA-PhysicalAI-AV-NCore
```

KITTI-360 is backed by `VBOGS_DATASETS_VOLUME`. NVIDIA NCore is backed by
`VBOGS_NVIDIA_NCORE_VOLUME`.



## Drive IDs

Most examples use KITTI-360 drive ids such as:

```text
2013_05_28_drive_0007_sync
2013_05_28_drive_0008_sync
```

Pass the chosen drive consistently with `--drive`, `VBOGS_DRIVE`, or the
`pipeline.drive` key in a config profile.

## NVIDIA PhysicalAI AV NCore

NVIDIA data should first be converted to NCore V4 outside VBOGS after accepting
the dataset license and providing a Hugging Face token as required by NVIDIA's
tools. Place converted clips under:

```text
data/NVIDIA-PhysicalAI-AV-NCore/
  <clip-or-sequence>/
    ...
```

Use the NVIDIA profile as a starting point:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/nvidia_ncore_dev.yaml \
  --scene-id <clip-id> \
  --use-service-labels
```

The default NVIDIA camera subset is `camera_front_wide_120fov`. The default
point source is LiDAR; switch to camera depth with `--point-source camera_depth`.
