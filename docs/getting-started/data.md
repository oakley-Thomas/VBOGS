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
TODO

## Docker Mounts

The compose stack mounts source datasets at:

```text
/workspace/VBOGS/data/KITTI-360
/workspace/VBOGS/data/NVIDIA-PhysicalAI-AV-NCore
```

KITTI-360 is backed by `VBOGS_DATASETS_VOLUME`. NVIDIA NCore is backed by
`VBOGS_NVIDIA_NCORE_VOLUME`.
