# Data Setup

VBOGS currently targets KITTI-360 perspective stereo. The pipeline expects
rectified stereo images, camera poses, and calibration text files.

## Download Helper

```bash
# Download KITTI-360 Dataset
bash scripts/download_kitti_360.sh
```

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

The compose stack mounts the source KITTI-360 data at:

```text
/workspace/VBOGS/data/KITTI-360
```

This path is backed by the persistent Docker volume named: `VBOGS_DATASETS_VOLUME`





## Drive IDs

Most examples use KITTI-360 drive ids such as:

```text
2013_05_28_drive_0007_sync
2013_05_28_drive_0008_sync
```

Pass the chosen drive consistently with `--drive`, `VBOGS_DRIVE`, or the
`pipeline.drive` key in a config profile.
