# Data Setup

VBOGS currently targets KITTI-360 perspective stereo. The pipeline expects
rectified stereo images, camera poses, and calibration text files.

## Expected Layout

The preferred repo-local layout is:

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

The helper `vbogs.data_layout.resolve_kitti360_path` also accepts a few
alternate historical layouts:

| Input kind | Candidate paths |
| --- | --- |
| Raw images | `data/KITTI-360/images`, `data/KITTI-360/data_2d_raw`, `data/KITTI-360/data_2d_test`, `data/KITTI-360/data_2d_test_slam`, `data/data_2d_raw`, `data/data_2d_test`, `data/data_2d_test_slam` |
| Poses | `data/KITTI-360/data_poses`, `data/data_poses` |
| Calibration | `data/KITTI-360/calibration`, `data/calibration/calibration`, `data/calibration` |

You can override discovery with `--raw-root`, `--poses-root`, and
`--calibration-dir`.

## Docker Mounts

The compose stack mounts the source KITTI-360 data at:

```text
/workspace/VBOGS/data/KITTI-360
```

This path is backed by the persistent Docker volume named by
`VBOGS_DATASETS_VOLUME`, defaulting to `KITTI-360`. The stack also uses
compose-managed Docker volumes for generated artifacts; they are persistent but
not marked `external`.

When running inside Docker, make sure the KITTI-360 layout above is present in
that dataset volume.

The main Docker volumes are:

| Volume | Container path | Purpose |
| --- | --- |
| `KITTI-360` or `VBOGS_DATASETS_VOLUME` | `/workspace/VBOGS/data/KITTI-360` | Source KITTI-360 images, poses, and calibration |
| `vbogs-data` | `/workspace/VBOGS/data` | VBOGS stage artifacts such as point clouds, buckets, fits, and `U.npy` |
| `COLMAP` | `/data/COLMAP` | Prepared COLMAP-style inputs for Octree-AnyGS |
| `OCTREE-ANYGS` | `/data/OCTREE-ANYGS` | Trained Octree-AnyGS runs and checkpoints |
| `vbogs-outputs` | `/workspace/VBOGS/outputs` | Curated render, NBV, bundle, and zip outputs |
| `vbogs-generated-configs` | `/workspace/VBOGS/generated_configs` | Generated Octree-AnyGS config files |

## Download Helpers

There are two repo-owned helpers:

```bash
python scripts/download_kitti360.py \
  --manifest data/KITTI-360/download_manifest.json \
  --data-root data/KITTI-360 \
  --skip-existing
```

This Python helper is manifest-driven and uses only the standard library.
Copy the example manifest, fill in the source URLs, then run it.

```bash
export KITTI_CALIBRATION_LINK='https://.../calibration.zip'
export KITTI_POSES_LINK='https://.../data_poses.zip'
export KITTI_IMAGES_LINK='https://.../data_2d_test_slam.zip'
bash data/download_kitti_360.sh
```

The shell helper normalizes archives into the preferred `data/KITTI-360/`
layout. It uses `KITTI_IMAGES_LINK` for image archives such as
`data_2d_test_slam.zip`; `VBOGS_DRIVE` is only
used later to select a drive for pipeline execution.

## Drive IDs

Most examples use KITTI-360 drive ids such as:

```text
2013_05_28_drive_0007_sync
2013_05_28_drive_0008_sync
```

Pass the chosen drive consistently with `--drive`, `VBOGS_DRIVE`, or the
`pipeline.drive` key in a config profile.
