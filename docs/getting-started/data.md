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
| Raw images | `data/KITTI-360/images`, `data/KITTI-360/data_2d_raw`, `data/KITTI-360/data_2d_test`, `data/data_2d_raw`, `data/data_2d_test` |
| Poses | `data/KITTI-360/data_poses`, `data/data_poses` |
| Calibration | `data/KITTI-360/calibration`, `data/calibration/calibration`, `data/calibration` |

You can override discovery with `--raw-root`, `--poses-root`, and
`--calibration-dir`.

## Docker Volumes

The base compose stack mounts the source KITTI-360 data at:

```text
/workspace/VBOGS/data/KITTI-360
```

In Portainer deployments this path is expected to be backed by the external
Docker volume named `KITTI-360`.

The main generated data volumes are:

| Volume/path | Purpose |
| --- | --- |
| `COLMAP` mounted at `/data/COLMAP` | Prepared COLMAP-style inputs for Octree-AnyGS |
| `OCTREE-ANYGS` mounted at `/data/OCTREE-ANYGS` | Trained Octree-AnyGS runs and checkpoints |
| `vbogs-data` mounted at `/workspace/VBOGS/data` | VBOGS stage artifacts such as point clouds, buckets, fits, and `U.npy` |
| `vbogs-outputs` mounted at `/workspace/VBOGS/outputs` | Curated render, NBV, bundle, and zip outputs |

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
bash data/download_kitti_360.sh
```

The shell helper normalizes archives into the preferred `data/KITTI-360/`
layout.

## Drive IDs

Most examples use KITTI-360 drive ids such as:

```text
2013_05_28_drive_0007_sync
2013_05_28_drive_0008_sync
```

Pass the chosen drive consistently with `--drive`, `VBOGS_DRIVE`, or the
`pipeline.drive` key in a config profile.
