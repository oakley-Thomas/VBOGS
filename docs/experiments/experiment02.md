# Experiment 02: 3D Baseline Drive Sweep

Experiment 02 runs the KITTI-360 pipeline across one or more drives with the
experiment-specific baseline profiles, then creates compact review videos and
zip artifacts.

Run these commands from inside the Portainer `vbogs-pipeline` container after
pulling the latest repo changes.

## Preflight

Start in the repository root:

```bash
cd /workspace/VBOGS
```

Confirm the KITTI-360 data is mounted and the experiment configs exist:

```bash
ls data/KITTI-360/images
ls data/KITTI-360/data_poses
ls outputs/experiments/exp02_B_implicit3d_baseline_pipeline_config.yaml
```

The optional explicit baseline uses:

```text
outputs/experiments/exp02-A_explicit3d_baseline_pipeline_config.yaml
```

## List Drives

Preview the KITTI-360 drives that have both stereo images and camera poses:

```bash
scripts/run_exp02_all_drives.py --list-drives
```

If the dataset is mounted in a non-default location, pass the roots explicitly:

```bash
scripts/run_exp02_all_drives.py \
  --raw-root /workspace/VBOGS/data/KITTI-360/images \
  --poses-root /workspace/VBOGS/data/KITTI-360/data_poses \
  --list-drives
```

## Dry Run

Preview the commands for the default `implicit` profile:

```bash
scripts/run_exp02_all_drives.py --variant implicit --dry-run
```

Preview a single drive:

```bash
scripts/run_exp02_all_drives.py \
  --variant implicit \
  --drive 2013_05_28_drive_0007_sync \
  --dry-run
```

Preview both Experiment 02 profiles:

```bash
scripts/run_exp02_all_drives.py --variant both --dry-run
```

## Full Run

Run the active implicit3D baseline across all discovered drives:

```bash
scripts/run_exp02_all_drives.py --variant implicit
```

Run one selected drive:

```bash
scripts/run_exp02_all_drives.py \
  --variant implicit \
  --drive 2013_05_28_drive_0007_sync
```

Run both explicit and implicit baselines:

```bash
scripts/run_exp02_all_drives.py --variant both
```

To keep later drives running after one drive/profile fails:

```bash
scripts/run_exp02_all_drives.py \
  --variant both \
  --continue-on-error
```

Extra pipeline arguments can be passed after `--`:

```bash
scripts/run_exp02_all_drives.py \
  --variant implicit \
  --drive 2013_05_28_drive_0007_sync \
  -- \
  --render-max-views 5 \
  --nbv-max-candidates 8
```

## Outputs

The default review roots are:

```text
outputs/experiments/exp02_explicit3d_baseline/<drive>/
outputs/experiments/exp02_implicit3d_baseline/<drive>/
```

The helper scripts expect each completed drive output to include:

```text
views/train/side_by_side/
views/test/side_by_side/
pointclouds/anchors/
```

If the pipeline configs write to different roots, pass `--explicit-root` or
`--implicit-root` to the video and packaging steps.

## Review Videos

Encode train and test side-by-side frames into per-drive MP4 files:

```bash
scripts/make_exp02_side_by_side_videos.py --variant implicit --dry-run
scripts/make_exp02_side_by_side_videos.py --variant implicit
```

For both variants:

```bash
scripts/make_exp02_side_by_side_videos.py --variant both
```

The default outputs are named like:

```text
outputs/experiments/exp02_implicit3d_baseline/<drive>-implicit-train.mp4
outputs/experiments/exp02_implicit3d_baseline/<drive>-implicit-test.mp4
```

Use `--overwrite` to replace existing videos.

## Package Artifacts

Create compact zip files for review:

```bash
scripts/package_exp02_artifacts.py --variant implicit --dry-run
scripts/package_exp02_artifacts.py --variant implicit
```

For both variants:

```bash
scripts/package_exp02_artifacts.py --variant both
```

The default zip output is named like:

```text
outputs/experiments/exp02_implicit3d_baseline/<drive>-implicit.zip
```

Each zip contains the pipeline config, train/test side-by-side images, and
anchor uncertainty point-cloud artifacts. Use `--overwrite` to replace an
existing zip.
