# Experiment 03: Stereo Training Comparison

Run this from inside the Portainer `vbogs-pipeline` container after pulling the
latest repo changes.

## Dry Run

Preview the two commands:

```bash
cd /workspace/VBOGS
scripts/experiment03-stereo-comparison --dry-run
```

## Full Run

Start the left-only run, followed by the stereo-training run:

```bash
cd /workspace/VBOGS
scripts/experiment03-stereo-comparison
```

By default this uses KITTI-360 drive:

```text
2013_05_28_drive_0004_sync
```

To run a different KITTI-360 clip:

```bash
scripts/experiment03-stereo-comparison 2013_05_28_drive_0008_sync
```

## Outputs

Results are written to:

```text
outputs/experiments/experiment03-stereo-comparison/left/
outputs/experiments/experiment03-stereo-comparison/stereo/
```

Check the prepared metadata:

```bash
cat outputs/experiments/experiment03-stereo-comparison/left/2013_05_28_drive_0004_sync/prepared/metadata.json
cat outputs/experiments/experiment03-stereo-comparison/stereo/2013_05_28_drive_0004_sync/prepared/metadata.json
```

Expected values:

```text
"training_cameras": "left"
"training_cameras": "stereo"
```
