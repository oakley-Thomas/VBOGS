# Full Drive Runs

This page describes the full curated run path for a KITTI-360 drive.

## Recommended Full Run

From inside `vbogs-pipeline`:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --gpu 0 \
  --jax-device 0 \
  --start-at prepare \
  --stop-after bundle \
  --run-output-root outputs/v1_0 \
  --use-service-labels
```

For the server/Portainer profile:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/portainer.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --gpu 0 \
  --jax-device 0 \
  --start-at prepare \
  --stop-after bundle \
  --run-output-root outputs/v1_0 \
  --use-service-labels
```

## Expected Outputs

With `--run-output-root outputs/v1_0`, curated outputs land at:

```text
outputs/v1_0/<drive>/
outputs/v1_0/<drive>.zip
```

The bundle includes:

- prepared-data metadata and generated Octree-AnyGS config;
- stereo point-cloud sidecars;
- colored anchor uncertainty PLY files;
- RGB and uncertainty render diagnostics;
- NBV score files and visualizations;
- `uncertainty/U.npy`, uncertainty metadata, and histogram when present;
- `run_manifest.json`.

Large checkpoint and posterior artifacts remain in their native data volumes
and are referenced by path in the manifest.

## Important Runtime Knobs

| Goal | Arguments |
| --- | --- |
| Choose drive | `--drive <drive-id>` |
| Choose GPU | `--gpu 0`, `--jax-device 0` |
| Reduce prepared frames | `--frame-step <N>`, `--max-frames <N>` |
| Reduce Octree-AnyGS memory | `--resolution <N>`, `--feat-dim <N>`, `--base-layer <N>` |
| Reduce stereo size | `--pixel-step <N>`, `--max-points-per-frame <N>` |
| Cap M4/M4b scale | `--bucket-max-points <N>` |
| Use explicit model path | `--model-path /data/OCTREE-ANYGS/<drive>/<run>` |
| Render fewer diagnostics | `--render-max-views <N>` |
| Score fewer NBV candidates | `--nbv-max-candidates <N>` |

See [Pipeline Arguments](../documentation/RUN_DRIVE_PIPELINE_ARGS.md) for the
complete argument reference.

## Development Smoke Run

Use this before committing to a long run:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --gpu 0 \
  --jax-device 0 \
  --start-at prepare \
  --stop-after render \
  --frame-step 20 \
  --max-frames 30 \
  --resolution 4 \
  --iterations 7000 \
  --max-points-per-frame 50000 \
  --render-max-views 2 \
  --use-service-labels
```

## Before M7

Do not treat a completed bundle as validated science output until a human has:

1. Inspected a sample of anchor posteriors after M4b.
2. Checked tight, sparse, and noisy anchors against expected uncertainty.
3. Rendered uncertainty overlays on held-out views.
4. Confirmed the NBV winner matches scene intuition.

The implementation can produce plausible files without guaranteeing that the
uncertainty is calibrated for the scene.
