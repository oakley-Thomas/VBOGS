# Full Drive Runs

This page describes the full curated run path for a KITTI-360 drive.

## Recommended Full Run

From inside `vbogs-pipeline`:

```bash
scripts/run_pipeline.sh \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --gpu 0 \
  --jax-device 0 \
  --start-at prepare \
  --stop-after bundle \
  --run-output-root outputs/v1_0
```

For the server/Portainer profile, run this from inside the stack's
`vbogs-pipeline` container:

```bash
scripts/run_pipeline.sh \
  --config configs/pipeline/portainer.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --gpu 0 \
  --jax-device 0 \
  --start-at prepare \
  --stop-after bundle \
  --run-output-root outputs/v1_0
```

## Expected Outputs

With `--run-output-root outputs/v1_0`, curated outputs land at:

```text
outputs/v1_0/<scene-id>/
outputs/v1_0/<scene-id>/<scene-id>.zip
```

The bundle includes:

- prepared-data metadata and generated Octree-AnyGS config;
- stereo point-cloud sidecars;
- colored anchor uncertainty PLY files;
- NBV score files and visualizations;
- `uncertainty/U.npy`, uncertainty metadata, and histogram when present;
- `local_viewer/`, a portable local-render export with the model checkpoint,
  prepared camera metadata, uncertainty array, and `VIEWER_COMMANDS.md`;
- `run_manifest.json`.

Rendered RGB/uncertainty diagnostics may remain on the run host under
`outputs/v1_0/<scene-id>/views/`, but the bundle zip excludes that directory.

Download `outputs/v1_0/<scene-id>/<scene-id>.zip` when you want compact
diagnostics or when you want to extract the run on another machine with a
checked-out VBOGS repo and use the realtime browser viewer or REST API locally.
Full posterior artifacts remain in their native data paths and are referenced
by path in the manifest.

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
| Skip portable viewer export | `--skip-local-viewer-export` |

See [Pipeline Arguments](../documentation/RUN_DRIVE_PIPELINE_ARGS.md) for the
complete argument reference.

## Development Smoke Run

Use this before committing to a long run:

```bash
scripts/run_pipeline.sh \
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
  --render-max-views 2
```

## Before M7

Do not treat a completed bundle as validated science output until a human has:

1. Inspected a sample of anchor posteriors after M4b.
2. Checked tight, sparse, and noisy anchors against expected uncertainty.
3. Rendered uncertainty overlays on held-out views.
4. Confirmed the NBV winner matches scene intuition.

The implementation can produce plausible files without guaranteeing that the
uncertainty is calibrated for the scene.
