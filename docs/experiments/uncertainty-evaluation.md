# Uncertainty-Evaluation

## Purpose

Uncertainty-Evaluation measures two related but distinct properties of a VBOGS scene:

1. **RGB reconstruction quality** — how closely an Octree-AnyGS render matches a held-out photograph, measured with PSNR, SSIM, and LPIPS.
2. **Uncertainty calibration** — whether views with high alpha-normalized VBOGS uncertainty also have high held-out rendering error, measured with Spearman correlation and normalized AUSE.

The experiment enforces a three-way data boundary:

- `train` frames may create the Octree scene, sparse seed, world-point cloud, anchor buckets, and VBGS posteriors;
- `validation` frames may select an Octree profile/checkpoint and then a VBOGS uncertainty profile;
- `test` frames are read only after both choices are written into an immutable, hash-verified selection lock.

The selected model is not retrained with validation data. The exact validation-selected Octree checkpoint and posterior are evaluated once on test.

## Runtime and preflight

Run the experiment from the `vbogs-pipeline` container. It resolves the active `vbogs-torch` and `vbogs-jax` services through Docker Compose labels, like the main pipeline.

```bash
cd /workspace/VBOGS
```

Confirm that the chosen KITTI-360 drive or NCore clip is present before starting. The full default sweep trains four 90k-iteration Octree scenes and performs five full-scene VBGS fits, so budget multiple GPU-hours and substantial `/data` space.

## Dry run and smoke run

Preview every command without creating an experiment directory:

```bash
scripts/uncertainty-evaluation \
  --dataset-name kitti360 \
  --scene-id 2013_05_28_drive_0008_sync \
  --run-id kitti-dry-run \
  --dry-run
```

Run a reduced end-to-end smoke experiment with 16 selected frames, one 2k-iteration Octree profile, three uncertainty profiles, at most 200k bucketed points, and four views per held-out split:

```bash
scripts/uncertainty-evaluation \
  --dataset-name kitti360 \
  --scene-id 2013_05_28_drive_0008_sync \
  --run-id smoke-01 \
  --smoke
```

NCore uses the same interface:

```bash
scripts/uncertainty-evaluation \
  --dataset-name nvidia_ncore \
  --scene-id <clip_uuid> \
  --run-id smoke-01 \
  --smoke
```

Use repeated `--camera-id` options to replace the NCore camera list in the experiment config.

## Full run

```bash
scripts/uncertainty-evaluation \
  --dataset-name kitti360 \
  --scene-id 2013_05_28_drive_0008_sync \
  --run-id full-01
```

The default profile definitions are in `configs/experiments/uncertainty-evaluation.yaml`. The Octree sweep changes resolution, base layer, or visibility pruning one factor at a time around the production profile. Every saved 10k checkpoint is evaluated. The uncertainty sweep changes mixture capacity, the observed-anchor point threshold, or the ELBO growth tolerance around the PLAN.md defaults.

Override common run controls without editing the config:

```bash
scripts/uncertainty-evaluation \
  --dataset-name nvidia_ncore \
  --scene-id <clip_uuid> \
  --run-id full-01 \
  --gpu 1 \
  --jax-device 1 \
  --frame-step 2 \
  --max-frames 400
```

## Resume and stage control

Every completed stage writes a marker under `.stages/`. Resume requires the original run ID and refuses to continue if the effective config hash, dataset, scene, or prepared split differs:

```bash
scripts/uncertainty-evaluation \
  --dataset-name kitti360 \
  --scene-id 2013_05_28_drive_0008_sync \
  --run-id full-01 \
  --resume
```

To run a bounded part of the state machine, use `--start-at` and `--stop-after`. The stages are:

```text
prepare
octree-train
octree-validation
octree-select
points
bucket
uncertainty-fit
uncertainty-validation
uncertainty-select
test
export
report
```

Do not start at `test` without the existing experiment manifest and `selection.lock.json`. The test stage re-hashes the selected config, checkpoint directory, posterior, and `U.npy` before rendering.

## Selection rules

Octree selection uses primary-camera validation views. KITTI’s primary camera is `image_00`; NCore uses `primary_camera_id` from prepared metadata. Candidates are ordered by:

1. highest mean PSNR;
2. lowest mean LPIPS;
3. highest mean SSIM;
4. fewer training iterations;
5. profile order in the YAML file.

After that checkpoint is fixed, every VBOGS profile renders the same validation views. For each image metric, per-view errors are normalized to `[0,1]` over the common primary-camera views. PSNR and SSIM are inverted; LPIPS is not. Constant metrics are omitted with a warning.

The uncertainty profile with the lowest mean normalized AUSE across the available PSNR, SSIM, and LPIPS errors wins. Higher mean Spearman correlation and then YAML profile order break ties.

No absolute quality threshold is imposed. Selection ranks the tested candidates; human M7 review still decides whether the scene and uncertainty are scientifically acceptable.

## Outputs

Reports are written to:

```text
outputs/experiments/uncertainty-evaluation/<dataset>/<scene>/<run-id>/
```

Important files are:

```text
experiment_manifest.json
effective_config.yaml
prepared_metadata.json
validation/octree_metrics.md
validation/uncertainty_metrics.md
octree_selection.json
uncertainty_selection.json
selection.lock.json
test/summary.json
test/per_view.json
test/renders/
plots/
export/
report.md
```

## Exported scene and uncertainty

The `export` stage copies the validation-selected splat and uncertainty into the report
directory so a single download is enough to visualize the run locally:

```text
export/README.md
export/VIEWER_COMMANDS.md
export/export_manifest.json
export/splat/config.yaml
export/splat/original_config.yaml
export/splat/point_cloud/iteration_<N>/point_cloud_gs.ply
export/splat/point_cloud/iteration_<N>/point_cloud_anchor.ply
export/prepared/sparse/0/
export/prepared/metadata.json
export/uncertainty/U.npy
export/uncertainty/uncertainty_anchors.ply
export/uncertainty/uncertainty_metadata.json
```

With the default `explicit3D` gaussian type, `point_cloud_gs.ply` holds expanded Gaussians
and opens in a standard 3DGS viewer. `point_cloud_anchor.ply` holds Octree anchors and
needs Octree-AnyGS to render. `export/splat` mirrors the Octree model layout, so it can be
passed back as `--model-path` without rearranging anything.

### Interactive viewer

`export/` is self-contained enough to open in the realtime viewer. The trained config
records an absolute `source_path` into the server's `/data/COLMAP` tree, so the export
ships the prepared COLMAP cameras under `export/prepared/` and rewrites
`export/splat/config.yaml` to the relative `source_path: ../prepared`. The untouched
config stays as `export/splat/original_config.yaml`, and that is what the `model_config`
hash in `export_manifest.json` covers.

```bash
EXPORT_DIR=/workspace/VBOGS/local_viewer_exports/<run-id>/export

python scripts/view_octree_anygs.py \
  --model-path "${EXPORT_DIR}/splat" \
  --u-path "${EXPORT_DIR}/uncertainty/U.npy" \
  --iteration <N> \
  --camera-source train \
  --resolution 4
```

`--camera-source train` is required, and this is the one place the experiment's export
differs from `scripts/export_local_viewer_run.py`. The `prepare` stage enforces a leakage
gate that keeps held-out frames out of COLMAP entirely, and the trained config sets
`eval: false`, so Octree-AnyGS loads every prepared camera as a train camera and
`getTestCameras()` returns nothing. The viewer's default `--camera-source test` fails with
"No test cameras available".

Held-out poses still ship, as `export/prepared/metadata.json`, but they are not viewer
cameras. Use them through `scripts/evaluate_uncertainty_views.py --selection-metadata`, or
navigate to an arbitrary pose in the viewer with `--initial-pose` or the `pose` field on
`POST /api/rendered-anchors`. `export/prepared/images/` is omitted: the viewer uses
metadata-only camera loading. Set `export.include_prepared_images: true` if you need
`--load-source-images`.

Setting `export.include_prepared_colmap: false` restores the previous diagnostics-only
bundle, with an unmodified `config.yaml` and no `prepared/` tree.

`uncertainty_anchors.ply` is the anchor cloud colored by per-anchor `U`, viewable in
CloudCompare or MeshLab. It keeps the raw scalar as a per-vertex `uncertainty` property,
and the color range is written into the ply `obj_info` header. Unobserved anchors take the
maximum uncertainty rather than a fitted value, so read `unobserved_anchor_count` in
`uncertainty_metadata.json` before interpreting the high end of the ramp.

Export re-verifies `selection.lock.json` before copying, and repeats the locked hashes in
`export_manifest.json`. The `export:` config block turns the stage off, adds the full
`anchor_posterior.npz`, or changes the colormap and clip percentile.

Large working data stays outside the report directory:

```text
/data/OCTREE-ANYGS/uncertainty-evaluation/<dataset>/<scene>/<run-id>/
data/experiments/uncertainty-evaluation/<dataset>/<scene>/<run-id>/
```

The manifest and selection records preserve the absolute source paths and hashes.

## Reading the results

PSNR and SSIM are higher-is-better; LPIPS is lower-is-better. These evaluate the RGB renderer, not the Bayesian uncertainty model.

For uncertainty calibration:

- positive Spearman means more-uncertain views tend to have larger rendering errors;
- normalized AUSE of zero means uncertainty ranks the views in the same order as the oracle rendering error;
- larger AUSE means poorer error ranking.

`report.md` contains headline primary-camera and all-camera test results. The plots show per-view uncertainty/error scatter, sparsification against the oracle, and the selected per-anchor uncertainty distribution.

## Limitations

- The deterministic timeline-uniform split mostly measures interpolation between nearby driving frames and can be optimistic about distant novel viewpoints.
- Metrics are computed on clamped sRGB images with Octree’s resize convention and no dynamic-object or alpha mask. Pose/calibration error, exposure changes, moving objects, and vegetation motion can reduce PSNR even when static reconstruction is reasonable.
- Image metrics do not establish geometric accuracy. Use held-out depth or LiDAR metrics for that claim.
- VBOGS scalar rendering only covers existing Octree anchors. The experiment does not fix or measure uncertainty in truly unmodeled empty space.
- The default fit seed is zero. Broader statistical studies should repeat the experiment with independently declared seeds rather than treating one fit as a variance estimate.
