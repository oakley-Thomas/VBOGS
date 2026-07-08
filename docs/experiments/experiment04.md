# Experiment 04: Camera-Count Comparison

Evaluate how Octree-AnyGS reconstruction quality and VB uncertainty change
when the same NVIDIA NCore scene is reconstructed with different numbers of
camera sensors (cam1 = `camera_front_wide_120fov`, cam2 = wide + tele, and so
on if the clip exposes more cameras).

Run this from inside the Portainer `vbogs-pipeline` container after pulling
the latest repo changes.

## 1. Download the Scene

The experiment needs the clip's **full** component set — the per-camera
`.zarr.itar` files are required for any camera beyond the core imagery. In
the `vbogs-torch` container:

```bash
cd /workspace/VBOGS
python scripts/download_nvidia_ncore_dataset.py --scene-id <clip_uuid> --mode full
```

Set `HF_TOKEN` if the Hugging Face dataset requires authentication.

## 2. Discover Cameras

List the clip's cameras, per-camera frame counts, and the ready-to-copy
`--camera-id` flag sets (also in `vbogs-torch`, which has the NCore loader):

```bash
python scripts/inspect_nvidia_ncore_clip.py --scene-id <clip_uuid> --frame-step 2
```

The output includes a suggested `--max-frames` per camera. Experiment 04
requires `--max-frames` to be a **multiple of 8** — see the fairness contract
below. The experiment script runs this discovery step itself before the
variants unless `--skip-discovery` is passed.

## 3. Dry Run

Preview every command from inside `vbogs-pipeline`:

```bash
cd /workspace/VBOGS
scripts/experiment04-camera-count --scene-id <clip_uuid> --dry-run
```

## 4. Smoke Run

Verify the full path end-to-end in minutes rather than hours:

```bash
scripts/experiment04-camera-count --scene-id <clip_uuid> \
  --max-frames 16 -- --iterations 2000 --render-max-views 2
```

Then confirm `analysis/comparison.json` exists and its `fairness` block lists
the shared primary-camera test views, and that `metrics_table.md` has one row
per variant.

## 5. Full Run

```bash
scripts/experiment04-camera-count --scene-id <clip_uuid>
```

Defaults: variants `cam1,cam2`, 200 frames at step 2, the 48 GB
high-quality profile from `configs/pipeline/experiment04_ncore_portainer.yaml`
(resolution 2, 90k iterations, explicit3D), stages `prepare → bundle` per
variant, followed by the cross-variant analysis.

For a clip with more than two cameras, extend the variants:

```bash
scripts/experiment04-camera-count --scene-id <clip_uuid> --cameras-list 1,2,3
```

!!! warning "Variants must run sequentially"
    Each variant overwrites `/data/COLMAP/<scene>` and `data/m4/<scene>`.
    Never launch two variants of the same scene in parallel; the script
    already runs them one after another and snapshots per-variant artifacts
    into the run bundle before starting the next variant.

## Fairness Contract

Octree-AnyGS sorts training images by path (grouping them per camera) and
holds out every `llffhold`-th index as the test split. Experiment 04 pins
`llffhold=8` and requires the per-camera frame count (`--max-frames`) to be a
multiple of 8, so **every variant holds out the identical primary-camera
frames**. A post-prepare gate inside the script verifies this against
`/data/COLMAP/<scene>/metadata.json` before training starts, and the analysis
hard-fails if the primary-camera test views ever differ across variants.

Consequently, the `*_wide` columns in the analysis (metrics restricted to
`camera_front_wide_120fov` test views) are the like-for-like comparison
across camera counts; the overall columns additionally average each extra
camera's own held-out views.

## Outputs

```text
outputs/experiments/experiment04-camera-count/<scene>/
  cam1/<scene>/            # per-variant run bundle (+ <scene>.zip)
    run_manifest.json
    prepared/metadata.json
    octree/{config.yaml,results.json,per_view.json}
    uncertainty/{U.npy,uncertainty_metadata.json,...}
    nbv/nbv_scores.json    # per-test-view uncertainty scores
  cam2/<scene>/
  analysis/
    metrics_table.md       # PSNR/SSIM/LPIPS per variant, overall + wide-only
    metrics.csv
    comparison.json        # fairness block, uncertainty stats, calibration
    uncertainty_hist_overlay.png
    calibration_scatter_cam<N>.png
    sparsification_{PSNR,SSIM,LPIPS}.png
```

The calibration outputs correlate per-view uncertainty (the alpha-normalized
render of per-anchor `U` from the NBV stage) with per-view rendering error:
Spearman rank correlations plus sparsification curves with an AUSE-style
area. Uncertainty that decreases — and stays correlated with error — as
cameras are added indicates a well-behaved estimate.

The analysis can be re-run standalone (in `vbogs-torch`):

```bash
python scripts/analyze_experiment04.py \
  --experiment-root outputs/experiments/experiment04-camera-count/<scene>
```

## Pulling Results Locally

Each variant bundle is zipped (`cam<N>/<scene>/<scene>.zip`). Download it and
the `analysis/` directory through the File Browser service on port `8088`
(credentials via `scripts/get_filebrowser_login.py`) or `scp`, following
[Download and View Server Artifacts Locally](../getting-started/local-artifact-viewing.md).
