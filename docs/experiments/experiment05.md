# Experiment 05: Seed-Source Comparison (SGBM vs LiDAR)

## What the Experiment Is

Experiment 05 answers one question: **does seeding Octree-AnyGS from LiDAR
points instead of stereo SGBM depth improve the reconstruction?**

The sparse seed cloud matters more in Octree-AnyGS than in vanilla 3DGS:
anchor positions are voxelized from the seed and never move during training
(`position_lr_init: 0`), so noisy or missing seed geometry is baked into the
anchor lattice — showing up as blur where seeds were missing and floaters
where stereo depth was wrong.

The experiment trains the same scene twice, once per *variant*:

| Variant | Seed source |
| --- | --- |
| `sgbm` | Per-frame stereo SGBM depth (`--seed-mode stereo`) |
| `lidar` | LiDAR scans (`--seed-mode lidar`) |

On KITTI-360 the `lidar` variant seeds from raw velodyne scans
(`data_3d_raw`), colored by projecting each point into the left camera. On
NVIDIA NCore it seeds from `lidar_top_360fov`; the `sgbm` variant rectifies
the front-wide + front-tele camera pair and runs SGBM on it.

Both variants use identical frames, cameras, and training settings — only the
seed differs — and hold out the **same test frames**, so metrics are directly
comparable. A final analysis step writes a head-to-head table with the
`lidar − sgbm` delta per metric.

## How to Run It on a Scene

### KITTI-360

**1. Download the drive's velodyne scans** — in the `vbogs-pipeline`
container. The images/poses/calibration download is unchanged; LiDAR seeding
additionally needs the drive's raw velodyne archive (KITTI-360 ships these
per drive; the full `data_3d_raw` set is ~119 GB, so download only the drives
you need):

```bash
export KITTI_VELODYNE_LINK='https://.../2013_05_28_drive_0004_sync_velodyne.zip'
bash scripts/download_kitti_360.sh
```

**2. Smoke run** — in the `vbogs-pipeline` container. Takes minutes and
verifies the whole path (including the velodyne transform chain) before you
commit to hours of training:

```bash
scripts/experiment05-seed-comparison \
  --max-frames 16 -- --iterations 2000 --render-max-views 2
```

After each prepare stage, check the printed fairness-gate line and
`prepared/metadata.json`: the lidar variant's
`seed_metadata.colored_point_fraction` should be roughly 0.3–0.6 (the share
of 360° LiDAR points visible to the forward camera). A near-zero fraction
means the velodyne→camera transform chain is off — inspect the seed
`points3D.ply` against the images before training.

**3. Full run:**

```bash
# default drive: 2013_05_28_drive_0004_sync
scripts/experiment05-seed-comparison

# another drive
scripts/experiment05-seed-comparison 2013_05_28_drive_0008_sync
```

### NVIDIA NCore

```bash
scripts/experiment05-seed-comparison --dataset ncore --scene-id <scene-id>
```

The NCore `sgbm` variant approximates the FTheta cameras as pinholes and
rectifies the front-wide/front-tele pair; pass
`-- --seed-stereo-pair <left>,<right>` to try a different pair.

Each variant is a full 90k-iteration training run; budget several hours per
variant. Variants run sequentially by design — each one overwrites
`/data/COLMAP/<scene>` and `data/m4/<scene>` before its results are
snapshotted into the bundle, so never launch both variants of the same scene
at once. If a variant fails, rerun just that one with `--variant sgbm` or
`--variant lidar`, or pass `--continue-on-error` up front.

Useful flags: `--dry-run` prints every command without executing;
`--max-frames N` changes how many frames are used (must be a multiple of 8
and the same for both variants).

## Reading the Results

Everything lands under
`outputs/experiments/experiment05-seed-comparison/<dataset>/<scene>/`:

```text
sgbm/<scene>/    # one bundle per variant (+ <scene>.zip)
lidar/<scene>/
analysis/        # the head-to-head comparison (start here)
```

### `analysis/metrics_table.md` — reconstruction quality

One row per variant plus a `lidar − sgbm` delta section. The key columns:

| Column | Meaning |
| --- | --- |
| `seed_mode`, `seed_points` | The seed source and how many seed points were written to `points3D.ply`. |
| `PSNR`, `SSIM`, `LPIPS` | Standard image-quality metrics on the shared held-out test views. PSNR/SSIM higher is better; LPIPS lower is better. |
| `GS_NUMS` | Average number of Gaussians rendered per test view (model size/complexity). |
| `observed_anchor_fraction` | Fraction of octree anchors the VB fit actually observed. |
| `U_mean`, `U_mean_observed` | Mean per-anchor uncertainty, over all anchors and over observed anchors. |

If LiDAR seeding helps, the delta row shows positive PSNR/SSIM and negative
LPIPS. `comparison.json` holds the same numbers plus a `fairness` block
listing the shared test views and any warnings.

### Uncertainty calibration

As in Experiment 04, the analysis also checks whether per-view uncertainty
predicts per-view error for each variant:
`calibration_scatter_<variant>.png`, `sparsification_{PSNR,SSIM,LPIPS}.png`,
AUSE values in `comparison.json`, and `uncertainty_hist_overlay.png`
comparing the per-anchor uncertainty distributions of the two seeds.

The analysis can be re-run on its own (in `vbogs-torch`):

```bash
python scripts/analyze_experiment05.py \
  --experiment-root outputs/experiments/experiment05-seed-comparison/<dataset>/<scene>
```

For a qualitative look, encode the rendered test views of both variants into
videos with `scripts/make_view_videos.py` and compare them side by side.

## Why the Comparison Is Fair

Octree-AnyGS holds out every 8th image (by sorted order) as the test set.
Both variants prepare the identical image set — same drive/clip, same
`--frame-step`, same `--max-frames` (a multiple of 8), same cameras — so the
held-out test frames are **the same in both variants**, and neither trains
on them. Only `sparse/0/points3D.ply` differs. The script verifies this
after every prepare stage: it checks the prepared `seed_mode`, and it
fingerprints the selected frames and cameras
(`fairness_fingerprint.json`, written by the first variant) and fails if the
second variant's prepared metadata deviates. The analysis independently
refuses to compare variants whose test-view sets differ.

## Pulling Results to Your Machine

Each variant bundle is zipped as `<variant>/<scene>/<scene>.zip`; grab it
and the `analysis/` directory through the File Browser service on port
`8088` (credentials via `scripts/get_filebrowser_login.py`) or `scp`. See
[Download and View Server Artifacts Locally](../getting-started/local-artifact-viewing.md).
