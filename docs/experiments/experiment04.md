# Experiment 04: Camera-Count Comparison

## What the Experiment Is

Experiment 04 answers one question: **does adding cameras improve the
reconstruction of an NVIDIA NCore scene, and does the VB uncertainty estimate
respond correctly?**

It trains the same scene several times, once per *variant*, where each
variant uses one more camera than the last:

| Variant | Cameras used for training |
| --- | --- |
| `cam1` | front wide |
| `cam2` | front wide + front tele |
| `cam3` | + cross left |
| `cam4` | + cross right |
| `cam5`–`cam7` | + rear left, rear right, rear tele |

Every variant runs the full VBOGS pipeline (prepare → train → uncertainty →
bundle) and holds out the **same front-wide test frames**, so metrics on
those frames are directly comparable across variants. A final analysis step
collects everything into one comparison table and a set of plots.

## How to Run It on a Scene

All commands take the bare clip UUID as `<scene-id>` (no `pai_` prefix).

**1. Download the clip** — in the `vbogs-torch` container. `--mode full` is
required: each camera ships as its own `.zarr.itar` component.

```bash
cd /workspace/VBOGS
python scripts/download_nvidia_ncore_dataset.py --scene-id <scene-id> --mode full
```

**2. See what the clip contains** (optional but recommended):

```bash
python scripts/inspect_nvidia_ncore_clip.py --scene-id <scene-id> --frame-step 2
```

This lists the cameras, frame counts, and the largest usable `--max-frames`
value. The experiment script runs the same discovery itself, so this step is
only for choosing your sweep.

**3. Smoke run** — in the `vbogs-pipeline` container. Takes minutes and
verifies the whole path before you commit to hours of training:

```bash
scripts/experiment04-camera-count --scene-id <scene-id> \
  --max-frames 16 -- --iterations 2000 --render-max-views 2
```

**4. Full run:**

```bash
# default: cam1 and cam2
scripts/experiment04-camera-count --scene-id <scene-id>

# sweep every camera the clip has
scripts/experiment04-camera-count --scene-id <scene-id> --cameras-list 1,2,3,4,5,6,7
```

Each variant is a full 90k-iteration training run; budget several hours per
variant. Variants run sequentially by design — each one overwrites
`/data/COLMAP/<scene>` and `data/m4/<scene>` before its results are
snapshotted into the bundle, so never launch two variants of the same scene
at once. If a variant fails, rerun just that one with `--variant cam<N>`, or
pass `--continue-on-error` up front.

Useful flags: `--dry-run` prints every command without executing;
`--max-frames N` changes how many frames per camera are used (must be a
multiple of 8 and the same for the whole sweep).

## Reading the Results

Everything lands under
`outputs/experiments/experiment04-camera-count/<scene-id>/`:

```text
cam1/<scene-id>/    # one bundle per variant (+ <scene-id>.zip)
cam2/<scene-id>/
analysis/           # the cross-variant comparison (start here)
```

### `analysis/metrics_table.md` — reconstruction quality

One row per variant. The key columns:

| Column | Meaning |
| --- | --- |
| `PSNR`, `SSIM`, `LPIPS` | Standard image-quality metrics on held-out test views. PSNR/SSIM higher is better; LPIPS lower is better. |
| `PSNR_wide`, `SSIM_wide`, `LPIPS_wide` | The same metrics restricted to the front-wide test views that are **identical in every variant**. These are the columns to compare across rows — if adding cameras helps, `*_wide` improves down the table. |
| `GS_NUMS` | Average number of Gaussians rendered per test view (model size/complexity). |
| `observed_anchor_fraction` | Fraction of octree anchors the VB fit actually observed. More cameras should observe more of the scene. |
| `U_mean`, `U_mean_observed` | Mean per-anchor uncertainty, over all anchors and over observed anchors. A trustworthy estimate should decrease as cameras are added. |

The plain columns average over *all* test views, including each extra
camera's own held-out frames — useful, but not like-for-like across rows.

### Uncertainty calibration — is the uncertainty estimate meaningful?

An uncertainty estimate is useful only if high uncertainty actually predicts
high rendering error. The analysis checks this per variant:

- **`calibration_scatter_cam<N>.png`** — per-view uncertainty vs per-view
  error, one panel per metric, with the Spearman rank correlation in the
  title. Positive correlation = the model knows where it is wrong.
- **`sparsification_{PSNR,SSIM,LPIPS}.png`** — remove the most-uncertain
  views first and plot the mean error of what remains (solid line), next to
  the best possible ordering by true error (dashed "oracle"). The closer the
  solid line tracks the dashed one, the better calibrated the uncertainty.
- **AUSE** (in `comparison.json`) — the area between those two curves.
  0 means perfectly calibrated; larger means worse.

- **`uncertainty_hist_overlay.png`** — the per-anchor uncertainty
  distribution of every variant on shared axes. Expect the mass to shift
  left (lower uncertainty) as cameras are added.

`comparison.json` holds all of the above as numbers, plus a `fairness` block
listing the shared front-wide test views, and any warnings.

The analysis can be re-run on its own (in `vbogs-torch`):

```bash
python scripts/analyze_experiment04.py \
  --experiment-root outputs/experiments/experiment04-camera-count/<scene-id>
```

## Why the Comparison Is Fair

Octree-AnyGS holds out every 8th image (by sorted order) as the test set.
Because images sort grouped by camera, and the experiment pins the frame
count per camera to a multiple of 8, every camera block holds out its local
frames 0, 8, 16, … — so the front-wide test frames are **the same in every
variant**, and no variant ever trains on them. The script verifies this
after every prepare stage, and the analysis refuses to compare variants
whose front-wide test sets differ.

## Pulling Results to Your Machine

Each variant bundle is zipped as `cam<N>/<scene-id>/<scene-id>.zip`; grab it
and the `analysis/` directory through the File Browser service on port
`8088` (credentials via `scripts/get_filebrowser_login.py`) or `scp`. See
[Download and View Server Artifacts Locally](../getting-started/local-artifact-viewing.md).
