# VBOGS Implementation Plan

Actionable plan derived from [Algorithm.txt](Algorithm.txt). Check items off as they complete.

---

## 0. Blocking decisions (you)

These gate delegation. Each LLM task below needs its answer before it can run.

- [x] **Stereo data source** — KITTI-360 perspective stereo (`image_00` / `image_01`); poses = shipped GT (fused GPS+IMU + laser); native KITTI-360 layout → needs adapter to Octree-AnyGS's COLMAP-style ingest. Chosen drive: `2013_05_28_drive_0008_sync`.
- [x] **Stereo matcher** — Start with OpenCV `StereoSGBM` as the baseline implementation. Keep the M3 interface provider-agnostic so we can later swap in RAFT-Stereo or another backend without changing downstream file formats.
- [x] **Octree-AnyGS training budget** — hard cap: `46 GB` VRAM max for any training/inference step. Strategy: `train-per-scene` on the chosen KITTI-360 drive rather than reusing a checkpoint.
- [x] **NBV candidate pose set** — reachable set from planner. Initial implementation may approximate this with a ground-vehicle local lattice, but the M6 interface should accept planner-produced candidate poses directly later.
- [x] **Starting hyperparameters** (commit to values before M4b)
  - [x] `K_INIT = pc.n_offsets = 10` (matching the current Octree-AnyGS default)
  - [x] `K_MAX = 4 * K_INIT = 40`
  - [x] `K_GROWTH_FACTOR = 2` (`K -> 2K`)
  - [x] `MIN_POINTS_PER_ANCHOR = 20`
  - [x] `ELBO_IMPROVEMENT_TOL = 0.01` nats/point
- [x] **Entropy definition** — use π-weighted per-component entropy as the Stage 4/5 scalar uncertainty definition; do not use total mixture entropy `H(q)` in the initial implementation.

---

## 1. Execution graph

```
M1 ── M2 ──┐
           ├── M4a ── M4b ── M5 ── M6 ── M7
    M3 ────┘
```

M2 and M3 are independent — run in parallel. M4a onward is strictly linear.

---

## 2. Milestones

Each milestone is self-contained once its dependencies and decisions above are resolved. "LLM" = delegable with the spec in [Algorithm.txt](Algorithm.txt) plus the files listed.

### M1 — Environment setup [LLM]

Two conda envs required (JAX/PyTorch CUDA conflict is real; don't try to unify).

- [x] Create `vbogs-torch` env (Octree-AnyGS deps; see `Octree-AnyGS/environment.yml`)
- [x] Create `vbogs-jax` env (vbgs deps; see `vbgs/install_deps.sh`)
- [x] Smoke test: `vbogs-torch` runs `Octree-AnyGS/render.py --help`
- [x] Smoke test: `vbogs-jax` imports `vbgs.model.train.fit_gmm_step` without error
- [x] Document activation commands in a `scripts/envs.sh`

### M2 — Train Octree-AnyGS [LLM, mostly ops]

Depends on: M1, stereo data source, training budget.

Local-dev note: the original scene-training budget was `46 GB`, but the
repo now includes a conservative `16 GB` dev-machine workflow. Use the local
path first, then scale the config back up on the server if needed.

- [x] Prepare input in Octree-AnyGS's expected format (COLMAP-style posed RGB)
- [x] Pick a config from `Octree-AnyGS/config/` that stays within the `46 GB` VRAM budget
- [x] Run training to convergence
- [x] Save checkpoint (`.ply` + MLP weights)
- [x] Sanity render a held-out view; confirm photometric quality

Completed on local dev machine with the conservative `16 GB` preset:
`render_mode=RGB`, `add_prefilter=false`, `densification=false`,
`resolution=4`, `feat_dim=16`, `base_layer=9`, `iterations=15000`.
New training runs live under `/data/OCTREE-ANYGS/<drive>/<timestamp>/`.
The original local-dev artifact was
`outputs/kitti360/2013_05_28_drive_0008_sync/2026-04-22_15:47:13`.

### M3 — Stereo → world point cloud [LLM]

Depends on: M1, stereo data source, stereo matcher choice.

- [x] Script `scripts/stereo_to_pointcloud.py` (runs in `vbogs-torch`)
- [x] Define a matcher abstraction / CLI flag (`--matcher`) so disparity can come from `sgbm`, `raft`, or another future provider while preserving the same `points_world.npz` output contract
- [x] For each stereo pair: disparity → depth → unproject → world-frame
- [x] Apply validity mask (left-right consistency, texture threshold)
- [x] Concat across frames; save `points_world.npz` with keys `xyz`, `rgb`, `frame_id`
- [x] Sanity check: visualize point cloud in a viewer; should match scene geometry

### M4a — Point → anchor bucketing [LLM]

Depends on: M2, M3. Runs in `vbogs-torch` (needs Octree-AnyGS checkpoint).

Reference: [Octree-AnyGS/scene/basic_model.py:100-120](Octree-AnyGS/scene/basic_model.py#L100-L120) (`octree_sample` — grid discretization to match exactly).

- [x] Script `scripts/bucket_points.py`
- [x] Load checkpoint; read `pc._anchor`, `pc._level`, `pc.voxel_size`, `pc.fork`, `pc.init_pos`
- [x] Build `anchor_index: (level, grid_coord) -> anchor_id`
- [x] Bucket each world-frame point at **every** level it falls into (not just finest)
- [x] Apply `normalize_data` from vbgs to produce `points_norm`
- [x] Save `pts_by_anchor.npz`: per-anchor arrays of indices into `points_norm`
- [x] Save `points_norm.npz` + `norm_params.json`
- [x] Sanity check: print histogram of per-anchor point counts; inspect a few anchors

Completed on the bundled dev scene. Current M4a artifacts report `12,792,935`
points, `267,830` anchors across `9` levels, and `104,577` anchors with at
least `20` assigned points.

### M4b — Per-anchor VBGS fit [LLM, heaviest task]

Depends on: M4a, starting hyperparameters. Runs in `vbogs-jax`.

Reference: [vbgs/vbgs/model/train.py](vbgs/vbgs/model/train.py) (`fit_gmm_step`, `compute_elbo_delta`), [vbgs/scripts/model_volume.py](vbgs/scripts/model_volume.py) (`get_volume_delta_mixture`).

- [x] Script `scripts/fit_anchors.py`
- [x] Implement `FitAnchor(pts_a, K)` per Stage 3 of Algorithm.txt
- [x] Implement K-growth loop with ELBO comparison
- [x] Unobserved (pts < `MIN_POINTS_PER_ANCHOR`) → emit `None`/sentinel
- [x] Save `anchor_posterior.npz` — per-anchor `(mean, kappa, u, n)` for likelihood + delta, plus Dirichlet `alpha`, plus final `K`, plus an `is_observed` mask
- [ ] Manual validation pass (see "Don't delegate" below) **before** running M5
- [ ] Decide: loop vs `jax.vmap` across anchors (defer until N_anchors is known)

Implementation is in place and smoke-tested in `vbogs-jax`, but the full-scene
fit has not been run to completion yet. Current smoke artifacts live under
`data/m4/2013_05_28_drive_0008_sync/` as `anchor_posterior.smoke.npz` and
`fit_metadata.smoke.json`.

### M5 — Posterior → scalar uncertainty [LLM]

Depends on: M4b, entropy definition.

- [x] Script `scripts/compute_uncertainty.py` (runs in `vbogs-jax` or pure numpy)
- [x] Closed-form Normal-Wishart entropy from `(kappa, u, n)`
- [x] Closed-form Dirichlet entropy from `alpha`
- [x] Closed-form delta MVN entropy
- [x] Combine per chosen definition; emit `U.npy` of shape `[N_anchors]`
- [x] Unobserved anchors → `U_MAX`
- [ ] Sanity check: plot histogram of `U`; tails should be fat, not uniform

### M6 — `render_scalar` + NBV loop [LLM]

Depends on: M2, M5, candidate pose set. Runs in `vbogs-torch`.

Reference: [Octree-AnyGS/gaussian_renderer/render.py](Octree-AnyGS/gaussian_renderer/render.py), [Octree-AnyGS/scene/implicit_model/base_model.py:460-534](Octree-AnyGS/scene/implicit_model/base_model.py#L460-L534) (`generate_gaussians`).

- [ ] Implement `render_scalar(cam, pc, per_anchor_scalar)` per Stage 5
- [ ] Return `(unc_image, alpha_image)` — both needed for the score
- [ ] Implement candidate pose generator for a planner-reachable set; first pass can be a ground-plane local lattice with yaw samples, but keep the input interface compatible with future planner-emitted poses
- [ ] NBV loop: `score = sum(unc_image) / (sum(alpha_image) + EPS)`
- [ ] Return best pose + diagnostic dump of top-K candidates

### M7 — End-to-end viz + validation [you]

Depends on: M6.

- [ ] Pick a scene you understand well (known uncertain regions: far / glass / textureless)
- [ ] Run full pipeline
- [ ] Overlay `U` as a heatmap on a held-out training view
- [ ] Confirm NBV pick visually matches intuition
- [ ] Document failure modes observed

---

## 3. Do not delegate

Even if an LLM could produce plausible output, these need human judgment:

- [ ] **Validate a sample of anchor posteriors after M4b.** Tight-cluster anchor → low entropy; sparse/noisy anchor → high. Spot-check before running M5
- [ ] **First scene choice for M7** — pick one where you already know which regions *should* be uncertain
- [ ] **Hyperparameter retuning after first run** — the LLM has no frame of reference for "nats/point"
- [ ] **Accepting that an anchor looks "fine"** without looking at at least 5 posterior fits by eye

---

## 4. Known risks / flags

- [ ] **Scale of N_anchors.** Octree-AnyGS scenes routinely have 10⁵–10⁶ anchors. Per-anchor fits at ~10ms each ≈ 3 hrs. `jax.vmap` across anchors is the 10–100× fix; plan to invest in it after M4a shows the actual N.
- [ ] **Empty-region blindness.** `render_scalar` only splats through existing anchors — NBV cannot be drawn to never-seen volumes. If your AV use case needs exploration of empty space, add a follow-on M8 (volumetric occupancy prior or per-pixel unknown-ray penalty).
- [ ] **ELBO-as-K-selection is biased.** KL term scales with K. Per-point mean ELBO is defensible but not principled. Swap in held-out log-likelihood or BIC if model selection seems off.
- [ ] **Normalization-coord entropy comparison.** Stage 4 entropies are comparable across anchors *because* coordinates are globally normalized. If you switch to per-anchor normalization later, entropy values stop being directly comparable.
- [ ] **Score semantics choice.** Alpha-normalized score is "direct me to the most uncertain thing." Unnormalized sum is "direct me to where I'll learn the most in aggregate." Algorithm uses the former; flip if your planner prefers the latter.

---

## 5. Prompt template for delegating a milestone

When handing a milestone to an LLM, the prompt should include:

1. Link to this file and [Algorithm.txt](Algorithm.txt) for context
2. The specific stage's pseudocode excerpt
3. The relevant Octree-AnyGS / vbgs files listed in the milestone
4. The filesystem contract (inputs read, outputs written, formats)
5. "Test plan: call the entry point on the artifacts produced by M{N-1}; expected output shape is X"
6. Which conda env the script runs in
