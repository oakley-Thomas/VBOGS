# VBOGS Implementation Plan

Actionable plan derived from [docs/manuscript/Algorithm.tex](docs/manuscript/Algorithm.tex). Check items off as they complete.

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
           ├── M4a ── M4b ── M5 ── M6 ── M7 ── M8
    M3 ────┘
```

M2 and M3 are independent — run in parallel. M4a onward is strictly linear.
M8 depends on a validated M7 scene and does not retrain or densify
Octree-AnyGS online.

---

## 2. Milestones

Each milestone is self-contained once its dependencies and decisions above are resolved. "LLM" = delegable with the spec in [docs/manuscript/Algorithm.tex](docs/manuscript/Algorithm.tex) plus the files listed.

### M1 — Docker runtime setup [LLM]

PyTorch and JAX still need separate CUDA runtime stacks, but those stacks are
Docker services rather than locally managed conda environments. Do not try to
merge them.

- [x] Build `vbogs-torch` image (Octree-AnyGS deps; see `docker/torch.Dockerfile`)
- [x] Build `vbogs-jax` image (vbgs deps; see `docker/jax.Dockerfile`)
- [x] Build `vbogs-pipeline` image for orchestration and artifact handling
- [x] Smoke test: `vbogs-torch` runs `scripts/check_torch_stack.py`
- [x] Smoke test: `vbogs-jax` imports `vbgs.model.train.fit_gmm_step` without error
- [x] Document Docker Compose startup and service commands in `docs/getting-started/`

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
- [x] Optional dataset-neutral dynamic-mask preprocessing: mirror static alpha
  masks into the prepared COLMAP layout and enable Octree-AnyGS's native mask loss

Completed on local dev machine with the conservative `16 GB` preset:
`render_mode=RGB`, `add_prefilter=false`, `densification=false`,
`resolution=4`, `feat_dim=16`, `base_layer=9`, `iterations=15000`.
New training runs live under `/data/OCTREE-ANYGS/<drive>/<timestamp>/`.
The original local-dev artifact was
`outputs/kitti360/2013_05_28_drive_0008_sync/2026-04-22_15:47:13`.

### M3 — Stereo → world point cloud [LLM]

Depends on: M1, stereo data source, stereo matcher choice.

- [x] Script `scripts/stereo_to_pointcloud.py` (runs in the `vbogs-torch` service)
- [x] Define a matcher abstraction / CLI flag (`--matcher`) so disparity can come from `sgbm`, `raft`, or another future provider while preserving the same `points_world.npz` output contract
- [x] For each stereo pair: disparity → depth → unproject → world-frame
- [x] Apply validity mask (left-right consistency, texture threshold)
- [x] Concat across frames; save `points_world.npz` with keys `xyz`, `rgb`, `frame_id`
- [x] Sanity check: visualize point cloud in a viewer; should match scene geometry
- [x] Apply the shared dynamic-mask artifact before stereo/LiDAR/camera-depth
  unprojection so confirmed movers never reach anchor bucketing

### M4a — Point → anchor bucketing [LLM]

Depends on: M2, M3. Runs in the `vbogs-torch` service (needs Octree-AnyGS checkpoint).

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

Depends on: M4a, starting hyperparameters. Runs in the `vbogs-jax` service.

Reference: [vbgs/vbgs/model/train.py](vbgs/vbgs/model/train.py) (`fit_gmm_step`, `compute_elbo_delta`), [vbgs/scripts/model_volume.py](vbgs/scripts/model_volume.py) (`get_volume_delta_mixture`).

- [x] Script `scripts/fit_anchors.py`
- [x] Implement `FitAnchor(pts_a, K)` per Stage 3 of [docs/manuscript/Algorithm.tex](docs/manuscript/Algorithm.tex)
- [x] Implement K-growth loop with ELBO comparison
- [x] Unobserved (pts < `MIN_POINTS_PER_ANCHOR`) → emit `None`/sentinel
- [x] Save `anchor_posterior.npz` — per-anchor `(mean, kappa, u, n)` for likelihood + delta, plus Dirichlet `alpha`, plus final `K`, plus an `is_observed` mask
- [x] Deterministic shard mode for parallel anchor fitting, plus shard merge back to `anchor_posterior.npz`
- [ ] Manual validation pass (see "Don't delegate" below) **before** running M5
- [x] Implement grouped batched fitting with `jax.vmap`; keep the one-anchor loop as a debugging fallback

Implementation is in place and smoke-tested in the `vbogs-jax` service. The default path is
now grouped/batched fitting, with point-count buckets controlling padding and
memory use. The full-scene fit still needs a completion/quality pass before M7.
Current smoke artifacts live under
`data/m4/2013_05_28_drive_0008_sync/` as `anchor_posterior.smoke.npz` and
`fit_metadata.smoke.json`.

### M5 — Posterior → scalar uncertainty [LLM]

Depends on: M4b, entropy definition.

- [x] Script `scripts/compute_uncertainty.py` (runs in the `vbogs-jax` service or pure numpy)
- [x] Closed-form Normal-Wishart entropy from `(kappa, u, n)`
- [x] Closed-form Dirichlet entropy from `alpha`
- [x] Closed-form delta MVN entropy
- [x] Combine per chosen definition; emit `U.npy` of shape `[N_anchors]`
- [x] Unobserved anchors → `U_MAX`
- [x] Sanity check: plot histogram of `U`; tails should be fat, not uniform

### M6 — `render_scalar` + NBV loop [LLM]

Depends on: M2, M5, candidate pose set. Runs in the `vbogs-torch` service.

Reference: [Octree-AnyGS/gaussian_renderer/render.py](Octree-AnyGS/gaussian_renderer/render.py), [Octree-AnyGS/scene/implicit_model/base_model.py:460-534](Octree-AnyGS/scene/implicit_model/base_model.py#L460-L534) (`generate_gaussians`).

- [x] Implement `render_scalar(cam, pc, per_anchor_scalar)` per Stage 5
- [x] Return `(unc_image, alpha_image)` — both needed for the score
- [x] Implement candidate pose generator for a planner-reachable set; first pass can be a ground-plane local lattice with yaw samples, but keep the input interface compatible with future planner-emitted poses
- [x] NBV loop: `score = sum(unc_image) / (sum(alpha_image) + EPS)`
- [x] Return best pose + diagnostic dump of top-K candidates

Initial implementation lives in `vbogs/render.py` and `scripts/score_nbv.py`,
with diagnostic render, map visualization, NBV visualization, and bundle stages
available through `scripts/run_drive_pipeline.py`. It has syntax/CLI
verification, but still needs a full torch/GPU render validation pass on a
completed M5 `U.npy`.

### M7 — End-to-end viz + validation [you]

Depends on: M6.

The reproducible validation workflow is documented in
[`docs/experiments/uncertainty-evaluation.md`](docs/experiments/uncertainty-evaluation.md).
It selects checkpoints and uncertainty settings on validation views before a
hash-locked, one-time test evaluation. Running the workflow does not replace
the human posterior and scene review below.

- [ ] Pick a scene you understand well (known uncertain regions: far / glass / textureless)
- [ ] Run full pipeline
- [ ] Overlay `U` as a heatmap on a held-out training view
- [ ] Confirm NBV pick visually matches intuition
- [ ] Document failure modes observed

### M8 — Real-time ROS2 uncertainty/NBV loop [LLM + ops]

Depends on: M7. Runs as a split Torch/ROS2 + JAX updater workflow with a fixed
Octree-AnyGS scene and filesystem handoff.

- [x] Config `configs/online/ros2_default.yaml` for ROS2 topics, online bundle paths, stereo settings, deadline, and candidate cap
- [x] Script `scripts/build_online_state.py` to package `online_manifest.json`, `anchor_grid_cache.npz`, `vbgs_online_state.npz`, `U_online.npy`, `norm_params.json`, and handoff directories
- [x] Reusable online helpers under `vbogs/online/` for cached multi-level bucketing, fixed normalization, score ranking, atomic state writes, and touched-anchor updates
- [x] Script `scripts/online_jax_updater.py` to consume `batches/<seq>.npz`, update touched anchors with fixed-K posterior moments, refresh `U_online.npy`, and write `updates/<seq>.npz`
- [x] Script `scripts/ros2_online_nbv_node.py` for ROS2 Humble-style stereo/pose/candidate subscriptions and best-pose/diagnostic publications
- [x] Script `scripts/benchmark_online_loop.py` for KITTI replay-style latency measurement through normalization, bucketing, optional updater, and total loop timing
- [ ] Run in a ROS2 Humble environment with live topics or bag replay
- [x] Replace the first fixed-K moment updater with exact fixed-scaffold VBGS updates; keep the original moment updater as a fallback mode
- [ ] Validate p95 frame-to-NBV latency under `1.0 s` on the target GPU server with `max_candidates <= 32`

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

1. Link to this file and [docs/manuscript/Algorithm.tex](docs/manuscript/Algorithm.tex) for context
2. The specific stage's pseudocode excerpt
3. The relevant Octree-AnyGS / vbgs files listed in the milestone
4. The filesystem contract (inputs read, outputs written, formats)
5. "Test plan: call the entry point on the artifacts produced by M{N-1}; expected output shape is X"
6. Which Docker service the script runs in
