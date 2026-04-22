# Agent guide — VBOGS

Orientation for coding agents working on this repo.

## What this project is

VBOGS combines Octree-AnyGS (scalable Gaussian Splatting with LOD) with Variational Bayes Gaussian Splatting (per-Gaussian Bayesian posteriors) to produce a scene representation with calibrated per-anchor uncertainty, used to pick **next-best views** for autonomous-vehicle mapping.

It is **pre-implementation**: the root repo currently contains only the spec ([Algorithm.txt](Algorithm.txt)), the implementation plan ([PLAN.md](PLAN.md)), and two git submodules providing the upstream components. No `scripts/`, `vbogs/`, or tests exist yet — they will be built out per PLAN.md.

## Read these first

1. **[Algorithm.txt](Algorithm.txt)** — authoritative pseudocode. Five stages. Every design decision is captured here.
2. **[PLAN.md](PLAN.md)** — milestone breakdown (M1–M7) with checkboxes, dependencies, and which env each task runs in.
3. This file — repo conventions and pitfalls.

If a user request conflicts with Algorithm.txt, flag it and ask — don't silently diverge.

## Repository layout

```
VBOGS/
├── Algorithm.txt          # spec — the source of truth
├── PLAN.md                # milestone checklist
├── AGENTS.md              # this file
├── README.md
├── Octree-AnyGS/          # submodule — DO NOT EDIT
└── vbgs/                  # submodule — DO NOT EDIT
```

**The two submodules are read-only dependencies.** Don't edit files under `Octree-AnyGS/` or `vbgs/`. If upstream behavior is wrong for our use case, wrap/subclass from our own code rather than patching the submodule. Changes there won't survive a submodule update and make the project hard to maintain.

## Upstream orientation

- **Octree-AnyGS** (PyTorch) — primary scene representation. Key files:
  - [Octree-AnyGS/scene/basic_model.py](Octree-AnyGS/scene/basic_model.py) — `octree_sample` defines the grid discretization we must match for point→anchor bucketing
  - [Octree-AnyGS/scene/implicit_model/base_model.py](Octree-AnyGS/scene/implicit_model/base_model.py) — `generate_gaussians` at line 460; the geometry path we reuse for `render_scalar`
  - [Octree-AnyGS/scene/implicit_model/lod_model.py](Octree-AnyGS/scene/implicit_model/lod_model.py) — `set_anchor_mask` for LOD-aware visibility
  - [Octree-AnyGS/gaussian_renderer/render.py](Octree-AnyGS/gaussian_renderer/render.py) — the render function we sibling for scalar rendering
- **vbgs** (JAX) — the uncertainty head. Key files:
  - [vbgs/vbgs/model/train.py](vbgs/vbgs/model/train.py) — `fit_gmm_step` and `compute_elbo_delta`
  - [vbgs/scripts/model_volume.py](vbgs/scripts/model_volume.py) — `get_volume_delta_mixture` factory

## Environment

**Two conda envs are required.** JAX and PyTorch's CUDA builds conflict. Don't try to unify them — the vbgs README is explicit about this. Data flows between envs as `.npz` / `.npy` files on disk.

- `vbogs-torch` — Octree-AnyGS, stereo matching, `render_scalar`, bucketing
- `vbogs-jax` — vbgs fits, posterior computations

Both envs live alongside one another; each script declares which one it needs (see PLAN.md's per-milestone annotations).

## Terminology

- **Anchor = voxel.** Octree-AnyGS stores a flat tensor of anchors, each with a level. The anchor's cell size is `voxel_size / fork^level`. There is no separate "voxel" object. When Algorithm.txt or code says "voxel," it means "anchor at that level's cell size."
- **Per-anchor uncertainty** — a scalar `U[i]` on each anchor derived from the VBGS posterior for points falling in that anchor's cell.
- **Next-best view (NBV)** — output of Stage 5; a camera pose chosen to maximize alpha-normalized expected posterior entropy.

## Implementation conventions

Code we write lives outside the submodules:

```
scripts/          # entry points per PLAN milestone (stereo_to_pointcloud.py, ...)
vbogs/            # shared library code (when enough accumulates to warrant it)
tests/            # integration tests against small fixtures
```

Create these as needed — don't scaffold empty dirs ahead of demand.

- **Framework boundary is the filesystem.** PyTorch scripts read/write `.npz`; JAX scripts read/write `.npz`. No in-process IPC, no cross-framework tensor sharing.
- **Coordinate frames matter.** Stage 3 buckets points in **world** coords (to match Octree-AnyGS's grid) but fits VBGS in **normalized** coords. Mixing these silently produces nonsense grid buckets. Make the frame explicit in variable names (`points_world` vs `points_norm`).
- **Don't modify the submodules to add functionality.** Example: for `render_scalar`, write a sibling function in `vbogs/render.py` that imports and reuses `generate_gaussians` from Octree-AnyGS; do NOT edit `Octree-AnyGS/gaussian_renderer/render.py`.

## Common pitfalls

- **"Leaf voxel" doesn't exist.** Octree-AnyGS has no leaf concept — anchors at different levels coexist, LOD selection picks among them at render time. Per-anchor GMMs receive points from **every level that contains them**, not just the finest (otherwise coarse anchors in well-sampled regions look spuriously uncertain).
- **`generate_gaussians` color comes from an MLP.** There is no aux channel on anchors. To render a scalar (like uncertainty), reuse `generate_gaussians` for geometry, then substitute your own color tensor before calling `gsplat.rasterization`. See Stage 5 in Algorithm.txt.
- **Padding in `octree_sample`.** Anchor positions are stored as `grid_coord * cur_size + init_pos + padding * cur_size`. When bucketing points, don't subtract padding — just recompute grid coords for both points and anchors; the offset cancels.
- **ELBO across K is biased.** The KL term scales with component count. Per-point mean ELBO is our pragmatic choice, not a principled one. If model selection looks wrong, the fix is held-out log-likelihood or BIC, not tuning `ELBO_IMPROVEMENT_TOL` into oblivion.
- **NBV can't see empty space.** `render_scalar` only splats through existing anchors. Truly unobserved volumes don't contribute to the score. This is a known limitation documented in Algorithm.txt's Notes; don't try to "fix" it without talking to the user — the extension is non-trivial.

## When in doubt

Ask. The spec has intentional gaps (see PLAN.md §0 "Blocking decisions") that belong to the human, not to you.
