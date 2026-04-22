# VBOGS
Combining Octree-GS's scene scalability with Variational Bayes GS uncertainty for better autonomous vehicle mapping

## Environment Notes

VBOGS uses two separate conda environments because the PyTorch and JAX CUDA stacks
conflict in practice:

- `vbogs-torch` for Octree-AnyGS, stereo, point bucketing, and rendering
- `vbogs-jax` for VBGS fitting and posterior computations

The repo helper script is [scripts/envs.sh](/home/oakley/ub/advanced_robotics/VBOGS/scripts/envs.sh:1).

Common commands:

```bash
bash scripts/envs.sh create-torch
bash scripts/envs.sh create-jax
bash scripts/envs.sh check-torch-stack
bash scripts/envs.sh smoke-test-jax
```

## M2 Training Workflow

M2 now has a repo-owned local workflow for preparing KITTI-360 drive
`2013_05_28_drive_0008_sync` into the COLMAP-style ingest that Octree-AnyGS
expects and then launching a conservative LoD training run that fits inside a
16 GB dev GPU budget.

Prepare the dataset:

```bash
python scripts/prepare_kitti360_colmap.py \
  --drive 2013_05_28_drive_0008_sync \
  --frame-step 10 \
  --max-frames 160
```

That writes a prepared dataset under
`data/octree_anygs_colmap/2013_05_28_drive_0008_sync/` with:

- `images/`
- `sparse/0/cameras.txt`
- `sparse/0/images.txt`
- `sparse/0/points3D.ply`

Generate a 16 GB-safe config and launch training:

```bash
python scripts/train_octree_anygs.py \
  --dataset-path data/octree_anygs_colmap/2013_05_28_drive_0008_sync \
  --gpu 0
```

The default local preset intentionally trades fidelity for safety:

- `resolution: 4`
- `feat_dim: 16`
- `base_layer: 9`
- `iterations: 15000`
- `render_mode: RGB`
- `add_prefilter: false`
- `densification: false`

If you have headroom after a first successful run, the least risky upgrades are
to lower `--resolution` from `4` to `2` and increase `--iterations`.

The local preset also disables Octree-AnyGS densification because the current
upstream stats path is incompatible with the installed `gsplat` tensor shapes
on this machine. That keeps M2 stable on the dev box at the cost of some final
scene quality.

Use `--write-config-only` to inspect the generated YAML without starting
training.

## M3 Point Cloud Export

M3 exports a dense-ish world-frame stereo point cloud from the same KITTI-360
drive using the `vbogs-torch` env:

```bash
bash -lc 'source scripts/envs.sh activate-torch >/dev/null && \
python scripts/stereo_to_pointcloud.py \
  --drive 2013_05_28_drive_0008_sync \
  --selection-metadata data/octree_anygs_colmap/2013_05_28_drive_0008_sync/metadata.json \
  --write-ply'
```

That writes artifacts under `data/points_world/2013_05_28_drive_0008_sync/`:

- `points_world.npz` with keys `xyz`, `rgb`, and `frame_id`
- `points_world_metadata.json` with the matcher and filtering settings
- `points_world.ply` when `--write-ply` is passed for quick viewer sanity checks

The current implementation ships with an `sgbm` provider and a forward-looking
`--matcher` interface so a future RAFT-Stereo backend can preserve the same
output contract. The validity mask keeps only pixels that pass:

- minimum disparity / depth bounds
- left-right consistency
- a local grayscale texture threshold

Use `--pixel-step` and `--max-points-per-frame` to trade off density vs runtime
and file size on the dev machine.

### Torch Stack

The current `vbogs-torch` setup is intentionally pinned to a CUDA 12.8 PyTorch
wheel stack:

- Python `3.10`
- PyTorch `2.7.1+cu128`
- torchvision `0.22.1+cu128`
- torchaudio `2.7.1+cu128`
- `torch_scatter` wheel matched to `torch 2.7 / cu128`
- `gsplat` installed from the official `pt27cu128` wheel index

This configuration is chosen because it works on the local RTX 5080 dev machine
and is a reasonable deployment target for the Quadro RTX 8000 server, assuming
the server's NVIDIA driver is new enough for CUDA 12.8-era PyTorch wheels.

### Validation

Use the following command after provisioning `vbogs-torch`:

```bash
bash scripts/envs.sh check-torch-stack
```

It verifies:

- CUDA visibility in PyTorch
- a real CUDA tensor operation
- `torch_scatter` CUDA execution
- `gsplat` import
- `gaussian_renderer.render` import through `Octree-AnyGS`

## M4 Point Bucketing And Anchor Fits

M4 is split into:

- `M4a` in `vbogs-torch`: bucket each stereo point into every Octree-AnyGS
  anchor cell that contains it, while also writing the globally normalized
  `(xyz, rgb)` rows that VBGS expects.
- `M4b` in `vbogs-jax`: fit a post-hoc VBGS posterior per anchor from those
  packed point assignments.

Run M4a:

```bash
bash -lc 'source scripts/envs.sh activate-torch >/dev/null && \
python scripts/bucket_points.py \
  --drive 2013_05_28_drive_0008_sync'
```

This writes to `data/m4/2013_05_28_drive_0008_sync/`:

- `points_norm.npz` with `points_norm` plus raw sidecar arrays
- `pts_by_anchor.npz` with packed `anchor_offsets` / `point_indices`
- `norm_params.json`
- `bucket_metadata.json`

The current dev-scene summary on the bundled artifacts is:

- `12,792,935` stereo points
- `267,830` anchors across `9` levels
- `104,577` anchors with at least `20` assigned points

Run a smoke test for M4b:

```bash
bash -lc 'source scripts/envs.sh activate-jax >/dev/null && \
python scripts/fit_anchors.py \
  --drive 2013_05_28_drive_0008_sync \
  --max-observed-anchors 5 \
  --log-every 1'
```

Smoke runs write:

- `anchor_posterior.smoke.npz`
- `fit_metadata.smoke.json`

Run the full M4b fit by omitting `--max-observed-anchors`; that writes:

- `anchor_posterior.npz`
- `fit_metadata.json`

The initial hyperparameter defaults match `PLAN.md`:

- `K_INIT=10`
- `K_MAX=40`
- `K_GROWTH_FACTOR=2`
- `MIN_POINTS_PER_ANCHOR=20`
- `ELBO_IMPROVEMENT_TOL=0.01`
