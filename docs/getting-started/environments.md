# Environments

VBOGS keeps PyTorch and JAX in separate environments. Do not merge them. The
framework boundary is the filesystem.

## Docker Environments

The normal path is Docker Compose:

| Service | Runtime | Use |
| --- | --- | --- |
| `vbogs-torch` | PyTorch/CUDA plus Octree-AnyGS | preparation, training, stereo, bucketing, rendering, NBV scoring |
| `vbogs-jax` | JAX/CUDA plus VBGS | anchor fitting and uncertainty computation |
| `vbogs-pipeline` | lightweight orchestration image | running stage commands in sibling containers, bundling, uploads |

Build with:

```bash
bash scripts/build_stack_serial.sh
```

Start local dev containers with:

```bash
docker compose --project-directory . \
  -f docker/compose/compose.yml \
  -f docker/compose/dev.yml \
  up -d --no-build
```

## Conda Helpers

For non-Docker work, `scripts/envs.sh` creates and activates the two conda
environments:

```bash
bash scripts/envs.sh create-all
source scripts/envs.sh activate-torch
source scripts/envs.sh activate-jax
```

Smoke tests:

```bash
bash scripts/envs.sh smoke-test-torch
bash scripts/envs.sh smoke-test-jax
bash scripts/envs.sh smoke-test-all
```

Torch GPU checks:

```bash
bash scripts/envs.sh verify-torch-gpu
bash scripts/envs.sh check-torch-stack
```

## What Runs Where

| Stage | Entry point | Environment |
| --- | --- | --- |
| M2 train scene | `scripts/train_octree_anygs.py` | `vbogs-torch` |
| M3 stereo point cloud | `scripts/stereo_to_pointcloud.py` | `vbogs-torch` |
| M4a bucket points | `scripts/bucket_points.py` | `vbogs-torch` |
| M4b fit anchors | `scripts/fit_anchors.py` | `vbogs-jax` |
| M5 compute uncertainty | `scripts/compute_uncertainty.py` | `vbogs-jax` or NumPy-compatible Python |
| M6 render/score NBV | `scripts/render_uncertainty_views.py`, `scripts/score_nbv.py` | `vbogs-torch` |

## Filesystem Contract

PyTorch and JAX do not share in-process tensors. Stage outputs are files:

- `.npz` for structured NumPy arrays.
- `.npy` for dense arrays such as `U.npy`.
- `.json` for metadata and normalization parameters.
- `.yaml` for generated Octree-AnyGS and pipeline config.
- `.ply` for point-cloud inspection.
- `.png` and `.mp4` for diagnostics.
