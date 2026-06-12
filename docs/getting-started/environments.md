# Runtime Services

VBOGS keeps PyTorch and JAX in separate Docker services. Do not merge them. The
framework boundary is the filesystem.

## Services

The normal path is Docker Compose:

| Service | Runtime | Use |
| --- | --- | --- |
| `vbogs-torch` | PyTorch/CUDA plus Octree-AnyGS | preparation, training, stereo, bucketing, rendering, NBV scoring |
| `vbogs-jax` | JAX/CUDA plus VBGS | anchor fitting and uncertainty computation |
| `vbogs-pipeline` | lightweight orchestration image | running stage commands in sibling containers, bundling, uploads |
| `vbogs-filebrowser` | File Browser sidecar | browser access to project files and artifacts; `/outputs` is writable for uploads |

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

The dev overlay bind-mounts the local checkout. Pull-only and Portainer stacks
do not bake VBOGS into the images; run `vbogs-bootstrap-repo` from
`vbogs-pipeline` after the stack starts. Server stacks share that bootstrapped
checkout through the `vbogs-repo` volume so File Browser and all runtime
services see the same project files.

## What Runs Where

| Stage | Entry point | Docker service |
| --- | --- | --- |
| M2 train scene | `scripts/train_octree_anygs.py` | `vbogs-torch` |
| M3 world point cloud | `scripts/export_points_world.py` | `vbogs-torch` |
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
