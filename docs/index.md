# VBOGS Documentation

VBOGS combines Octree-AnyGS, a scalable Gaussian-splatting scene
representation, with VBGS, a per-anchor Bayesian uncertainty head.

## Start Here

| Need | Page |
| --- | --- |
| Build containers and run test commands | [Quickstart](getting-started/index.md) |
| Download KITTI-360 data | [Data Setup](getting-started/data.md) |
| Understand the Docker services | [Environments](getting-started/environments.md) |
| Download a server run and view/query it locally | [Download and View Artifacts](getting-started/local-artifact-viewing.md) |
| Query the render server for RGB, uncertainty, and scores | [Render Server API](running/realtime-viewer.md) |

## Core Mental Model

VBOGS has two runtime stacks because PyTorch CUDA and JAX CUDA dependencies are
kept separate:

| Runtime | Owns | Main stages |
| --- | --- | --- |
| `vbogs-torch` | Octree-AnyGS, stereo, bucketing, scalar rendering | M2, M3, M4a, M6 |
| `vbogs-jax` | VBGS posterior fitting and uncertainty reduction | M4b, M5 |
| `vbogs-pipeline` | Orchestration, packaging, uploads | Full pipeline |
| `vbogs-filebrowser` | Browser access to project files and artifacts; `/outputs` is writable for uploads | Operator access |

Data crosses that framework boundary only through `.npz`, `.npy`, `.json`,
`.yaml`, `.ply`, and image files on disk.

## Most Common Commands

Build the local Docker images:

```bash
bash scripts/build_stack_serial.sh
```

Start the development stack:

```bash
docker compose --project-directory . \
  -f docker/compose/compose.yml \
  -f docker/compose/dev.yml \
  up -d --no-build
```

Enter `vbogs-pipeline`, then run the configured dev pipeline:

```bash
./dc_bash.sh
scripts/run_pipeline.sh \
  --config configs/pipeline/dev.yaml
```

Preview commands without running expensive stages from inside `vbogs-pipeline`:

```bash
scripts/run_pipeline.sh \
  --config configs/pipeline/dev.yaml \
  --dry-run
```
