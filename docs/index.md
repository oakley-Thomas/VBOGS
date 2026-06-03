# VBOGS Documentation

VBOGS combines Octree-AnyGS, a scalable Gaussian-splatting scene
representation, with VBGS, a per-anchor Bayesian uncertainty head. The pipeline
trains a scene, assigns exported world points to Octree-AnyGS anchors, fits posterior
models per anchor, renders uncertainty, and scores next-best camera views.

## Start Here

| Need | Page |
| --- | --- |
| Build containers and run test commands | [Quickstart](getting-started/index.md) |
| Download Kitti-360 Data | [Data Setup](getting-started/data.md) |
| Run a complete drive through the pipeline | [Full Drive Runs](running/full-drive.md) |
| Use the browser viewer or viewer APIs | [Realtime Viewer](running/realtime-viewer.md) |
| Understand what each stage is doing | [Algorithm Overview](algorithm/index.md) |
| Look up every `run_drive_pipeline.py` argument | [Pipeline Arguments](documentation/RUN_DRIVE_PIPELINE_ARGS.md) |
| Find output files and artifact contracts | [Artifacts and Data Layout](reference/artifacts.md) |
| Debug common failures | [Troubleshooting](reference/troubleshooting.md) |

## Core Mental Model

VBOGS has two runtime stacks because PyTorch CUDA and JAX CUDA dependencies are
kept separate:

| Runtime | Owns | Main stages |
| --- | --- | --- |
| `vbogs-torch` | Octree-AnyGS, stereo, bucketing, scalar rendering | M2, M3, M4a, M6 |
| `vbogs-jax` | VBGS posterior fitting and uncertainty reduction | M4b, M5 |
| `vbogs-pipeline` | Orchestration, packaging, uploads | Full pipeline |
| `vbogs-filebrowser` | Read-only browser access to project files and artifacts | Operator access |

Data crosses that framework boundary only through `.npz`, `.npy`, `.json`,
`.yaml`, `.ply`, and image files on disk.

## Current Status

The repo has implemented entry points for M1 through M6. M7 is the remaining
human validation pass: choose a scene with known uncertain regions, inspect
anchor posteriors, render uncertainty overlays, and confirm that the selected
next-best view matches intuition. See [Status and Milestones](algorithm/status.md).

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

Run the configured dev pipeline from the Docker host:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --compose-file docker/compose/compose.yml \
  --compose-file docker/compose/dev.yml \
  --compose-project-directory .
```

Preview commands without running expensive stages:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --dry-run \
  --compose-file docker/compose/compose.yml \
  --compose-file docker/compose/dev.yml \
  --compose-project-directory .
```
