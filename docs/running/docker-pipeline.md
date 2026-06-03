# Docker Pipeline

`scripts/run_drive_pipeline.py` is the main operator entry point. It loads a
YAML config, applies CLI overrides, and runs selected stages in the correct
Docker service.

## Stage Order

```text
prepare -> train -> stereo -> bucket -> fit -> inspect -> uncertainty -> map-viz -> render -> nbv -> nbv-viz -> bundle
```

Use `--start-at` and `--stop-after` to run a slice.

## Common Invocation

Run pipeline orchestration from the Docker host. VBOGS containers no longer
mount the host Docker socket, so `vbogs-pipeline` does not control sibling
containers.

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --gpu 0 \
  --jax-device 0 \
  --start-at prepare \
  --stop-after bundle \
  --compose-file docker/compose/compose.yml \
  --compose-file docker/compose/dev.yml \
  --compose-project-directory .
```

For non-dev stacks where the repo is not bind-mounted, run
`vbogs-bootstrap-repo` from `vbogs-pipeline` once so the shared `vbogs-repo`
volume contains the checkout. Then invoke pipeline scripts from the Docker
host with the compose file for that stack.

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/portainer.yaml \
  --compose-file docker/compose/portainer.yml \
  --compose-project-directory . \
  --drive 2013_05_28_drive_0007_sync
```

## Dry Runs

Use `--dry-run` whenever you change drive ids, profiles, stage slices, or
output roots:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --dry-run \
  --compose-file docker/compose/compose.yml \
  --compose-file docker/compose/dev.yml \
  --compose-project-directory .
```

## Config Profiles

| Profile | Intended use |
| --- | --- |
| `configs/pipeline/dev.yaml` | Local Docker Compose development stack; repo checkout bind-mounted, artifacts in Docker volumes |
| `configs/pipeline/portainer.yaml` | Portainer deployment with compose-managed Docker volumes |
| `configs/pipeline/default.yaml` | Backward-compatible default profile |

CLI flags override config values. For the full mapping, see
[Configuration](../reference/configuration.md).

## Stage Slices

Prepare data only:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --start-at prepare \
  --stop-after prepare \
  --compose-file docker/compose/compose.yml \
  --compose-file docker/compose/dev.yml \
  --compose-project-directory .
```

Train only, using already prepared data:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --start-at train \
  --stop-after train \
  --gpu 0 \
  --compose-file docker/compose/compose.yml \
  --compose-file docker/compose/dev.yml \
  --compose-project-directory .
```

Resume after training:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --start-at stereo \
  --stop-after bundle \
  --gpu 0 \
  --jax-device 0 \
  --compose-file docker/compose/compose.yml \
  --compose-file docker/compose/dev.yml \
  --compose-project-directory .
```

Run only visualization and packaging after M5/M6 artifacts exist:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --start-at map-viz \
  --stop-after bundle \
  --compose-file docker/compose/compose.yml \
  --compose-file docker/compose/dev.yml \
  --compose-project-directory .
```
