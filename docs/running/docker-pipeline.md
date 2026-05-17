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

From inside `vbogs-pipeline`:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --gpu 0 \
  --jax-device 0 \
  --start-at prepare \
  --stop-after bundle \
  --use-service-labels
```

From the host, omit `--use-service-labels` and provide compose details when
you need something other than the defaults:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --compose-file docker/compose/compose.yml \
  --compose-project-directory . \
  --drive 2013_05_28_drive_0007_sync
```

## Dry Runs

Use `--dry-run` whenever you change drive ids, profiles, stage slices, or
output roots:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --use-service-labels \
  --dry-run
```

## Config Profiles

| Profile | Intended use |
| --- | --- |
| `configs/pipeline/dev.yaml` | Local Docker Compose development stack; outputs are visible in local `outputs/` |
| `configs/pipeline/portainer.yaml` | Portainer deployment with stack-managed volumes |
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
  --use-service-labels
```

Train only, using already prepared data:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --start-at train \
  --stop-after train \
  --gpu 0 \
  --use-service-labels
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
  --use-service-labels
```

Run only visualization and packaging after M5/M6 artifacts exist:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --start-at map-viz \
  --stop-after bundle \
  --use-service-labels
```
