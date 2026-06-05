# Docker Pipeline

`scripts/run_pipeline.sh` is the operator entry point. Run it from inside
`vbogs-pipeline`; it forwards arguments to the internal Python runner, resolves
the sibling Torch and JAX containers, and executes each stage in the correct
Docker service.

## Stage Order

```text
prepare -> train -> stereo -> bucket -> fit -> inspect -> uncertainty -> map-viz -> render -> nbv -> nbv-viz -> bundle
```

Use `--start-at` and `--stop-after` to run a slice.

## Common Invocation

For local development, enter `vbogs-pipeline`. The dev overlay mounts the host
Docker socket into that container so the wrapper can resolve sibling containers
by Compose label.

```bash
scripts/run_pipeline.sh \
  --drive 2013_05_28_drive_0007_sync \
  --gpu 0 \
  --jax-device 0 \
  --start-at prepare \
  --stop-after bundle
```

For non-dev stacks where the repo is not bind-mounted, run
`vbogs-bootstrap-repo` from `vbogs-pipeline` once so the shared `vbogs-repo`
volume contains the checkout. Then run the same pipeline command from inside
`vbogs-pipeline`:

```bash
scripts/run_pipeline.sh \
  --config configs/pipeline/portainer.yaml \
  --drive 2013_05_28_drive_0007_sync
```

## Dry Runs

Use `--dry-run` whenever you change drive ids, profiles, stage slices, or
output roots:

```bash
scripts/run_pipeline.sh \
  --config configs/pipeline/dev.yaml \
  --dry-run
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
scripts/run_pipeline.sh \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --start-at prepare \
  --stop-after prepare
```

Train only, using already prepared data:

```bash
scripts/run_pipeline.sh \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --start-at train \
  --stop-after train \
  --gpu 0
```

Resume after training:

```bash
scripts/run_pipeline.sh \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --start-at stereo \
  --stop-after bundle \
  --gpu 0 \
  --jax-device 0
```

Run only visualization and packaging after M5/M6 artifacts exist:

```bash
scripts/run_pipeline.sh \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --start-at map-viz \
  --stop-after bundle
```
