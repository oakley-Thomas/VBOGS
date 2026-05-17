# VBOGS Usage

This is the short operator guide for the repo-owned Docker pipeline. For the
complete argument reference, see
[docs/documentation/RUN_DRIVE_PIPELINE_ARGS.md](docs/documentation/RUN_DRIVE_PIPELINE_ARGS.md).

## Build the Stack

```bash
bash scripts/build_stack_serial.sh
```

To rebuild one service:

```bash
bash scripts/build_stack_serial.sh vbogs-torch
bash scripts/build_stack_serial.sh vbogs-jax
bash scripts/build_stack_serial.sh vbogs-pipeline
```

## Start Local Development Containers

Plain Docker Compose automatically reads `docker-compose.override.yml`, which
bind-mounts this checkout into the containers and maps local `outputs/` to
`/workspace/VBOGS/outputs`.

```bash
docker compose up -d --no-build
docker compose exec vbogs-pipeline bash
```

From inside `vbogs-pipeline`, run the default configured pipeline:

```bash
python scripts/run_drive_pipeline.py --config pipeline_config.dev.yaml --use-service-labels
```

## Run a Full Drive

```bash
python scripts/run_drive_pipeline.py \
  --config pipeline_config.dev.yaml \
  --drive 2013_05_28_drive_0000_sync \
  --gpu 0 \
  --jax-device 0 \
  --start-at prepare \
  --stop-after bundle \
  --use-service-labels
```

Stage order:

```text
prepare -> train -> stereo -> bucket -> fit -> inspect -> uncertainty -> map-viz -> render -> nbv -> nbv-viz -> bundle
```

Use `--dry-run` to print the container commands without launching expensive
work.

## Development Smoke Run

This keeps data volume, training time, and renders small enough for quick
verification:

```bash
python scripts/run_drive_pipeline.py \
  --config pipeline_config.dev.yaml \
  --drive 2013_05_28_drive_0000_sync \
  --gpu 0 \
  --jax-device 0 \
  --start-at prepare \
  --stop-after render \
  --frame-step 20 \
  --max-frames 30 \
  --resolution 4 \
  --iterations 7000 \
  --max-points-per-frame 50000 \
  --render-max-views 2 \
  --use-service-labels
```

## Config Profiles

Use the profile that matches the runtime environment:

| File | Intended use |
| --- | --- |
| `pipeline_config.dev.yaml` | Local Docker Compose development stack |
| `pipeline_config.portainer.yaml` | Portainer deployment with stack-managed volumes |
| `pipeline_config.yaml` | Backward-compatible default profile |

Curated outputs are written under `outputs/v1_0/<drive>/` when
`outputs.run_root` is set. The final `bundle` stage also creates
`outputs/v1_0/<drive>.zip`.
