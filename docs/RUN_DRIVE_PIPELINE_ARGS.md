# `scripts/run_drive_pipeline.py` arguments

`scripts/run_drive_pipeline.py` orchestrates the implemented VBOGS stages across
the Docker Compose stack. Torch stages run in `vbogs-torch`; JAX stages run in
`vbogs-jax`. Values in `pipeline_config.yaml` become defaults, and explicit CLI
arguments override the config.

Common full-pipeline command from inside `vbogs-pipeline`:

```bash
python scripts/run_drive_pipeline.py \
  --drive 2013_05_28_drive_0008_sync \
  --gpu 0 \
  --jax-device 0 \
  --start-at prepare \
  --stop-after render \
  --use-service-labels
```

Use `--dry-run` to print the commands that would be run without launching the
expensive work.

## Pipeline Selection

| Argument | Default | Description |
| --- | --- | --- |
| `--config CONFIG` | `pipeline_config.yaml` | YAML file used for defaults. Pass an empty string (`--config ""`) to disable config loading. |
| `--drive DRIVE` | Config: `pipeline.drive` | KITTI-360 drive id, for example `2013_05_28_drive_0008_sync`. Required if not set in config. |
| `--start-at {prepare,train,stereo,bucket,fit,inspect,uncertainty,render}` | `prepare` | First stage to run. |
| `--stop-after {prepare,train,stereo,bucket,fit,inspect,uncertainty,render}` | `inspect` | Last stage to run. Use `render` for the full implemented pipeline. |
| `--dry-run` | `false` | Print the Docker/stage commands without executing them. |

Stage order is:

```text
prepare -> train -> stereo -> bucket -> fit -> inspect -> uncertainty -> render
```

## Orchestration

| Argument | Default | Description |
| --- | --- | --- |
| `--compose-command COMPOSE_COMMAND` | `docker compose` | Compose command used when running from the host. |
| `--compose-file COMPOSE_FILE` | `docker-compose.yml` | Compose file used when running from the host. |
| `--project-name PROJECT_NAME` | Empty | Optional Compose/Portainer stack project name passed as `-p`. |
| `--torch-container TORCH_CONTAINER` | Empty | Concrete container name/id for Torch stages. When set, the runner uses `docker exec` instead of `docker compose exec` for Torch. |
| `--jax-container JAX_CONTAINER` | Empty | Concrete container name/id for JAX stages. When set, the runner uses `docker exec` instead of `docker compose exec` for JAX. |
| `--use-service-labels` | `false` | Resolve sibling containers by Docker Compose labels. Use this from inside `vbogs-pipeline`. |
| `--label-project LABEL_PROJECT` | `VBOGS_COMPOSE_PROJECT` or auto-detected | Compose project label to use with `--use-service-labels`. Usually unnecessary inside the stack. |
| `--skip-up` | `false` | Do not run `docker compose up -d` before executing selected stages. Automatically skipped with `--use-service-labels`. |

## KITTI-360 Inputs

These override the default source-data discovery used by the prep and stereo
stages.

| Argument | Default | Description |
| --- | --- | --- |
| `--raw-root RAW_ROOT` | Auto-detect | Root containing KITTI-360 rectified stereo images. |
| `--poses-root POSES_ROOT` | Auto-detect | Root containing KITTI-360 pose text files. |
| `--calibration-dir CALIBRATION_DIR` | Auto-detect | Directory containing KITTI-360 calibration text files. |

## `prepare`

Runs `scripts/prepare_kitti360_colmap.py` in `vbogs-torch` and writes a
COLMAP-style dataset under `/data/COLMAP/<drive>`.

| Argument | Default | Description |
| --- | --- | --- |
| `--frame-step FRAME_STEP` | `10` | Keep every Nth frame from the drive. Higher values are faster and smaller. |
| `--max-frames MAX_FRAMES` | `0` | Maximum number of frames to prepare. `0` means no cap. |
| `--copy-mode {symlink,copy}` | `symlink` | How images are placed in the prepared dataset. `symlink` is faster and saves space when supported. |
| `--seed-mode {stereo,random}` | `stereo` | How the initial point cloud is seeded for Octree-AnyGS ingest. |

## `train`

Runs `scripts/train_octree_anygs.py` in `vbogs-torch`. The generated config is
written under `generated_configs/`, and Octree-AnyGS outputs go under
`/data/OCTREE-ANYGS/<drive>/<timestamp>/`.

| Argument | Default | Description |
| --- | --- | --- |
| `--gpu GPU` | `0` | GPU id passed to the Octree-AnyGS training wrapper. |
| `--resolution RESOLUTION` | `4` | Octree-AnyGS image divisor. Higher values reduce memory use and image fidelity. |
| `--iterations ITERATIONS` | `15000` | Number of training iterations. |
| `--llffhold LLFFHOLD` | `8` | Held-out test frame cadence used by the Octree-AnyGS data loader. |
| `--feat-dim FEAT_DIM` | `16` | Anchor feature dimension. Lower values reduce VRAM pressure. |
| `--base-layer BASE_LAYER` | `9` | LoD base layer. Lower values reduce anchor count and memory. |
| `--visible-threshold VISIBLE_THRESHOLD` | `0.02` | LoD pruning visibility threshold. |
| `--write-config-only` | `false` | Generate the Octree-AnyGS YAML config and skip training. |

## `stereo`

Runs `scripts/stereo_to_pointcloud.py` in `vbogs-torch` and writes world-frame
stereo points under `data/points_world/<drive>/`.

| Argument | Default | Description |
| --- | --- | --- |
| `--matcher {sgbm,raft}` | `sgbm` | Stereo matcher backend. `raft` is reserved for a future provider unless installed/implemented. |
| `--pixel-step PIXEL_STEP` | `1` | Pixel subsampling step for point export. Higher values reduce density and runtime. |
| `--max-points-per-frame MAX_POINTS_PER_FRAME` | `250000` | Per-frame cap on exported stereo points. |
| `--write-ply` | `false` | Also write a PLY point cloud for quick visual inspection. |

The stereo stage also receives `--max-frames` and any KITTI-360 input override
arguments.

## `bucket`

Runs `scripts/bucket_points.py` in `vbogs-torch` and writes packed point-to-anchor
assignments under `data/m4/<drive>/`.

| Argument | Default | Description |
| --- | --- | --- |
| `--model-path MODEL_PATH` | Latest run under `/data/OCTREE-ANYGS/<drive>` | Explicit Octree-AnyGS model/run directory to bucket against. |
| `--bucket-iteration BUCKET_ITERATION` | `-1` | Checkpoint iteration to load. `-1` means use the latest available checkpoint. |

## `fit`

Runs `scripts/fit_anchors.py` in `vbogs-jax` and writes VBGS posterior files
under `data/m4/<drive>/`.

| Argument | Default | Description |
| --- | --- | --- |
| `--jax-device JAX_DEVICE` | `0` | JAX device index used for VBGS fitting. |
| `--fit-mode {batched,loop}` | `batched` | Fit implementation. `batched` is the normal path; `loop` is simpler but slower. |
| `--batch-size BATCH_SIZE` | `5000` | Number of anchors/assignments processed per batch, depending on fit mode internals. |
| `--vmap-group-size VMAP_GROUP_SIZE` | `64` | Group size for vectorized JAX fitting work. |
| `--log-every LOG_EVERY` | `100` | Progress logging interval. |
| `--max-observed-anchors MAX_OBSERVED_ANCHORS` | `0` | Smoke-test cap for observed anchors. `0` means full fit. Positive values write `anchor_posterior.smoke.npz`. |

## `inspect`

Runs `scripts/inspect_anchor_fits.py` in `vbogs-jax`.

| Argument | Default | Description |
| --- | --- | --- |
| `--inspect-top-k INSPECT_TOP_K` | `5` | Number of anchors shown per inspection heuristic list. |
| `--inspect-sample-points INSPECT_SAMPLE_POINTS` | `5` | Number of assigned points printed when `--inspect-anchor-id` is used. |
| `--inspect-anchor-id INSPECT_ANCHOR_ID` | Empty | Explicit anchor id to inspect. |
| `--inspect-export-ply INSPECT_EXPORT_PLY` | Empty | Optional PLY export path for assigned points from `--inspect-anchor-id`. |

## `uncertainty`

Runs `scripts/compute_uncertainty.py` in `vbogs-jax` and writes `U.npy` under
`data/m4/<drive>/`.

| Argument | Default | Description |
| --- | --- | --- |
| `--uncertainty-u-max UNCERTAINTY_U_MAX` | Maximum finite observed uncertainty | Value assigned to unobserved anchors. |
| `--uncertainty-no-histogram` | `false` | Skip writing the M5 uncertainty histogram PNG. |

## `render`

Runs `scripts/render_uncertainty_views.py` in `vbogs-torch` and writes diagnostic
RGB/uncertainty views.

| Argument | Default | Description |
| --- | --- | --- |
| `--render-split {train,test,both}` | `both` | Camera split to render. |
| `--render-max-views RENDER_MAX_VIEWS` | `0` | Per-split cap for render smoke tests. `0` renders all views. |
| `--render-colormap RENDER_COLORMAP` | `turbo` | Matplotlib colormap for uncertainty heatmaps. |
| `--render-vmin RENDER_VMIN` | Auto | Lower bound for uncertainty colormap normalization. |
| `--render-vmax RENDER_VMAX` | Auto | Upper bound for uncertainty colormap normalization. |
| `--render-output-dir RENDER_OUTPUT_DIR` | `outputs/uncertainty_views/<drive>` | Optional render output root in the Torch container. |

## Config Mapping

The default config file uses section names that map to CLI arguments:

| Config section | Example keys |
| --- | --- |
| `pipeline` | `drive`, `start_at`, `stop_after`, `dry_run`, `skip_up` |
| `inputs` | `raw_root`, `poses_root`, `calibration_dir` |
| `prepare` | `frame_step`, `max_frames`, `copy_mode`, `seed_mode` |
| `train` | `gpu`, `resolution`, `iterations`, `llffhold`, `feat_dim`, `base_layer`, `visible_threshold`, `write_config_only` |
| `stereo` | `matcher`, `pixel_step`, `max_points_per_frame`, `write_ply` |
| `bucket` | `model_path`, `bucket_iteration` |
| `fit` | `jax_device`, `fit_mode`, `batch_size`, `vmap_group_size`, `log_every`, `max_observed_anchors` |
| `inspect` | `top_k`, `sample_points`, `anchor_id`, `export_ply` |
| `uncertainty` | `u_max`, `no_histogram` |
| `render` | `split`, `max_views`, `colormap`, `vmin`, `vmax`, `output_dir` |
| `orchestration` | `compose_command`, `compose_file`, `project_name`, `torch_container`, `jax_container`, `use_service_labels`, `label_project` |

