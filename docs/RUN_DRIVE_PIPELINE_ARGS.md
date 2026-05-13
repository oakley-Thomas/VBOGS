# `scripts/run_drive_pipeline.py` arguments

`scripts/run_drive_pipeline.py` orchestrates the implemented VBOGS stages across
the Docker Compose stack. Torch stages run in `vbogs-torch`; JAX stages run in
`vbogs-jax`. Values in `pipeline_config.yaml` become defaults, and explicit CLI
arguments override the config.

Common full-pipeline command from inside `vbogs-pipeline`:

```bash
python scripts/run_drive_pipeline.py \
  --drive 2013_05_28_drive_0000_sync \
  --gpu 0 \
  --jax-device 0 \
  --start-at prepare \
  --stop-after bundle \
  --run-output-root outputs/v1_0 \
  --use-service-labels
```

Use `--dry-run` to print the commands that would be run without launching the
expensive work.

## Config Profiles

Use the profile config that matches where the stack is running:

| File | Intended use | Output location |
| --- | --- | --- |
| `pipeline_config.dev.yaml` | Local Docker Compose development stack | `outputs/v1_0/<drive>/` in this checkout, via the dev compose bind mount |
| `pipeline_config.portainer.yaml` | Portainer deployment | `outputs/v1_0/<drive>/` inside the `vbogs-outputs` Docker volume |
| `pipeline_config.yaml` | Backward-compatible default | Depends on the active compose mounts |

For local development, plain `docker compose up` auto-loads
`docker-compose.override.yml`, which bind-mounts `${VBOGS_LOCAL_OUTPUTS:-./outputs}`
to `/workspace/VBOGS/outputs`. The same setup is available explicitly with:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --no-build
```

From inside `vbogs-pipeline`, run the local-dev profile with:

```bash
python scripts/run_drive_pipeline.py --config pipeline_config.dev.yaml --use-service-labels
```

## Pipeline Selection

| Argument | Default | Description |
| --- | --- | --- |
| `--config CONFIG` | `pipeline_config.yaml` | YAML file used for defaults. Pass an empty string (`--config ""`) to disable config loading. |
| `--drive DRIVE` | Config: `pipeline.drive` | KITTI-360 drive id, for example `2013_05_28_drive_0008_sync`. Required if not set in config. |
| `--start-at {prepare,train,stereo,bucket,fit,inspect,uncertainty,map-viz,render,nbv,nbv-viz,bundle}` | `prepare` | First stage to run. |
| `--stop-after {prepare,train,stereo,bucket,fit,inspect,uncertainty,map-viz,render,nbv,nbv-viz,bundle}` | `inspect` | Last stage to run. Use `bundle` for the full curated run output. |
| `--run-output-root RUN_OUTPUT_ROOT` | Config: `outputs.run_root` | Optional root for curated outputs. When set, stage outputs are derived under `<root>/<drive>/`. |
| `--dry-run` | `false` | Print the Docker/stage commands without executing them. |

Stage order is:

```text
prepare -> train -> stereo -> bucket -> fit -> inspect -> uncertainty -> map-viz -> render -> nbv -> nbv-viz -> bundle
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

The `vbogs-pipeline` service is also GPU-enabled, so after the stack is
running you can check host GPU visibility from the pipeline container:

```bash
docker compose exec vbogs-pipeline nvidia-smi
```

## Google Drive Upload

The pipeline image includes `rclone` and `scripts/upload_google_drive.py`.
When `VBOGS_GDRIVE_UPLOAD=1`, `scripts/run_pipeline_from_env.py` uploads the
curated archive after a successful pipeline run. By default that source is:

```text
outputs/v1_0/<drive>.zip
```

Recommended service-account setup for a private Google Drive folder:

1. Create a Google service account.
2. Share the private Drive folder with the service account email.
3. Copy the folder id from the Drive URL.
4. Set these stack environment variables:

```bash
VBOGS_GDRIVE_UPLOAD=1
VBOGS_GDRIVE_REMOTE=vbogs_gdrive
VBOGS_GDRIVE_FOLDER_ID=<google-drive-folder-id>
VBOGS_GDRIVE_SERVICE_ACCOUNT_CREDENTIALS={"type":"service_account",...}
```

If you mount the JSON credentials file into the container instead, set:

```bash
VBOGS_GDRIVE_SERVICE_ACCOUNT_FILE=/run/secrets/vbogs-google-drive-service-account.json
```

Optional upload controls:

| Environment variable | Description |
| --- | --- |
| `VBOGS_GDRIVE_DEST` | Destination path inside the configured remote/root folder. Empty means the folder root. |
| `VBOGS_GDRIVE_SOURCE` | Override the upload source file or directory. |
| `VBOGS_GDRIVE_SCOPE` | rclone Drive scope. Defaults to `drive` for service-account uploads. |
| `VBOGS_GDRIVE_RCLONE_ARGS` | Extra arguments appended to the rclone command, for example `--progress --checksum`. |
| `VBOGS_GDRIVE_DRY_RUN` | Set to `1` to print the upload command without transferring. |

Manual upload example from inside `vbogs-pipeline`:

```bash
python scripts/upload_google_drive.py \
  --config pipeline_config.portainer.yaml \
  --folder-id <google-drive-folder-id> \
  --service-account-file /run/secrets/vbogs-google-drive-service-account.json
```

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
| `--frame-step FRAME_STEP` | Config: `1` | Keep every Nth frame from the drive. Higher values are faster and smaller. |
| `--max-frames MAX_FRAMES` | Config: `1000` | Maximum number of frames to prepare. `0` means no cap. |
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
| `--iterations ITERATIONS` | Config: `30000` | Number of training iterations. |
| `--llffhold LLFFHOLD` | `8` | Held-out test frame cadence used by the Octree-AnyGS data loader. |
| `--gaussian-type {implicit3D,explicit3D}` | `implicit3D` | Octree-AnyGS Gaussian representation. `implicit3D` is the neural default; `explicit3D` uses explicit SH 3D Gaussians. |
| `--feat-dim FEAT_DIM` | `16` | Neural anchor feature dimension. Lower values reduce VRAM pressure. Ignored for `explicit3D`. |
| `--base-layer BASE_LAYER` | `9` | LoD base layer. Lower values reduce anchor count and memory. |
| `--visible-threshold VISIBLE_THRESHOLD` | `0.02` | LoD pruning visibility threshold. |
| `--train-port TRAIN_PORT` | Auto | Octree-AnyGS network GUI port. The wrapper defaults to `6009 + GPU index`, so GPU 1 uses `6010`. |
| `--write-config-only` | `false` | Generate the Octree-AnyGS YAML config and skip training. |

## `stereo`

Runs `scripts/stereo_to_pointcloud.py` in `vbogs-torch` and writes world-frame
stereo points under `data/points_world/<drive>/`.

| Argument | Default | Description |
| --- | --- | --- |
| `--matcher {sgbm,raft}` | `sgbm` | Stereo matcher backend. `raft` is reserved for a future provider unless installed/implemented. |
| `--pixel-step PIXEL_STEP` | `1` | Pixel subsampling step for point export. Higher values reduce density and runtime. |
| `--max-points-per-frame MAX_POINTS_PER_FRAME` | `250000` | Per-frame cap on exported stereo points. |
| `--write-ply` | Config: `true` | Also write a PLY point cloud for quick visual inspection and the curated bundle. |

The stereo stage also receives `--max-frames` and any KITTI-360 input override
arguments.

## `bucket`

Runs `scripts/bucket_points.py` in `vbogs-torch` and writes packed point-to-anchor
assignments under `data/m4/<drive>/`.

| Argument | Default | Description |
| --- | --- | --- |
| `--model-path MODEL_PATH` | Latest run under `/data/OCTREE-ANYGS/<drive>` | Explicit Octree-AnyGS model/run directory to bucket against. |
| `--bucket-iteration BUCKET_ITERATION` | `-1` | Checkpoint iteration to load. `-1` means use the latest available checkpoint. |
| `--bucket-point-chunk-size BUCKET_POINT_CHUNK_SIZE` | `1000000` | Number of stereo points processed per bucketing chunk. Lower values reduce peak memory. |
| `--bucket-max-points BUCKET_MAX_POINTS` | `0` | Optional deterministic cap on stereo points used for M4 bucketing/fitting. `0` keeps all points. |

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

## `map-viz`

Runs `scripts/export_uncertainty_map.py` in `vbogs-torch` and writes
CloudCompare-friendly colored anchor PLYs. With `--run-output-root outputs/v1_0`,
the default output directory is `outputs/v1_0/<drive>/pointclouds/anchors`.

| Argument | Default | Description |
| --- | --- | --- |
| `--map-viz-output-dir MAP_VIZ_OUTPUT_DIR` | Derived from run root | Optional map-scale PLY output directory. |
| `--map-viz-vmin MAP_VIZ_VMIN` | Auto | Explicit lower bound for anchor uncertainty colors. |
| `--map-viz-vmax MAP_VIZ_VMAX` | Auto | Explicit upper bound for anchor uncertainty colors. |
| `--map-viz-percentile-low MAP_VIZ_PERCENTILE_LOW` | `2.0` | Lower observed-anchor percentile used for automatic color scale. |
| `--map-viz-percentile-high MAP_VIZ_PERCENTILE_HIGH` | `98.0` | Upper observed-anchor percentile used for automatic color scale. |
| `--map-viz-observed-only` | `false` | Export only observed anchors. |
| `--map-viz-no-split-levels` | `false` | Only write the combined all-levels PLY. |
| `--map-viz-no-trajectory` | `false` | Skip `camera_trajectory.ply`. |

## `render`

Runs `scripts/render_uncertainty_views.py` in `vbogs-torch` and writes diagnostic
RGB/uncertainty views. With `--run-output-root outputs/v1_0`, the default output
directory is `outputs/v1_0/<drive>/views`.

| Argument | Default | Description |
| --- | --- | --- |
| `--render-split {train,test,both}` | `both` | Camera split to render. |
| `--render-resolution RENDER_RESOLUTION` | Config: `2` | Octree-AnyGS image divisor/target width for diagnostic renders. Smaller divisors produce higher-resolution views; `1` is full input resolution. |
| `--render-max-views RENDER_MAX_VIEWS` | `0` | Per-split cap for render smoke tests. `0` renders all views. |
| `--render-colormap RENDER_COLORMAP` | `turbo` | Matplotlib colormap for uncertainty heatmaps. |
| `--render-vmin RENDER_VMIN` | Auto | Lower bound for uncertainty colormap normalization. |
| `--render-vmax RENDER_VMAX` | Auto | Upper bound for uncertainty colormap normalization. |
| `--render-output-dir RENDER_OUTPUT_DIR` | Derived from run root | Optional render output root in the Torch container. |

## `nbv`

Runs `scripts/score_nbv.py` in `vbogs-torch` and writes NBV scores plus top
uncertainty/alpha arrays. With `--run-output-root outputs/v1_0`, the default
output directory is `outputs/v1_0/<drive>/nbv`.

| Argument | Default | Description |
| --- | --- | --- |
| `--nbv-candidate-source {test,train,lattice}` | `test` | Candidate camera set used for scoring. |
| `--nbv-max-candidates NBV_MAX_CANDIDATES` | `0` | Optional candidate cap. `0` scores all selected candidates. |
| `--nbv-top-k NBV_TOP_K` | `10` | Number of ranked candidates stored in `nbv_scores.json`. |
| `--nbv-save-top-images NBV_SAVE_TOP_IMAGES` | `5` | Number of top uncertainty/alpha arrays saved for visualization. |
| `--nbv-force-all-levels` | `false` | Force all Octree-AnyGS levels active during scalar renders. |
| `--nbv-output-dir NBV_OUTPUT_DIR` | Derived from run root | Optional NBV output directory. |

## `nbv-viz`

Runs `scripts/visualize_m6.py` in `vbogs-torch` and converts saved top NBV
uncertainty/alpha arrays into PNG diagnostics under `<nbv-output-dir>/viz`.

## `bundle`

Runs `scripts/bundle_run_outputs.py` in `vbogs-torch`. It copies curated,
user-facing artifacts into `outputs/v1_0/<drive>` and writes
`run_manifest.json`, then zips that output folder to
`outputs/v1_0/<drive>.zip`. Bulky Octree-AnyGS checkpoints and full VBGS
posterior artifacts remain in their native data volumes and are referenced by
path.

Bundled outputs include:

- `pointclouds/stereo/points_world.npz`, optional `points_world.ply`, and metadata
- `pointclouds/anchors/` generated by `map-viz`
- `views/` generated by `render`
- `nbv/` generated by `nbv` and `nbv-viz`
- `uncertainty/U.npy`, uncertainty metadata/components, and histogram when present
- `prepared/metadata.json`, `octree/config.yaml`, and `run_manifest.json`

## Config Mapping

The default config file uses section names that map to CLI arguments:

| Config section | Example keys |
| --- | --- |
| `pipeline` | `drive`, `start_at`, `stop_after`, `dry_run`, `skip_up` |
| `inputs` | `raw_root`, `poses_root`, `calibration_dir` |
| `prepare` | `frame_step`, `max_frames`, `copy_mode`, `seed_mode` |
| `train` | `gpu`, `resolution`, `iterations`, `llffhold`, `gaussian_type`, `feat_dim`, `base_layer`, `visible_threshold`, `port`, `write_config_only` |
| `stereo` | `matcher`, `pixel_step`, `max_points_per_frame`, `write_ply` |
| `bucket` | `model_path`, `bucket_iteration`, `point_chunk_size`, `max_points` |
| `fit` | `jax_device`, `fit_mode`, `batch_size`, `vmap_group_size`, `log_every`, `max_observed_anchors` |
| `inspect` | `top_k`, `sample_points`, `anchor_id`, `export_ply` |
| `uncertainty` | `u_max`, `no_histogram` |
| `map_viz` | `output_dir`, `vmin`, `vmax`, `percentile_low`, `percentile_high`, `observed_only`, `no_split_levels`, `no_trajectory` |
| `render` | `split`, `resolution`, `max_views`, `colormap`, `vmin`, `vmax`, `output_dir` |
| `nbv` | `candidate_source`, `max_candidates`, `top_k`, `save_top_images`, `force_all_levels`, `output_dir` |
| `outputs` | `run_root` |
| `orchestration` | `compose_command`, `compose_file`, `project_name`, `torch_container`, `jax_container`, `use_service_labels`, `label_project` |
