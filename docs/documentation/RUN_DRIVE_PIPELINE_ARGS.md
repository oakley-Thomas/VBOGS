# `scripts/run_pipeline.sh` arguments

`scripts/run_pipeline.sh` is the container-side operator entry point. Run it
from inside `vbogs-pipeline`; it forwards arguments to the internal Python
runner, which orchestrates implemented VBOGS stages across the Docker Compose
stack. Torch stages run in `vbogs-torch`; JAX stages run in `vbogs-jax`.
Values in `configs/pipeline/default.yaml` become defaults, and explicit CLI
arguments override the config.

Common full-pipeline command from inside `vbogs-pipeline`:

```bash
scripts/run_pipeline.sh \
  --drive 2013_05_28_drive_0000_sync \
  --gpu 0 \
  --jax-device 0 \
  --start-at prepare \
  --stop-after bundle \
  --run-output-root outputs/v1_0
```

Use `--dry-run` to print the commands that would be run without launching the
expensive work.

## Config Profiles

Use the profile config that matches where the stack is running:

| File | Intended use | Output location |
| --- | --- | --- |
| `configs/pipeline/dev.yaml` | Local Docker Compose development stack | `outputs/v1_0/<drive>/` inside the `vbogs-outputs` Docker volume |
| `configs/pipeline/portainer.yaml` | Portainer deployment | `outputs/v1_0/<drive>/` inside the `vbogs-outputs` Docker volume |
| `configs/pipeline/nvidia_ncore_dev.yaml` | NVIDIA PhysicalAI AV NCore development | `outputs/v1_0/<scene-id>/` |
| `configs/pipeline/default.yaml` | Backward-compatible default | Depends on the active compose mounts |

For local development, use the base compose file plus the dev overlay, which
bind-mounts this checkout to `/workspace/VBOGS` while keeping generated
artifacts in Docker volumes:

```bash
docker compose --project-directory . -f docker/compose/compose.yml -f docker/compose/dev.yml up -d --no-build
```

Enter `vbogs-pipeline`, then run the local-dev profile with:

```bash
./dc_bash.sh
scripts/run_pipeline.sh \
  --config configs/pipeline/dev.yaml
```

## Pipeline Selection

| Argument | Default | Description |
| --- | --- | --- |
| `--config CONFIG` | `configs/pipeline/default.yaml` | YAML file used for defaults. Pass an empty string (`--config ""`) to disable config loading. |
| `--dataset-name {kitti360,nvidia_ncore}` | Config: `dataset.name`; parser: `kitti360` | Dataset adapter used by `prepare` and point-cloud export. |
| `--scene-id SCENE_ID` | Config: `dataset.scene_id` | Dataset scene/clip id. Defaults to `--drive` for backward compatibility. |
| `--drive DRIVE` | Config: `pipeline.drive` | Backward-compatible scene id alias. For KITTI-360 this is the drive id. |
| `--ncore-root NCORE_ROOT` | Config: `dataset.ncore_root` | Root containing converted NVIDIA NCore clips. |
| `--camera-id CAMERA_ID` | Config: `dataset.camera_ids` | NVIDIA camera id. Repeat or pass comma-separated ids. |
| `--point-source {stereo,lidar,camera_depth}` | Config: `dataset.point_source` or `points.point_source` | Point source. Defaults to `stereo` for KITTI-360 and `lidar` for NVIDIA NCore. |
| `--camera-depth-pair LEFT,RIGHT` | Config: `dataset.camera_depth_pair` | NVIDIA camera pair used by `camera_depth`. |
| `--start-at {prepare,train,stereo,bucket,fit,inspect,uncertainty,map-viz,render,nbv,nbv-viz,bundle}` | `prepare` | First stage to run. |
| `--stop-after {prepare,train,stereo,bucket,fit,inspect,uncertainty,map-viz,render,nbv,nbv-viz,bundle}` | Parser: `inspect`; profile configs usually use `bundle` | Last stage to run. Use `bundle` for the full curated run output. |
| `--run-output-root RUN_OUTPUT_ROOT` | Config: `outputs.run_root` | Optional root for curated outputs. When set, stage outputs are derived under `<root>/<drive>/`. |
| `--dry-run` | `false` | Print the Docker/stage commands without executing them. |

Stage order is:

```text
prepare -> train -> stereo -> bucket -> fit -> inspect -> uncertainty -> map-viz -> render -> nbv -> nbv-viz -> bundle
```

## Orchestration

| Argument | Default | Description |
| --- | --- | --- |
| `--torch-container TORCH_CONTAINER` | Empty | Concrete container name/id for Torch stages. When set, the runner uses `docker exec` instead of `docker compose exec` for Torch. |
| `--jax-container JAX_CONTAINER` | Empty | Concrete container name/id for JAX stages. When set, the runner uses `docker exec` instead of `docker compose exec` for JAX. |
| `--use-service-labels` | `true` | Resolve containers by Docker Compose labels and use `docker exec`. This is the default container-side mode. |
| `--label-project LABEL_PROJECT` | `VBOGS_COMPOSE_PROJECT` or auto-detected | Compose project label to use with `--use-service-labels`. |

The `vbogs-pipeline` service is also GPU-enabled, so after the stack is
running you can check host GPU visibility from the pipeline container:

```bash
docker compose --project-directory . -f docker/compose/compose.yml -f docker/compose/dev.yml exec vbogs-pipeline nvidia-smi
```

## Google Drive Upload

The pipeline image includes `rclone` and `scripts/upload_google_drive.py`.
Pass `--upload-google-drive` to `scripts/run_pipeline.sh` or set
`upload.enabled: true` in the pipeline config. The upload runs after all
selected stages succeed. By default the source is:

```text
outputs/v1_0/<drive>.zip
```

Config example:

```yaml
upload:
  enabled: true
  remote: vbogs_gdrive
  dest: experiments
  folder_id: <google-drive-folder-id>
  service_account_file: /run/secrets/vbogs-google-drive-service-account.json
  scope: drive
  rclone_args: --progress --checksum
  dry_run: false
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

| CLI/config/env | Description |
| --- | --- |
| `--gdrive-dest` / `upload.dest` / `VBOGS_GDRIVE_DEST` | Destination path inside the configured remote/root folder. Empty means the folder root. |
| `--gdrive-source` / `upload.source` / `VBOGS_GDRIVE_SOURCE` | Override the upload source file or directory. |
| `--gdrive-folder-id` / `upload.folder_id` / `VBOGS_GDRIVE_FOLDER_ID` | Private Drive folder id to use as the rclone root. |
| `--gdrive-service-account-file` / `upload.service_account_file` / `VBOGS_GDRIVE_SERVICE_ACCOUNT_FILE` | Service-account JSON path inside the pipeline container. |
| `VBOGS_GDRIVE_SERVICE_ACCOUNT_CREDENTIALS` | Raw service-account JSON. Keep this in environment/secrets, not in pipeline configs. |
| `--gdrive-scope` / `upload.scope` / `VBOGS_GDRIVE_SCOPE` | rclone Drive scope. Defaults to `drive` for service-account uploads. |
| `--gdrive-rclone-args` / `upload.rclone_args` / `VBOGS_GDRIVE_RCLONE_ARGS` | Extra arguments appended to the rclone command, for example `--progress --checksum`. |
| `--gdrive-dry-run` / `upload.dry_run` / `VBOGS_GDRIVE_DRY_RUN` | Print the rclone upload command without transferring. |

Manual upload example from inside `vbogs-pipeline`:

```bash
python scripts/upload_google_drive.py \
  --config configs/pipeline/portainer.yaml \
  --folder-id <google-drive-folder-id> \
  --service-account-file /run/secrets/vbogs-google-drive-service-account.json
```

## Realtime Renderer Viewer

The compose stack publishes `vbogs-vbgs-render` for browser-based RGB and
uncertainty inspection after the scene and uncertainty artifacts exist. Start
the viewer inside `vbogs-vbgs-render`:

```bash
python scripts/view_octree_anygs.py \
  --drive 2013_05_28_drive_0007_sync \
  --resolution 4
```

The default host URL is:

```text
http://<host>:8071
```

Set `VBOGS_RENDER_VIEWER_HOST_BIND` or `VBOGS_RENDER_VIEWER_HOST_PORT` when the
viewer should bind to a different interface or host port.

## File Browser Access

The compose stack includes a `vbogs-filebrowser` sidecar using the
`filebrowser/filebrowser` image. It provides browser-based, read-only access to
the shared project and artifact volumes in both local Docker Compose and
Portainer deployments.

The default URL is:

```text
http://<host>:8088
```

Relevant stack variables:

| Variable | Description |
| --- | --- |
| `VBOGS_FILEBROWSER_IMAGE` | File Browser image, default `filebrowser/filebrowser:v2-s6`. |
| `VBOGS_FILEBROWSER_HOST_BIND` | Host bind address, default `0.0.0.0`. |
| `VBOGS_FILEBROWSER_HOST_PORT` | Host HTTP port, default `8088`. |
| `VBOGS_FILEBROWSER_BASE_URL` | Optional reverse-proxy subpath. |
| `VBOGS_FILEBROWSER_PUID` / `VBOGS_FILEBROWSER_PGID` | Optional UID/GID for the File Browser process and its database/config volumes. |

On first boot, read the generated `admin` password from the
`vbogs-filebrowser` logs. The mounted paths are `/project`, `/outputs`,
`/data`, `/COLMAP`, `/OCTREE-ANYGS`, and `/generated_configs`.

## KITTI-360 Inputs

These override the default source-data discovery used by the prep and stereo
stages.

| Argument | Default | Description |
| --- | --- | --- |
| `--raw-root RAW_ROOT` | Auto-detect | Root containing KITTI-360 rectified stereo images. |
| `--poses-root POSES_ROOT` | Auto-detect | Root containing KITTI-360 pose text files. |
| `--calibration-dir CALIBRATION_DIR` | Auto-detect | Directory containing KITTI-360 calibration text files. |

## NVIDIA NCore Inputs

NVIDIA PhysicalAI AV clips must already be converted to NCore V4. The compose
stack mounts them at `/workspace/VBOGS/data/NVIDIA-PhysicalAI-AV-NCore`.

| Argument | Default | Description |
| --- | --- | --- |
| `--ncore-root NCORE_ROOT` | Auto-detect | Root containing converted NCore clips. |
| `--camera-id CAMERA_ID` | `camera_front_wide_120fov` | Camera subset used for training and LiDAR coloring. |
| `--ncore-lidar-id LIDAR_ID` | `lidar_top_360fov` | LiDAR sensor used for sparse seeding and LiDAR point export. |
| `--camera-depth-pair LEFT,RIGHT` | `camera_front_wide_120fov,camera_front_tele_30fov` | Camera pair for camera-depth export. |

## `prepare`

Runs the selected dataset adapter in `vbogs-torch` and writes a COLMAP-style
dataset under `/data/COLMAP/<scene-id>`. KITTI uses
`scripts/prepare_kitti360_colmap.py`; NVIDIA NCore uses
`scripts/prepare_nvidia_ncore_colmap.py`.

| Argument | Default | Description |
| --- | --- | --- |
| `--frame-step FRAME_STEP` | Config: `1` | Keep every Nth frame from the drive. Higher values are faster and smaller. |
| `--max-frames MAX_FRAMES` | Config: `1000` | Maximum number of frames to prepare. `0` means no cap. |
| `--copy-mode {symlink,copy}` | `symlink` | How images are placed in the prepared dataset. `symlink` is faster and saves space when supported. |
| `--seed-mode {stereo,lidar,random}` | `stereo` | How the initial point cloud is seeded for Octree-AnyGS ingest. `stereo` maps to LiDAR for NVIDIA NCore. |

## `train`

Runs `scripts/train_octree_anygs.py` in `vbogs-torch`. The generated config is
written under `generated_configs/`, and Octree-AnyGS outputs go under
`/data/OCTREE-ANYGS/<drive>/<timestamp>/`.

| Argument | Default | Description |
| --- | --- | --- |
| `--gpu GPU` | `0` | GPU id passed to the Octree-AnyGS training wrapper. |
| `--resolution RESOLUTION` | `4` | Octree-AnyGS image divisor. Higher values reduce memory use and image fidelity. |
| `--iterations ITERATIONS` | Profile-specific config; parser: `15000` | Number of training iterations. |
| `--llffhold LLFFHOLD` | `8` | Held-out test frame cadence used by the Octree-AnyGS data loader. |
| `--gaussian-type {implicit3D,explicit3D}` | `implicit3D` | Octree-AnyGS Gaussian representation. `implicit3D` is the neural default; `explicit3D` uses explicit SH 3D Gaussians. |
| `--feat-dim FEAT_DIM` | `16` | Neural anchor feature dimension. Lower values reduce VRAM pressure. Ignored for `explicit3D`. |
| `--base-layer BASE_LAYER` | `9` | LoD base layer. Lower values reduce anchor count and memory. |
| `--visible-threshold VISIBLE_THRESHOLD` | `0.02` | LoD pruning visibility threshold. |
| `--train-port TRAIN_PORT` | Auto | Octree-AnyGS network GUI port. The wrapper defaults to `6009 + GPU index`, so GPU 1 uses `6010`. |
| `--write-config-only` | `false` | Generate the Octree-AnyGS YAML config and skip training. |

## `stereo` / Point-Cloud Export

Runs `scripts/export_points_world.py` in `vbogs-torch` and writes world-frame
points under `data/points_world/<scene-id>/`. The stage name remains `stereo`
for backward compatibility with existing command slices.

| Argument | Default | Description |
| --- | --- | --- |
| `--matcher {sgbm,raft}` | `sgbm` | Stereo matcher backend. `raft` is reserved for a future provider unless installed/implemented. |
| `--pixel-step PIXEL_STEP` | `1` | Pixel subsampling step for point export. Higher values reduce density and runtime. |
| `--max-points-per-frame MAX_POINTS_PER_FRAME` | `250000` | Per-frame cap on exported world points. |
| `--write-ply` | Config: `true` | Also write a PLY point cloud for quick visual inspection and the curated bundle. |
| `--point-source {stereo,lidar,camera_depth}` | Dataset default | Selects KITTI stereo, NVIDIA LiDAR, or NVIDIA camera-depth export. |

The point-cloud stage also receives `--max-frames`, KITTI input overrides, and
NVIDIA NCore sensor options as applicable.

## `bucket`

Runs `scripts/bucket_points.py` in `vbogs-torch` and writes packed point-to-anchor
assignments under `data/m4/<drive>/`.

| Argument | Default | Description |
| --- | --- | --- |
| `--model-path MODEL_PATH` | Latest run under `/data/OCTREE-ANYGS/<drive>` | Explicit Octree-AnyGS model/run directory to bucket against. |
| `--bucket-iteration BUCKET_ITERATION` | `-1` | Checkpoint iteration to load. `-1` means use the latest available checkpoint. |
| `--bucket-point-chunk-size BUCKET_POINT_CHUNK_SIZE` | `1000000` | Number of world points processed per bucketing chunk. Lower values reduce peak memory. |
| `--bucket-max-points BUCKET_MAX_POINTS` | `0` | Optional deterministic cap on exported points used for M4 bucketing/fitting. `0` keeps all points. |

## `fit`

Runs `scripts/fit_anchors.py` in `vbogs-jax` and writes VBGS posterior files
under `data/m4/<drive>/`.

| Argument | Default | Description |
| --- | --- | --- |
| `--jax-device JAX_DEVICE` | `0` | JAX device index used for VBGS fitting. |
| `--fit-mode {batched,loop}` | `batched` | Fit implementation. `batched` is the normal path; `loop` is simpler but slower. |
| `--batch-size BATCH_SIZE` | `5000` | VBEM sufficient-stat batch size and default padded-point budget input for batched fitting. |
| `--batch-buckets BATCH_BUCKETS` | `64,128,256,512,1024,2048,4096,5000` | Comma-separated point-count buckets used by `batched` mode. Anchors are padded to the smallest bucket that fits. |
| `--no-auto-extend-buckets` | `false` | Disable automatic dense-tail bucket extension for anchors above the largest configured bucket. |
| `--vmap-group-size VMAP_GROUP_SIZE` | `64` | Group size for vectorized JAX fitting work. |
| `--max-padded-points-per-group MAX_PADDED_POINTS_PER_GROUP` | `0` | Maximum padded anchor-points per batched call. `0` means `vmap_group_size * batch_size`. |
| `--log-every LOG_EVERY` | `100` | Progress logging interval. |

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
posterior artifacts remain in their native data paths and are referenced by
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
| `pipeline` | `drive`, `start_at`, `stop_after`, `dry_run` |
| `inputs` | `raw_root`, `poses_root`, `calibration_dir` |
| `prepare` | `frame_step`, `max_frames`, `copy_mode`, `seed_mode` |
| `train` | `gpu`, `resolution`, `iterations`, `llffhold`, `gaussian_type`, `feat_dim`, `base_layer`, `visible_threshold`, `port`, `write_config_only`, `skip_stack_check` |
| `stereo` | `matcher`, `pixel_step`, `max_points_per_frame`, `write_ply` |
| `bucket` | `model_path`, `bucket_iteration`, `point_chunk_size`, `max_points` |
| `fit` | `jax_device`, `fit_mode`, `batch_size`, `batch_buckets`, `no_auto_extend_buckets`, `vmap_group_size`, `max_padded_points_per_group`, `log_every` |
| `inspect` | `top_k`, `sample_points`, `anchor_id`, `export_ply` |
| `uncertainty` | `u_max`, `no_histogram` |
| `map_viz` | `output_dir`, `vmin`, `vmax`, `percentile_low`, `percentile_high`, `observed_only`, `no_split_levels`, `no_trajectory` |
| `render` | `split`, `resolution`, `max_views`, `colormap`, `vmin`, `vmax`, `output_dir` |
| `nbv` | `candidate_source`, `max_candidates`, `top_k`, `save_top_images`, `force_all_levels`, `output_dir` |
| `outputs` | `run_root` |
| `upload` | `enabled`, `source`, `remote`, `dest`, `folder_id`, `service_account_file`, `scope`, `rclone_args`, `dry_run` |
| `orchestration` | `torch_container`, `jax_container`, `use_service_labels`, `label_project` |
