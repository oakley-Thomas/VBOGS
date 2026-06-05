# Configuration

`scripts/run_pipeline.sh` loads `configs/pipeline/default.yaml` through the
internal pipeline runner. Pass a different file with `--config`, or disable
config loading with `--config ""`.

CLI flags override config values.

## Profiles

| File | Intended use | Typical outputs |
| --- | --- | --- |
| `configs/pipeline/dev.yaml` | Local Docker Compose development stack | `outputs/v1_0/<drive>/` inside the `vbogs-outputs` volume |
| `configs/pipeline/portainer.yaml` | Portainer deployment | `outputs/v1_0/<drive>/` inside the `vbogs-outputs` volume |
| `configs/pipeline/nvidia_ncore_dev.yaml` | Local NVIDIA PhysicalAI AV NCore development | `outputs/v1_0/<scene-id>/` inside the `vbogs-outputs` volume |
| `configs/pipeline/default.yaml` | Backward-compatible default profile | depends on active compose mounts |

## Top-Level Sections

| Section | Purpose |
| --- | --- |
| `dataset` | dataset adapter, scene id, NVIDIA NCore root, camera ids, point source |
| `pipeline` | drive id, stage slice, dry-run behavior |
| `outputs` | curated run output root |
| `upload` | Google Drive/rclone upload behavior |
| `inputs` | KITTI-360 source-data overrides |
| `prepare` | COLMAP-style data preparation |
| `train` | Octree-AnyGS training |
| `points` | preferred point-cloud export settings |
| `stereo` | backward-compatible alias for KITTI stereo point-cloud settings |
| `bucket` | point-to-anchor bucketing |
| `fit` | VBGS fitting |
| `inspect` | posterior inspection |
| `uncertainty` | scalar uncertainty computation |
| `map_viz` | anchor uncertainty PLY exports |
| `render` | RGB/uncertainty diagnostic views |
| `nbv` | candidate scoring |
| `orchestration` | compose/container lookup behavior |

## Minimal Local Override Example

```yaml
pipeline:
  drive: 2013_05_28_drive_0007_sync
  start_at: prepare
  stop_after: bundle

outputs:
  run_root: outputs/v1_0

train:
  gpu: "0"
  resolution: 4
  iterations: 30000

fit:
  jax_device: 0
```

Run it with:

```bash
scripts/run_pipeline.sh \
  --config configs/pipeline/my-local.yaml
```

## Dev vs Portainer Training Defaults

| Setting | Dev profile | Portainer profile |
| --- | --- | --- |
| `train.resolution` | `4` | `2` |
| `train.iterations` | `30000` | `90000` |
| `train.gaussian_type` | `explicit3D` | `explicit3D` |
| `train.base_layer` | `9` | `10` |
| `train.visible_threshold` | `0.02` | `0.01` |
| `bucket.max_points` | `10000000` | `0` |

The dev profile is meant to finish on smaller local hardware. The Portainer
profile is the higher-quality server path.

## Environment Variables

The compose stack reads these frequently used variables:

| Variable | Use |
| --- | --- |
| `VBOGS_DRIVE` | Drive id used by manual upload helpers |
| `VBOGS_PIPELINE_CONFIG` | Default config path available inside `vbogs-pipeline` |
| `VBOGS_TORCH_IMAGE` | Torch image name |
| `VBOGS_JAX_IMAGE` | JAX image name |
| `VBOGS_VBGS_RENDER_IMAGE` | Base VBGS render image name |
| `VBOGS_PIPELINE_IMAGE` | Pipeline image name |
| `VBOGS_TORCH_CUDA_ARCH_LIST` | CUDA arch list for Torch image build; `auto` detects GPU 0 |
| `VBOGS_TORCH_MAX_JOBS` | Torch build parallelism |
| `VBOGS_RENDER_CUDA_ARCH_LIST` | CUDA arch list for the base VBGS render image |
| `VBOGS_RENDER_MAX_JOBS` | Base VBGS render image build parallelism |
| `VBOGS_RENDER_VIEWER_HOST_BIND` | Host bind address for the browser viewer served from `vbogs-vbgs-render`, default `0.0.0.0` |
| `VBOGS_RENDER_VIEWER_HOST_PORT` | Host HTTP port for the browser viewer served from `vbogs-vbgs-render`, default `8071` |
| `VBOGS_GDRIVE_UPLOAD` | Enable post-run Google Drive upload |
| `VBOGS_FILEBROWSER_IMAGE` | File Browser sidecar image, default `filebrowser/filebrowser:v2-s6` |
| `VBOGS_FILEBROWSER_HOST_BIND` | File Browser host bind address, default `0.0.0.0` |
| `VBOGS_FILEBROWSER_HOST_PORT` | File Browser host HTTP port, default `8088` |
| `VBOGS_FILEBROWSER_BASE_URL` | Optional reverse-proxy subpath for File Browser |
| `VBOGS_FILEBROWSER_PUID` / `VBOGS_FILEBROWSER_PGID` | Optional UID/GID overrides for File Browser database/config volumes |

Pull-only and Portainer stacks share a `vbogs-repo` volume mounted at
`/workspace/VBOGS`. Run `vbogs-bootstrap-repo` once after stack creation so the
runtime services and File Browser see the same checkout. The local dev overlay
bind-mounts only the working tree instead.

## Argument Reference

For every config key and CLI override, use
[Pipeline Arguments](../documentation/RUN_DRIVE_PIPELINE_ARGS.md).
