# Configuration

`scripts/run_drive_pipeline.py` loads `configs/pipeline/default.yaml` by
default. Pass a different file with `--config`, or disable config loading with
`--config ""`.

CLI flags override config values.

## Profiles

| File | Intended use | Typical outputs |
| --- | --- | --- |
| `configs/pipeline/dev.yaml` | Local Docker Compose development stack | local `outputs/v1_0/<drive>/` via bind mount |
| `configs/pipeline/portainer.yaml` | Portainer deployment | `vbogs-outputs` volume at `/workspace/VBOGS/outputs` |
| `configs/pipeline/default.yaml` | Backward-compatible default profile | depends on active compose mounts |

## Top-Level Sections

| Section | Purpose |
| --- | --- |
| `pipeline` | drive id, stage slice, dry-run behavior |
| `outputs` | curated run output root |
| `upload` | Google Drive/rclone upload behavior |
| `inputs` | KITTI-360 source-data overrides |
| `prepare` | COLMAP-style data preparation |
| `train` | Octree-AnyGS training |
| `stereo` | stereo point-cloud export |
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
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/my-local.yaml \
  --use-service-labels
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
| `VBOGS_DRIVE` | Drive id for autorun |
| `VBOGS_PIPELINE_AUTORUN` | Set to `1` to run pipeline on service start |
| `VBOGS_PIPELINE_CONFIG` | Config path for autorun |
| `VBOGS_PIPELINE_ARGS` | Extra CLI args for autorun |
| `VBOGS_TORCH_IMAGE` | Torch image name |
| `VBOGS_JAX_IMAGE` | JAX image name |
| `VBOGS_PIPELINE_IMAGE` | Pipeline image name |
| `VBOGS_TORCH_CUDA_ARCH_LIST` | CUDA arch list for Torch image build; `auto` detects GPU 0 |
| `VBOGS_TORCH_MAX_JOBS` | Torch build parallelism |
| `VBOGS_GDRIVE_UPLOAD` | Enable post-run Google Drive upload |
| `VBOGS_TRANSFER_AUTHORIZED_KEYS` | Enable read-only SSH/SFTP transfer service |

## Argument Reference

For every config key and CLI override, use
[Pipeline Arguments](../documentation/RUN_DRIVE_PIPELINE_ARGS.md).
